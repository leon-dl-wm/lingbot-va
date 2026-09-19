# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Standalone LeRobot v2.1-format reader (decoupled from the lerobot package,
# see project.md 11.4 Plan B): reads meta/info.json + meta/episodes.jsonl
# (including the custom `action_config` field) and per-episode action parquet
# files directly, so it works with any installed lerobot version. Videos are
# consumed as pre-extracted latents (.pth) and never decoded here.
"""LeRobot v2.1 latent dataset reader (the core of the training data pipeline).

Position in the architecture: provides training samples for ``wan_va/train.py``. Videos are
never decoded here — they have been encoded offline by the Wan2.2 VAE into
``latents/chunk-XXX/<cam>/episode_{i}_{s}_{e}.pth``, and text embeddings (UMT5) are likewise
pre-computed offline inside the .pth files. This module only reads the latents, loads raw
actions from parquet, and performs three key processing steps:

1. **Action alignment** (``_action_post_process``): aligns the action sequence with the
   sampled frame_ids of the latent frames; pads ``frame_stride*4`` zero actions at the
   beginning (latent frame 0 corresponds to "history" actions executed before the window
   starts, which are unobservable); robotwin data is first converted to relative poses via
   ``get_relative_pose``;
2. **Channel remapping**: ``inverse_used_action_channel_ids`` scatters the dataset's own
   16-dim actions into the correct slots of the 30-dim unified action space; remaining slots
   are zeroed with mask=False (excluded from the loss); q01/q99 normalization to [-1,1]
   followed by clipping to ±1.5;
3. **Multi-camera T-shape mosaic** (``_cat_video_latents``, env_type='robotwin_tshape'):
   cam_high at full resolution + both wrist cameras at half resolution are stitched into one
   large (3h/2, w) image, so a single stream handles all three cameras.

Sample granularity: every ``action_config`` segment (start_frame/end_frame) in episodes.jsonl
is one sample (``parse_meta`` expands them into ``new_metas``); with probability ``cfg_prob``
the text_emb is replaced by the empty embedding to train the CFG unconditional branch.

Public entry point: ``MultiLatentLeRobotDataset`` (aggregates all LeRobot repos under
dataset_path). The file ships a ``__main__`` self-test printing field shapes and action
statistics.
"""
import json
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
import multiprocessing
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
import pyarrow.parquet as pq

def recursive_find_file(directory, filename='info.json'):
    """Recursively find all files with the given name under a directory (used to scan the
    meta/info.json of multiple LeRobot repos).

    Args:
        directory: root directory to walk.
        filename: target file name, default 'info.json'.

    Returns:
        list[str]: full paths of all matching files; permission/other errors are printed,
        not raised.
    """
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    """Build a LatentLeRobotDataset for a single LeRobot repo (with config bound via partial,
    so it can be called by a process pool's map).

    Args:
        repo_id: repo root directory (must contain meta/info.json).
        config: dataset configuration (obs_cam_keys, norm_stat, cfg_prob, etc.).

    Returns:
        A LatentLeRobotDataset instance.
    """
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    """Scan all LeRobot repos under config.dataset_path and build their datasets (in parallel).

    Repos are discovered by recursively finding meta/info.json; building each repo (especially
    _load_actions reading all parquet files) is slow, so when more than 2 repos are found the
    datasets are constructed in a spawn process pool.

    Args:
        config: dataset configuration; must contain dataset_path.
        num_init_worker: maximum number of parallel construction processes.

    Returns:
        list[LatentLeRobotDataset]: one entry per discovered repo.
    """
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    if len(repo_list) <= 2:
        # Build in-process: forking pool workers from a parent that already has
        # torch/NCCL threads can deadlock (children inherit held locks).
        return [construct_func(repo_id) for repo_id in repo_list]
    # 'spawn' workers start clean without inherited locks (fork is unsafe here).
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(min(num_init_worker, len(repo_list))) as pool:
        datasets_out_lst = pool.map(construct_func, repo_list)
    return datasets_out_lst

def get_relative_pose(pose):
    """Convert a sequence of absolute poses into poses relative to the first one (robotwin action representation).

    Motivation: the numeric range of absolute poses depends on the robot's initial placement;
    relative poses have a more concentrated distribution and are more consistent across
    episodes, which helps normalization and learning.

    Args:
        pose: [N,7] array/tensor; each row is (x,y,z,qx,qy,qz,qw) — first 3 dims translation,
            last 4 dims quaternion (scipy xyzw order).

    Returns:
        torch.Tensor [N,7]: translation minus the first frame's translation; rotation is
        first_rot⁻¹ * rot (i.e. each frame's rotational delta relative to the first frame),
        still concatenated as (trans3, quat4).
    """
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    # Replicate the first frame's rotation N times for batched relative-rotation computation
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    # Relative rotation: left-multiply by the inverse of the first frame's rotation
    # (pulls the coordinate frame back to the first frame)
    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    """Aggregated dataset over multiple LeRobot repos: behaves like a single Dataset externally.

    Concatenates the LatentLeRobotDataset instances built by ``construct_lerobot_multi_processor``
    into one global index space: global idx -> (sub-dataset id, local idx), so the DataLoader /
    DistributedSampler only need to face this single dataset.

    Args:
        config: dataset configuration (must contain dataset_path pointing to the parent
            directory of multiple LeRobot repos).
        num_init_worker: number of processes for parallel sub-dataset construction.
    """
    def __init__(
        self,
        config,
        num_init_worker=128,
    ):
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        """Global sample count = sum of all sub-dataset sample counts (i.e. action_config segments)."""
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        """Build the global index mappings.

        Returns:
            (item_id_to_dataset_id, acc_dset_num):
            - item_id_to_dataset_id: {global idx: sub-dataset id};
            - acc_dset_num: {sub-dataset id: global idx of that dataset's first sample}
              (prefix sums, used to convert a global idx into a local idx).
        """
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        """Route the global idx to the local idx of the corresponding sub-dataset and fetch the sample (return format: see LatentLeRobotDataset.__getitem__)."""
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(torch.utils.data.Dataset):
    """Latent dataset for a single LeRobot v2.1 repo.

    One sample = one ``action_config`` segment (start_frame~end_frame) in episodes.jsonl.
    All metadata and actions (parquet) are loaded once at init; video latents are loaded
    on demand from .pth in __getitem__ (to avoid blowing up memory).

    Args:
        repo_id: repo root directory (must contain meta/info.json, meta/episodes.jsonl,
            data/chunk-*/, latents/chunk-*/).
        config: dataset configuration; key fields:
            - obs_cam_keys: list of camera keys (the 0th must be the full-resolution main
              camera cam_high);
            - norm_stat: {'q01','q99'} action normalization quantiles (from
              evaluation/robotwin/calc_stat.py);
            - cfg_prob: probability of replacing text_emb with the empty embedding (trains
              the CFG unconditional branch);
            - empty_emb_path: path to the empty text embedding file;
            - env_type: 'robotwin_tshape' enables the T-shape multi-camera mosaic and the
              relative-pose conversion;
            - inverse_used_action_channel_ids: scatter indices mapping the 16-dim dataset
              actions into the 30-dim unified space.

    __getitem__ returns a dict:
        - latents [48,F,H,W]: multi-camera concatenated VAE latent (C=48, F=latent frames);
        - text_emb [L,D]: UMT5 text embedding (or the empty embedding with prob cfg_prob);
        - actions [30,F,16,1]: normalized unified-space actions (16 = action sub-steps per
          latent frame);
        - actions_mask [30,F,16,1] bool: valid-channel mask.
    """
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = Path(repo_id)
        if not (self.root / 'meta' / 'info.json').is_file():
            raise FileNotFoundError(
                f"meta/info.json not found under {self.root}; "
                "dataset_path must point to a local LeRobot v2.1-format repo")

        with open(self.root / 'meta' / 'info.json') as f:
            self.info = json.load(f)
        # chunks_size: number of episodes per chunk directory (LeRobot v2.1 directory layout)
        self.chunks_size = self.info.get('chunks_size', 1000)
        # parquet relative-path template (can be overridden by data_path in info.json)
        self.data_path_tpl = self.info.get(
            'data_path',
            'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet')

        # metadata → (episode, global frame offset) index → load all actions once
        self.episodes = self._load_episodes()
        self.episode_data_index = self._build_episode_data_index()
        self.all_actions = self._load_actions()

        self.latent_path = self.root / 'latents'
        # Empty text embedding: replaces text_emb with probability cfg_prob to train the
        # CFG unconditional branch
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        # Action normalization statistics (q01/q99 quantiles), shaped [1,C] for per-channel broadcasting
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        # Expand action_config segments into the sample list (validating latent files exist)
        self.parse_meta()

    def _load_episodes(self):
        """Read meta/episodes.jsonl (one JSON per episode, including the custom action_config field).

        Returns:
            list[dict]: episode metadata sorted by episode_index (containing length, tasks,
            action_config=[{start_frame,end_frame,...}, ...]).
        """
        episodes = []
        with open(self.root / 'meta' / 'episodes.jsonl') as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        episodes.sort(key=lambda ep: ep['episode_index'])
        return episodes

    def _build_episode_data_index(self):
        """Build the episode → global frame-range index (prefix sums of episode lengths).

        all_actions is one big array of all episodes' actions concatenated in order; this
        index converts (episode_index, local frame number) into a global row number in
        all_actions.

        Returns:
            dict: {'from': [n_ep] first global frame number of each episode,
                   'to':   [n_ep] end global frame number of each episode (exclusive)}.
        """
        starts = np.zeros(len(self.episodes), dtype=np.int64)
        acc = 0
        for i, ep in enumerate(self.episodes):
            starts[i] = acc
            acc += ep['length']
        return {'from': starts, 'to': starts + np.array(
            [ep['length'] for ep in self.episodes], dtype=np.int64)}

    def get_episode_chunk(self, episode_index):
        """Return the chunk number an episode belongs to (episode_index // chunks_size).

        LeRobot v2.1 stores files in per-chunk directories (data/chunk-XXX/,
        latents/chunk-XXX/); both the parquet and the latent paths depend on this number.
        """
        return episode_index // self.chunks_size

    def _episode_parquet_path(self, episode_index):
        """Build the action parquet file path of an episode from the data_path template."""
        rel = self.data_path_tpl.format(
            episode_chunk=self.get_episode_chunk(episode_index),
            episode_index=episode_index)
        return self.root / rel

    def _load_actions(self):
        """Read the 'action' column of every episode parquet once and concatenate into a global action array.

        Only the action column is read (pyarrow column pruning avoids loading unrelated
        fields like video paths); per-episode row counts are validated against the meta
        length to prevent metadata/data misalignment.

        Returns:
            np.ndarray [total_frames, action_dim] float32: concatenated in episode order;
            use the from/to of episode_data_index to locate a single episode's row range.
        """
        actions = []
        for ep in tqdm(self.episodes, desc=f'loading actions [{self.root.name}]'):
            idx = ep['episode_index']
            table = pq.read_table(self._episode_parquet_path(idx),
                                  columns=['action'])
            arr = np.stack(table.column('action').to_pylist()).astype(np.float32)
            assert len(arr) == ep['length'], (
                f"episode {idx}: parquet rows {len(arr)} != meta length {ep['length']}")
            actions.append(arr)
        return np.concatenate(actions, axis=0)

    def parse_meta(self):
        """Expand episodes.jsonl into the sample list self.new_metas.

        Each episode's action_config may contain multiple time segments (sub-tasks); every
        segment (start_frame, end_frame, task text) forms one training sample. During
        expansion, _check_meta validates that the corresponding latent files are complete;
        segments with missing files are skipped (tolerating datasets whose offline latent
        extraction is incomplete).
        """
        out = []
        for value in self.episodes:
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        """Check whether all camera latent files required by a segment exist.

        Latent file naming convention: latents/chunk-{c:03d}/{cam}/episode_{i:06d}_{s}_{e}.pth,
        in one-to-one correspondence with action_config segments.

        Returns:
            bool: True if the .pth exists for every key in used_video_keys, else False
            (the segment is skipped).
        """
        episode_chunk = self.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        """Convert (episode, local frame number) into a global row number in the all_actions array."""
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        """Slice the raw high-frequency actions by the global frame range [start_frame, end_frame).

        Returns:
            dict: {'action': torch.Tensor [end-start, action_dim]} (in the dataset's raw
            action dimensionality).
        """
        return {'action': torch.from_numpy(self.all_actions[start_frame:end_frame])}

    def _flatten_latent_dict(self, latent_dict):
        """Flatten the two-level dict {cam_key: {field: value}} into a single-level {"cam_key.field": value} dict."""
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        """Load the latent .pth files of all cameras for one sample segment and flatten them.

        Each .pth contains latent (flattened [F*H*W, C]), latent_num_frames/height/width,
        text_emb, frame_ids (the original frame numbers of the segment's video sampled at
        the target fps), etc. (full field table in README "Step 3: Extract video latents").

        Returns:
            dict: {"{cam}.latent", "{cam}.latent_num_frames", "{cam}.frame_ids",
                   "{cam}.text_emb", ...}.
        """
        episode_chunk = self.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        """Multi-camera latent concatenation + random CFG text-embedding dropout.

        Each camera's latent is stored flattened as (f h w) c in the .pth; it is first
        restored to [F,H,W,C], then concatenated according to env_type:
        - 'robotwin_tshape' (T-shape mosaic): used_video_keys[0]=cam_high at full
          resolution, the two wrist cameras at half resolution each — the wrist cameras
          are first concatenated along width into a row as wide as cam_high, then stacked
          with cam_high along height ⇒ one "large image" [F, 3H/2, W, C] enters the
          single-stream transformer, so all three cameras share one set of attention
          (strictly consistent with the mosaic built by server._encode_obs at inference);
        - otherwise: all cameras are simply concatenated along width.

        Additionally, with probability cfg_prob the text_emb is replaced by the empty
        embedding (empty_emb) to train the classifier-free-guidance unconditional branch.

        Args:
            data_dict: output of _flatten_latent_dict (per-camera latents and text_emb).

        Returns:
            dict: {'latents': [F,H,W,C] concatenated latent, 'text_emb': [L,D]}.
        """
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        if self.config.env_type == 'robotwin_tshape':
            # T-shape mosaic: the (half-resolution) wrist cameras are concatenated along
            # width (dim=2) into one row, then stacked with cam_high (full resolution)
            # along height (dim=1) ⇒ [F,3H/2,W,C]
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            # Plain mode: concatenate all cameras horizontally along width
            cat_latent = torch.cat(latent_lst, dim=2)

        # All camera .pth files store the same text_emb; taking the first camera key's is enough
        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        # Drop the text condition with probability cfg_prob (swap in the empty embedding)
        # to train the CFG unconditional branch
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict
    
    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        """Action post-processing: temporal alignment → relative poses → 30-dim unified-space channel remapping → normalization (the most critical part of the data pipeline).

        Turns the dataset's raw high-frequency actions [T, act_dim] into the layout the model
        expects, [30, F, 16, 1] (30 = unified action space, F = latent frames, 16 = action
        sub-steps per latent frame), and produces a valid-channel mask of the same shape.

        Args:
            local_start_frame: segment start frame number (episode-local coordinates).
            local_end_frame: segment end frame number (not used directly; truncation is
                decided by required_action_num).
            latent_frame_ids: list of original frame numbers of the segment's video sampled
                at the target fps (length = number of sampled pixel frames N, usually N=4k+1).
            action: np.ndarray [T, act_dim], raw actions of the segment (16-dim for robotwin).

        Returns:
            (actions, actions_mask):
            - actions torch.FloatTensor [30, F, 16, 1]: unified-space actions normalized to
              [-1,1] (clipped to ±1.5) with invalid channels zeroed;
            - actions_mask torch.BoolTensor [30, F, 16, 1]: True = valid channel.
        """
        # frame_ids are original frame numbers after sampling: the first sampled frame may
        # lag behind the segment start, so drop the actions before it to align the action
        # sequence with the video sampling window
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        # Temporal sampling stride: how many raw action steps between two consecutive sampled
        # frames (= ori_fps/target_fps; 4 for robotwin ⇒ each latent frame covers 4*4=16
        # action sub-steps, i.e. action_per_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            # robotwin 16-dim layout: [left-arm EEF pose 7 | left gripper 1 | right-arm EEF
            # pose 7 | right gripper 1]. Poses are converted to poses relative to the first
            # frame (more concentrated distribution, consistent across episodes, better for
            # normalization); gripper openings keep their original values, then everything is
            # concatenated back into 16 dims in the original order
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        # Pad frame_stride*4 zero actions at the beginning (= the number of action sub-steps
        # per latent frame, 16): the action slot of latent frame 0 corresponds to the
        # "history" actions executed before the first observation frame was produced; they
        # lie outside this segment's window and are unobservable, so zeros are padded. After
        # padding, action block f aligns exactly with latent frame f (causal alignment:
        # action block f = the actions executed before reaching frame f)
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

        # Wan VAE temporal compression is 4x (causal compression: N=4k+1 pixel frames → k+1 latent frames)
        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        # Each latent frame covers 4 sampled pixel frames, each sampled frame spans frame_stride raw action steps
        required_action_num = latent_frame_num * frame_stride * 4

        # Truncate to whole blocks (trailing actions shorter than one latent frame are
        # dropped) and assert the length matches exactly
        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num


        # Append one all-zero column at the end (the 17th column, index 16) as the common
        # "fill value" for invalid slots
        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        # Channel remapping (scatter): inverse_used_action_channel_ids is a length-30 index
        # table, unified-space channel j → dataset action index i; unused slots hold
        # len(used)=16, pointing exactly at the zero column appended above ⇒ value 0 and
        # mask=False (robotwin uses only 16 of the 30 dims: 0~6 left-arm EEF + 28 left
        # gripper + 7~13 right-arm EEF + 29 right gripper)
        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        # q01/q99 quantile normalization to [-1,1], then clip to ±1.5 (tolerates slight
        # out-of-quantile overflow: keeps genuine action information beyond q01/q99 while
        # preventing extreme values from destabilizing training)
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = np.clip(action_aligned, -1.5, 1.5)
        # [F*16, 30] → [30, F, 16, 1]: C=30 unified channels, F=latent frames, 16=action
        # sub-steps per frame; the trailing 1 is the spatial-dim placeholder (action tokens
        # have no spatial structure, i.e. H=W=1)
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        # Force invalid channels to 0 (double insurance with the mask handling in
        # _add_noise/compute_loss)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def __getitem__(self, idx) -> dict:
        """Fetch sample idx (one action_config segment): latents + text_emb + aligned actions.

        Flow: read all cameras' latent .pth for the segment → convert local frame numbers to
        global row numbers and slice the actions → multi-camera concatenation
        (_cat_video_latents, incl. CFG text dropout) → action post-processing
        (_action_post_process) → permute latents from [F,H,W,C] to [C,F,H,W].

        Args:
            idx: sample index (taken modulo len(new_metas) as a defensive wrap-around).

        Returns:
            dict:
            - latents [48,F,H,W]: concatenated multi-camera VAE latent (after DataLoader
              batching: [B,48,F,H,W], fed directly to _add_noise in train.py);
            - text_emb [L,D]: UMT5 text embedding (or the empty embedding with prob cfg_prob);
            - actions [30,F,16,1]: normalized unified-space actions;
            - actions_mask [30,F,16,1] bool: valid-channel mask.
        """
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        # Keep local frame-number copies: latent files are named by local frame numbers,
        # while action slicing uses global row numbers
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        # Sampled frame numbers (in original-fps coordinates): the time base for action alignment
        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        # Local frame numbers → global row numbers into all_actions
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict['action'])

        # [F,H,W,C] → [C,F,H,W]: matches the model's channel-first input convention
        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        return out_dict

    def __len__(self):
        """Total sample count = number of action_config segments that passed latent-file validation."""
        return len(self.new_metas)

if __name__ == '__main__':
    # Self-test entry: build the demo dataset, print each field's shape/dtype for one sample
    # plus the total sample count, then iterate the DataLoader to collect the maximum token
    # count (F*H*W) and per-channel action mean/min/max (a quick sanity check that action
    # normalization lands within the expected [-1.5, 1.5] range)
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['demo_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
