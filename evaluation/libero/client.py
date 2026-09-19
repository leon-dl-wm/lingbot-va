"""LIBERO simulation evaluation client: runs LIBERO benchmark tests by connecting to the
LingBot-VA inference server over websocket.

Role in the evaluation loop (server side is ``wan_va/wan_va_server.py``, launched with the
libero config):
    1. Build the LIBERO off-screen rendering environment (OffScreenRenderEnv) and reset it
       to the benchmark-provided initial state;
    2. ``infer(dict(reset=True, prompt=...))``: the server clears its KV cache and encodes
       the language instruction;
    3. Send the initial observation (agentview + eye-in-hand camera images) to get the
       action chunk ``action`` with shape [C, F, N]: C=7 (LIBERO single-arm EEF 6 dims +
       1 gripper dim; the server maps the 30-dim unified action space back to these 7 dims
       via ``used_action_channel_ids`` and denormalizes with q01/q99), F=number of latent
       frames (2), N=control substeps per latent frame (16);
    4. Execute the actions substep by substep, collecting one real-observation keyframe
       every N/4 substeps (the Wan VAE downsamples time by 4x: 1 latent frame corresponds
       to 4 real video frames);
    5. Feed the keyframes back with ``compute_kv_cache=True``: the server first discards
       the imagined-frame cache, then writes the real observations (update_cache=2),
       completing closed-loop correction, and the next infer round begins; this repeats
       until done or the environment timestep exceeds 800;
    6. Save the side-by-side two-camera video (filename carries the True/False success
       marker) and write per-task success-rate JSON files.

Typical usage (see launch_client.sh)::

    python evaluation/libero/client.py --libero-benchmark libero_10 \
        --port 29056 --test-num 50 --task-range 0 10 --out-dir outputs/libero
"""
import numpy as np
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
import argparse
from libero.libero import benchmark
import time
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.utils import write_json
import os
import imageio
import cv2


def save_video(real_obs_list, save_path, fps=15, video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]):
    """Save the observation sequence as an mp4 video with cameras concatenated horizontally.

    Args:
        real_obs_list: List of observation dicts, each containing [H,W,3] uint8 images per camera.
        save_path: Output mp4 path.
        fps: Video frame rate.
        video_names: Camera key names to concatenate, ordered left to right.
    """
    if not real_obs_list:
        print("❌ No real observation frames")
        return

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)
    
    print(f"Saving video: {len(real_obs_list)} frames...")

    # Per frame: resize every camera image to the agentview size, then hstack them into one row
    final_frames = [
        np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)
        for obs in real_obs_list
    ]

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Video saved to: {save_path}")


def construct_single_env(env_args):
    """Build the LIBERO off-screen rendering environment, retrying up to 5 times on failure
    (5 seconds apart).

    Args:
        env_args: OffScreenRenderEnv constructor arguments (bddl file path, camera height/width).
    Returns:
        The environment instance; None if all 5 attempts fail (rendering resources
        occasionally fail to initialize, and retrying usually recovers).
    """
    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as e:
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    """
    Extract agentview and eye_in_hand images from raw env obs dict.

    Avoids torch round-trip: the env already returns uint8 numpy arrays [H, W, C].
    We just flip the vertical axis ([::-1]) and make a contiguous copy once.

    Notes: Extracts the two camera images from the raw env obs and renames them to the keys
    agreed with the server. LIBERO's MuJoCo-rendered images are upside down, so they are
    flipped along the vertical axis ([::-1]); this operates directly on the numpy uint8
    arrays ([H,W,3]) with a single contiguous copy, avoiding a torch tensor round-trip.
    """
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {"observation.images.agentview_rgb": agentview, "observation.images.eye_in_hand_rgb": eye_in_hand}


def init_single_env(env_in, init_state):
    """Reset the environment to the benchmark-provided initial state and return the initial
    observation.

    First env.reset(), then set_init_state (guaranteeing reproducible object placement),
    then execute 5 all-zero 7-dim actions to let the simulation settle; the last frame's
    observation is used as the model's first input.

    Args:
        env_in: LIBERO environment instance.
        init_state: A single initial-state array from benchmark.get_task_init_states.
    Returns:
        The initial observation dict (format see :func:`_extract_obs`).
    """
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.] * 7)
    return _extract_obs(obs)


def env_one_step(env_in, action):
    """Execute a single environment step (7-dim EEF control: 6-dim pose delta + gripper);
    returns (extracted observation, done)."""
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def run_one(model, libero_benchmark, task_idx, out_dir, episode_idx):
    """Run a single evaluation episode of the given task and save its video.

    Flow: build env -> reset to the initial state and grab the initial observation ->
    server reset(prompt) -> main loop {infer to get an action chunk -> execute substep by
    substep -> collect a keyframe every N/4 substeps -> feed back via compute_kv_cache},
    until done or the environment timestep exceeds 800 (LIBERO's per-episode step limit).

    Args:
        model: WebsocketClientPolicy client.
        libero_benchmark: Benchmark name (e.g. "libero_10").
        task_idx: Task index within the benchmark.
        out_dir: Result root directory.
        episode_idx: Episode index (taken modulo the initial-state array so different
            episodes use different initial placements).
    Returns:
        bool: whether this episode succeeded (done).
    """
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    prompt = benchmark_instance.get_task(task_idx).language
    env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    first_obs = init_single_env(cur_env, init_states[episode_idx % init_states.shape[0]])

    # Make the server clear its KV cache and encode the language instruction
    # (the return value is unused; this call only resets the session)
    ret = model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = []
    done = False
    first = True
    # LIBERO caps each episode at 800 environment timesteps; exceeding it counts as failure
    while cur_env.env.timestep < 800:
        # Only the first round carries the initial observation; afterwards first_obs stays
        # None and the server continues autoregressive inference purely from the real-obs
        # KV cache written via compute_kv_cache
        ret = model.infer(dict(obs=first_obs, prompt=prompt))
        action = ret['action']

        key_frame_list = []
        # action shape [C=7, F, N]: F=number of latent frames, N=control substeps per latent frame.
        # The Wan VAE downsamples time by 4x: 1 latent frame corresponds to 4 real video frames,
        # so one real-observation keyframe is collected every N/4 substeps, keeping the
        # feedback rate aligned with the latent frame rate.
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        # The first round skips latent frame 0: its actions correspond to the "history"
        # (aligned with the initial observation; the training data pipeline pads zero
        # actions at the beginning), so executing them would repeat; all frames of later
        # rounds are future actions.
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = action[:, i, j]
                observes, done = env_one_step(cur_env, ee_action)
                if done:
                    break
                if (j+1) % action_per_frame == 0:
                    full_obs_list.append(observes)
                    key_frame_list.append(observes)

            if done:
                break

        first = False

        if done:
            break
        else:
            # Feed the real-observation keyframes back: the server first drops imagined
            # frames via clear_pred_cache, then writes the real observations with
            # update_cache=2 (closed-loop correction); state carries the action chunk just
            # executed, serving as the clean action-condition segment in the KV cache
            # (aligned with the training sequence layout).
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))

    # The video filename carries the True/False success marker so statistics scripts can
    # aggregate success rates directly from filenames
    out_file = Path(out_dir) / libero_benchmark / f"{task_idx}_{prompt.replace(' ', '_')}" / f"{episode_idx}_{done}.mp4"
    out_file.parent.mkdir(exist_ok=True, parents=True)

    save_video(
        real_obs_list=full_obs_list,
        save_path=out_file,
        fps=60,
        video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]
    )

    cur_env.close()
    return done


def run(libero_benchmark, port, out_dir, test_num, task_range=None):
    '''
        task_range: [start, end) for splitting tasks

        Notes: Main evaluation loop — iterates over every task in task_range, runs
        test_num episodes per task, and after each episode prints the running success
        rate and writes ``<out_dir>/<benchmark>_<task_idx>.json``.
        task_range enables multi-process work splitting: different workers evaluate
        disjoint task intervals. When video_save_root_dict is not None, evaluation can
        resume (scanning saved videos to restore the success count and skipping finished
        episodes); it defaults to None, i.e. the feature is disabled.

        Args:
            libero_benchmark: Benchmark name (libero_10/libero_goal/libero_spatial/libero_object).
            port: Inference server websocket port.
            out_dir: Result root directory.
            test_num: Number of episodes to evaluate per task.
            task_range: [start, end) task index interval; None means evaluate all tasks.
    '''
    if task_range is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        num_tasks = benchmark_instance.get_num_tasks()
        progress_bar = tqdm(range(num_tasks), total=num_tasks)
    else:
        assert len(task_range) == 2, f'task_range: [start, end) for splitting tasks, however, task_range: {task_range}'
        num_tasks = task_range[1] - task_range[0]
        progress_bar = tqdm(range(task_range[0], task_range[1]), total=num_tasks)

    print(f"#################### Use benchmark: {libero_benchmark}, num_tasks: {num_tasks} #############")
    # Establish the websocket connection (blocks internally until the server is ready)
    model = WebsocketClientPolicy(port=port)

    video_save_root_dict = None

    episode_list = range(test_num)
    for task_idx in progress_bar:
        if video_save_root_dict is not None and task_idx in video_save_root_dict:
            video_save_list = os.listdir(os.path.join(out_dir, libero_benchmark, video_save_root_dict[task_idx]))
            video_states = [1 for file in video_save_list if file.split('_')[1].split('.')[0] == 'True']
            succ_num = float(len(video_states))
            episode_list = range(len(video_save_list), test_num)
        else:
            succ_num = 0.

        for episode_idx in tqdm(episode_list, total=len(episode_list)):
            res_i = run_one(model, libero_benchmark, task_idx, out_dir, episode_idx)
            succ_num += res_i
            succ_rate = succ_num / (episode_idx + 1)
            print(f"Success rate: {succ_rate}, success num: {succ_num}, total num: {episode_idx + 1}")
            out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            write_json({
                "succ_num": succ_num,
                "total_num": episode_idx + 1.,
                "succ_rate": succ_rate,
                }, out_file
            )


def main():
    """Command-line entry: parse arguments (benchmark / task range / port / episode count /
    output directory) and start the evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    args = parser.parse_args()
    run(**vars(args))
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
