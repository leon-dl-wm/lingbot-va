"""RoboTwin 2.0 simulation evaluation client: runs dual-arm manipulation task evaluations
by connecting to the LingBot-VA inference server over websocket.

Role in the evaluation loop (server side is ``wan_va/wan_va_server.py``, launched by
launch_server.sh): this script is the "environment side" — it builds the RoboTwin/Sapien
simulation environment, captures 3-camera observations (cam_high + left/right wrist),
formats them and sends them to the server; the server runs the "imagine future frames ->
infer actions" AR diffusion inference and returns an action chunk; this script executes the
actions substep by substep and periodically feeds real observations back to update the
server's KV cache (closed-loop correction), forming the asynchronous execution protocol::

    reset(prompt) -> infer(initial obs) returns 32 actions (2 latent frames x 16 substeps)
    -> execute substep by substep (collect 1 real keyframe every 4 substeps, since the VAE
       downsamples time by 4x)
    -> compute_kv_cache feeds the keyframes back (replacing imagined frames) -> loop

Main logic:
- ``main``: loads task/embodiment/camera configs, creates result directories, connects to
  the server, and starts the evaluation;
- ``eval_policy``: the multi-seed loop — first verifies with the expert policy (play_once)
  that the task is solvable for the seed (expert_check), then evaluates the model; counts
  success rates and saves visualization videos and metrics JSON;
- Action format: the server returns ``action`` with shape [C, F, N] (C=action channels,
  F=latent frames, N=substeps per frame). C=16 is the dual-arm relative-pose
  representation ([xyz+quaternion+gripper]x2), which must be composed with the episode's
  initial EEF pose via ``add_init_pose`` to recover absolute poses; C=14 is the dual-arm
  Euler-angle representation ([xyz+euler+gripper]x2), which needs ``euler2quat`` to
  convert to quaternions and rearrange into 16 dims.

Typical usage (see launch_client.sh / launch_client_multigpus.sh)::

    python -m evaluation.robotwin.eval_polict_client_openpi \
        --config policy/ACT/deploy_policy.yml \
        --overrides --task_name adjust_bottle --task_config demo_clean \
        --ckpt_setting 0 --seed 0 --policy_name ACT --save_root ./results \
        --video_guidance_scale 5 --action_guidance_scale 1 \
        --test_num 100 --port 29056
"""
import sys
import os
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from pathlib import Path

robowin_root = Path("/path/to/your/robowin")
# The RoboTwin repo root must be added to sys.path and set as cwd: the task environment
# classes (envs.*) and the relative task_config paths both assume the robowin root as the
# working directory
if str(robowin_root) not in sys.path:
    sys.path.insert(0, str(robowin_root))


import os
os.chdir(robowin_root)

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb
from evaluation.robotwin.geometry import euler2quat
import numpy as np

from description.utils.generate_episode_instructions import *
import traceback

import imageio
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
from evaluation.robotwin.test_render import Sapien_TEST

def write_json(data: dict, fpath: Path) -> None:
    """Write data to a JSON file.

    Creates parent directories if they don't exist.

    Args:
        data (dict): The dictionary to write.
        fpath (Path): The path to the output JSON file.

    Notes: writes a result dict (e.g. success-rate metrics) to a JSON file, creating parent
    directories automatically; during evaluation it is overwritten after every episode so
    external scripts can monitor progress in real time.
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def add_title_bar(img, text, font_scale=0.8, thickness=2):
    """Add a black title bar with text above the image"""
    # Notes: overlays a 40-pixel-high black title bar on top of the image with centered
    # white text, used in the comparison video to label each row (real observation /
    # imagined video stream).
    # Args: img input image [H, W, 3] (uint8); text title string; font_scale/thickness font
    # scaling and line width. Returns: image with the title bar [H+40, W, 3].
    h, w, _ = img.shape
    bar_height = 40
    
    # Create black background bar
    title_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    
    # Calculate text position to center it
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (w - text_w) // 2
    text_y = (bar_height + text_h) // 2 - 5
    
    cv2.putText(title_bar, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return np.vstack([title_bar, img])

def quaternion_to_euler(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    quat: [rx, ry, rz, rw] format
    Return: [roll, pitch, yaw] (radians)

    Notes: scipy-based quaternion -> Euler angle conversion. The input quaternion is in
    **xyzw** order (scipy convention, opposite to geometry.py's wxyz); returns
    [roll, pitch, yaw] (radians) with the xyz rotation order. Used for visualization
    plotting only, not for control.
    """
    # scipy uses [x, y, z, w] format
    rotation = R.from_quat(quat)
    euler = rotation.as_euler('xyz', degrees=False)  # returns [roll, pitch, yaw]
    return euler

def visualize_action_step(action_history, step_idx, window=50):
    """
    Plot dual-arm action curves:
    Subplot 1: Left arm XYZ Position + Gripper
    Subplot 2: Left arm Euler angles (Roll, Pitch, Yaw) - converted from quaternion
    Subplot 3: Right arm XYZ Position + Gripper
    Subplot 4: Right arm Euler angles (Roll, Pitch, Yaw) - converted from quaternion
    
    Input data format: [left_x, left_y, left_z, left_rx, left_ry, left_rz, left_rw, left_gripper,
                   right_x, right_y, right_z, right_rx, right_ry, right_rz, right_rw, right_gripper]
    Total 16 dimensions

    Notes: plots the 16-dim dual-arm action history within a sliding window (the last
    `window` steps) as a 2x2 subplot grid (left/right arm XYZ position + gripper, and RPY
    angles converted from quaternions), renders the matplotlib canvas into a uint8 numpy
    image and returns it, for embedding into comparison video frames.
    """
    # Create four subplots, sharing the X-axis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8), dpi=100, sharex=True)
    
    # 1. Determine slice range
    start = max(0, step_idx - window)
    end = step_idx + 1
    
    # 2. Get data subset
    history_subset = np.array(action_history)[start:end]
    
    # 3. Generate X-axis based on actual data length
    actual_len = len(history_subset)
    x_axis = range(start, start + actual_len)
    
    if actual_len > 0 and history_subset.shape[1] >= 16:
        # Convert quaternions to Euler angles
        left_euler = []
        right_euler = []
        
        for action in history_subset:
            # Left arm quaternion to Euler angles
            left_quat = action[3:7]  # [rx, ry, rz, rw]
            left_rpy = quaternion_to_euler(left_quat)
            left_euler.append(left_rpy)
            
            # Right arm quaternion to Euler angles
            right_quat = action[11:15]  # [rx, ry, rz, rw]
            right_rpy = quaternion_to_euler(right_quat)
            right_euler.append(right_rpy)
        
        left_euler = np.array(left_euler)
        right_euler = np.array(right_euler)
        
        # --- Left Arm ---
        # Subplot 1: Left Arm Translation (XYZ) + Gripper
        ax1.plot(x_axis, history_subset[:, 0], label='left_x', color='r', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 1], label='left_y', color='g', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 2], label='left_z', color='b', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 7], label='left_grip', color='orange', 
                 linestyle=':', linewidth=2, alpha=0.8)
        ax1.set_ylabel('Position (m)')
        ax1.legend(loc='upper right', fontsize='x-small', ncol=4)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Step {step_idx}: Left Arm Position & Gripper")

        # Subplot 2: Left Arm Euler Angles (Roll, Pitch, Yaw)
        ax2.plot(x_axis, left_euler[:, 0], label='left_roll', color='c', linewidth=1.5)
        ax2.plot(x_axis, left_euler[:, 1], label='left_pitch', color='m', linewidth=1.5)
        ax2.plot(x_axis, left_euler[:, 2], label='left_yaw', color='y', linewidth=1.5)
        ax2.set_ylabel('Rotation (rad)')
        ax2.legend(loc='upper right', fontsize='x-small', ncol=3)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Left Arm Rotation (RPY from Quaternion)")

        # --- Right Arm ---
        # Subplot 3: Right Arm Translation (XYZ) + Gripper
        ax3.plot(x_axis, history_subset[:, 8], label='right_x', color='r', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 9], label='right_y', color='g', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 10], label='right_z', color='b', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 15], label='right_grip', color='orange', 
                 linestyle=':', linewidth=2, alpha=0.8)
        ax3.set_ylabel('Position (m)')
        ax3.legend(loc='upper right', fontsize='x-small', ncol=4)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Right Arm Position & Gripper")

        # Subplot 4: Right Arm Euler Angles (Roll, Pitch, Yaw)
        ax4.plot(x_axis, right_euler[:, 0], label='right_roll', color='c', linewidth=1.5, linestyle='--')
        ax4.plot(x_axis, right_euler[:, 1], label='right_pitch', color='m', linewidth=1.5, linestyle='--')
        ax4.plot(x_axis, right_euler[:, 2], label='right_yaw', color='y', linewidth=1.5, linestyle='--')
        ax4.set_ylabel('Rotation (rad)')
        ax4.legend(loc='upper right', fontsize='x-small', ncol=3)
        ax4.grid(True, alpha=0.3)
        ax4.set_title("Right Arm Rotation (RPY from Quaternion)")

    # Set X-axis display range to maintain sliding window effect
    ax1.set_xlim(max(0, step_idx - window), max(window, step_idx))
    ax3.set_xlabel('Step')
    ax4.set_xlabel('Step')
    
    plt.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())
    img = img[:, :, :3]
    
    # Convert to uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
        
    plt.close(fig)
    return img


def save_comparison_video(real_obs_list, imagined_video, action_history, save_path, fps=15):
    """Save a top/bottom comparison mp4 video: "real observation vs model-imagined video".

    Top row: the 3 cameras (cam_high / left wrist / right wrist) resized to a common height
    and hstacked, with a title bar; bottom row: the imagined video frames returned by the
    server (a gray "Coming soon" placeholder is shown when missing). The total frame count
    follows the real observations.

    Args:
        real_obs_list: List of real observation dicts (outputs of format_obs).
        imagined_video: List of imagined video segments (concatenable along axis 0 into
            [T,H,W,3]), or None.
        action_history: Action history list (currently unused; reserved for action-curve
            visualization).
        save_path: Output mp4 path.
        fps: Video frame rate.
    """
    if not real_obs_list:
        return

    n_real = len(real_obs_list)
    if imagined_video is not None:
        imagined_video = np.concatenate(imagined_video, 0)
        n_imagined = len(imagined_video) 
    else:
        n_imagined = 0
    n_frames = n_real # Based on real observation frames
    
    print(f"Saving video: Real {n_real} frames, Imagined {n_imagined} frames...")

    final_frames = []

    for i in range(n_frames):
        obs = real_obs_list[i]
        cam_high = obs["observation.images.cam_high"]
        cam_left = obs["observation.images.cam_left_wrist"]
        cam_right = obs["observation.images.cam_right_wrist"]

        base_h = cam_high.shape[0]
        
        def resize_h(img, h):
            # Scale each camera image proportionally to cam_high's height so the horizontal
            # concatenation aligns; also make it contiguous + uint8 (required by imageio)
            if img.shape[0] != h:
                w = int(img.shape[1] * h / img.shape[0])
                img = cv2.resize(img, (w, h))
            img = np.ascontiguousarray(img)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            return img

        row_real = np.hstack([
            resize_h(cam_high, base_h), 
            resize_h(cam_left, base_h), 
            resize_h(cam_right, base_h)
        ])
        
        row_real = np.ascontiguousarray(row_real)

        row_real = add_title_bar(row_real, "Real Observation (High / Left / Right)")

        target_width = row_real.shape[1]

        if imagined_video is not None and i < n_imagined:
            img_frame = imagined_video[i]
            # Imagined frames may be [0,1] floats or [0,255] values; normalize to uint8
            if img_frame.dtype != np.uint8 and img_frame.max() <= 1.0001:
                img_frame = (img_frame * 255).astype(np.uint8)
            elif img_frame.dtype != np.uint8:
                img_frame = img_frame.astype(np.uint8)

            h = int(img_frame.shape[0] * target_width / img_frame.shape[1])
            row_imagined = cv2.resize(img_frame, (target_width, h))
        else:
            # When imagined frames are missing (e.g. server started without
            # save_visualization), use a gray placeholder image
            row_imagined = np.zeros((300, target_width, 3), dtype=np.uint8)
            cv2.putText(row_imagined, "Coming soon", (target_width//2 - 100, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        row_imagined = np.ascontiguousarray(row_imagined)
        row_imagined = add_title_bar(row_imagined, "Imagined Video Stream")
        full_frame = np.vstack([row_real, row_imagined])
        full_frame = np.ascontiguousarray(full_frame)
        final_frames.append(full_frame)

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"Combined video saved to: {save_path}")


def class_decorator(task_name):
    """Dynamically import the ``envs.<task_name>`` module from the RoboTwin repo and
    instantiate the task environment class of the same name.

    RoboTwin convention: one module per task, containing an environment class named after
    the task (inheriting from _base_task) with interfaces like setup_demo/play_once/
    check_success.

    Args:
        task_name: Task name (e.g. "adjust_bottle"), which is both the module and class name.
    Returns:
        The task environment instance.
    Raises:
        SystemExit("No Task"): If the module is missing, has no same-named class, or
            instantiation fails.
    """
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    """Dynamically import a policy module and fetch the given model factory function.

    A generic interface kept for loading local models by policy name; the current main
    flow uses WebsocketClientPolicy remote inference instead, so main does not call this.
    """
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    """Read the configuration of the given camera model from RoboTwin's
    ``task_config/_camera_config.yml``.

    Args:
        camera_type: Camera model name (e.g. "D435").
    Returns:
        The model's config dict (resolution w/h etc., used to determine the evaluation
        video's video_size).
    """
    camera_config_path = os.path.join(robowin_root, "task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    """Read the ``config.yml`` under the given embodiment (robot body) directory and
    return its config dict.

    The config contains fields like arm_joints_name, used to infer the left/right arm
    degrees of freedom.
    """
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    """Evaluation entry: load configs -> create result directories -> connect to the
    inference server -> run the evaluation and write results.

    Steps:
    1. Read the task config yml (``task_config/<task_config>.yml``) and merge command-line
       overrides;
    2. Parse the embodiment config: list length 1 = both arms use the same robot; length 3
       = [left arm, right arm, dual-arm distance];
    3. Create the ``eval_result/<task>/<policy>/<config>/<ckpt>/<timestamp>/`` result dir;
    4. Print the domain-randomization (cluttered table / random background / light / table
       height / camera distance), camera, and embodiment config summary;
    5. Establish the websocket connection (blocks until the server is ready);
    6. Compute the starting seed ``st_seed = 10000*(1+seed)`` from ``seed``: parallel
       workers with different --seed use non-overlapping 10000-wide seed intervals,
       enabling multi-GPU work splitting;
    7. Call :func:`eval_policy` to complete test_num valid episodes and write the success
       rate to ``_result.txt``.

    Args:
        usr_args: Dict merged from command-line arguments and the yml config (see
            parse_args_and_config), containing task_name/task_config/ckpt_setting/
            save_root/policy_name/video_guidance_scale/action_guidance_scale/seed/
            test_num/port, etc.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    save_root = usr_args["save_root"]
    policy_name = usr_args["policy_name"]
    video_guidance_scale = usr_args["video_guidance_scale"]
    action_guidance_scale = usr_args["action_guidance_scale"]
    # Language instruction type is fixed to 'seen': sample from the instruction candidates
    # "seen during training" produced by the task description generator
    instruction_type = 'seen'
    save_dir = None
    video_save_dir = None
    video_size = None

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["save_root"] = save_root

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        # Single entry: both arms use the same robot model (standard dual-arm mode)
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        # Three entries: [left arm model, right arm model, dual-arm distance], supporting
        # heterogeneous dual arms
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    # Different --seed values map to seed intervals 10000 apart: in multi-GPU parallel
    # evaluation (launch_client_multigpus.sh assigns each worker a different seed/task
    # group), episode seeds never overlap between workers
    st_seed = 10000 * (1 + seed)
    suc_nums = []
    test_num = usr_args["test_num"]

    
    # Establish the websocket connection; the constructor blocks and polls until the
    # inference server is ready
    model = WebsocketClientPolicy(port=usr_args['port'])

    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   save_visualization=True,
                                   video_guidance_scale=video_guidance_scale,
                                   action_guidance_scale=action_guidance_scale)
    suc_nums.append(suc_num)

    # Write the final success-rate summary text (_result.txt), content = success count / test_num
    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")

def format_obs(observation, prompt):
    """Flatten the RoboTwin environment's raw observation into the dict format agreed with
    the server.

    Args:
        observation: Raw obs returned by ``TASK_ENV.get_obs()``, containing the 3-camera
            RGB images and joint state.
        prompt: Language instruction text.
    Returns:
        dict with:
        - ``observation.images.cam_high/cam_left_wrist/cam_right_wrist``: the 3 camera
          images [H, W, 3] uint8 (the server stitches them into a "T-shaped" big image
          before VAE encoding: cam_high at full resolution, wrist cameras at half
          resolution);
        - ``observation.state``: dual-arm joint + gripper state vector (joint_action["vector"]);
        - ``task``: the language instruction.
    """
    return {
                "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"], # H,W,3
                "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
                "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
                "observation.state": observation["joint_action"]["vector"],
                "task": prompt,
            }

def add_eef_pose(new_pose, init_pose):
    """Compose a single-arm relative EEF pose onto the initial absolute pose
    (relative pose -> absolute pose).

    Why this is needed: the training data pipeline uses get_relative_pose to convert
    robotwin actions into a representation relative to the episode's initial pose
    (translations become more concentrated, which helps normalization), so the model also
    outputs relative poses; they must be composed back into absolute poses before execution.

    Composition rules:
    - Translation: direct addition ``out_trans = new_trans + init_trans``;
    - Rotation: quaternion multiplication ``init_R * new_R`` (scipy, xyzw order), i.e. the
      relative rotation is applied on top of the initial orientation;
    - Gripper: passed through from new_pose (open/close is an absolute quantity with no
      relative semantics).

    Args:
        new_pose: Relative pose [x,y,z, qx,qy,qz,qw, gripper] (8 dims).
        init_pose: The episode's initial absolute pose (same 8-dim format).
    Returns:
        The composed absolute pose [x,y,z, qx,qy,qz,qw, gripper] (8 dims).
    """
    new_pose_R = R.from_quat(new_pose[3:7][None])
    init_pose_R = R.from_quat(init_pose[3:7][None])
    out_rot = (init_pose_R * new_pose_R).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])

def add_init_pose(new_pose, init_pose):
    """Apply :func:`add_eef_pose` to the dual-arm 16-dim relative pose (first 8 dims = left
    arm, last 8 dims = right arm).

    Args:
        new_pose: 16-dim relative pose [left arm xyz+quat+gripper, right arm xyz+quat+gripper].
        init_pose: 16-dim initial absolute pose (same layout).
    Returns:
        The 16-dim absolute pose.
    """
    left_pose = add_eef_pose(new_pose[:8], init_pose[:8])
    right_pose = add_eef_pose(new_pose[8:], init_pose[8:])
    return np.concatenate([left_pose, right_pose])

def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                save_visualization=False,
                video_guidance_scale=5.0,
                action_guidance_scale=5.0):
    """Single-task evaluation main loop: runs test_num "valid" episodes and returns the
    success count.

    Multi-seed loop structure (why not a simple `for seed in range(test_num)`):
    - For each episode, the RoboTwin built-in expert policy first runs the current seed
      (``play_once`` motion-planning replay + ``check_success``); only seeds verified as
      "the task itself is solvable" count toward test_num. Seeds where environment setup
      fails (UnStableError) or the expert fails are skipped with seed+1 — this prevents
      unsolvable configurations from diluting the model's success rate, so the number of
      seeds actually consumed exceeds test_num;
    - For each valid seed: rebuild the environment -> randomly pick a language instruction
      -> server reset(prompt) -> the inner while loop runs "infer to get an action chunk ->
      execute substep by substep -> feed real observations back via compute_kv_cache"
      until success or step_lim is reached -> save the comparison video, update the
      res.json success rate, and periodically clear the simulation cache according to
      clear_cache_freq.

    Args:
        task_name: Task name (e.g. "adjust_bottle").
        TASK_ENV: Task environment instance (output of class_decorator).
        args: Full config dict (contains render_freq/clear_cache_freq/step_lim/eval_video_log etc.).
        model: WebsocketClientPolicy client.
        st_seed: Starting random seed.
        test_num: Number of valid episodes required.
        video_size: "WxH" string used by ffmpeg for environment video recording.
        instruction_type: Language instruction type (e.g. 'seen'), deciding which candidate
            instruction group to sample from.
        save_visualization: Whether the server should return imagined video for comparison
            visualization.
        video_guidance_scale: CFG guidance scale for the video branch (forwarded to the server).
        action_guidance_scale: CFG guidance scale for the action branch (forwarded to the server).
    Returns:
        (now_seed, suc): the next unused seed and the number of successful episodes.
    """
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    # Expert-verification switch: first confirm the seed is solvable with the built-in
    # expert policy, then evaluate the model (see the function docstring)
    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []


    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    # Loop until test_num "expert-solvable" valid seeds have been collected; invalid seeds
    # only increment now_seed and are skipped
    while succ_seed < test_num:
        render_freq = args["render_freq"]
        # Temporarily disable rendering during expert verification for speed; restored afterwards
        args["render_freq"] = 0

        if expert_check:
            try:
                # Build the environment with the current seed and let the expert policy
                # (motion planning) replay the whole task once
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print(f"error occurs ! {e}")
                traceback.print_exc()
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            # Expert planning succeeded and the task is judged complete -> the seed is
            # valid and counts toward the evaluation quota
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            # Even the expert cannot solve it (unsolvable object placement, etc.); move to
            # the next seed
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        # Rebuild the environment with the same seed (identical object placement to the
        # expert verification) and start the real evaluation
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        # Randomly pick one language instruction from the generated candidates and set it
        # on the environment (instruction_type selects the seen/unseen group)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            # Launch an ffmpeg subprocess that receives rawvideo (rgb24) frames rendered by
            # the environment through its stdin pipe and encodes them to h264 mp4 in real
            # time, saving the RoboTwin environment-view evaluation recording
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False

        prompt = TASK_ENV.get_instruction()
        # Reset the server session: clear the KV cache, encode the language instruction and
        # the initial frame (the return value is not used here)
        ret = model.infer(dict(reset = True, prompt=prompt, save_visualization=save_visualization))
        
        first = True
        full_obs_list = []
        gen_video_list = []
        full_action_history = []

        # Record the episode's initial dual-arm EEF absolute poses, 16 dims total:
        # [left arm xyz(3) + quaternion(4) + gripper(1), right arm xyz(3) + quaternion(4) +
        # gripper(1)]. The model outputs actions as poses relative to this initial pose, so
        # they must be composed back with add_init_pose before execution
        initial_obs = TASK_ENV.get_obs() 
        inint_eef_pose = initial_obs['endpose']['left_endpose'] + \
        [initial_obs['endpose']['left_gripper']] + \
        initial_obs['endpose']['right_endpose'] + \
        [initial_obs['endpose']['right_gripper']]
        inint_eef_pose = np.array(inint_eef_pose, dtype=np.float64)
        initial_formatted_obs = format_obs(initial_obs, prompt)
        full_obs_list.append(initial_formatted_obs)
        first_obs = None
        # Execution main loop: runs until the environment's executed-action count reaches
        # the step_lim upper bound (timeout counts as failure)
        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim:
            if first:
                # Only the first round fetches an observation as inference input; afterwards
                # first_obs stays None and the server continues autoregressive inference
                # purely from the real-observation KV cache written via compute_kv_cache
                observation = TASK_ENV.get_obs()
                first_obs = format_obs(observation, prompt)

            ret = model.infer(dict(obs=first_obs, prompt=prompt, save_visualization=save_visualization, video_guidance_scale=video_guidance_scale, action_guidance_scale=action_guidance_scale)) #(TASK_ENV, model, observation)
            action = ret['action']
            if 'video' in ret:
                # Imagined video frames returned by the server (when save_visualization=True),
                # used for the comparison visualization
                imagined_video = ret['video']
                gen_video_list.append(imagined_video)
            key_frame_list = []

            # action shape [C, F, N]: C=action channels (14=Euler-angle representation /
            # 16=relative-pose representation), F=latent frames (2), N=control substeps per
            # latent frame (16). The Wan VAE downsamples time by 4x: 1 latent frame
            # corresponds to 4 real video frames, so one real-observation keyframe is
            # collected every N/4 substeps to stay aligned with the latent frame rate
            assert action.shape[2] % 4 == 0
            action_per_frame = action.shape[2] // 4

            # The first round skips latent frame 0: its actions correspond to the "history"
            # (the training data pipeline pads zero actions at the beginning to align with
            # the initial observation), so executing them would repeat; all frames of later
            # rounds are future actions
            start_idx = 1 if first else 0
            for i in range(start_idx, action.shape[1]):
                for j in range(action.shape[2]):
                    raw_action_step = action[:, i, j].flatten() 
                    full_action_history.append(raw_action_step)

                    ee_action = action[:, i, j]
                    if action.shape[0] == 14:
                        # 14-dim Euler-angle representation: [left xyz, left euler(3), left
                        # gripper, right xyz, right euler(3), right gripper]. Use
                        # euler2quat (geometry.py) to convert Euler angles to quaternions
                        # and rearrange into the 16-dim [xyz+quat+gripper]x2 layout the
                        # environment expects
                        ee_action = np.concatenate([
                            ee_action[:3],
                            euler2quat(ee_action[3], ee_action[4], ee_action[5]),
                            ee_action[6:10],
                            euler2quat(ee_action[10], ee_action[11], ee_action[12]),
                            ee_action[13:14]
                        ])
                    elif action.shape[0] == 16:
                        # 16-dim relative-pose representation: first compose with the
                        # initial absolute pose to recover absolute poses, then renormalize
                        # both quaternions (guarding against numerical drift from the
                        # diffusion output producing invalid rotations)
                        ee_action =  add_init_pose(ee_action, inint_eef_pose)
                        ee_action = np.concatenate([
                            ee_action[:3],
                            ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
                            ee_action[7:11],
                            ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
                            ee_action[15:16]
                        ])
                    else:
                        raise NotImplementedError
                    # Execute one EEF pose control substep (the environment internally
                    # converts it to joint commands via motion planning)
                    TASK_ENV.take_action(ee_action, action_type='ee')
                   
                    if (j+1) % action_per_frame == 0:
                        # Collect one real-observation keyframe every action_per_frame
                        # substeps: it goes both into the comparison video and into the
                        # batch later fed back to update the server's KV cache
                        obs = format_obs(TASK_ENV.get_obs(), prompt)
                        full_obs_list.append(obs)
                        key_frame_list.append(obs)
                    
            first = False

            # Feed this chunk's real-observation keyframes back: the server first drops
            # the imagined frames via clear_pred_cache, then writes the real observations
            # with update_cache=2 (closed-loop correction; imagined frames never enter the
            # long-term history). state carries the action chunk just executed, serving as
            # the clean action-condition segment in the KV cache
            model.infer(dict(obs = key_frame_list, compute_kv_cache=True, imagine=False, save_visualization=save_visualization, state=action))
  
            if TASK_ENV.eval_success:
                # The environment judges the task successful (e.g. objects reached their
                # target poses); end this episode early
                succ = True
                break
      

        # Save the "real observation / imagined video" comparison video; the filename ends
        # with the True/False success marker so statistics scripts like calc_stat.py can
        # aggregate success rates directly from filenames
        vis_dir = Path(args['save_root']) / f'stseed-{st_seed}' / 'visualization' / task_name
        vis_dir.mkdir(parents=True, exist_ok=True)
        video_name = f"{TASK_ENV.test_num}_{prompt.replace(' ', '_')}_{succ}.mp4"
        out_img_file = vis_dir / video_name
        save_comparison_video(
            real_obs_list=full_obs_list,
            imagined_video=None, #gen_video_list,
            action_history=full_action_history,
            save_path=str(out_img_file),
            fps=15 # Suggest adjusting fps based on simulation step
        )
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        # Close the environment; every clear_cache_freq episodes do a thorough simulation
        # cache cleanup to prevent memory leaks in the long-running evaluation process
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        # Overwrite the success-rate JSON after every episode so external monitoring
        # scripts can read the progress in real time
        save_dir = Path(args['save_root']) / f'stseed-{st_seed}' / 'metrics' / task_name
        save_dir.mkdir(parents=True, exist_ok=True)
        out_json_file = save_dir / 'res.json'
        write_json({
          "succ_num": float(TASK_ENV.suc),
          "total_num": float(TASK_ENV.test_num),
          "succ_rate": float(TASK_ENV.suc / TASK_ENV.test_num),
        }, out_json_file)
        
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    """Parse command-line arguments and merge them with the yml deployment config,
    returning the final config dict.

    - ``--config``: path to the policy deployment config file (e.g. policy/ACT/deploy_policy.yml);
    - ``--overrides``: followed by pairs of ``--key value`` tokens that override same-named
      entries in the yml; each value is attempted to be converted to a Python literal via
      eval (numbers/booleans/lists, etc.), falling back to the raw string on failure;
    - Other fixed arguments: port (server port), save_root (result root directory),
      video/action_guidance_scale (CFG guidance scales), test_num (number of valid episodes).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    parser.add_argument("--port", type=int, default=8000, help='remote policy socket port.')
    parser.add_argument("--save_root", type=str, default="results/default_vis_path")
    parser.add_argument("--video_guidance_scale", type=float, default=5.0)
    parser.add_argument("--action_guidance_scale", type=float, default=5.0)
    parser.add_argument("--test_num", type=int, default=100)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        # Parse ["--key1", "v1", "--key2", "v2", ...] into {key1: v1, ...}; eval lets values
        # be literals such as numbers/booleans/lists, invalid expressions stay raw strings
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    
    # Run the Sapien rendering self-check first: if the rendering backend is broken, exit
    # immediately instead of wasting a whole evaluation run
    Sapien_TEST()
    usr_args = parse_args_and_config()
    main(usr_args)

