"""Sapien rendering environment self-check script.

Role in the evaluation loop: RoboTwin evaluation depends on Sapien's ray-tracing renderer
to produce camera images; if the rendering backend (GPU driver / Vulkan / shader config) is
broken, evaluation would produce corrupted images or fail silently. This script builds a
minimal scene to verify rendering capability: on success it prints a green "Render Well",
on failure it prints a red "Render Error" and exits the process immediately (fail-fast).

Typical usage::

    python -m evaluation.robotwin.test_render
    # eval_polict_client_openpi.py also runs Sapien_TEST() automatically at startup

Note: the duplicated imports at the top of this file are historical artifacts and are
intentionally kept as-is.
"""
import sys
import warnings
import os

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_dir, "../../tools"))
import numpy as np
import pdb
import json
import torch
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import gymnasium as gym
import toppra as ta
import transforms3d as t3d
from collections import OrderedDict

import sys
import warnings
import os

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_dir, "../../tools"))
import numpy as np
import pdb
import json
import torch
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import gymnasium as gym
import toppra as ta
import transforms3d as t3d
from collections import OrderedDict


class Sapien_TEST(gym.Env):
    """Sapien rendering self-check environment: attempts to build a ray-tracing scene and
    judges rendering capability by whether an exception is raised.

    Instantiation alone triggers the check (see __init__); no step/reset calls are needed.
    ``eval_polict_client_openpi.py`` constructs this class once before main as a precondition check.
    """

    def __init__(self):
        """Initialize: silence third-party library logging, then try to build the scene;
        on failure print a red error and exit()."""
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging
        try:
            self.setup_scene()
            print("\033[32m" + "Render Well" + "\033[0m")
        except:
            print("\033[31m" + "Render Error" + "\033[0m")
            exit()

    def setup_scene(self, **kwargs):
        """
        Set the scene
            - Set up the basic scene: light source, viewer.

        Notes (Chinese-original explanation, translated): creates the Sapien physics engine
        and the ray-tracing ("rt") renderer, and configures:
        - material/texture count limits (50000; evaluation scenes contain many objects, so
          the defaults are not enough);
        - ray-tracing parameters: 32 samples per pixel, path depth 8, OIDN denoiser — kept
          identical to the real evaluation rendering quality, so passing this self-check
          means evaluation rendering will work.
        Any exception in these steps counts as a rendering self-check failure (caught by
        __init__, which then exits).
        """
        self.engine = sapien.Engine()
        # declare sapien renderer
        from sapien.render import set_global_config

        set_global_config(max_num_materials=50000, max_num_textures=50000)
        self.renderer = sapien.SapienRenderer()
        # give renderer to sapien sim
        self.engine.set_renderer(self.renderer)

        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

        # declare sapien scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)


if __name__ == "__main__":
    a = Sapien_TEST()
