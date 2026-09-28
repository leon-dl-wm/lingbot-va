# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

import torch
from easydict import EasyDict

# Root of the shared CFS Turbo mount (TI-ONE injects STORAGE_MOUNT_PATH).
# `or` (not just a get default) so an empty value cannot yield relative paths.
STORAGE_MOUNT_PATH = os.environ.get('STORAGE_MOUNT_PATH') or '/home/tione/notebook'

va_shared_cfg = EasyDict()

va_shared_cfg.host = '0.0.0.0'
va_shared_cfg.port = 29536

va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.save_root = './train_out'

va_shared_cfg.patch_size = (1, 2, 2)

va_shared_cfg.enable_offload = False