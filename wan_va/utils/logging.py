# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Global logging configuration: provides the root logger and the ``init_logger`` setup function.

The whole repo (train.py / wan_va_server.py / sever_utils.py, etc.) shares the ``logger``
defined here; call ``init_logger`` once at process startup (see the ``__main__`` block of
wan_va_server.py).
"""

import logging
import os

logger = logging.getLogger()


def init_logger():
    """Initialize the root logger: INFO level, console output, uniform timestamp format.

    Also sets KINETO_LOG_LEVEL=5 to suppress verbose torch.profiler logging.
    """
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"
