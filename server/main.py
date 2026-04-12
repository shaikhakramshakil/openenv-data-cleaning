# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Server entry point for the Data Cleaning environment.

Provides a main function for running the server directly:
    uv run --project . server
    python -m server.app
"""

import os

from .app import app, main

__all__ = ["app", "main"]
