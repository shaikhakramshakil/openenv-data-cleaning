# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Main FastAPI application for the Data Cleaning environment.

This module contains the core application logic. The `app` object
is imported by `app.py` which serves as the server entry point.
"""

# Re-export the app instance from app.py
# app.py contains the full FastAPI application with all routes,
# WebSocket handlers, and the session store.
from .app import app  # noqa: F401

__all__ = ["app"]
