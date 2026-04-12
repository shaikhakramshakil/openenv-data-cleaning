# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Data Cleaning Environment.

This module creates an HTTP server that exposes the DataCleaningEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 2

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies with '\n    pip install openenv\n'"
    ) from e

# Import from local models.py (PYTHONPATH includes /app in Docker)
try:
    from models import DataCleaningAction, DataCleaningObservation
except ImportError:
    from ..models import DataCleaningAction, DataCleaningObservation

from .environment import DataCleaningEnvironment, TASKS

# Import Web UI
try:
    from .web_ui import WEB_UI_HTML
except ImportError:
    try:
        from server.web_ui import WEB_UI_HTML
    except ImportError:
        WEB_UI_HTML = "<html><body><h1>Data Cleaning Environment</h1></body></html>"


# Create the app with web interface using the standard OpenEnv factory
app = create_app(
    DataCleaningEnvironment,
    DataCleaningAction,
    DataCleaningObservation,
    env_name="data-cleaning-env",
    max_concurrent_envs=100,
)


# -- Custom Routes (added on top of standard OpenEnv endpoints) ---------------

from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the web UI."""
    return RedirectResponse(url="/web")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/info")
async def info():
    """Return environment metadata."""
    return {
        "name": "data-cleaning-env",
        "version": "1.1.0",
        "description": "AI agent data quality assessment environment with tool-use",
        "tasks": list(TASKS.keys()),
        "tools": ["check_schema", "search_reference", "run_statistics"],
    }


@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """Premium Web UI for interacting with the environment."""
    return WEB_UI_HTML


def main():
    """Entry point for the server."""
    import uvicorn
    import os
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
