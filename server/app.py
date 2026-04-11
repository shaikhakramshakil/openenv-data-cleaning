# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
FastAPI server for the Data Cleaning OpenEnv environment.

Exposes the environment via HTTP and WebSocket endpoints following
the OpenEnv specification.

HTTP endpoints support stateful sessions via session_id for multi-step episodes.
"""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from models import DataCleaningAction, DataCleaningObservation, DataCleaningState

try:
    # Module execution: python -m server.app
    from .environment import DataCleaningEnvironment, TASKS
    from .web_ui import WEB_UI_HTML
except ImportError:
    # Script execution: python server/app.py
    from server.environment import DataCleaningEnvironment, TASKS
    from server.web_ui import WEB_UI_HTML

# Configuration
DEFAULT_TASK = os.environ.get("OPENENV_TASK", "task_1_identify")
MAX_CONCURRENT_ENVS = int(os.environ.get("MAX_CONCURRENT_ENVS", "100"))

# In-memory session store for stateful HTTP sessions
SESSIONS: Dict[str, DataCleaningEnvironment] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    print(f"Data Cleaning Environment starting (default task: {DEFAULT_TASK})")
    print(f"Max concurrent environments: {MAX_CONCURRENT_ENVS}")
    yield
    print("Data Cleaning Environment shutting down")
    SESSIONS.clear()


app = FastAPI(
    title="OpenEnv Data Cleaning Environment",
    description="A real-world environment for training AI agents on data quality assessment",
    version="1.1.0",
    lifespan=lifespan,
)


# ─── HTTP Endpoints (stateful via session_id) ────────────────────────


@app.get("/")
async def root():
    """Redirect root to the web UI."""
    from fastapi.responses import RedirectResponse
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
        "default_task": DEFAULT_TASK,
    }


@app.post("/reset")
async def http_reset(data: Dict[str, Any] = None):
    """
    Reset the environment and create a new session.

    Returns a session_id that must be used in subsequent /step calls
    to maintain state across the multi-step episode.
    """
    if data is None:
        data = {}
    task_name = data.get("task_name", DEFAULT_TASK)

    # Create new environment instance and store in session
    session_id = str(uuid.uuid4())[:12]
    env = DataCleaningEnvironment(task_name=task_name)
    obs = env.reset(**data)
    SESSIONS[session_id] = env

    result = _serialize_observation(obs)
    result["session_id"] = session_id
    return result


@app.post("/step")
async def http_step(data: Dict[str, Any]):
    """
    Execute a step in an existing session.

    Requires session_id from a previous /reset call to maintain
    stateful multi-step episode support.
    """
    session_id = data.get("session_id")
    if not session_id or session_id not in SESSIONS:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Session not found or expired",
                "hint": "Call /reset first to create a new session",
            },
        )

    env = SESSIONS[session_id]
    action_type = data.get("action_type", "")
    value = data.get("value", "")

    action = DataCleaningAction(action_type=action_type, value=value)
    obs = env.step(action)

    result = _serialize_observation(obs)

    # Clean up session when episode is done
    if obs.done:
        del SESSIONS[session_id]

    return result


@app.get("/state")
async def http_state(session_id: Optional[str] = None):
    """Get state for a session. Pass session_id as a query parameter."""
    if not session_id or session_id not in SESSIONS:
        # Return a default state template when no session is active
        state = DataCleaningState(
            episode_id="",
            step_count=0,
            task_name=DEFAULT_TASK,
            total_errors=12,
            cumulative_reward=0.0,
        )
        return state.model_dump()

    state = SESSIONS[session_id].state
    return state.model_dump()


# ─── WebSocket Endpoint (stateful, persistent session) ───────────────


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for persistent environment sessions.

    Each connection gets its own isolated environment instance.
    Protocol:
        Client sends: {"type": "reset"|"step"|"state"|"close", "data": {...}}
        Server responds: {"type": "result"|"state"|"error", "data": {...}}
    """
    await websocket.accept()

    env = None

    try:
        while True:
            raw = await websocket.receive_text()
            message = json.loads(raw)
            msg_type = message.get("type", "")
            msg_data = message.get("data", {})

            if msg_type == "reset":
                task_name = msg_data.get("task_name", DEFAULT_TASK)
                env = DataCleaningEnvironment(task_name=task_name)
                obs = env.reset(**msg_data)
                response = {
                    "type": "result",
                    "data": _serialize_observation(obs),
                }
                await websocket.send_text(json.dumps(response))

            elif msg_type == "step":
                if env is None:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"message": "Environment not initialized. Call reset first.", "code": "NOT_INITIALIZED"},
                    }))
                    continue

                action = _deserialize_action(msg_data)
                obs = env.step(action)
                response = {
                    "type": "result",
                    "data": _serialize_observation(obs),
                }
                await websocket.send_text(json.dumps(response))

            elif msg_type == "state":
                if env is None:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"message": "Environment not initialized. Call reset first.", "code": "NOT_INITIALIZED"},
                    }))
                    continue

                state = env.state
                response = {
                    "type": "state",
                    "data": state.model_dump(),
                }
                await websocket.send_text(json.dumps(response))

            elif msg_type == "close":
                break

            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": {"message": f"Unknown message type: {msg_type}", "code": "UNKNOWN_TYPE"},
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"message": str(e), "code": "INTERNAL_ERROR"},
            }))
        except Exception:
            pass


# ─── Web UI ─────────────────────────────────────────────────────────


@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """Premium Web UI for interacting with the environment."""
    return WEB_UI_HTML


# ─── Serialization Helpers ──────────────────────────────────────────


def _serialize_observation(obs) -> Dict[str, Any]:
    """Convert observation dataclass to JSON-serializable dict."""
    return {
        "observation": {
            "dataset_text": obs.dataset_text,
            "task_name": obs.task_name,
            "task_description": obs.task_description,
            "available_actions": obs.available_actions,
            "feedback": obs.feedback,
            "tool_output": obs.tool_output,
            "step_number": obs.step_number,
            "max_steps": obs.max_steps,
            "num_rows": obs.num_rows,
            "num_columns": obs.num_columns,
            "column_names": obs.column_names,
            "done": obs.done,
            "reward": obs.reward,
            "metadata": obs.metadata,
        },
        "reward": obs.reward,
        "done": obs.done,
    }


def _deserialize_action(data: Dict[str, Any]) -> DataCleaningAction:
    """Convert JSON dict to DataCleaningAction."""
    return DataCleaningAction(
        action_type=data.get("action_type", ""),
        value=data.get("value", ""),
    )


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
