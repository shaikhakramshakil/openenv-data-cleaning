"""
FastAPI server for the Data Cleaning OpenEnv environment.

Exposes the environment via HTTP and WebSocket endpoints following
the OpenEnv specification.
"""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from .environment import DataCleaningEnvironment, TASKS

# Read task name from environment variable (default: task_1_identify)
DEFAULT_TASK = os.environ.get("OPENENV_TASK", "task_1_identify")
MAX_CONCURRENT_ENVS = int(os.environ.get("MAX_CONCURRENT_ENVS", "100"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    print(f"Data Cleaning Environment starting (default task: {DEFAULT_TASK})")
    print(f"Max concurrent environments: {MAX_CONCURRENT_ENVS}")
    yield
    print("Data Cleaning Environment shutting down")


app = FastAPI(
    title="OpenEnv Data Cleaning Environment",
    description="A real-world environment for training AI agents on data quality assessment",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── HTTP Endpoints (stateless, for debugging) ───────────────────────

@app.get("/")
async def root():
    """Redirect root to the web UI so the HF Space doesn't show 'Not Found'."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/web")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/info")
async def info():
    """Return environment info."""
    return {
        "name": "data-cleaning-env",
        "version": "1.0.0",
        "description": "AI agent data quality assessment environment",
        "tasks": list(TASKS.keys()),
        "default_task": DEFAULT_TASK,
    }


@app.post("/reset")
async def http_reset(data: Dict[str, Any] = None):
    """Reset the environment (stateless — creates a new env each time)."""
    if data is None:
        data = {}
    task_name = data.get("task_name", DEFAULT_TASK)
    env = DataCleaningEnvironment(task_name=task_name)
    obs = env.reset(**data)
    return _serialize_observation(obs)


@app.post("/step")
async def http_step(data: Dict[str, Any]):
    """Execute a single step (stateless — creates a fresh env per request)."""
    task_name = data.get("task_name", DEFAULT_TASK)
    action_type = data.get("action_type", "")
    value = data.get("value", "")

    env = DataCleaningEnvironment(task_name=task_name)
    env.reset()

    from models import DataCleaningAction
    action = DataCleaningAction(action_type=action_type, value=value)
    obs = env.step(action)
    return _serialize_observation(obs)


@app.get("/state")
async def http_state():
    """Get state (stateless — returns default state template). Use /ws for real sessions."""
    from models import DataCleaningState
    state = DataCleaningState(
        episode_id="",
        step_count=0,
        task_name=DEFAULT_TASK,
        total_errors=10,
        errors_found=0,
        cumulative_reward=0.0,
    )
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
                    "data": {
                        "episode_id": state.episode_id,
                        "step_count": state.step_count,
                        "task_name": state.task_name,
                        "total_errors": state.total_errors,
                        "errors_found": state.errors_found,
                        "cumulative_reward": state.cumulative_reward,
                    },
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


# ─── Web UI ───────────────────────────────────────────────────────────


@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """Simple web UI for interacting with the environment."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Data Cleaning Environment</title></head>
    <body>
        <h1>Data Cleaning Environment</h1>
        <p>This is an OpenEnv environment for AI agent data quality assessment.</p>
        <h2>Available Tasks</h2>
        <ul>
            <li><strong>task_1_identify</strong> (Easy): Find which rows have errors</li>
            <li><strong>task_2_classify</strong> (Medium): Find and classify errors</li>
            <li><strong>task_3_fix</strong> (Hard): Find, classify, and fix errors</li>
        </ul>
        <h2>API Endpoints</h2>
        <ul>
            <li>GET /health -- Health check</li>
            <li>GET /info -- Environment info</li>
            <li>POST /reset -- Reset environment</li>
            <li>WS /ws -- WebSocket for stateful sessions</li>
            <li>GET /docs -- OpenAPI documentation</li>
        </ul>
    </body>
    </html>
    """


# ─── Serialization Helpers ────────────────────────────────────────────


def _serialize_observation(obs) -> Dict[str, Any]:
    """Convert observation dataclass to JSON-serializable dict."""
    return {
        "observation": {
            "dataset_text": obs.dataset_text,
            "task_name": obs.task_name,
            "task_description": obs.task_description,
            "available_actions": obs.available_actions,
            "feedback": obs.feedback,
            "step_number": obs.step_number,
            "max_steps": obs.max_steps,
            "num_rows": obs.num_rows,
            "num_columns": obs.num_columns,
            "column_names": obs.column_names,
            "metadata": obs.metadata,
        },
        "reward": obs.reward,
        "done": obs.done,
    }


def _deserialize_action(data: Dict[str, Any]):
    """Convert JSON dict to DataCleaningAction."""
    from models import DataCleaningAction
    return DataCleaningAction(
        action_type=data.get("action_type", ""),
        value=data.get("value", ""),
    )
