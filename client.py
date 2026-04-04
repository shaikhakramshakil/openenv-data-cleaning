"""
OpenEnv client for the Data Cleaning environment.

Provides both async (EnvClient) and sync wrappers for interacting
with the deployed environment server.
"""

import asyncio
import json
from dataclasses import asdict
from typing import Any, Dict, Optional

import websockets

from models import DataCleaningAction, DataCleaningObservation, DataCleaningState


class DataCleaningClient:
    """
    Client for interacting with the Data Cleaning environment server.

    Supports both WebSocket (for multi-step sessions) and HTTP (for debugging).

    Usage (sync):
        client = DataCleaningClient(base_url="http://localhost:8000")
        obs = client.reset(task_name="task_1_identify")
        while not obs.done:
            action = DataCleaningAction(action_type="identify_errors", value='{"row_ids": [2,3]}')
            obs = client.step(action)
        print(f"Final reward: {obs.reward}")

    Usage (async):
        client = DataCleaningClient(base_url="http://localhost:8000")
        obs = await client.async_reset(task_name="task_1_identify")
        while not obs.done:
            action = DataCleaningAction(action_type="identify_errors", value='...')
            obs = await client.async_step(action)
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.

        Args:
            base_url: The HTTP URL of the environment server (e.g., http://localhost:8000).
                      WebSocket URL is derived automatically.
        """
        self.base_url = base_url.rstrip("/")
        self.ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self._ws = None

    # ─── Async WebSocket Methods ──────────────────────────────────────

    async def async_connect(self):
        """Establish a WebSocket connection."""
        self._ws = await websockets.connect(self.ws_url)

    async def async_close(self):
        """Close the WebSocket connection."""
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "close", "data": {}}))
            except Exception:
                pass
            await self._ws.close()
            self._ws = None

    async def async_reset(self, task_name: str = "task_1_identify", **kwargs) -> DataCleaningObservation:
        """
        Reset the environment and return initial observation.

        Args:
            task_name: One of task_1_identify, task_2_classify, task_3_fix.
        """
        if self._ws is None:
            await self.async_connect()

        message = {
            "type": "reset",
            "data": {"task_name": task_name, **kwargs},
        }
        await self._ws.send(json.dumps(message))
        response = json.loads(await self._ws.recv())

        if response.get("type") == "error":
            raise RuntimeError(f"Reset failed: {response['data']['message']}")

        return self._parse_result(response["data"])

    async def async_step(self, action: DataCleaningAction) -> DataCleaningObservation:
        """
        Execute an action in the environment.

        Args:
            action: The action to execute.

        Returns:
            Observation with feedback and reward.
        """
        if self._ws is None:
            raise RuntimeError("Not connected. Call async_reset() first.")

        message = {
            "type": "step",
            "data": {
                "action_type": action.action_type,
                "value": action.value,
            },
        }
        await self._ws.send(json.dumps(message))
        response = json.loads(await self._ws.recv())

        if response.get("type") == "error":
            raise RuntimeError(f"Step failed: {response['data']['message']}")

        return self._parse_result(response["data"])

    async def async_state(self) -> DataCleaningState:
        """Get the current episode state."""
        if self._ws is None:
            raise RuntimeError("Not connected. Call async_reset() first.")

        message = {"type": "state", "data": {}}
        await self._ws.send(json.dumps(message))
        response = json.loads(await self._ws.recv())

        if response.get("type") == "error":
            raise RuntimeError(f"State failed: {response['data']['message']}")

        return self._parse_state(response["data"])

    # ─── Sync Wrappers ────────────────────────────────────────────────

    def reset(self, task_name: str = "task_1_identify", **kwargs) -> DataCleaningObservation:
        """Synchronous reset."""
        return asyncio.get_event_loop().run_until_complete(
            self.async_reset(task_name=task_name, **kwargs)
        )

    def step(self, action: DataCleaningAction) -> DataCleaningObservation:
        """Synchronous step."""
        return asyncio.get_event_loop().run_until_complete(self.async_step(action))

    def state(self) -> DataCleaningState:
        """Synchronous state."""
        return asyncio.get_event_loop().run_until_complete(self.async_state())

    def close(self):
        """Synchronous close."""
        return asyncio.get_event_loop().run_until_complete(self.async_close())

    # ─── Parsing Helpers ──────────────────────────────────────────────

    def _parse_result(self, payload: Dict[str, Any]) -> DataCleaningObservation:
        """Parse a server response into an Observation dataclass."""
        obs_data = payload.get("observation", {})
        return DataCleaningObservation(
            dataset_text=obs_data.get("dataset_text", ""),
            task_name=obs_data.get("task_name", ""),
            task_description=obs_data.get("task_description", ""),
            available_actions=obs_data.get("available_actions", []),
            feedback=obs_data.get("feedback", ""),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 10),
            num_rows=obs_data.get("num_rows", 0),
            num_columns=obs_data.get("num_columns", 0),
            column_names=obs_data.get("column_names", []),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DataCleaningState:
        """Parse a state response into a State dataclass."""
        return DataCleaningState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", ""),
            total_errors=payload.get("total_errors", 0),
            errors_found=payload.get("errors_found", 0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )

    # ─── Context Manager ─────────────────────────────────────────────

    async def __aenter__(self):
        await self.async_connect()
        return self

    async def __aexit__(self, *args):
        await self.async_close()
