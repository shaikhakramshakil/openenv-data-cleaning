# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data Cleaning Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DataCleaningAction, DataCleaningObservation


class DataCleaningEnv(EnvClient[DataCleaningAction, DataCleaningObservation, State]):
    """
    Client for the Data Cleaning Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> from openenv_data_cleaning import DataCleaningEnv, DataCleaningAction
        >>>
        >>> env = DataCleaningEnv(base_url="http://localhost:7860")
        >>> result = env.reset(task_name='task_1_identify')
        >>> print(f"Task: {result.observation.task_description}")
        >>>
        >>> # Use a tool
        >>> result = env.step(DataCleaningAction(action_type='check_schema', value=''))
        >>> print(f"Schema: {result.observation.tool_output}")
        >>>
        >>> # Submit answer
        >>> result = env.step(DataCleaningAction(
        ...     action_type='identify_errors',
        ...     value='{"row_ids": [2, 5, 8]}'
        ... ))
        >>> print(f"Score: {result.observation.reward}")
        >>> env.close()

    Example with Docker:
        >>> client = DataCleaningEnv.from_docker_image("data-cleaning-env:latest")
        >>> try:
        ...     result = client.reset(task_name='task_1_identify')
        ...     result = client.step(DataCleaningAction(
        ...         action_type='identify_errors',
        ...         value='{"row_ids": [2, 5, 8]}'
        ...     ))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DataCleaningAction) -> Dict:
        """
        Convert DataCleaningAction to JSON payload for step message.

        Args:
            action: DataCleaningAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "value": action.value,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DataCleaningObservation]:
        """
        Parse server response into StepResult[DataCleaningObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with DataCleaningObservation
        """
        obs_data = payload.get("observation", {})
        observation = DataCleaningObservation(
            dataset_text=obs_data.get("dataset_text", ""),
            task_name=obs_data.get("task_name", ""),
            task_description=obs_data.get("task_description", ""),
            available_actions=obs_data.get("available_actions", []),
            feedback=obs_data.get("feedback", ""),
            tool_output=obs_data.get("tool_output"),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 15),
            num_rows=obs_data.get("num_rows", 0),
            num_columns=obs_data.get("num_columns", 0),
            column_names=obs_data.get("column_names", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


# Backwards compatibility alias
DataCleaningClient = DataCleaningEnv
