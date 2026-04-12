# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Data Cleaning Environment.

These models define the action, observation, and state types used by the
OpenEnv integration for the data cleaning server.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class DataCleaningAction(Action):
    """Action for the Data Cleaning environment."""

    action_type: str = Field(..., description="Type of action to perform")
    value: str = Field(default="", description="Action payload (JSON string or text)")


class DataCleaningObservation(Observation):
    """Observation returned by the Data Cleaning environment."""

    dataset_text: str = Field(default="", description="Formatted dataset table")
    task_name: str = Field(default="", description="Current task identifier")
    task_description: str = Field(default="", description="Task instructions")
    available_actions: List[str] = Field(
        default_factory=list, description="Valid actions for current task"
    )
    feedback: str = Field(default="", description="Feedback from last action")
    tool_output: Optional[Any] = Field(
        default=None, description="Output from tool-use actions"
    )
    step_number: int = Field(default=0, description="Current step in episode")
    max_steps: int = Field(default=15, description="Maximum steps allowed")
    num_rows: int = Field(default=0, description="Number of rows in dataset")
    num_columns: int = Field(default=0, description="Number of columns in dataset")
    column_names: List[str] = Field(
        default_factory=list, description="Column names in dataset"
    )


class DataCleaningState(State):
    """State for the Data Cleaning environment with task-specific fields."""

    task_name: str = Field(default="", description="Current task name")
    total_errors: int = Field(default=0, description="Total errors in dataset")
    cumulative_reward: float = Field(
        default=0.0, description="Cumulative reward this episode"
    )


__all__ = [
    "DataCleaningAction",
    "DataCleaningObservation",
    "DataCleaningState",
]
