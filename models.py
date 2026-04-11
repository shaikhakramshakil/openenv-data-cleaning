# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Type-safe models for the Data Cleaning OpenEnv environment.

Uses Pydantic BaseModel as required by the OpenEnv spec for typed
Action, Observation, and State models between client and server.

Uses openenv.core types with fallback for standalone operation.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Try to use openenv.core types if available, otherwise use standalone models
try:
    from openenv.core.env_server.types import Action as BaseAction
    from openenv.core.env_server.types import Observation as BaseObservation
    from openenv.core.env_server.types import State as BaseState
    from openenv.core.env_server.types import StepResult as BaseStepResult

    # Create Action type
    class DataCleaningAction(BaseAction):
        action_type: str
        value: str = ""

    # Create Observation type
    class DataCleaningObservation(BaseObservation):
        dataset_text: str = ""
        task_name: str = ""
        task_description: str = ""
        available_actions: List[str] = Field(default_factory=list)
        feedback: str = ""
        tool_output: Optional[Any] = None
        step_number: int = 0
        max_steps: int = 15
        num_rows: int = 0
        num_columns: int = 0
        column_names: List[str] = Field(default_factory=list)
        done: bool = False
        reward: float = 0.0
        metadata: Dict[str, Any] = Field(default_factory=dict)

    # Create State type
    class DataCleaningState(BaseState):
        episode_id: str = ""
        step_count: int = 0
        task_name: str = ""
        total_errors: int = 0
        cumulative_reward: float = 0.0

    # Create StepResult type
    class StepResult(BaseStepResult):
        observation: Optional[DataCleaningObservation] = None
        reward: float = 0.0
        done: bool = False

    USES_OPENENV_CORE = True

except ImportError:
    # Fallback to standalone pydantic models when openenv.core is not available

    class DataCleaningAction(BaseModel):
        action_type: str
        value: str = ""

    class DataCleaningObservation(BaseModel):
        dataset_text: str = ""
        task_name: str = ""
        task_description: str = ""
        available_actions: List[str] = Field(default_factory=list)
        feedback: str = ""
        tool_output: Optional[Any] = None
        step_number: int = 0
        max_steps: int = 15
        num_rows: int = 0
        num_columns: int = 0
        column_names: List[str] = Field(default_factory=list)
        done: bool = False
        reward: float = 0.0
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class DataCleaningState(BaseModel):
        episode_id: str = ""
        step_count: int = 0
        task_name: str = ""
        total_errors: int = 0
        cumulative_reward: float = 0.0

    class StepResult(BaseModel):
        observation: Optional[DataCleaningObservation] = None
        reward: float = 0.0
        done: bool = False

    USES_OPENENV_CORE = False


class EvaluationResult(BaseModel):
    """Result from running an evaluation scenario."""

    task_name: str = ""
    reward: float = 0.0
    done: bool = False
    steps: int = 0
    errors_found: int = 0
    message: str = ""
