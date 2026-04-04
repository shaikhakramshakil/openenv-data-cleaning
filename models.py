"""
Type-safe models for the Data Cleaning OpenEnv environment.

Defines the Action, Observation, and State dataclasses that form
the contract between the client and server.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DataCleaningAction:
    """
    Action sent by the agent to the environment.

    Attributes:
        action_type: The type of action to perform.
            - "identify_errors": Report which rows contain errors.
                value should be JSON: {"row_ids": [2, 3, 5]}
            - "classify_errors": Report errors with their types.
                value should be JSON: {"errors": [{"row_id": 2, "column": "email", "error_type": "invalid_format"}, ...]}
            - "fix_errors": Report errors with proposed fixes.
                value should be JSON: {"fixes": [{"row_id": 2, "column": "email", "error_type": "invalid_format", "current_value": "jane@email", "corrected_value": "jane@email.com"}, ...]}
            - "submit": Finalize submission. No value needed.
        value: JSON string containing the action payload.
    """
    action_type: str
    value: str = ""


@dataclass
class DataCleaningObservation:
    """
    Observation returned by the environment to the agent.

    Attributes:
        dataset_text: The dataset rendered as a formatted text table.
        task_name: Name of the current task (task_1_identify, task_2_classify, task_3_fix).
        task_description: Human-readable description of what the agent should do.
        available_actions: List of valid action_type strings for this task.
        feedback: Feedback from the previous action (empty on reset).
        step_number: Current step number in the episode.
        max_steps: Maximum number of steps allowed.
        num_rows: Number of rows in the dataset.
        num_columns: Number of columns in the dataset.
        column_names: List of column names in the dataset.
        done: Whether the episode is finished.
        reward: Reward for the last action taken.
        metadata: Additional metadata.
    """
    dataset_text: str = ""
    task_name: str = ""
    task_description: str = ""
    available_actions: List[str] = field(default_factory=list)
    feedback: str = ""
    step_number: int = 0
    max_steps: int = 10
    num_rows: int = 0
    num_columns: int = 0
    column_names: List[str] = field(default_factory=list)
    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataCleaningState:
    """
    Episode state metadata.

    Attributes:
        episode_id: Unique identifier for this episode.
        step_count: Number of steps taken so far.
        task_name: Name of the current task.
        total_errors: Total number of planted errors in the dataset.
        errors_found: Number of errors correctly identified so far.
        cumulative_reward: Total reward accumulated in this episode.
    """
    episode_id: str = ""
    step_count: int = 0
    task_name: str = ""
    total_errors: int = 0
    errors_found: int = 0
    cumulative_reward: float = 0.0
