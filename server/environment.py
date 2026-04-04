"""
Core environment logic for the Data Cleaning environment.

Implements the OpenEnv Environment interface with reset(), step(), and state.
Manages the data cleaning tasks, validates agent actions, and calculates rewards.
"""

import json
import uuid
from typing import Any, Dict, List, Optional

from data import (
    COLUMNS,
    format_dataset_as_table,
    get_clean_dataset,
    get_dataset_summary,
    get_dirty_dataset,
    get_error_row_ids,
    get_ground_truth_errors,
)
from models import DataCleaningAction, DataCleaningObservation, DataCleaningState


# Task definitions
TASKS = {
    "task_1_identify": {
        "name": "task_1_identify",
        "description": (
            "TASK 1 — Error Identification (Easy)\n"
            "Examine the dataset below and identify ALL rows that contain data quality errors.\n"
            "Send your answer using the 'identify_errors' action with a JSON payload:\n"
            '  {"row_ids": [2, 5, 8]}  (list the 1-indexed row IDs with errors)\n'
            "When you are confident in your answer, use the 'submit' action to finalize.\n"
            "You can refine your answer before submitting."
        ),
        "available_actions": ["identify_errors", "submit"],
        "max_steps": 5,
        "difficulty": "easy",
    },
    "task_2_classify": {
        "name": "task_2_classify",
        "description": (
            "TASK 2 — Error Classification (Medium)\n"
            "Examine the dataset and identify ALL data quality errors.\n"
            "For each error, specify the row, column, and error type.\n"
            "Valid error types: missing_value, invalid_format, outlier, duplicate, type_error, inconsistency\n"
            "Send your answer using the 'classify_errors' action with a JSON payload:\n"
            '  {"errors": [{"row_id": 2, "column": "email", "error_type": "invalid_format"}, ...]}\n'
            "When you are confident, use 'submit' to finalize."
        ),
        "available_actions": ["classify_errors", "submit"],
        "max_steps": 5,
        "difficulty": "medium",
    },
    "task_3_fix": {
        "name": "task_3_fix",
        "description": (
            "TASK 3 — Error Detection and Correction (Hard)\n"
            "Examine the dataset and identify ALL data quality errors.\n"
            "For each error, specify the row, column, error type, the current wrong value, and the corrected value.\n"
            "Valid error types: missing_value, invalid_format, outlier, duplicate, type_error, inconsistency\n"
            "Send your answer using the 'fix_errors' action with a JSON payload:\n"
            '  {"fixes": [{"row_id": 2, "column": "email", "error_type": "invalid_format", '
            '"current_value": "bob.smith@yahoo", "corrected_value": "bob.smith@yahoo.com"}, ...]}\n'
            "When you are confident, use 'submit' to finalize."
        ),
        "available_actions": ["fix_errors", "submit"],
        "max_steps": 5,
        "difficulty": "hard",
    },
}


class DataCleaningEnvironment:
    """
    OpenEnv Environment for data quality assessment.

    The agent is presented with a 'dirty' dataset containing planted errors
    and must identify, classify, or fix them depending on the task.
    """

    def __init__(self, task_name: str = "task_1_identify"):
        """
        Initialize the environment.

        Args:
            task_name: Which task to run. One of: task_1_identify, task_2_classify, task_3_fix.
        """
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Must be one of: {list(TASKS.keys())}")

        self._task_name = task_name
        self._task_config = TASKS[task_name]
        self._dirty_dataset: List[Dict[str, Any]] = []
        self._ground_truth: List[Dict[str, Any]] = []
        self._error_row_ids: List[int] = []
        self._episode_id: str = ""
        self._step_count: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._last_reward: float = 0.0
        self._last_submission: Optional[Dict[str, Any]] = None
        self._best_score: float = 0.0
        self._feedback: str = ""

    def reset(self, **kwargs) -> DataCleaningObservation:
        """
        Reset the environment to start a new episode.

        Returns:
            Initial observation with the dirty dataset and task instructions.
        """
        self._dirty_dataset = get_dirty_dataset()
        self._ground_truth = get_ground_truth_errors()
        self._error_row_ids = get_error_row_ids()
        self._episode_id = kwargs.get("episode_id", str(uuid.uuid4())[:8])
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._last_reward = 0.0
        self._last_submission = None
        self._best_score = 0.0
        self._feedback = ""

        dataset_text = format_dataset_as_table(self._dirty_dataset)
        summary = get_dataset_summary()

        return DataCleaningObservation(
            dataset_text=f"{summary}\n\n{dataset_text}",
            task_name=self._task_config["name"],
            task_description=self._task_config["description"],
            available_actions=self._task_config["available_actions"],
            feedback="Episode started. Examine the dataset and find the errors.",
            step_number=0,
            max_steps=self._task_config["max_steps"],
            num_rows=len(self._dirty_dataset),
            num_columns=len(COLUMNS),
            column_names=list(COLUMNS),
            done=False,
            reward=0.0,
            metadata={"episode_id": self._episode_id, "task_difficulty": self._task_config["difficulty"]},
        )

    def step(self, action: DataCleaningAction) -> DataCleaningObservation:
        """
        Execute an agent action and return the resulting observation.

        Args:
            action: The action to execute.

        Returns:
            Observation with feedback and reward.
        """
        if self._done:
            return self._make_observation(
                feedback="Episode is already done. Call reset() to start a new episode.",
                reward=0.0,
            )

        self._step_count += 1

        # Check for step limit
        if self._step_count >= self._task_config["max_steps"]:
            self._done = True
            # Auto-submit the best submission if any
            if self._last_submission is not None:
                reward = self._best_score
                feedback = f"Step limit reached. Auto-submitting your best answer (score: {self._best_score:.2f})."
            else:
                reward = 0.0
                feedback = "Step limit reached with no submission. Score: 0.0."
            self._last_reward = reward
            return self._make_observation(feedback=feedback, reward=reward)

        # Validate action type
        valid_actions = self._task_config["available_actions"]
        if action.action_type not in valid_actions:
            self._last_reward = -0.05
            self._cumulative_reward += self._last_reward
            feedback = (
                f"Invalid action type '{action.action_type}'. "
                f"Valid actions for this task: {valid_actions}"
            )
            return self._make_observation(feedback=feedback, reward=self._last_reward)

        # Process action
        if action.action_type == "submit":
            return self._handle_submit()
        elif action.action_type == "identify_errors":
            return self._handle_identify(action.value)
        elif action.action_type == "classify_errors":
            return self._handle_classify(action.value)
        elif action.action_type == "fix_errors":
            return self._handle_fix(action.value)
        else:
            self._last_reward = -0.05
            self._cumulative_reward += self._last_reward
            return self._make_observation(
                feedback=f"Unrecognized action: {action.action_type}",
                reward=self._last_reward,
            )

    @property
    def state(self) -> DataCleaningState:
        """Return current episode state metadata."""
        return DataCleaningState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            total_errors=len(self._ground_truth),
            errors_found=0,  # Updated during grading
            cumulative_reward=self._cumulative_reward,
        )

    # ─── Action Handlers ──────────────────────────────────────────────

    def _handle_identify(self, value: str) -> DataCleaningObservation:
        """Handle 'identify_errors' action for Task 1."""
        try:
            data = json.loads(value)
            row_ids = data.get("row_ids", [])
            if not isinstance(row_ids, list):
                raise ValueError("'row_ids' must be a list of integers")
            row_ids = [int(r) for r in row_ids]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self._last_reward = -0.05
            self._cumulative_reward += self._last_reward
            return self._make_observation(
                feedback=f"Invalid JSON format. Error: {e}. Expected: {{\"row_ids\": [2, 5, 8]}}",
                reward=self._last_reward,
            )

        # Grade the identification
        true_ids = set(self._error_row_ids)
        predicted_ids = set(row_ids)

        true_positives = len(true_ids & predicted_ids)
        false_positives = len(predicted_ids - true_ids)
        false_negatives = len(true_ids - predicted_ids)

        precision = true_positives / max(len(predicted_ids), 1)
        recall = true_positives / max(len(true_ids), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        # Partial reward based on F1 score
        reward = round(f1, 3)
        self._last_reward = reward
        self._cumulative_reward += reward

        # Track best submission
        if reward > self._best_score:
            self._best_score = reward
            self._last_submission = {"row_ids": row_ids}

        # Build feedback
        feedback_parts = [
            f"Identification score: {reward:.2f} (F1)",
            f"  Correct identifications: {true_positives}/{len(true_ids)}",
            f"  False alarms: {false_positives}",
            f"  Missed errors: {false_negatives}",
        ]

        if false_negatives > 0:
            feedback_parts.append("  Hint: There are more errors in the dataset. Look carefully at data types, formats, and consistency.")

        if false_positives > 0:
            feedback_parts.append("  Hint: Some rows you flagged are actually clean. Double-check the validation rules.")

        return self._make_observation(
            feedback="\n".join(feedback_parts),
            reward=reward,
        )

    def _handle_classify(self, value: str) -> DataCleaningObservation:
        """Handle 'classify_errors' action for Task 2."""
        try:
            data = json.loads(value)
            errors = data.get("errors", [])
            if not isinstance(errors, list):
                raise ValueError("'errors' must be a list")
            # Validate each error entry has required fields
            for e in errors:
                if not all(k in e for k in ("row_id", "column", "error_type")):
                    raise ValueError("Each error must have 'row_id', 'column', and 'error_type'")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self._last_reward = -0.05
            self._cumulative_reward += self._last_reward
            return self._make_observation(
                feedback=f"Invalid JSON format. Error: {e}. Expected: {{\"errors\": [{{\"row_id\": 2, \"column\": \"email\", \"error_type\": \"invalid_format\"}}]}}",
                reward=self._last_reward,
            )

        # Grade classification
        total_gt = len(self._ground_truth)
        location_matches = 0
        full_matches = 0

        for pred in errors:
            pred_row = int(pred["row_id"])
            pred_col = pred["column"].strip().lower()
            pred_type = pred["error_type"].strip().lower()

            for gt in self._ground_truth:
                if gt["row_id"] == pred_row and gt["column"].lower() == pred_col:
                    location_matches += 1
                    if gt["error_type"].lower() == pred_type:
                        full_matches += 1
                    break

        # Score: 50% for finding the right location, 50% for correct type
        location_score = location_matches / max(total_gt, 1)
        type_score = full_matches / max(total_gt, 1)
        reward = round(0.5 * location_score + 0.5 * type_score, 3)

        # Penalize false positives lightly
        false_positives = max(0, len(errors) - location_matches)
        penalty = min(0.1, false_positives * 0.02)
        reward = max(0.0, round(reward - penalty, 3))

        self._last_reward = reward
        self._cumulative_reward += reward

        if reward > self._best_score:
            self._best_score = reward
            self._last_submission = {"errors": errors}

        feedback_parts = [
            f"Classification score: {reward:.2f}",
            f"  Errors found at correct location: {location_matches}/{total_gt}",
            f"  Errors with correct type: {full_matches}/{total_gt}",
            f"  False alarms: {false_positives}",
        ]

        missed = total_gt - location_matches
        if missed > 0:
            feedback_parts.append(f"  You missed {missed} errors. Check all columns carefully.")
        if location_matches > full_matches:
            feedback_parts.append(f"  {location_matches - full_matches} errors found but misclassified. Review error types.")

        return self._make_observation(
            feedback="\n".join(feedback_parts),
            reward=reward,
        )

    def _handle_fix(self, value: str) -> DataCleaningObservation:
        """Handle 'fix_errors' action for Task 3."""
        try:
            data = json.loads(value)
            fixes = data.get("fixes", [])
            if not isinstance(fixes, list):
                raise ValueError("'fixes' must be a list")
            for f in fixes:
                required = ("row_id", "column", "error_type", "current_value", "corrected_value")
                if not all(k in f for k in required):
                    raise ValueError(f"Each fix must have: {required}")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self._last_reward = -0.05
            self._cumulative_reward += self._last_reward
            return self._make_observation(
                feedback=f"Invalid JSON format. Error: {e}.",
                reward=self._last_reward,
            )

        # Grade fixes
        total_gt = len(self._ground_truth)
        location_matches = 0
        type_matches = 0
        fix_matches = 0

        for pred in fixes:
            pred_row = int(pred["row_id"])
            pred_col = pred["column"].strip().lower()
            pred_type = pred["error_type"].strip().lower()
            pred_fix = str(pred["corrected_value"]).strip().lower()

            for gt in self._ground_truth:
                if gt["row_id"] == pred_row and gt["column"].lower() == pred_col:
                    location_matches += 1
                    if gt["error_type"].lower() == pred_type:
                        type_matches += 1
                    # Check fix value (fuzzy match for strings, exact for numbers)
                    gt_fix = str(gt["corrected_value"]).strip().lower()
                    if pred_fix == gt_fix:
                        fix_matches += 1
                    elif _is_close_numeric(pred["corrected_value"], gt["corrected_value"]):
                        fix_matches += 0.5  # Partial credit for close numeric values
                    break

        # Score: 30% location, 30% type, 40% fix quality
        location_score = location_matches / max(total_gt, 1)
        type_score = type_matches / max(total_gt, 1)
        fix_score = fix_matches / max(total_gt, 1)
        reward = round(0.3 * location_score + 0.3 * type_score + 0.4 * fix_score, 3)

        # Penalize false positives
        false_positives = max(0, len(fixes) - location_matches)
        penalty = min(0.1, false_positives * 0.02)
        reward = max(0.0, round(reward - penalty, 3))

        self._last_reward = reward
        self._cumulative_reward += reward

        if reward > self._best_score:
            self._best_score = reward
            self._last_submission = {"fixes": fixes}

        feedback_parts = [
            f"Fix score: {reward:.2f}",
            f"  Errors located: {location_matches}/{total_gt}",
            f"  Correct error types: {type_matches}/{total_gt}",
            f"  Correct fixes: {fix_matches}/{total_gt}",
            f"  False alarms: {false_positives}",
        ]

        missed = total_gt - location_matches
        if missed > 0:
            feedback_parts.append(f"  You missed {missed} errors.")
        if type_matches < location_matches:
            feedback_parts.append(f"  Some error types are wrong. Review the error type categories.")
        if fix_matches < type_matches:
            feedback_parts.append(f"  Some corrections are not quite right. Check values carefully.")

        return self._make_observation(
            feedback="\n".join(feedback_parts),
            reward=reward,
        )

    def _handle_submit(self) -> DataCleaningObservation:
        """Handle 'submit' action — finalize the episode."""
        self._done = True

        if self._last_submission is not None:
            final_reward = self._best_score
            feedback = f"Submission accepted! Final score: {final_reward:.3f}"
        else:
            final_reward = 0.0
            feedback = "Submitted with no answer. Score: 0.0. You need to provide an answer before submitting."

        self._last_reward = final_reward
        return self._make_observation(feedback=feedback, reward=final_reward)

    # ─── Helpers ──────────────────────────────────────────────────────

    def _make_observation(self, feedback: str, reward: float) -> DataCleaningObservation:
        """Build an observation object with current state."""
        dataset_text = format_dataset_as_table(self._dirty_dataset)
        summary = get_dataset_summary()

        return DataCleaningObservation(
            dataset_text=f"{summary}\n\n{dataset_text}",
            task_name=self._task_config["name"],
            task_description=self._task_config["description"],
            available_actions=self._task_config["available_actions"],
            feedback=feedback,
            step_number=self._step_count,
            max_steps=self._task_config["max_steps"],
            num_rows=len(self._dirty_dataset),
            num_columns=len(COLUMNS),
            column_names=list(COLUMNS),
            done=self._done,
            reward=reward,
            metadata={
                "episode_id": self._episode_id,
                "task_difficulty": self._task_config["difficulty"],
                "best_score": self._best_score,
                "cumulative_reward": self._cumulative_reward,
            },
        )


def _is_close_numeric(a: Any, b: Any) -> bool:
    """Check if two values are numerically close."""
    try:
        return abs(float(a) - float(b)) < 0.01
    except (ValueError, TypeError):
        return False
