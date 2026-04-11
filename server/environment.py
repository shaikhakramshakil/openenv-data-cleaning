# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Core environment logic for the Data Cleaning environment.

Implements the OpenEnv Environment interface with reset(), step(), and state.
Manages the data cleaning tasks, validates agent actions, and calculates rewards.
Now includes Tool-Use capabilities and complex statistical insight tasks.
"""

import json
import uuid
import math
from typing import Any, Dict, List, Optional

from data import (
    COLUMNS,
    format_dataset_as_table,
    get_dataset_summary,
    get_dirty_dataset,
    get_error_row_ids,
    get_ground_truth_errors,
    get_validation_rules,
    PLAN_PRICING,
)
from models import DataCleaningAction, DataCleaningObservation, DataCleaningState


SCORE_EPSILON = 0.001

# Task definitions
TASKS = {
    "task_1_identify": {
        "name": "task_1_identify",
        "description": (
            "TASK 1 — Error Identification\n"
            "Identify ALL rows containing errors. Use tools to verify your findings if needed."
        ),
        "available_actions": ["identify_errors", "check_schema", "search_reference", "run_statistics", "submit"],
        "max_steps": 15,
        "difficulty": "easy",
    },
    "task_2_classify": {
        "name": "task_2_classify",
        "description": (
            "TASK 2 — Error Classification\n"
            "Identify and classify ALL errors into: missing_value, invalid_format, outlier, duplicate, type_error, inconsistency."
        ),
        "available_actions": ["classify_errors", "check_schema", "search_reference", "run_statistics", "submit"],
        "max_steps": 15,
        "difficulty": "medium",
    },
    "task_3_fix": {
        "name": "task_3_fix",
        "description": (
            "TASK 3 — Error Correction\n"
            "Identify, classify, and provide the correct value for ALL errors."
        ),
        "available_actions": ["fix_errors", "check_schema", "search_reference", "run_statistics", "submit"],
        "max_steps": 20,
        "difficulty": "hard",
    },
    "task_4_insight": {
        "name": "task_4_insight",
        "description": (
            "TASK 4 — Quality Insights (Expert)\n"
            "Calculate the total 'monthly_amount' only for 'active' users AFTER removing all pricing inconsistencies. "
            "Use tools to run statistics or check the schema. Provide your answer as a numeric string."
        ),
        "available_actions": ["answer_insight", "check_schema", "search_reference", "run_statistics", "submit"],
        "max_steps": 15,
        "difficulty": "expert",
    },
}


class DataCleaningEnvironment:
    """
    OpenEnv Environment with Tool-Use for data quality assessment.
    """

    def __init__(self, task_name: str = "task_1_identify"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}")

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
        self._tool_output: Optional[Any] = None

    def reset(self, **kwargs) -> DataCleaningObservation:
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
        self._tool_output = None

        return self._make_observation(
            feedback=f"New episode for {self._task_name}. Dataset loaded with {len(self._dirty_dataset)} rows.",
            reward=0.0
        )

    def step(self, action: DataCleaningAction) -> DataCleaningObservation:
        if self._done:
            return self._make_observation("Episode finished.", 0.0)

        self._step_count += 1
        self._tool_output = None # Clear previous tool output

        # Check for step limit
        if self._step_count >= self._task_config["max_steps"]:
            self._done = True
            return self._handle_submit()

        # Handle Actions
        ac = action.action_type
        val = action.value

        if ac == "submit":
            return self._handle_submit()
        elif ac == "check_schema":
            return self._handle_check_schema()
        elif ac == "run_statistics":
            return self._handle_run_stats()
        elif ac == "search_reference":
            return self._handle_search(val)
        elif ac == "identify_errors":
            return self._handle_identify(val)
        elif ac == "classify_errors":
            return self._handle_classify(val)
        elif ac == "fix_errors":
            return self._handle_fix(val)
        elif ac == "answer_insight":
            return self._handle_insight(val)
        else:
            self._last_reward = -0.01
            return self._make_observation(f"Unknown action '{ac}'",-0.01)

    # ─── Tool Handlers ────────────────────────────────────────────────

    def _handle_check_schema(self) -> DataCleaningObservation:
        rules = get_validation_rules()
        self._tool_output = rules
        self._last_reward = 0.01 # Small reward for exploring tools
        return self._make_observation("Retrieved schema and validation rules.", 0.01)

    def _handle_run_stats(self) -> DataCleaningObservation:
        import statistics
        amounts = []
        ages = []
        for r in self._dirty_dataset:
            try: amounts.append(float(r["monthly_amount"]))
            except: pass
            try: ages.append(float(r["age"]))
            except: pass
        
        stats = {
            "row_count": len(self._dirty_dataset),
            "monthly_amount": {
                "mean": statistics.mean(amounts) if amounts else 0,
                "sum": sum(amounts) if amounts else 0,
                "min": min(amounts) if amounts else 0,
                "max": max(amounts) if amounts else 0,
            },
            "age": {
                "mean": statistics.mean(ages) if ages else 0,
                "median": statistics.median(ages) if ages else 0
            }
        }
        self._tool_output = stats
        self._last_reward = 0.01
        return self._make_observation("Calculated dataset statistics.", 0.01)

    def _handle_search(self, query: str) -> DataCleaningObservation:
        rules = get_validation_rules()
        query = query.lower().strip()
        results = []
        if "city" in query or "cities" in query:
            results = rules["cities"]
        elif "plan" in query or "pricing" in query:
            results = [f"{k}: ${v}" for k, v in rules["pricing"].items()]
        else:
            results = ["No specific reference found for query. Try 'cities' or 'pricing'."]
        
        self._tool_output = results
        self._last_reward = 0.01
        return self._make_observation(f"Search results for '{query}'.", 0.01)

    # ─── Task Handlers ────────────────────────────────────────────────

    def _handle_identify(self, value: str) -> DataCleaningObservation:
        try:
            data = json.loads(value)
            pred_ids = set(data.get("row_ids", []))
            true_ids = set(self._error_row_ids)
            f1 = self._calc_f1(pred_ids, true_ids)
            reward = _normalize_task_score(f1)
            self._update_best(reward, data)
            return self._make_observation(f"Identification F1: {f1:.2f}. Correct: {len(pred_ids & true_ids)}/{len(true_ids)}", reward)
        except Exception as e:
            return self._make_observation(f"JSON Error: {e}", -0.05)

    def _handle_classify(self, value: str) -> DataCleaningObservation:
        try:
            data = json.loads(value)
            preds = data.get("errors", [])
            matches = 0
            for p in preds:
                for gt in self._ground_truth:
                    if p["row_id"] == gt["row_id"] and p["column"] == gt["column"]:
                        if p["error_type"] == gt["error_type"]:
                            matches += 1
            score = matches / max(len(self._ground_truth), 1)
            reward = _normalize_task_score(score)
            self._update_best(reward, data)
            return self._make_observation(f"Classification Score: {score:.2f}", reward)
        except Exception as e:
            return self._make_observation(f"JSON Error: {e}", -0.05)

    def _handle_fix(self, value: str) -> DataCleaningObservation:
        try:
            data = json.loads(value)
            preds = data.get("fixes", [])
            score = 0
            for p in preds:
                for gt in self._ground_truth:
                    if p["row_id"] == gt["row_id"] and p["column"] == gt["column"]:
                        if str(p["corrected_value"]).lower() == str(gt["corrected_value"]).lower():
                            score += 1
                        elif _is_close_numeric(p["corrected_value"], gt["corrected_value"]):
                            score += 0.5
            final = score / max(len(self._ground_truth), 1)
            reward = _normalize_task_score(final)
            self._update_best(reward, data)
            return self._make_observation(f"Fix Score: {final:.2f}", reward)
        except Exception as e:
            return self._make_observation(f"JSON Error: {e}", -0.05)

    def _handle_insight(self, value: str) -> DataCleaningObservation:
        # Calculate Ground Truth for Task 4
        # "monthly_amount" for "active" users after fixing ALL inconsistencies
        # Important: we must use the CLEAN (corrected) values for BOTH status and amount
        total = 0.0
        for r in self._dirty_dataset:
            # Determine the CORRECTED status value
            corrected_status = r["status"]
            corrected_amount = r["monthly_amount"]

            for gt in self._ground_truth:
                if gt["row_id"] == r["id"]:
                    if gt["column"] == "status":
                        corrected_status = gt["corrected_value"]
                    elif gt["column"] == "monthly_amount":
                        corrected_amount = gt["corrected_value"]

            # Only count users whose CORRECTED status is "active"
            if str(corrected_status).lower().strip() == "active":
                total += float(corrected_amount)

        try:
            pred = float(value.replace("$", "").strip())
            diff = abs(pred - total)
            if diff < 0.01: score = 1.0
            elif diff < 10: score = 0.5
            else: score = 0.0
            reward = _normalize_task_score(score)
            self._update_best(reward, {"answer": value})
            return self._make_observation(f"Insight Accuracy Reward: {reward:.2f}", reward)
        except:
            return self._make_observation("Please provide a numeric value.", -0.05)

    def _handle_submit(self) -> DataCleaningObservation:
        self._done = True
        return self._make_observation(f"Episode complete. Final best score: {self._best_score:.3f}", self._best_score)

    def _calc_f1(self, pred, true):
        if not pred: return 0.0
        tp = len(pred & true)
        prec = tp / len(pred)
        rec = tp / len(true)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    def _update_best(self, score, data):
        if score > self._best_score:
            self._best_score = score
            self._last_submission = data

    def _make_observation(self, feedback: str, reward: float) -> DataCleaningObservation:
        self._cumulative_reward += reward
        return DataCleaningObservation(
            dataset_text=format_dataset_as_table(self._dirty_dataset),
            task_name=self._task_name,
            task_description=self._task_config["description"],
            available_actions=self._task_config["available_actions"],
            feedback=feedback,
            tool_output=self._tool_output,
            step_number=self._step_count,
            max_steps=self._task_config["max_steps"],
            num_rows=len(self._dirty_dataset),
            num_columns=len(COLUMNS),
            column_names=COLUMNS,
            done=self._done,
            reward=reward,
            metadata={"episode_id": self._episode_id}
        )

    @property
    def state(self) -> DataCleaningState:
        return DataCleaningState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            total_errors=len(self._ground_truth),
            cumulative_reward=self._cumulative_reward
        )

def _is_close_numeric(a, b):
    try: return abs(float(a) - float(b)) < 0.01
    except: return False

def _normalize_task_score(score: float) -> float:
    """Clamp and round task scores so they are always strictly between 0 and 1."""
    return round(min(max(float(score), SCORE_EPSILON), 1.0), 3)
