# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Tests for the Data Cleaning environment.

Run with: pytest tests/test_environment.py -v
"""

import json
import os
import sys

import pytest

# Add parent directory to path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import DataCleaningAction, DataCleaningObservation, DataCleaningState
from server.environment import DataCleaningEnvironment, TASKS


class TestEnvironmentReset:
    """Test environment reset functionality."""

    @pytest.mark.parametrize("task_name", list(TASKS.keys()))
    def test_reset_returns_observation(self, task_name):
        env = DataCleaningEnvironment(task_name=task_name)
        obs = env.reset()
        assert isinstance(obs, DataCleaningObservation)
        assert obs.task_name == task_name
        assert obs.num_rows > 0
        assert obs.num_columns > 0

    def test_reset_initializes_ground_truth(self):
        env = DataCleaningEnvironment(task_name="task_1_identify")
        env.reset()
        state = env.state
        assert state.total_errors > 0


class TestEnvironmentStep:
    """Test environment step functionality."""

    def setup_method(self):
        self.env = DataCleaningEnvironment(task_name="task_1_identify")
        self.env.reset()

    def test_check_schema_tool(self):
        action = DataCleaningAction(action_type="check_schema")
        obs = self.env.step(action)
        assert obs.tool_output is not None
        assert "columns" in obs.tool_output
        assert obs.reward >= 0

    def test_run_statistics_tool(self):
        action = DataCleaningAction(action_type="run_statistics")
        obs = self.env.step(action)
        assert obs.tool_output is not None
        assert "monthly_amount" in obs.tool_output
        assert obs.reward >= 0

    def test_search_reference_tool(self):
        action = DataCleaningAction(action_type="search_reference", value="pricing")
        obs = self.env.step(action)
        assert obs.tool_output is not None
        assert obs.reward >= 0

    def test_unknown_action_returns_negative_reward(self):
        action = DataCleaningAction(action_type="invalid_action")
        obs = self.env.step(action)
        assert obs.reward < 0


class TestTaskIdentification:
    """Test Task 1: Error Identification."""

    def setup_method(self):
        self.env = DataCleaningEnvironment(task_name="task_1_identify")
        self.env.reset()

    def test_correct_identification(self):
        # Get actual error row IDs from the environment
        state = self.env.state
        action = DataCleaningAction(
            action_type="identify_errors",
            value=json.dumps({"row_ids": list(range(1, state.total_errors + 1))}),
        )
        obs = self.env.step(action)
        assert 0 <= obs.reward <= 1

    def test_empty_identification(self):
        action = DataCleaningAction(
            action_type="identify_errors",
            value=json.dumps({"row_ids": []}),
        )
        obs = self.env.step(action)
        assert obs.reward == pytest.approx(0.001, abs=0.001)  # epsilon

    def test_invalid_json_returns_error(self):
        action = DataCleaningAction(
            action_type="identify_errors",
            value="not valid json",
        )
        obs = self.env.step(action)
        assert obs.reward < 0


class TestTaskClassification:
    """Test Task 2: Error Classification."""

    def setup_method(self):
        self.env = DataCleaningEnvironment(task_name="task_2_classify")
        self.env.reset()

    def test_classification_returns_reward(self):
        action = DataCleaningAction(
            action_type="classify_errors",
            value=json.dumps({"errors": []}),
        )
        obs = self.env.step(action)
        assert 0 <= obs.reward <= 1


class TestTaskFix:
    """Test Task 3: Error Correction."""

    def setup_method(self):
        self.env = DataCleaningEnvironment(task_name="task_3_fix")
        self.env.reset()

    def test_fix_returns_reward(self):
        action = DataCleaningAction(
            action_type="fix_errors",
            value=json.dumps({"fixes": []}),
        )
        obs = self.env.step(action)
        assert 0 <= obs.reward <= 1


class TestTaskInsight:
    """Test Task 4: Quality Insights."""

    def setup_method(self):
        self.env = DataCleaningEnvironment(task_name="task_4_insight")
        self.env.reset()

    def test_numeric_answer_returns_reward(self):
        action = DataCleaningAction(action_type="answer_insight", value="100.0")
        obs = self.env.step(action)
        assert 0 <= obs.reward <= 1

    def test_invalid_answer_returns_negative(self):
        action = DataCleaningAction(action_type="answer_insight", value="not_a_number")
        obs = self.env.step(action)
        assert obs.reward < 0


class TestState:
    """Test environment state functionality."""

    def test_state_returns_dataclass(self):
        env = DataCleaningEnvironment(task_name="task_1_identify")
        env.reset()
        state = env.state
        assert isinstance(state, DataCleaningState)
        assert state.episode_id != ""
        assert state.step_count == 0

    def test_step_count_increments(self):
        env = DataCleaningEnvironment(task_name="task_1_identify")
        env.reset()
        env.step(DataCleaningAction(action_type="check_schema"))
        state = env.state
        assert state.step_count == 1

    def test_done_after_max_steps(self):
        env = DataCleaningEnvironment(task_name="task_1_identify")
        env.reset()
        # Take max steps
        for _ in range(15):
            obs = env.step(DataCleaningAction(action_type="check_schema"))
        assert obs.done


class TestModels:
    """Test Pydantic model validation."""

    def test_action_creation(self):
        action = DataCleaningAction(action_type="test_action", value="test_value")
        assert action.action_type == "test_action"
        assert action.value == "test_value"

    def test_observation_defaults(self):
        obs = DataCleaningObservation()
        assert obs.dataset_text == ""
        assert obs.task_name == ""
        assert obs.done is False
        assert obs.reward == 0.0

    def test_state_defaults(self):
        state = DataCleaningState()
        assert state.episode_id == ""
        assert state.step_count == 0
        assert state.cumulative_reward == 0.0


class TestScoreNormalization:
    """Test score normalization function."""

    def test_score_within_range(self):
        from server.environment import _normalize_task_score
        assert 0.001 <= _normalize_task_score(0.5) <= 1.0

    def test_negative_score_clamped(self):
        from server.environment import _normalize_task_score
        result = _normalize_task_score(-0.5)
        assert result >= 0.001

    def test_score_above_one_clamped(self):
        from server.environment import _normalize_task_score
        result = _normalize_task_score(2.0)
        assert result <= 1.0
