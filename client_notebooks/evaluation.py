# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Client Evaluation Notebook for Data Cleaning Environment.

This script demonstrates how to interact with the Data Cleaning environment
programmatically for evaluation and testing purposes.

Usage:
    python client_notebooks/evaluation.py

Environment variables:
    ENV_URL - Environment server URL (default: http://localhost:7860)
"""

import json
import os
import sys

import requests

# ─── Configuration ────────────────────────────────────────────────────

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
TASKS = ["task_1_identify", "task_2_classify", "task_3_fix", "task_4_insight"]


# ─── Helper Functions ─────────────────────────────────────────────────

def reset(task_name: str = "task_1_identify") -> dict:
    """Reset the environment and create a new session."""
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_name": task_name},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def step(session_id: str, action_type: str, value: str = "") -> dict:
    """Execute a step in an existing session."""
    resp = requests.post(
        f"{ENV_URL}/step",
        json={
            "session_id": session_id,
            "action_type": action_type,
            "value": value,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def run_evaluation():
    """Run a simple evaluation across all tasks."""
    print("=" * 60)
    print("Data Cleaning Environment — Client Evaluation")
    print(f"Environment: {ENV_URL}")
    print("=" * 60)

    # Check health
    health = requests.get(f"{ENV_URL}/health", timeout=10)
    print(f"\nHealth check: {health.json()}")

    # Check info
    info = requests.get(f"{ENV_URL}/info", timeout=10)
    info_data = info.json()
    print(f"Environment: {info_data.get('name')} v{info_data.get('version')}")
    print(f"Available tasks: {info_data.get('tasks')}")
    print(f"Available tools: {info_data.get('tools')}")

    # Run each task
    for task in TASKS:
        print(f"\n{'─' * 40}")
        print(f"Evaluating: {task}")
        print(f"{'─' * 40}")

        result = reset(task)
        session_id = result.get("session_id")
        obs = result.get("observation", {})
        reward = result.get("reward", 0.0)

        print(f"  Session: {session_id}")
        print(f"  Task: {obs.get('task_name')}")
        print(f"  Rows: {obs.get('num_rows')}")
        print(f"  Available actions: {obs.get('available_actions')}")

        # Example: use check_schema tool
        tool_result = step(session_id, "check_schema")
        tool_reward = tool_result.get("reward", 0.0)
        print(f"  check_schema reward: {tool_reward:.3f}")

        # Example: use run_statistics tool
        stats_result = step(session_id, "run_statistics")
        stats_reward = stats_result.get("reward", 0.0)
        print(f"  run_statistics reward: {stats_reward:.3f}")

        # Submit
        submit_result = step(session_id, "submit")
        final_reward = submit_result.get("reward", 0.0)
        done = submit_result.get("done", False)
        print(f"  Final reward: {final_reward:.3f} (done={done})")

    print(f"\n{'=' * 60}")
    print("Evaluation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_evaluation()
