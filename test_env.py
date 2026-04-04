"""Quick smoke test for the data cleaning environment logic."""

import sys
import json
sys.path.insert(0, ".")

from data import get_dirty_dataset, get_ground_truth_errors, format_dataset_as_table, get_error_row_ids
from models import DataCleaningAction, DataCleaningObservation
from server.environment import DataCleaningEnvironment

def test_task_1():
    """Test Task 1: Error Identification."""
    print("=" * 50)
    print("TEST: Task 1 — Error Identification")
    print("=" * 50)
    
    env = DataCleaningEnvironment(task_name="task_1_identify")
    obs = env.reset()
    
    print(f"Task: {obs.task_name}")
    print(f"Available actions: {obs.available_actions}")
    print(f"Num rows: {obs.num_rows}")
    print(f"Max steps: {obs.max_steps}")
    
    # Submit perfect answer
    true_ids = get_error_row_ids()
    print(f"\nGround truth error rows: {true_ids}")
    
    action = DataCleaningAction(
        action_type="identify_errors",
        value=json.dumps({"row_ids": true_ids})
    )
    obs = env.step(action)
    print(f"\nPerfect answer reward: {obs.reward}")
    print(f"Feedback: {obs.feedback}")
    assert obs.reward == 1.0, f"Expected 1.0 for perfect answer, got {obs.reward}"
    
    # Submit and finalize
    action = DataCleaningAction(action_type="submit", value="")
    obs = env.step(action)
    print(f"Submit reward: {obs.reward}")
    print(f"Done: {obs.done}")
    assert obs.done == True
    print("✅ Task 1 PASSED\n")


def test_task_2():
    """Test Task 2: Error Classification."""
    print("=" * 50)
    print("TEST: Task 2 — Error Classification")
    print("=" * 50)
    
    env = DataCleaningEnvironment(task_name="task_2_classify")
    obs = env.reset()
    print(f"Task: {obs.task_name}")
    
    # Submit perfect answer
    gt = get_ground_truth_errors()
    errors = [{"row_id": e["row_id"], "column": e["column"], "error_type": e["error_type"]} for e in gt]
    
    action = DataCleaningAction(
        action_type="classify_errors",
        value=json.dumps({"errors": errors})
    )
    obs = env.step(action)
    print(f"Perfect answer reward: {obs.reward}")
    print(f"Feedback: {obs.feedback}")
    assert obs.reward == 1.0, f"Expected 1.0, got {obs.reward}"

    action = DataCleaningAction(action_type="submit", value="")
    obs = env.step(action)
    assert obs.done == True
    print("✅ Task 2 PASSED\n")


def test_task_3():
    """Test Task 3: Error Correction."""
    print("=" * 50)
    print("TEST: Task 3 — Error Correction")
    print("=" * 50)
    
    env = DataCleaningEnvironment(task_name="task_3_fix")
    obs = env.reset()
    print(f"Task: {obs.task_name}")
    
    # Submit perfect answer
    gt = get_ground_truth_errors()
    fixes = [{
        "row_id": e["row_id"],
        "column": e["column"],
        "error_type": e["error_type"],
        "current_value": str(e["current_value"]),
        "corrected_value": str(e["corrected_value"])
    } for e in gt]
    
    action = DataCleaningAction(
        action_type="fix_errors",
        value=json.dumps({"fixes": fixes})
    )
    obs = env.step(action)
    print(f"Perfect answer reward: {obs.reward}")
    print(f"Feedback: {obs.feedback}")
    assert obs.reward == 1.0, f"Expected 1.0, got {obs.reward}"

    action = DataCleaningAction(action_type="submit", value="")
    obs = env.step(action)
    assert obs.done == True
    print("✅ Task 3 PASSED\n")


def test_partial_credit():
    """Test that partial answers get partial credit."""
    print("=" * 50)
    print("TEST: Partial Credit")
    print("=" * 50)
    
    env = DataCleaningEnvironment(task_name="task_1_identify")
    obs = env.reset()
    
    # Submit only half the errors
    true_ids = get_error_row_ids()
    partial_ids = true_ids[:5]
    
    action = DataCleaningAction(
        action_type="identify_errors",
        value=json.dumps({"row_ids": partial_ids})
    )
    obs = env.step(action)
    print(f"Half answers reward: {obs.reward}")
    assert 0.0 < obs.reward < 1.0, f"Expected partial credit, got {obs.reward}"
    print(f"Feedback: {obs.feedback}")
    print("✅ Partial Credit PASSED\n")


def test_invalid_action():
    """Test that invalid actions are handled gracefully."""
    print("=" * 50)
    print("TEST: Invalid Action Handling")
    print("=" * 50)
    
    env = DataCleaningEnvironment(task_name="task_1_identify")
    obs = env.reset()
    
    # Send invalid action type
    action = DataCleaningAction(action_type="invalid_action", value="")
    obs = env.step(action)
    print(f"Invalid action reward: {obs.reward}")
    assert obs.reward == -0.05, f"Expected -0.05 penalty, got {obs.reward}"
    print(f"Feedback: {obs.feedback}")
    
    # Send bad JSON
    action = DataCleaningAction(action_type="identify_errors", value="not json")
    obs = env.step(action)
    print(f"Bad JSON reward: {obs.reward}")
    assert obs.reward == -0.05
    print("✅ Invalid Action PASSED\n")


def test_dataset_display():
    """Test that the dataset formats correctly."""
    print("=" * 50)
    print("TEST: Dataset Display")
    print("=" * 50)
    
    dataset = get_dirty_dataset()
    table = format_dataset_as_table(dataset)
    print(table[:500])
    print("...")
    print("✅ Dataset Display PASSED\n")


if __name__ == "__main__":
    test_dataset_display()
    test_task_1()
    test_task_2()
    test_task_3()
    test_partial_credit()
    test_invalid_action()
    
    print("=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 50)
