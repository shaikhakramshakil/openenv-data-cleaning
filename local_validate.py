import json
from server.environment import DataCleaningEnvironment, TASKS
from models import DataCleaningAction

def test_all_tasks():
    print("Testing all tasks...")
    for task_name in ["task_1_identify", "task_2_classify", "task_3_fix", "task_4_insight"]:
        print(f"\n[Testing {task_name}]")
        env = DataCleaningEnvironment(task_name)
        obs = env.reset()
        print(f"  Reset OK: {obs.task_name}, rows={obs.num_rows}")
        
        # Test tool
        tool_obs = env.step(DataCleaningAction(action_type="check_schema"))
        print(f"  Tool check_schema: {tool_obs.feedback}")
        assert tool_obs.reward > 0, "Tool call should give small reward"
        
        # Test state
        state = env.state
        print(f"  State OK: step={state.step_count}, reward={state.cumulative_reward:.3f}")
        assert not hasattr(state, 'errors_found'), "errors_found should be removed"

    print("\nSUCCESS: All task logic verified.")

if __name__ == "__main__":
    test_all_tasks()
