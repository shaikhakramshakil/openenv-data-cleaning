"""
Quick test script - tests all 3 tasks work correctly.
Run this while the server is running on port 7860.

Usage:
    1. Start server:  uvicorn server.app:app --host 0.0.0.0 --port 7860
    2. Run this:       python quick_test.py
"""
import asyncio
import json
import aiohttp

SERVER = "http://localhost:7860"


async def test_task(session, task_name, action_type, payload):
    """Test a single task end-to-end via WebSocket."""
    ws_url = SERVER.replace("http://", "ws://") + "/ws"

    print(f"\n{'='*50}")
    print(f"TESTING: {task_name}")
    print(f"{'='*50}")

    async with session.ws_connect(ws_url) as ws:
        # Step 1: Reset
        await ws.send_str(json.dumps({
            "type": "reset",
            "data": {"task_name": task_name}
        }))
        reset_resp = json.loads(await ws.receive_str())

        if reset_resp.get("type") == "error":
            print(f"  ❌ Reset FAILED: {reset_resp['data']['message']}")
            return False

        obs = reset_resp["data"]["observation"]
        print(f"  ✅ Reset OK")
        print(f"     Task: {obs['task_name']}")
        print(f"     Dataset rows: {obs['num_rows']}")
        print(f"     Max steps: {obs['max_steps']}")

        # Step 2: Send an action
        await ws.send_str(json.dumps({
            "type": "step",
            "data": {"action_type": action_type, "value": json.dumps(payload)}
        }))
        step_resp = json.loads(await ws.receive_str())

        if step_resp.get("type") == "error":
            print(f"  ❌ Step FAILED: {step_resp['data']['message']}")
            return False

        step_data = step_resp["data"]
        reward = step_data.get("reward", -1)
        done = step_data.get("done", False)
        feedback = step_data["observation"].get("feedback", "")

        print(f"  ✅ Step OK")
        print(f"     Reward: {reward}")
        print(f"     Done: {done}")
        print(f"     Feedback: {feedback[:200]}...")

        # Verify reward is in valid range
        if not (0.0 <= reward <= 1.0 or reward == -0.05):
            print(f"  ⚠️  WARNING: Reward {reward} may be outside expected range!")

        # Step 3: Check state
        await ws.send_str(json.dumps({"type": "state", "data": {}}))
        state_resp = json.loads(await ws.receive_str())

        if state_resp.get("type") == "error":
            print(f"  ❌ State FAILED: {state_resp['data']['message']}")
            return False

        state = state_resp["data"]
        print(f"  ✅ State OK")
        print(f"     Episode: {state['episode_id']}")
        print(f"     Steps: {state['step_count']}")
        print(f"     Cumulative reward: {state['cumulative_reward']}")

        # Step 4: Submit
        await ws.send_str(json.dumps({
            "type": "step",
            "data": {"action_type": "submit", "value": ""}
        }))
        submit_resp = json.loads(await ws.receive_str())
        submit_data = submit_resp["data"]
        final_reward = submit_data.get("reward", -1)
        final_done = submit_data.get("done", False)

        print(f"  ✅ Submit OK")
        print(f"     Final reward: {final_reward}")
        print(f"     Done: {final_done}")

        if not final_done:
            print(f"  ❌ PROBLEM: Done should be True after submit!")
            return False

        if not (0.0 <= final_reward <= 1.0):
            print(f"  ❌ PROBLEM: Final reward {final_reward} is outside 0.0-1.0!")
            return False

        print(f"  ✅ ALL CHECKS PASSED for {task_name}")
        return True


async def test_graders_vary():
    """Make sure graders don't always return the same score."""
    print(f"\n{'='*50}")
    print(f"TESTING: Graders produce different scores")
    print(f"{'='*50}")

    ws_url = SERVER.replace("http://", "ws://") + "/ws"

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            # Test with perfect answer
            await ws.send_str(json.dumps({
                "type": "reset",
                "data": {"task_name": "task_1_identify"}
            }))
            await ws.receive_str()

            # Perfect answer (all error rows)
            await ws.send_str(json.dumps({
                "type": "step",
                "data": {
                    "action_type": "identify_errors",
                    "value": json.dumps({"row_ids": [2, 3, 5, 6, 8, 9, 10, 12, 13, 15]})
                }
            }))
            perfect_resp = json.loads(await ws.receive_str())
            perfect_reward = perfect_resp["data"].get("reward", 0)

        async with session.ws_connect(ws_url) as ws:
            # Test with wrong answer
            await ws.send_str(json.dumps({
                "type": "reset",
                "data": {"task_name": "task_1_identify"}
            }))
            await ws.receive_str()

            # Wrong answer (only 2 rows, some wrong)
            await ws.send_str(json.dumps({
                "type": "step",
                "data": {
                    "action_type": "identify_errors",
                    "value": json.dumps({"row_ids": [1, 4]})
                }
            }))
            wrong_resp = json.loads(await ws.receive_str())
            wrong_reward = wrong_resp["data"].get("reward", 0)

    print(f"  Perfect answer reward: {perfect_reward}")
    print(f"  Wrong answer reward:   {wrong_reward}")

    if perfect_reward == wrong_reward:
        print(f"  ❌ PROBLEM: Grader returns same score for different inputs!")
        return False
    else:
        print(f"  ✅ Graders vary correctly (different inputs → different scores)")
        return True


async def main():
    print("🧪 OpenEnv Data Cleaning — Quick Test Suite")
    print(f"Server: {SERVER}")
    print()

    results = []

    async with aiohttp.ClientSession() as session:
        # Test HTTP health
        print("TEST 0: Health check")
        try:
            async with session.get(f"{SERVER}/health") as resp:
                if resp.status == 200:
                    print(f"  ✅ Health: {await resp.json()}")
                    results.append(True)
                else:
                    print(f"  ❌ Health check failed: {resp.status}")
                    results.append(False)
        except aiohttp.ClientConnectorError:
            print(f"  ❌ Cannot connect to {SERVER}")
            print(f"     Make sure the server is running:")
            print(f"     uvicorn server.app:app --host 0.0.0.0 --port 7860")
            return

        # Test Task 1: Identify errors
        r = await test_task(
            session,
            "task_1_identify",
            "identify_errors",
            {"row_ids": [2, 3, 5, 6, 8, 9, 10, 12, 13, 15]}
        )
        results.append(r)

        # Test Task 2: Classify errors
        r = await test_task(
            session,
            "task_2_classify",
            "classify_errors",
            {"errors": [
                {"row_id": 2, "column": "email", "error_type": "invalid_format"},
                {"row_id": 3, "column": "signup_date", "error_type": "invalid_format"},
                {"row_id": 5, "column": "age", "error_type": "outlier"},
            ]}
        )
        results.append(r)

        # Test Task 3: Fix errors
        r = await test_task(
            session,
            "task_3_fix",
            "fix_errors",
            {"fixes": [
                {"row_id": 2, "column": "email", "error_type": "invalid_format",
                 "current_value": "bob.smith@yahoo", "corrected_value": "bob.smith@yahoo.com"},
                {"row_id": 5, "column": "age", "error_type": "outlier",
                 "current_value": "-3", "corrected_value": "31"},
            ]}
        )
        results.append(r)

    # Test grader variation
    r = await test_graders_vary()
    results.append(r)

    # Summary
    print(f"\n{'='*50}")
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print(f"{'='*50}")

    if all(results):
        print("🎉 ALL TESTS PASSED! Your environment is working correctly.")
    else:
        print("⚠️  Some tests failed. Fix the issues above before submitting.")


if __name__ == "__main__":
    asyncio.run(main())
