# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Pre-submission validation script.
Checks all mandatory requirements from the OpenEnv hackathon spec.
"""
import json
import os
import sys
import urllib.request


SPACE_URL = "https://shaikhakramshakil-openenv-data-cleaning.hf.space"
CHECKS = []


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    CHECKS.append((name, passed, detail))
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


def main():
    print("=" * 60)
    print("PRE-SUBMISSION VALIDATION v2")
    print("=" * 60)

    # 1. Check local files exist
    print("\n1. Project Structure:")
    required_files = [
        "models.py", "data.py", "client.py", "inference.py",
        "openenv.yaml", "Dockerfile", "README.md", "LICENSE",
        "requirements.txt", "pyproject.toml",
        "server/app.py", "server/environment.py",
    ]
    for f in required_files:
        check(f"File exists: {f}", os.path.exists(f))

    # 2. Check openenv.yaml has 4 tasks and spec_version
    print("\n2. OpenEnv Manifest:")
    import yaml
    with open("openenv.yaml", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    tasks = manifest.get("tasks", [])
    check("openenv.yaml has 4 tasks", len(tasks) >= 4, f"Found {len(tasks)} tasks")
    check("Has spec_version", "spec_version" in manifest)
    check("Has server config", "server" in manifest)
    check("Has type definitions", "types" in manifest)
    check("Has tools section", "tools" in manifest)

    # 3. Check environment core logic works locally
    print("\n3. Local Environment Tests:")
    try:
        sys.path.insert(0, os.getcwd())
        from server.environment import DataCleaningEnvironment, TASKS
        from models import DataCleaningAction

        for task_name in TASKS:
            env = DataCleaningEnvironment(task_name)
            obs = env.reset()
            check(f"reset({task_name})", obs.num_rows > 0, f"rows={obs.num_rows}")

        # Test tools
        env = DataCleaningEnvironment("task_1_identify")
        env.reset()
        obs = env.step(DataCleaningAction(action_type="check_schema"))
        check("check_schema tool works", obs.tool_output is not None)
        obs = env.step(DataCleaningAction(action_type="run_statistics"))
        check("run_statistics tool works", obs.tool_output is not None)
        obs = env.step(DataCleaningAction(action_type="search_reference", value="pricing"))
        check("search_reference tool works", obs.tool_output is not None)

        # Test reward range (strictly between 0 and 1)
        obs = env.step(DataCleaningAction(
            action_type="identify_errors",
            value='{"row_ids": [1,2,3,4,5,6,7,8,9,10,11,12]}',
        ))
        check("Reward is in (0,1)", 0 < obs.reward < 1, f"reward={obs.reward}")

        # Test Task 4 insight
        env4 = DataCleaningEnvironment("task_4_insight")
        env4.reset()
        obs4 = env4.step(DataCleaningAction(action_type="answer_insight", value="100.0"))
        check("Task 4 returns reward", obs4.reward >= 0, f"reward={obs4.reward}")

    except Exception as e:
        check("Local environment tests", False, str(e))

    # 4. Check HF Space is live
    print("\n4. HF Space Deployment:")
    try:
        r = urllib.request.urlopen(f"{SPACE_URL}/health", timeout=30)
        data = json.loads(r.read().decode())
        check("HF Space /health returns 200", r.status == 200, str(data))
    except Exception as e:
        check("HF Space /health returns 200", False, str(e))

    try:
        r = urllib.request.urlopen(f"{SPACE_URL}/info", timeout=30)
        info = json.loads(r.read().decode())
        check("/info returns metadata", "name" in info, info.get("name", ""))
        check("/info lists 4 tasks", len(info.get("tasks", [])) >= 4, str(info.get("tasks", [])))
        check("/info lists tools", "tools" in info)
    except Exception as e:
        check("/info returns metadata", False, str(e))

    try:
        r = urllib.request.urlopen(f"{SPACE_URL}/state", timeout=30)
        state = json.loads(r.read().decode())
        check("/state does not crash", True, str(state))
    except Exception as e:
        check("/state does not crash", False, str(e))

    # 4. Check reset works for all 4 tasks
    print("\n4. Task Endpoints:")
    for task in ["task_1_identify", "task_2_classify", "task_3_fix", "task_4_insight"]:
        try:
            req = urllib.request.Request(
                f"{SPACE_URL}/reset",
                data=json.dumps({"task_name": task}).encode(),
                headers={"Content-Type": "application/json"},
            )
            r = urllib.request.urlopen(req, timeout=30)
            data = json.loads(r.read().decode())
            obs = data.get("observation", {})
            has_obs = obs.get("task_name") == task
            check(f"reset({task})", has_obs, f"rows={obs.get('num_rows')}")
        except Exception as e:
            check(f"reset({task})", False, f"{task} reset failed: {e}")

    # 5. Check reward logic via WebSocket (Simulated)
    print("\n5. Reward Logic Verification:")
    try:
        # Instead of hardcoding, we verify reward is always strictly between 0 and 1
        from server.environment import DataCleaningEnvironment
        from models import DataCleaningAction
        env = DataCleaningEnvironment("task_1_identify")
        env.reset()
        # Mock a tool call to see if it gives small reward
        tool_obs = env.step(DataCleaningAction(action_type="check_schema"))
        valid_range = 0.0 < tool_obs.reward < 1.0
        check("Reward in valid range (0,1)", valid_range, f"sample={tool_obs.reward}")
    except Exception as e:
        check("Reward verification failed", False, str(e))

    # 6. Check inference script exists and has correct format
    print("\n6. Inference Script:")
    with open("inference.py", encoding="utf-8") as fh:
        content = fh.read()
    check("inference.py exists", True)
    check("Has [START] log format", "[START]" in content)
    check("Has [STEP] log format", "[STEP]" in content)
    check("Has [END] log format", "[END]" in content)
    check("Uses OpenAI Client", "from openai import OpenAI" in content)
    check("Has API_BASE_URL env var", "API_BASE_URL" in content)
    check("Has MODEL_NAME env var", "MODEL_NAME" in content)
    check("Has HF_TOKEN env var", "HF_TOKEN" in content)

    # 7. Check README has HF frontmatter
    print("\n7. README:")
    with open("README.md", encoding="utf-8") as fh:
        readme = fh.read()
    check("Has HF YAML frontmatter", readme.startswith("---"))
    check("Tagged with 'openenv'", "openenv" in readme)
    check("Has task_1_identify", "task_1_identify" in readme)
    check("Has task_4_insight", "task_4_insight" in readme)
    check("Has setup instructions", "pip install" in readme)

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, p, _ in CHECKS if p)
    total = len(CHECKS)
    print(f"RESULT: {passed}/{total} checks passed")
    if passed == total:
        print("ALL CHECKS PASSED! Ready for submission.")
    else:
        print("SOME CHECKS FAILED. Fix before submitting.")
        for name, p, detail in CHECKS:
            if not p:
                print(f"  FAILED: {name} - {detail}")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("Installing pyyaml...")
        os.system("pip install pyyaml --quiet")
        import yaml
    sys.exit(main())
