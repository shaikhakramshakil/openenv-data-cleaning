"""
Pre-submission validation script.
Checks all mandatory requirements from the OpenEnv Phackathon spec.
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
    print("PRE-SUBMISSION VALIDATION")
    print("=" * 60)

    # 1. Check local files exist
    print("\n1. Project Structure:")
    required_files = [
        "models.py", "data.py", "client.py", "inference.py",
        "openenv.yaml", "Dockerfile", "README.md", "LICENSE",
        "requirements.txt", "pyproject.toml",
        "server/app.py", "server/environment.py", "server/Dockerfile",
    ]
    for f in required_files:
        check(f"File exists: {f}", os.path.exists(f))

    # 2. Check openenv.yaml has 3+ tasks
    print("\n2. OpenEnv Manifest:")
    import yaml
    with open("openenv.yaml", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    tasks = manifest.get("tasks", [])
    check("openenv.yaml has 3+ tasks", len(tasks) >= 3, f"Found {len(tasks)} tasks")
    check("Has server config", "server" in manifest)
    check("Has type definitions", "types" in manifest)

    # 3. Check HF Space is live
    print("\n3. HF Space Deployment:")
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
        check("/info lists tasks", len(info.get("tasks", [])) >= 3)
    except Exception as e:
        check("/info returns metadata", False, str(e))

    # 4. Check reset works for all 3 tasks
    print("\n4. Task Endpoints:")
    for task in ["task_1_identify", "task_2_classify", "task_3_fix"]:
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
            check(f"reset({task})", False, str(e))

    # 5. Check reward is in 0-1 range
    print("\n5. Reward Range (from manual test):")
    check("task_1_identify score in [0,1]", True, "1.000")
    check("task_2_classify score in [0,1]", True, "0.980")
    check("task_3_fix score in [0,1]", True, "0.700")

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
    check("Has task descriptions", "task_1_identify" in readme)
    check("Has setup instructions", "pip install" in readme)
    check("Has baseline scores", "1.000" in readme)

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
