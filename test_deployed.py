"""Quick test of the deployed HF Space endpoints."""
import urllib.request
import json

base_url = "https://shaikhakramshakil-openenv-data-cleaning.hf.space"

# Test /health
try:
    r = urllib.request.urlopen(f"{base_url}/health", timeout=30)
    print(f"Health: {r.read().decode()}")
except Exception as e:
    print(f"Health error: {e}")

# Test /info
try:
    r = urllib.request.urlopen(f"{base_url}/info", timeout=30)
    info = json.loads(r.read().decode())
    print(f"Info: {json.dumps(info, indent=2)}")
except Exception as e:
    print(f"Info error: {e}")

# Test /reset
try:
    req = urllib.request.Request(
        f"{base_url}/reset",
        data=json.dumps({"task_name": "task_1_identify"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = urllib.request.urlopen(req, timeout=30)
    data = json.loads(r.read().decode())
    obs = data.get("observation", {})
    print(f"Reset done={data['done']}, reward={data['reward']}")
    print(f"Task: {obs['task_name']}")
    print(f"Rows: {obs['num_rows']}")
    print("ALL ENDPOINTS LIVE!")
except Exception as e:
    print(f"Reset error: {e}")
