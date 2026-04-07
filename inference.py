"""
Baseline inference script for the Data Cleaning OpenEnv environment.

This script runs a standard LLM agent against all 3 tasks and produces
scores. It uses the OpenAI Client with the required environment variables.

IMPORTANT: Start the environment server before running this script:
    # Option 1: Docker (recommended)
    docker build -t data-cleaning-env .
    docker run -p 7860:7860 data-cleaning-env

    # Option 2: Local development
    uvicorn server.app:app --host 0.0.0.0 --port 7860

Required environment variables:
    API_BASE_URL  - The API endpoint for the LLM
    MODEL_NAME    - The model identifier to use
    HF_TOKEN      - Your Hugging Face / API key

Optional:
    ENV_URL       - Environment server URL (default: http://localhost:7860)

Log format (mandatory):
    [START] task=<task> env=<env> model=<model>
    [STEP] step=<n> action=<action> reward=<r> done=<bool> error=<error|null>
    [END] success=<bool> steps=<n> score=<s> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import sys
import traceback

from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_NAME = "data-cleaning-env"
# Default to 7860 (HF Spaces port) - use 8000 for local dev with uvicorn
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

TASKS = ["task_1_identify", "task_2_classify", "task_3_fix"]


# ─── Logging ──────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ─── LLM Agent ────────────────────────────────────────────────────────

def call_llm(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Call the LLM and return the response text."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [WARN] LLM call failed: {e}", file=sys.stderr, flush=True)
        return f"ERROR: {e}"


def build_system_prompt(task_name: str) -> str:
    """Build the system prompt for the LLM agent based on the task."""
    base = (
        "You are a data quality analyst AI agent. You are given a dataset with data quality errors "
        "and must identify and fix them. Respond ONLY with valid JSON — no explanations, no markdown, no extra text.\n"
    )

    if task_name == "task_1_identify":
        return base + (
            "Your task is to identify which rows contain data quality errors.\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{"row_ids": [2, 5, 8]}\n'
            "List only the 1-indexed row IDs (matching the 'id' column) that have errors.\n"
            "The dataset has validation rules. Check for: missing values, invalid email formats, "
            "impossible dates, negative/impossible ages, wrong data types, duplicates, "
            "plan/amount inconsistencies, invalid status values, and typos."
        )
    elif task_name == "task_2_classify":
        return base + (
            "Your task is to identify ALL data quality errors and classify each one.\n"
            "Valid error types: missing_value, invalid_format, outlier, duplicate, type_error, inconsistency\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{"errors": [{"row_id": 2, "column": "email", "error_type": "invalid_format"}, ...]}\n'
            "Check every cell against the validation rules. Be thorough."
        )
    elif task_name == "task_3_fix":
        return base + (
            "Your task is to identify ALL errors, classify them, and propose fixes.\n"
            "Valid error types: missing_value, invalid_format, outlier, duplicate, type_error, inconsistency\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{"fixes": [{"row_id": 2, "column": "email", "error_type": "invalid_format", '
            '"current_value": "bob.smith@yahoo", "corrected_value": "bob.smith@yahoo.com"}, ...]}\n'
            "Check every cell, identify all issues, and provide corrected values."
        )
    return base


def parse_llm_response(response: str, task_name: str) -> tuple:
    """
    Parse the LLM response into an action_type and value.
    
    Returns:
        (action_type, value_json_string)
    """
    # Extract JSON from the response (handle markdown code blocks)
    text = response.strip()
    if text.startswith("```"):
        # Remove markdown code block
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        text = text.strip()

    # Try to parse as JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                return ("submit", "")
        else:
            return ("submit", "")

    # Determine action type from the response structure
    if task_name == "task_1_identify" and "row_ids" in data:
        return ("identify_errors", json.dumps(data))
    elif task_name == "task_2_classify" and "errors" in data:
        return ("classify_errors", json.dumps(data))
    elif task_name == "task_3_fix" and "fixes" in data:
        return ("fix_errors", json.dumps(data))
    else:
        return ("submit", "")


# ─── Environment Interaction ──────────────────────────────────────────

async def run_task(task_name: str, llm_client: OpenAI) -> float:
    """
    Run a single task episode against the environment.
    
    Uses HTTP for simplicity (environment runs locally).
    """
    import aiohttp

    log_start(task=task_name, env=ENV_NAME, model=MODEL_NAME)

    rewards = []
    step_count = 0
    success = False
    final_score = 0.0

    try:
        # Reset environment
        async with aiohttp.ClientSession() as session:
            # Reset
            async with session.post(
                f"{ENV_URL}/reset",
                json={"task_name": task_name},
            ) as resp:
                reset_data = await resp.json()

            obs = reset_data.get("observation", {})
            dataset_text = obs.get("dataset_text", "")
            task_description = obs.get("task_description", "")
            max_steps = obs.get("max_steps", 5)

            # Build prompt for the LLM
            system_prompt = build_system_prompt(task_name)
            user_prompt = f"{task_description}\n\n{dataset_text}"

            # We use a WebSocket for stateful interaction
            ws_url = ENV_URL.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

            async with session.ws_connect(ws_url) as ws:
                # Reset via WebSocket
                await ws.send_str(json.dumps({
                    "type": "reset",
                    "data": {"task_name": task_name},
                }))
                reset_response = json.loads(await ws.receive_str())
                obs = reset_response["data"].get("observation", {})
                dataset_text = obs.get("dataset_text", "")
                task_description = obs.get("task_description", "")
                feedback = obs.get("feedback", "")

                # Agent loop: call LLM, send action, get feedback, refine
                for step in range(1, max_steps + 1):
                    step_count = step

                    # Build user prompt with feedback from previous steps
                    if step == 1:
                        user_prompt = f"{task_description}\n\n{dataset_text}"
                    else:
                        user_prompt = (
                            f"{task_description}\n\n{dataset_text}\n\n"
                            f"FEEDBACK FROM PREVIOUS ATTEMPT:\n{feedback}\n\n"
                            "Please refine your answer based on this feedback."
                        )

                    # Call LLM
                    llm_response = call_llm(llm_client, system_prompt, user_prompt)

                    # Parse response into action
                    action_type, value = parse_llm_response(llm_response, task_name)

                    # Send action to environment
                    await ws.send_str(json.dumps({
                        "type": "step",
                        "data": {
                            "action_type": action_type,
                            "value": value,
                        },
                    }))
                    step_response = json.loads(await ws.receive_str())

                    if step_response.get("type") == "error":
                        error_msg = step_response["data"]["message"]
                        log_step(step=step, action=action_type, reward=0.0, done=False, error=error_msg)
                        rewards.append(0.0)
                        continue

                    step_data = step_response["data"]
                    reward = step_data.get("reward", 0.0)
                    done = step_data.get("done", False)
                    obs = step_data.get("observation", {})
                    feedback = obs.get("feedback", "")

                    rewards.append(reward)
                    log_step(step=step, action=action_type, reward=reward, done=done)

                    if done:
                        final_score = reward
                        success = True
                        break

                # If not done yet, submit
                if not done:
                    await ws.send_str(json.dumps({
                        "type": "step",
                        "data": {"action_type": "submit", "value": ""},
                    }))
                    submit_response = json.loads(await ws.receive_str())
                    submit_data = submit_response.get("data", {})
                    final_score = submit_data.get("reward", 0.0)
                    step_count += 1
                    rewards.append(final_score)
                    log_step(step=step_count, action="submit", reward=final_score, done=True)
                    success = True

    except Exception as e:
        error_msg = str(e)
        log_step(step=step_count, action="error", reward=0.0, done=True, error=error_msg)
        traceback.print_exc(file=sys.stderr)

    log_end(success=success, steps=step_count, score=final_score, rewards=rewards)
    return final_score


# ─── Main ─────────────────────────────────────────────────────────────

async def main():
    """Run all tasks and report scores."""
    print("=" * 60)
    print(f"Data Cleaning Environment — Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_URL}")
    print("=" * 60)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    scores = {}
    for task in TASKS:
        print(f"\n{'─' * 40}")
        print(f"Running: {task}")
        print(f"{'─' * 40}")
        score = await run_task(task, llm_client)
        scores[task] = score
        print(f"Task {task} score: {score:.3f}")

    print(f"\n{'=' * 60}")
    print("FINAL SCORES:")
    for task, score in scores.items():
        print(f"  {task}: {score:.3f}")
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  Average: {avg:.3f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
