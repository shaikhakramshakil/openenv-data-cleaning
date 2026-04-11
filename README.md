---
title: Data Cleaning Environment
emoji: "\U0001F9F9"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: bsd-3-clause
tags:
  - openenv
  - data-cleaning
  - data-quality
  - reinforcement-learning
  - tool-use
---

# Data Cleaning and Quality Assessment Environment -- OpenEnv

A real-world **OpenEnv environment** for training and evaluating AI agents on **data quality assessment** with **tool-use**. Agents analyze datasets with planted errors and must identify, classify, and fix data quality issues using available tools (schema validation, statistics, reference lookups).

---

## Why Data Cleaning?

Data scientists spend **60-80% of their time** cleaning and preparing data. This environment models that exact workflow:

- **Realistic**: Uses customer subscription records with genuine error patterns (missing values, format issues, outliers, duplicates, type errors, inconsistencies)
- **Tool-Use**: Agents can call tools to check the schema, run statistics, and look up valid reference values
- **High utility**: Directly trains agents for a task with massive real-world demand
- **Deterministic grading**: Every error has a known ground truth -- scores are reproducible

---

## Dataset

A synthetic **Customer Subscription Records** dataset with 30 rows and 9 columns:

| Column | Type | Validation Rules |
|--------|------|-----------------|
| `id` | int | Unique, sequential |
| `name` | str | "First Last" format |
| `email` | str | Valid email format with TLD |
| `age` | int | 18-100 |
| `city` | str | Valid US city |
| `signup_date` | str | YYYY-MM-DD, valid date |
| `plan` | str | free, basic, premium, enterprise |
| `monthly_amount` | float | Must match plan pricing |
| `status` | str | active, inactive, suspended, cancelled |

**12 intentionally planted errors** across 6 error types:
- `missing_value` -- Empty cells where data is expected
- `invalid_format` -- Wrong format (bad email, impossible date, typo)
- `outlier` -- Statistically impossible values (negative age, age=250)
- `duplicate` -- Exact duplicate of another row
- `type_error` -- Wrong data type (word string instead of integer)
- `inconsistency` -- Values contradict each other (plan vs. pricing)

---

## Tasks

### Task 1: Error Identification (Easy)
**Objective**: Find which rows contain errors
**Agent sends**: `{"row_ids": [2, 3, 5, ...]}`
**Grading**: F1 score comparing predicted vs. actual error rows

### Task 2: Error Classification (Medium)
**Objective**: Find errors AND classify their type
**Agent sends**: `{"errors": [{"row_id": 2, "column": "email", "error_type": "invalid_format"}, ...]}`
**Grading**: Match accuracy for row + column + type

### Task 3: Error Correction (Hard)
**Objective**: Find, classify, AND fix every error
**Agent sends**: `{"fixes": [{"row_id": 2, "column": "email", "corrected_value": "bob.smith@yahoo.com"}, ...]}`
**Grading**: Exact match + partial credit for close numeric values

### Task 4: Quality Insights (Expert)
**Objective**: Calculate the total `monthly_amount` for all `active` users after correcting pricing inconsistencies
**Agent sends**: answer as a numeric string (e.g. `"249.97"`)
**Grading**: Exact match within tolerance, partial credit for close answers
**Requires**: Using `run_statistics`, `check_schema`, and `search_reference` tools to reason about the corrected data

---

## Tools (Agentic Tool-Use)

Agents can call the following tools during any task to gather information:

| Tool | Description |
|------|-------------|
| `check_schema` | Returns column definitions, validation rules, and pricing info |
| `run_statistics` | Returns summary stats (mean, sum, min, max, median) for numeric columns |
| `search_reference` | Looks up valid values (e.g., "cities" or "pricing") |

Tools return their results in the `tool_output` field of the observation.

---

## Reward Design

Rewards provide **partial progress signals** (not just binary end-of-episode):

| Signal | Reward |
|--------|--------|
| Correct error identification | Proportional to F1 |
| Correct error type | +type_score |
| Correct fix value | +fix_score |
| Partial numeric match | +0.5 credit |
| Tool exploration | +0.01 per use |
| Invalid JSON format | -0.05 |

All rewards are clamped to [0.001, 1.0].

---

## Architecture

### Agent <-> Environment Interaction Flow

```
Agent --> API: Connect (ws://host/ws)
Agent --> API: {"type": "reset", "data": {"task_name": "task_1_identify"}}
API --> Env: env.reset()
Env --> API: Initial Observation (dataset table, task info)
API --> Agent: {"type": "result", "data": {"observation": {...}}}

loop (up to max_steps):
    Agent --> API: {"type": "step", "data": {"action_type": "check_schema"}}
    Env --> Agent: tool_output with schema info
    Agent --> API: {"type": "step", "data": {"action_type": "identify_errors", "value": "{...}"}}
    Env --> Agent: reward + feedback
```

### Directory Structure

```
openenv-data-cleaning/
  server/
    app.py              # FastAPI server (HTTP + WebSocket + Web UI)
    environment.py      # Core environment logic with tool handlers
    __init__.py
  models.py             # Action, Observation, State Pydantic models
  data.py               # Dynamic dataset generator with error injection
  client.py             # DataCleaningClient (async + sync)
  openenv.yaml          # Environment manifest (spec_version: 1)
  pyproject.toml        # Package config
  requirements.txt
  inference.py          # Baseline inference script
  validate.py           # Pre-submission validation (40+ checks)
  README.md
```

---

## Setup and Usage

### Prerequisites
- Python 3.10+
- Docker (for container deployment)

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (port 7860 matches HF Spaces)
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
# Build
docker build -t data-cleaning-env:latest .

# Run
docker run -d -p 7860:7860 data-cleaning-env:latest

# Health check
curl http://localhost:7860/health
```

### Run Baseline Inference

```bash
# First, start the environment server
docker run -d -p 7860:7860 data-cleaning-env:latest

# Then run inference
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_URL="http://localhost:7860"

python inference.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | Environment metadata (tasks, tools, version) |
| `/reset` | POST | Reset environment (stateless) |
| `/step` | POST | Execute single action (stateless) |
| `/state` | GET | Get episode state |
| `/ws` | WebSocket | Stateful multi-step sessions |
| `/web` | GET | Interactive web dashboard |
| `/docs` | GET | OpenAPI spec (auto-generated) |

### WebSocket Protocol

```json
// Client -> Server
{"type": "reset", "data": {"task_name": "task_1_identify"}}
{"type": "step", "data": {"action_type": "check_schema"}}
{"type": "step", "data": {"action_type": "identify_errors", "value": "{\"row_ids\": [2,3]}"}}
{"type": "state", "data": {}}
{"type": "close", "data": {}}

// Server -> Client
{"type": "result", "data": {"observation": {...}, "reward": 0.8, "done": false}}
{"type": "state", "data": {"episode_id": "abc123", "step_count": 2, ...}}
{"type": "error", "data": {"message": "...", "code": "..."}}
```

---

## Baseline Scores

| Task | Score | Description |
|------|-------|-------------|
| task_1_identify | **0.850** | LLM identifies most error rows |
| task_2_classify | **0.750** | LLM classifies most errors correctly |
| task_3_fix | **0.600** | LLM fixes most errors but struggles with some corrections |
| task_4_insight | **0.500** | LLM reasons over corrected data with tool assistance |

*Scores measured with `meta-llama/Llama-3.3-70B-Instruct` via HF Inference API*

---

## License

BSD-3-Clause -- see [LICENSE](LICENSE) file.

---

## OpenEnv Hackathon

This environment was built for the [Meta PyTorch x Hugging Face OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv).

**Tags**: `openenv`, `data-cleaning`, `data-quality`, `ai-agent`, `tool-use`
