# Starforge (RAG Assistant)

A local coding assistant for tool-driven development workflows, with:

- CLI chat mode
- OpenAI-compatible API server mode
- Workspace-safe file tools
- Hybrid memory (keyword + semantic retrieval)
- Deterministic DAG planning with topological step scheduling (`id/type/tool/args/depends_on/expected`)
- Autonomous loop with structured execution validation (`execute -> validate -> score -> decide`)
- Project-awareness bootstrap (`detect_project_context`) for autonomous runs
- Automatic skill learning from validated successful tool sequences
- Parameterized skill templates with reusable inputs and step placeholders
- Strategy memory for successful multi-step plans
- Root-cause memory with deterministic repair template execution
- Hypothesis-driven repair loop with duplicate-hypothesis and duplicate-fix prevention
- Hybrid strategy planner that reuses strategy skeletons and fills missing steps
- Reflection and reliability layers for tool retries and replanning
- Lightweight cost-awareness via token budgeting and per-run metrics
- Real-time trajectory logging for delayed offline fine-tuning
- Optional fine-tuning utilities

## Table of Contents

- [Starforge (RAG Assistant)](#starforge-rag-assistant)
  - [Table of Contents](#table-of-contents)
  - [What This Project Does](#what-this-project-does)
  - [Architecture Overview](#architecture-overview)
  - [Project Structure](#project-structure)
  - [Requirements](#requirements)
  - [Install](#install)
  - [Quick Start](#quick-start)
    - [1) Configure environment](#1-configure-environment)
    - [2) Run CLI](#2-run-cli)
    - [3) One-shot prompt](#3-one-shot-prompt)
    - [4) Start API server](#4-start-api-server)
  - [CLI Commands](#cli-commands)
  - [Model Providers](#model-providers)
    - [OpenRouter](#openrouter)
    - [Ollama](#ollama)
    - [Google](#google)
    - [Nvidia](#nvidia)
  - [Configuration](#configuration)
    - [Common runtime env vars](#common-runtime-env-vars)
    - [Provider-specific env vars](#provider-specific-env-vars)
  - [Tools](#tools)
    - [Tool reliability layer](#tool-reliability-layer)
  - [Memory System](#memory-system)
  - [Autonomous Mode](#autonomous-mode)
  - [Server Mode (OpenAI-Compatible)](#server-mode-openai-compatible)
    - [Example: health check](#example-health-check)
    - [Example: non-streaming completion](#example-non-streaming-completion)
    - [Example: streaming completion](#example-streaming-completion)
  - [Testing](#testing)
  - [Fine-Tuning](#fine-tuning)
    - [Build datasets](#build-datasets)
    - [Train LoRA SFT](#train-lora-sft)
  - [Troubleshooting](#troubleshooting)
    - [OpenRouter key missing](#openrouter-key-missing)
    - [402 payment required from provider](#402-payment-required-from-provider)
    - [Context or output too large/small](#context-or-output-too-largesmall)
    - [Repeated autonomous loops](#repeated-autonomous-loops)
  - [Notes and Constraints](#notes-and-constraints)

## What This Project Does

This assistant orchestrates LLM responses with local tools. Instead of only generating text, it can:

- inspect and edit files
- search code and symbols
- execute controlled terminal sessions
- retrieve local memory blocks
- search/read web sources
- create and execute reusable functions/tool-macros
- run autonomous iterative improvement loops

It supports provider backends including `openrouter`, `ollama`, `google`, and `nvidia`.

## Architecture Overview

At a high level:

1. `main.py` builds the model + tool system + chat engine.
2. `assistant/chat_engine.py` manages conversation flow, tool-call loops, autonomous runs, session persistence, and reflection.
3. `assistant/tools.py` dispatches tools, normalizes legacy tool names/args, and wraps calls with reliability retries/fallbacks.
4. `assistant/workspace_tools.py` handles file/project/plan/symbol operations with workspace path guards.
5. `assistant/memory.py` manages memory blocks, strategy memory, root-cause memory, and hybrid retrieval with recency/success ranking.
6. `assistant/server.py` exposes an OpenAI-compatible `/v1/chat/completions` API.
7. Autonomous runs use dependency-aware task queues, structured validator signals, reusable strategy recall, and token-budgeted runtime metrics.

## Project Structure

```text
assistant/
  chat_engine.py           # core conversation + tool loop + autonomous logic
  model.py                 # provider adapters and model routing
  tools.py                 # tool dispatcher + reliability wrapper
  workspace_tools.py       # file/project/plan/symbol tools
  memory.py                # memory blocks + strategy memory + semantic retrieval
  server.py                # OpenAI-compatible API
  prompt.py                # system prompt and tool protocol

functions/                 # persisted reusable functions/tool-macros/skills
memory/
  blocks/                  # memory blocks
  strategies/              # successful multi-step strategy snapshots
  root_causes/             # learned deterministic repair patterns/templates
  plans/                   # plan JSON files
  sessions/                # session logs

tests/                     # test suite
finetune/                  # synthetic dataset + LoRA training scripts
main.py                    # entry point
```

## Requirements

Runtime dependencies (`requirements.txt`):

- `fastapi`
- `uvicorn`
- `anyio`
- `pydantic`
- `requests`
- `python-dotenv`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Quick Start

### 1) Configure environment

Create/update `.env` in repo root.

Minimal OpenRouter example:

```bash
ASSISTANT_PROVIDER=openrouter
ASSISTANT_MODEL=arcee-ai/trinity-large-preview:free
OPENROUTER_API_KEY=your_key_here
```

Minimal Ollama example:

```bash
ASSISTANT_PROVIDER=ollama
ASSISTANT_MODEL=qwen3:8b
OLLAMA_URL=http://127.0.0.1:11434
```

### 2) Run CLI

```bash
python3 main.py
```

### 3) One-shot prompt

```bash
python3 main.py --once "How do I parse JSON in Python?"
```

### 4) Start API server

```bash
python3 main.py --server --host 0.0.0.0 --port 8000
```

## CLI Commands

In interactive CLI mode (`python3 main.py`), use:

- `/help`
- `/reset`
- `/status`
- `/maxout <n>` / `/maxout`
- `/ctx <n>` / `/ctx`
- `/autosize` / `/autosize status`
- `/stream <auto|native|chunk>` / `/stream`
- `/temperature <n>` / `/temperature`
- `/top_p <n>` / `/top_p`
- `/compact [on|off|status]`
- `/autostop [on|off]`
- `/session`
- `/session list`
- `/session new`
- `/session save <name>`
- `/session open <name>`
- `/auto`
- `/auto on [steps|infinite]`
- `/auto off`

## Model Providers

Set with `--provider` or `ASSISTANT_PROVIDER`:

- `openrouter`
- `ollama`
- `google`
- `nvidia`
- `auto`

### OpenRouter

- Uses `OPENROUTER_API_KEY`
- Supports model listing via:

```bash
python3 main.py --list-models
```

### Ollama

- Uses `OLLAMA_URL` (default `http://127.0.0.1:11434`)
- Auto limit tuning can be enabled with `ASSISTANT_AUTO_LIMITS=1`

### Google

- Uses `GOOGLE_API_KEY` (or `GEMINI_API_KEY` fallback)

### Nvidia

- Uses `NVIDIA_API_KEY`

## Configuration

### Common runtime env vars

- `ASSISTANT_PROVIDER` (default `openrouter`)
- `ASSISTANT_MODEL` (default `arcee-ai/trinity-large-preview:free`)
- `ASSISTANT_STREAM_MODE` (`auto|native|chunk`, default `auto`)
- `ASSISTANT_STREAM_TIMEOUT` (default `35` for online providers)
- `ASSISTANT_TEMPERATURE` (default `0.2`)
- `ASSISTANT_TOP_P` (default `0.95` where applicable)
- `ASSISTANT_NUM_PREDICT` (max output token setting)
- `ASSISTANT_NUM_CTX` (context window for local Ollama)
- `ASSISTANT_AUTO_LIMITS` (`1` enabled by default)
- `ASSISTANT_AUTONOMOUS_STEPS` (`0` means infinite)
- `ASSISTANT_COMPACT_CONTEXT` (default on)
- `ASSISTANT_MAX_CONTEXT_CHARS` (default `180000`)
- `ASSISTANT_TOOL_REFLECTION` (default on)
- `ASSISTANT_PLAN_STEP_CAP` (default `8`)
- `ASSISTANT_TOOL_RETRIES` (default `2`)
- `ASSISTANT_TOOL_RETRY_BACKOFF` (default `0.35`)
- `ASSISTANT_EXEC_VALIDATION_THRESHOLD` (default `0.55`)
- `ASSISTANT_AUTO_LEARN_SKILLS` (default on)
- `ASSISTANT_AUTO_VALIDATE_CHANGES` (default on)
- `ASSISTANT_AUTO_TEST_REPAIR_ATTEMPTS` (default `3`)
- `ASSISTANT_AUTO_TOKEN_BUDGET` (`0` disables budget enforcement)
- `ASSISTANT_LOG_INTERACTIONS` (default on)
- `ASSISTANT_LOG_FILE` (server mode sets `log.txt` if unset)

### Provider-specific env vars

OpenRouter:

- `OPENROUTER_URL` (default `https://openrouter.ai/api/v1`)
- `OPENROUTER_API_KEY`
- `OPENROUTER_FALLBACK_MODEL`
- `OPENROUTER_HTTP_REFERER`
- `OPENROUTER_APP_NAME`
- `OPENROUTER_PROVIDER`
- `OPENROUTER_PROVIDER_ONLY`

Ollama:

- `OLLAMA_URL`

Google:

- `GOOGLE_API_KEY`
- `GEMINI_API_KEY` (fallback alias)

Nvidia:

- `NVIDIA_API_KEY`
- `NVIDIA_ENABLE_REASONING`

## Tools

Registered tools include:

Memory/tools:

- `find_in_memory`
- `search_memory`
- `find_strategies`
- `record_strategy`
- `create_block`
- `create_function`
- `create_skill`
- `list_skills`
- `find_skills`
- `record_skill_outcome`
- `run_function`

Web/tools:

- `search_web`
- `read_web`
- `scrape_web`
- `extract_code_snippets`
- `get_current_datetime`

Workspace/tools:

- `list_files`
- `read_file`
- `create_file`
- `create_folder`
- `delete_file`
- `write_file` (legacy alias)
- `edit_file`
- `search_project`
- `index_symbols`
- `lookup_symbol`
- `summarize_file`

Planning/todo tools:

- `create_plan`
- `list_plans`
- `get_plan`
- `add_todo`
- `update_todo`

Terminal tool:

- `run_terminal(action="start|send|read|close", ...)`

### Tool reliability layer

`ToolSystem.safe_tool_call()` adds:

- retry for transient failures (timeouts/rate limits/network)
- fallback argument strategy for fragile web tools
- attempt metadata in tool results
- canonical alias resolution for common model drift (`google_search`, `web_search`, `search_manga`, etc.)
- argument-shape normalization for common malformed payloads (`queries` -> `query`, `limit` -> `max_results`)

Skills can now be stored as parameterized templates, for example:

```json
{
  "skill": "fix_import_error",
  "inputs": ["file_path", "search_query"],
  "steps_template": [
    {"tool": "search_project", "args": {"query": "${search_query}"}},
    {"tool": "edit_file", "args": {"path": "${file_path}"}}
  ],
  "match_conditions": ["import error", "module not found"]
}
```

## Memory System

Memory is filesystem-backed in `memory/blocks`, `memory/strategies`, and `memory/root_causes`.

Blocks store metadata (`info.json`) + content (`knowledge.md`).

Retrieval:

- `find_in_memory(keywords)` uses hybrid scoring:
  - keyword overlap
  - semantic similarity (local hashed embedding vectors)
- `search_memory(query)` performs semantic search directly
- ranking includes recency and historical success weighting (`record_memory_feedback`)
- `record_strategy(goal, strategy, success)` stores normalized multi-step plans with success/failure counters
- `find_strategies(query)` retrieves prior successful plans for planning reuse before asking the model to synthesize a new plan
- root-cause memory stores deterministic repair templates in:
  - `memory/root_causes/import_errors.json`
  - `memory/root_causes/test_failures.json`
- root-cause APIs in `MemoryStore` include:
  - `find_root_causes(error_text, context)` for deterministic root-cause matching
  - `record_root_cause_feedback(root_cause_id, success, confidence)` for confidence/count updates
  - `upsert_root_cause(pattern, context, fix_template, success, confidence, source, bucket_hint?)` for create/merge learning
- `memory/interaction_trajectories.jsonl` stores compact run trajectories with prompt, plan, tool trace, result, success flag, score, retry count, and error count

This project does not do live weight updates on each user message. Immediate adaptation happens through memory retrieval, strategy reuse, and parameterized skills. Weight updates are intended to happen offline from filtered trajectory logs.

## Autonomous Mode

Run with:

```bash
python3 main.py --autonomous --autonomous-steps 8
# or infinite
python3 main.py --autonomous --autonomous-steps 0
```

Autonomous flow now includes:

1. deterministic planning: planner output is normalized into structured steps and topologically sorted into a strict dependency-aware execution queue
2. hybrid strategy planner: reusable strategy skeletons are ranked by goal/context/success, then missing capability steps are generated and merged (`inspect/modify/validate`)
3. project-awareness bootstrap: run `detect_project_context` once and inject framework/test context into each step prompt
4. execution: run current step with tools and reflection
5. execution validator: score each step with structured evidence checks (tool success, test pass/fail, exit codes, diff presence, expected-output coverage, failure markers)
6. workspace grounding: `run_tests` parses structured failure summaries, failed node IDs, and test counts
7. optional workspace validation: run `validate_workspace_changes` after edit-like steps
8. root-cause deterministic fast path: before model repair, check root-cause memory and apply template fixes when top match score is high (`>= 0.75`)
9. hypothesis-driven test repair loop: run `hypothesis -> fix -> test`, skip duplicate hypotheses, and skip repeated non-improving fix signatures
10. success-gated root-cause learning: when repair resolves failures, persist the failure pattern + successful fix sequence via `upsert_root_cause(...)`
11. decision gate: `advance/retry/replan/done/bored` is influenced by reflection and validator threshold
12. behavior learning: successful validated step tool sequences can be auto-saved as reusable parameterized skills
13. strategy reuse: successful autonomous runs can be recalled as reusable multi-step plans for future objectives
14. lightweight performance tracking: per-run metrics include success rate, retry count, replan count, test-repair attempts, tool failure rate, validator failures, and estimated output token usage
15. optional cost guard: `ASSISTANT_AUTO_TOKEN_BUDGET` can stop long autonomous runs once the estimated output budget is reached
16. interaction logging: chat turns and autonomous steps emit compact trajectories that can later be filtered into offline LoRA datasets

Per-tool reflection can also trigger retries with revised calls.

Planner schema accepted by the runtime:

```json
{
  "steps": [
    {
      "id": 1,
      "type": "tool_call",
      "tool": "search_project",
      "args": {"query": "login"},
      "depends_on": [],
      "expected": "list of matching files"
    }
  ]
}
```

Compatibility aliases are accepted: `step_id <-> id`, `action <-> tool`, `expected_output <-> expected`.

Step dependencies are enforced by the runtime. If the planner emits out-of-order but valid dependency references, the runtime reorders the plan via topological sort before execution. Cycles are broken deterministically as a fallback instead of allowing unsafe parallel or skipped execution.

## Server Mode (OpenAI-Compatible)

Start server:

```bash
python3 main.py --server --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

### Example: health check

```bash
curl -s http://127.0.0.1:8000/health
```

### Example: non-streaming completion

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Give me a Python hello world"}
    ]
  }'
```

### Example: streaming completion

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Explain quicksort briefly"}
    ]
  }'
```

## Testing

Run core tests:

```bash
PYTHONPATH=$PWD python3 -m pytest -q \
  tests/test_memory.py \
  tests/test_structured_planner.py \
  tests/test_tool_calls.py \
  tests/test_utils.py \
  tests/test_code_intelligence.py \
  tests/test_skills.py \
  tests/test_tool_reliability.py
```

Run all tests:

```bash
PYTHONPATH=$PWD python3 -m pytest -q tests
```

## Fine-Tuning

### Build datasets

```bash
python3 finetune/build_function_dataset.py
python3 finetune/build_interaction_dataset.py \
  --input memory/interaction_trajectories.jsonl \
  --output finetune/interaction_train.jsonl \
  --min-score 0.7 --max-retries 0 --max-errors 0
python3 finetune/generate_synthetic_tool_use.py --per-topic 120
python3 finetune/build_full_tool_dataset.py \
  --train-output finetune/train_tool_use_full.jsonl \
  --val-output finetune/val_tool_use_full.jsonl
```

`build_full_tool_dataset.py` now automatically folds in high-quality interaction trajectories from `memory/interaction_trajectories.jsonl`, filtered to successful low-retry low-error runs.

### Train LoRA SFT

```bash
python3 -m pip install -r finetune/requirements-train.txt

python3 finetune/train_lora_sft.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --train-file finetune/train_tool_use_full.jsonl \
  --val-file finetune/val_tool_use_full.jsonl \
  --output-dir finetune/output/lora_tool_use \
  --epochs 2 --batch-size 1 --grad-accum 16 --max-length 2048 --bf16
```

Or run pipeline:

```bash
./finetune/run_full_finetune.sh
```

## Troubleshooting

### OpenRouter key missing

Symptom: startup/chat mentions missing key.

Fix:

- set `OPENROUTER_API_KEY`
- or use `--provider ollama`

### 402 payment required from provider

Symptom: model returns payment required.

Fix:

- add provider credits
- set `OPENROUTER_FALLBACK_MODEL` to a free model

### Context or output too large/small

Use CLI runtime controls:

- `/ctx <n>`
- `/maxout <n>`
- `/autosize`

### Repeated autonomous loops

Use:

- `/autostop on`
- smaller `--autonomous-steps`
- refine objective to be more concrete

## Notes and Constraints

- Storage is filesystem-only (no database).
- Workspace path guard prevents file operations outside project root.
- System prompt asks the agent to place generated task artifacts in `workspaces/<task_name>/`.
- For time-sensitive factual queries, the assistant is configured to call date/time + web tools first.
