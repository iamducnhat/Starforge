# Local Coding Assistant

## Run

```bash
python3 main.py
```

## .env Defaults

The app auto-loads `.env` at startup. Current default backend/model:
- provider: `openrouter`
- model: `arcee-ai/trinity-large-preview:free`

Set your API key in `.env`:
```bash
OPENROUTER_API_KEY=your_key_here
```

Startup now prints model connection info (provider, endpoint, streaming mode, connect attempts).
If requested OpenRouter model is unavailable, the app auto-falls back (default: `arcee-ai/trinity-large-preview:free`).

## Optional

```bash
# choose local Ollama model
python3 main.py --model qwen2.5:3b-instruct

# explicitly use Ollama
python3 main.py --provider ollama --model qwen3:8b

# use OpenRouter online model
OPENROUTER_API_KEY=your_key \
python3 main.py --provider openrouter --model openai/gpt-4o-mini

# list OpenRouter models
OPENROUTER_API_KEY=your_key \
python3 main.py --list-models

# one-shot prompt
python3 main.py --once "How do I parse JSON in Python?"
```

Notes:
- Streaming is enabled for all providers.
- If native stream fails, the app falls back to chunked token streaming from a non-stream response.
- In CLI, use `/maxout <n>` to change max output tokens at runtime, or `/maxout` to view current value.
- In CLI, use `/stream <auto|native|chunk>` to change stream mode, or `/stream` to view current mode.
- Web tools now include:
  - `read_web(url, ...)`: fetch one page, return text + links + code snippets.
  - `scrape_web(start_url, ...)`: crawl a site and return discovered links/pages.

## Fine-tune Data (Function Tool Use)

```bash
# build focused training set for "create function" requests
python3 finetune/build_function_dataset.py
```

## Notes

- Storage is filesystem-only: `memory/blocks` and `functions`.
- If Ollama is unavailable, a fallback response is used.
- `search_web` supports `level`: `quick`, `balanced`, `deep`, `auto` (default).
- Workspace coding tools are available:
  - `list_files`, `read_file`, `write_file`, `edit_file`
  - `create_plan`, `list_plans`, `get_plan`, `add_todo`, `update_todo`
