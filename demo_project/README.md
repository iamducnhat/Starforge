# Starforge Learning Demo

This demo is designed to hit three reactions:

- "Wait... it fixed itself??"
- "This thing actually learns"
- "I need this in my workflow"

## Quick Run

From repo root:

```bash
bash demo_project/scripts/reset_demo.sh
pytest -q demo_project/tests
```

You should see one failing test (`test_add`).

Run autonomous repair:

```bash
python3 main.py \
  --autonomous \
  --autonomous-steps 8 \
  --autonomous-objective "Fix failing tests in demo_project only. Run pytest -q demo_project/tests, debug, patch only minimum files under demo_project, rerun tests, and finish when green."
```

Inspect learned root-cause memory:

```bash
bash demo_project/scripts/show_demo_memory.sh
```

Inject the same operator bug in another file:

```bash
bash demo_project/scripts/inject_repeat_bug.sh
pytest -q demo_project/tests
```

Run the exact autonomous command again and record the speed/attempt delta.

## Important Path Note

If your agent prompt enforces edits inside ``, keep all objectives and pytest paths under `demo_project`. Using the old `demo_project/...` path can cause plan/validation loops.

## Recording Notes

Use these moments as scene cuts:

1. Initial failing test output.
2. `[auto] test-driven repair attempt ...`
3. `[auto] hypothesis: ...`
4. `[auto] recording root cause: ...`
5. Second run with:
   - `[auto] root-cause candidate: ...`
   - `[auto] applying known fix template...`
   - `[auto] known root-cause fix solved failing tests`

## Suggested Title

`I built an AI that debugs code and learns from its own mistakes`