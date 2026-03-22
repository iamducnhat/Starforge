#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

cat >"$ROOT_DIR/demo_project/utils.py" <<'PY'
from __future__ import annotations


def add(a: int, b: int) -> int:
    return a - b  # WRONG (intentional demo bug)
PY

cat >"$ROOT_DIR/demo_project/ops.py" <<'PY'
from __future__ import annotations


def sum_numbers(a: int, b: int) -> int:
    return a + b
PY

python3 - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

path = Path("memory/root_causes/test_failures.json")
if not path.exists():
    raise SystemExit(0)

try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)

if not isinstance(payload, list):
    raise SystemExit(0)

def keep(entry: object) -> bool:
    if not isinstance(entry, dict):
        return True
    pattern = str(entry.get("pattern", ""))
    return "demo_project/tests/" not in pattern

filtered = [item for item in payload if keep(item)]
if len(filtered) != len(payload):
    path.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
PY

echo "Demo reset complete."
echo "Expected state:"
echo "- demo_project/utils.py has broken add()"
echo "- demo_project/ops.py is correct"
