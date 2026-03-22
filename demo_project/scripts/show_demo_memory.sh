#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MEMORY_FILE="$ROOT_DIR/memory/root_causes/test_failures.json"

if [[ ! -f "$MEMORY_FILE" ]]; then
  echo "No root-cause memory file found at $MEMORY_FILE"
  exit 0
fi

python3 - "$MEMORY_FILE" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(payload, list):
    print("Unexpected memory format.")
    raise SystemExit(0)

rows = []
for item in payload:
    if not isinstance(item, dict):
        continue
    pattern = str(item.get("pattern", ""))
    if "demo_project/tests/" not in pattern:
        continue
    rows.append(
        {
            "id": item.get("id", ""),
            "pattern": pattern,
            "confidence": item.get("confidence", 0),
            "success_count": item.get("success_count", 0),
            "fail_count": item.get("fail_count", 0),
            "fix_template": item.get("fix_template", []),
        }
    )

if not rows:
    print("No demo-specific root-cause entries yet.")
else:
    print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
