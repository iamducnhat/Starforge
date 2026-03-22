#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TARGET="$ROOT_DIR/demo_project/ops.py"

python3 - "$TARGET" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

target = Path(sys.argv[1])
text = target.read_text(encoding="utf-8")

if "return a - b" in text:
    print("Repeat bug already present.")
    raise SystemExit(0)

updated = text.replace("return a + b", "return a - b  # WRONG (repeat bug)")
if updated == text:
    raise SystemExit("Could not inject repeat bug: expected line not found.")

target.write_text(updated, encoding="utf-8")
print("Injected repeat bug into demo_project/ops.py")
PY
