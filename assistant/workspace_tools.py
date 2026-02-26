from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .utils import ensure_dir, read_json, redact_secrets_text, slugify, utc_now_iso, write_json, write_text


class WorkspaceTools:
    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.plans_dir = self.workspace_root / "memory" / "plans"
        ensure_dir(self.plans_dir)

    def _resolve(self, path: str) -> Path:
        raw = Path(path)
        resolved = raw.resolve() if raw.is_absolute() else (self.workspace_root / raw).resolve()
        root_str = str(self.workspace_root)
        resolved_str = str(resolved)
        if resolved_str != root_str and not resolved_str.startswith(root_str + os.sep):
            raise ValueError(f"path outside workspace: {path}")
        return resolved

    def _to_workspace_rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace_root))
        except Exception:
            return str(path)

    def list_files(
        self,
        path: str = ".",
        glob: str = "**/*",
        include_hidden: bool = False,
        max_entries: int = 200,
    ) -> dict[str, Any]:
        base = self._resolve(path)
        if not base.exists():
            return {"ok": False, "error": f"path not found: {path}"}
        if not base.is_dir():
            return {"ok": False, "error": f"not a directory: {path}"}

        entries: list[dict[str, Any]] = []
        pattern = glob.strip() or "**/*"

        for p in base.glob(pattern):
            if len(entries) >= max(1, max_entries):
                break

            rel = self._to_workspace_rel(p)
            if not include_hidden and any(part.startswith(".") for part in Path(rel).parts):
                continue
            if p.is_dir():
                continue

            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            entries.append({"path": rel, "size": size})

        entries.sort(key=lambda x: x["path"])
        return {"ok": True, "root": self._to_workspace_rel(base), "count": len(entries), "files": entries}

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        max_chars: int = 12000,
    ) -> dict[str, Any]:
        file_path = self._resolve(path)
        if not file_path.exists():
            return {"ok": False, "error": f"file not found: {path}"}
        if not file_path.is_file():
            return {"ok": False, "error": f"not a file: {path}"}

        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        s = max(1, start_line)
        e = len(lines) if end_line is None else max(s, min(end_line, len(lines)))
        selected = "\n".join(lines[s - 1 : e])
        truncated = False
        if len(selected) > max_chars:
            selected = selected[: max_chars - 3] + "..."
            truncated = True

        masked = redact_secrets_text(selected)

        return {
            "ok": True,
            "path": self._to_workspace_rel(file_path),
            "start_line": s,
            "end_line": e,
            "truncated": truncated,
            "content": masked,
        }

    def write_file(self, path: str, content: str, append: bool = False) -> dict[str, Any]:
        file_path = self._resolve(path)
        ensure_dir(file_path.parent)

        if append:
            with file_path.open("a", encoding="utf-8") as f:
                f.write(content)
        else:
            write_text(file_path, content)

        return {"ok": True, "path": self._to_workspace_rel(file_path), "bytes": len(content.encode("utf-8"))}

    def edit_file(
        self,
        path: str,
        find_text: str,
        replace_text: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        if not find_text:
            return {"ok": False, "error": "find_text must not be empty"}

        file_path = self._resolve(path)
        if not file_path.exists() or not file_path.is_file():
            return {"ok": False, "error": f"file not found: {path}"}

        original = file_path.read_text(encoding="utf-8", errors="replace")
        if find_text not in original:
            return {"ok": False, "error": "find_text not found"}

        if replace_all:
            updated = original.replace(find_text, replace_text)
            replacements = original.count(find_text)
        else:
            updated = original.replace(find_text, replace_text, 1)
            replacements = 1

        write_text(file_path, updated)
        return {"ok": True, "path": self._to_workspace_rel(file_path), "replacements": replacements}

    def _plan_path(self, plan_id: str) -> Path:
        return self.plans_dir / f"{slugify(plan_id)}.json"

    def _load_plan(self, plan_id: str) -> dict[str, Any] | None:
        path = self._plan_path(plan_id)
        if not path.exists():
            return None
        try:
            return read_json(path)
        except Exception:
            return None

    def create_plan(self, title: str, goal: str, steps: list[str]) -> dict[str, Any]:
        clean_steps = [s.strip() for s in steps if isinstance(s, str) and s.strip()]
        if not clean_steps:
            return {"ok": False, "error": "steps must contain at least one item"}

        base_id = slugify(title) or "plan"
        plan_id = base_id
        idx = 2
        while self._plan_path(plan_id).exists():
            plan_id = f"{base_id}_{idx}"
            idx += 1

        now = utc_now_iso()
        todos = []
        for i, step in enumerate(clean_steps, start=1):
            todos.append(
                {
                    "id": i,
                    "text": step,
                    "status": "pending",
                    "created_at": now,
                    "updated_at": now,
                }
            )

        plan = {
            "id": plan_id,
            "title": title.strip() or plan_id,
            "goal": goal.strip(),
            "created_at": now,
            "updated_at": now,
            "todos": todos,
        }
        write_json(self._plan_path(plan_id), plan)
        return {"ok": True, "plan_id": plan_id, "todo_count": len(todos), "plan": plan}

    def list_plans(self) -> dict[str, Any]:
        plans: list[dict[str, Any]] = []
        for p in sorted(self.plans_dir.glob("*.json")):
            try:
                plan = read_json(p)
            except Exception:
                continue
            todos = plan.get("todos", [])
            pending = 0
            done = 0
            if isinstance(todos, list):
                for todo in todos:
                    if not isinstance(todo, dict):
                        continue
                    if todo.get("status") == "done":
                        done += 1
                    else:
                        pending += 1
            plans.append(
                {
                    "id": plan.get("id", p.stem),
                    "title": plan.get("title", p.stem),
                    "goal": plan.get("goal", ""),
                    "updated_at": plan.get("updated_at", ""),
                    "pending": pending,
                    "done": done,
                }
            )
        return {"ok": True, "count": len(plans), "plans": plans}

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        plan = self._load_plan(plan_id)
        if not plan:
            return {"ok": False, "error": f"plan not found: {plan_id}"}
        return {"ok": True, "plan": plan}

    def add_todo(self, plan_id: str, text: str) -> dict[str, Any]:
        plan = self._load_plan(plan_id)
        if not plan:
            return {"ok": False, "error": f"plan not found: {plan_id}"}

        todos = plan.get("todos", [])
        if not isinstance(todos, list):
            todos = []
        next_id = max([int(t.get("id", 0)) for t in todos if isinstance(t, dict)] + [0]) + 1
        now = utc_now_iso()
        todo = {"id": next_id, "text": text.strip(), "status": "pending", "created_at": now, "updated_at": now}
        todos.append(todo)
        plan["todos"] = todos
        plan["updated_at"] = now
        write_json(self._plan_path(plan_id), plan)
        return {"ok": True, "plan_id": plan_id, "todo": todo}

    def update_todo(self, plan_id: str, todo_id: int, status: str) -> dict[str, Any]:
        plan = self._load_plan(plan_id)
        if not plan:
            return {"ok": False, "error": f"plan not found: {plan_id}"}

        valid = {"pending", "in_progress", "done"}
        if status not in valid:
            return {"ok": False, "error": f"invalid status: {status}"}

        todos = plan.get("todos", [])
        if not isinstance(todos, list):
            return {"ok": False, "error": "invalid plan todos"}

        updated = None
        now = utc_now_iso()
        for todo in todos:
            if not isinstance(todo, dict):
                continue
            if int(todo.get("id", -1)) == int(todo_id):
                todo["status"] = status
                todo["updated_at"] = now
                updated = todo
                break

        if updated is None:
            return {"ok": False, "error": f"todo not found: {todo_id}"}

        plan["updated_at"] = now
        write_json(self._plan_path(plan_id), plan)
        return {"ok": True, "plan_id": plan_id, "todo": updated}
