from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import unicodedata
from collections import deque
from pathlib import Path
from typing import Any

from .logging_config import get_logger
from .utils import (ensure_dir, read_json, redact_secrets_text, slugify,
                    utc_now_iso, write_json, write_text)

logger = get_logger(__name__)


class WorkspaceTools:
    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.plans_dir = self.workspace_root / "memory" / "plans"
        self._terminals: dict[str, dict[str, Any]] = {}
        self._terminal_buffer_lines = max(
            100, min(int(os.getenv("ASSISTANT_TERMINAL_BUFFER_LINES", "2000")), 20000)
        )
        self._terminal_line_max_chars = max(
            256,
            min(int(os.getenv("ASSISTANT_TERMINAL_LINE_MAX_CHARS", "4096")), 65536),
        )
        self._terminal_idle_timeout_s = max(
            60,
            min(int(os.getenv("ASSISTANT_TERMINAL_IDLE_TIMEOUT", "1800")), 86400),
        )
        ensure_dir(self.plans_dir)

    def _resolve(self, path: str) -> Path:
        """Resolve a path relative to workspace root with security validation."""
        raw = Path(path)
        try:
            resolved = (
                raw.resolve()
                if raw.is_absolute()
                else (self.workspace_root / raw).resolve()
            )
        except Exception as e:
            raise ValueError(f"invalid path: {path}") from e

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

    def _close_terminal_session(
        self, session_id: str, session: dict[str, Any], *, kill: bool = False
    ) -> None:
        proc = session.get("proc")
        if proc is not None:
            try:
                if proc.poll() is None:
                    if kill:
                        proc.kill()
                    else:
                        proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=1)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._terminals.pop(session_id, None)

    def close_idle_terminals(self, max_idle_s: int | None = None) -> int:
        idle_limit = (
            self._terminal_idle_timeout_s
            if max_idle_s is None
            else max(1, int(max_idle_s))
        )
        now = time.time()
        closed = 0
        for session_id, session in list(self._terminals.items()):
            proc = session.get("proc")
            last_used = float(session.get("last_used_at", now) or now)
            idle_for = max(0.0, now - last_used)
            should_close = idle_for >= idle_limit
            if proc is not None and proc.poll() is not None:
                should_close = True
            if not should_close:
                continue
            self._close_terminal_session(session_id, session, kill=idle_for >= idle_limit)
            closed += 1
        return closed

    def list_files(
        self,
        path: str = ".",
        glob: str = "**/*",
        include_hidden: bool = False,
        max_entries: int = 200,
    ) -> dict[str, Any]:
        """List files in a directory with comprehensive validation and error handling."""

        # Validate input parameters
        if not isinstance(path, str):
            return {"ok": False, "error": "path must be a string"}
        if not isinstance(glob, str):
            return {"ok": False, "error": "glob must be a string"}
        if not isinstance(include_hidden, bool):
            return {"ok": False, "error": "include_hidden must be a boolean"}
        if not isinstance(max_entries, int) or max_entries < 1:
            return {"ok": False, "error": "max_entries must be a positive integer"}

        try:
            base = self._resolve(path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        if not base.exists():
            return {"ok": False, "error": f"path not found: {path}"}
        if not base.is_dir():
            return {"ok": False, "error": f"not a directory: {path}"}

        entries: list[dict[str, Any]] = []
        pattern = glob.strip() or "**/*"

        try:
            for p in base.glob(pattern):
                if len(entries) >= max(1, max_entries):
                    break

                rel = self._to_workspace_rel(p)
                if not include_hidden and any(
                    part.startswith(".") for part in Path(rel).parts
                ):
                    continue
                if p.is_dir():
                    continue

                try:
                    size = p.stat().st_size
                except Exception:
                    size = 0
                entries.append({"path": rel, "size": size})

            entries.sort(key=lambda x: x["path"])
            return {
                "ok": True,
                "root": self._to_workspace_rel(base),
                "count": len(entries),
                "files": entries,
            }
        except Exception as e:
            logger.error(f"Failed to list files in {path}: {e}")
            return {
                "ok": False,
                "error": f"error listing files: {str(e)}",
                "hint": "Check permissions and path validity",
            }

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        max_chars: int = 12000,
    ) -> dict[str, Any]:
        file_path = self._resolve(path)
        if not file_path.exists():
            logger.warning(f"Read failed: file not found: {path}")
            return {"ok": False, "error": f"file not found: {path}"}
        if not file_path.is_file():
            logger.warning(f"Read failed: not a file: {path}")
            return {"ok": False, "error": f"not a file: {path}"}

        s = max(1, start_line)
        requested_end = None if end_line is None else max(s, int(end_line))
        max_len = max(1, int(max_chars))

        selected_parts: list[str] = []
        selected_chars = 0
        truncated = False
        actual_end = s

        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            for idx, raw_line in enumerate(f, start=1):
                if idx < s:
                    continue
                if requested_end is not None and idx > requested_end:
                    break

                line = raw_line.rstrip("\n")
                segment = line if not selected_parts else f"\n{line}"
                seg_len = len(segment)
                if selected_chars + seg_len > max_len:
                    remaining = max_len - selected_chars
                    if remaining > 0:
                        selected_parts.append(segment[:remaining])
                    truncated = True
                    break
                selected_parts.append(segment)
                selected_chars += seg_len
                actual_end = idx

        selected = "".join(selected_parts)
        if len(selected) > max_len:
            clip = max(1, max_len - 3)
            selected = selected[:clip] + "..."
            truncated = True
        elif requested_end is None and selected_chars >= max_len:
            truncated = True

        masked = redact_secrets_text(selected)

        return {
            "ok": True,
            "path": self._to_workspace_rel(file_path),
            "start_line": s,
            "end_line": max(s, actual_end),
            "truncated": truncated,
            "content": masked,
        }

    def create_file(
        self, path: str, content: str, overwrite: bool = False
    ) -> dict[str, Any]:
        file_path = self._resolve(path)
        ensure_dir(file_path.parent)

        if file_path.exists() and not overwrite:
            logger.warning(f"Create failed: file already exists: {path}")
            return {
                "ok": False,
                "error": f"file already exists: {path}",
                "hint": "Use edit_file to update existing files, or create_file(..., overwrite=true) to replace.",
            }
        write_text(file_path, content)

        return {
            "ok": True,
            "path": self._to_workspace_rel(file_path),
            "bytes": len(content.encode("utf-8")),
        }

    def create_folder(self, path: str) -> dict[str, Any]:
        folder_path = self._resolve(path)
        if folder_path.exists():
            return {"ok": False, "error": f"path already exists: {path}"}

        try:
            ensure_dir(folder_path)
            return {"ok": True, "path": self._to_workspace_rel(folder_path)}
        except Exception as e:
            return {"ok": False, "error": f"failed to create folder: {e}"}

    def delete_file(self, path: str) -> dict[str, Any]:
        target_path = self._resolve(path)
        if not target_path.exists():
            return {"ok": False, "error": f"path not found: {path}"}

        try:
            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
            return {
                "ok": True,
                "path": self._to_workspace_rel(target_path),
                "deleted": True,
            }
        except Exception as e:
            return {"ok": False, "error": f"failed to delete: {e}"}

    def run_terminal(
        self, action: str, cmd: str | None = None, session_id: str = "default"
    ) -> dict[str, Any]:
        actions = {"start", "send", "read", "close"}
        if action not in actions:
            return {
                "ok": False,
                "error": f"invalid action '{action}', must be one of {actions}",
            }

        self.close_idle_terminals()

        if action == "start":
            if session_id in self._terminals:
                return {"ok": False, "error": f"Session {session_id} already exists."}

            try:
                proc = subprocess.Popen(
                    ["/bin/bash"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(self.workspace_root),
                )
            except Exception as e:
                return {"ok": False, "error": f"failed to start session: {e}"}

            output_buffer: deque[str] = deque(maxlen=self._terminal_buffer_lines)
            buffer_lock = threading.Lock()
            buffer_stats = {"dropped_lines": 0}

            def reader(pipe, buffer_ref, buffer_lock_ref, buffer_stats_ref):
                try:
                    for line in iter(pipe.readline, ""):
                        clipped = (
                            line[-self._terminal_line_max_chars :]
                            if len(line) > self._terminal_line_max_chars
                            else line
                        )
                        with buffer_lock_ref:
                            if len(buffer_ref) >= self._terminal_buffer_lines:
                                buffer_stats_ref["dropped_lines"] += 1
                            buffer_ref.append(clipped)
                except Exception:
                    pass
                finally:
                    pipe.close()

            t = threading.Thread(
                target=reader,
                args=(proc.stdout, output_buffer, buffer_lock, buffer_stats),
                daemon=True,
            )
            t.start()
            self._terminals[session_id] = {
                "proc": proc,
                "buffer": output_buffer,
                "buffer_stats": buffer_stats,
                "lock": buffer_lock,
                "thread": t,
                "last_used_at": time.time(),
            }
            return {
                "ok": True,
                "session_id": session_id,
                "message": f"Started session: {session_id}",
            }

        s = self._terminals.get(session_id)
        if not s:
            return {"ok": False, "error": f"Session {session_id} not found."}

        if action == "send":
            if not cmd:
                return {"ok": False, "error": "No command provided for 'send' action."}
            try:
                # Add newline if not present
                if not cmd.endswith("\n"):
                    cmd += "\n"
                s["proc"].stdin.write(cmd)
                s["proc"].stdin.flush()
                s["last_used_at"] = time.time()
                return {
                    "ok": True,
                    "session_id": session_id,
                    "message": f"Sent command.",
                }
            except Exception as e:
                return {"ok": False, "error": f"failed to send command: {e}"}

        if action == "read":
            lines = []
            time.sleep(0.1)  # Brief wait for fast outputs
            buffer_lock = s.get("lock")
            buffer_ref = s.get("buffer")
            if isinstance(buffer_ref, deque) and buffer_lock is not None:
                with buffer_lock:
                    lines = list(buffer_ref)
                    buffer_ref.clear()
            s["last_used_at"] = time.time()
            output = "".join(lines)
            stats = s.get("buffer_stats", {})
            dropped_lines = (
                int(stats.get("dropped_lines", 0) or 0)
                if isinstance(stats, dict)
                else 0
            )
            if isinstance(stats, dict):
                stats["dropped_lines"] = 0
            return {
                "ok": True,
                "session_id": session_id,
                "output": output if output else "[No new output]",
                "truncated": dropped_lines > 0,
                "dropped_lines": dropped_lines,
            }

        if action == "close":
            try:
                self._close_terminal_session(session_id, s)
                return {
                    "ok": True,
                    "session_id": session_id,
                    "message": f"Closed session: {session_id}",
                }
            except Exception as e:
                return {"ok": False, "error": f"failed to close session: {e}"}

    def write_file(
        self, path: str, content: str, append: bool = False
    ) -> dict[str, Any]:
        # Backward-compatible shim. Prefer create_file + edit_file in tool protocol.
        file_path = self._resolve(path)
        ensure_dir(file_path.parent)
        if append:
            with file_path.open("a", encoding="utf-8") as f:
                f.write(content)
            return {
                "ok": True,
                "path": self._to_workspace_rel(file_path),
                "bytes": len(content.encode("utf-8")),
                "deprecated": True,
                "hint": "write_file is deprecated; prefer create_file/edit_file.",
            }
        result = self.create_file(path=path, content=content, overwrite=True)
        result["deprecated"] = True
        result["hint"] = "write_file is deprecated; prefer create_file/edit_file."
        return result

    @staticmethod
    def _normalize_edit_match_line(text: str) -> str:
        if not isinstance(text, str):
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = (
            normalized.replace("\u00a0", " ")
            .replace("\u2010", "-")
            .replace("\u2011", "-")
            .replace("\u2012", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u2212", "-")
        )
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @staticmethod
    def _split_line_ending(line: str) -> tuple[str, str]:
        if line.endswith("\r\n"):
            return line[:-2], "\r\n"
        if line.endswith("\n"):
            return line[:-1], "\n"
        if line.endswith("\r"):
            return line[:-1], "\r"
        return line, ""

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
            return {
                "ok": False,
                "error": f"file not found: {path}",
                "hint": "Create the file first with create_file(path, content), then call edit_file.",
            }

        original = file_path.read_text(encoding="utf-8", errors="replace")
        find_text_clean = find_text.rstrip("\r\n")
        if find_text in original:
            matched_text = find_text
        elif find_text_clean and find_text_clean in original:
            matched_text = find_text_clean
        else:
            matched_text = None

        if matched_text is None:
            # Fallback for single-line edits when model output drifts in whitespace
            # or Unicode punctuation (e.g., non-breaking spaces, en-dash vs hyphen).
            if "\n" not in find_text_clean and "\r" not in find_text_clean:
                target = self._normalize_edit_match_line(find_text_clean)
                if target:
                    lines = original.splitlines(keepends=True)
                    hit_indexes: list[int] = []
                    for idx, raw_line in enumerate(lines):
                        body, _ = self._split_line_ending(raw_line)
                        if self._normalize_edit_match_line(body) == target:
                            hit_indexes.append(idx)

                    if hit_indexes:
                        if replace_all:
                            for idx in hit_indexes:
                                body, line_ending = self._split_line_ending(lines[idx])
                                leading = re.match(r"[ \t]*", body)
                                indent = leading.group(0) if leading else ""
                                replacement = (
                                    replace_text
                                    if replace_text.startswith((" ", "\t"))
                                    else f"{indent}{replace_text.lstrip()}"
                                )
                                lines[idx] = f"{replacement}{line_ending}"
                            replacements = len(hit_indexes)
                        else:
                            idx = hit_indexes[0]
                            body, line_ending = self._split_line_ending(lines[idx])
                            leading = re.match(r"[ \t]*", body)
                            indent = leading.group(0) if leading else ""
                            replacement = (
                                replace_text
                                if replace_text.startswith((" ", "\t"))
                                else f"{indent}{replace_text.lstrip()}"
                            )
                            lines[idx] = f"{replacement}{line_ending}"
                            replacements = 1

                        write_text(file_path, "".join(lines))
                        return {
                            "ok": True,
                            "path": self._to_workspace_rel(file_path),
                            "replacements": replacements,
                            "match_mode": "normalized_line",
                        }

            return {
                "ok": False,
                "error": "find_text not found",
                "hint": "Exact match failed. Re-read the file and copy the snippet exactly, or use a shorter unique find_text.",
            }

        if replace_all:
            updated = original.replace(matched_text, replace_text)
            replacements = original.count(matched_text)
        else:
            updated = original.replace(matched_text, replace_text, 1)
            replacements = 1

        write_text(file_path, updated)
        return {
            "ok": True,
            "path": self._to_workspace_rel(file_path),
            "replacements": replacements,
        }

    def search_project(
        self,
        query: str,
        path: str = ".",
        glob: str = "**/*",
        case_sensitive: bool = False,
        regex: bool = False,
        max_matches: int = 200,
    ) -> dict[str, Any]:
        base = self._resolve(path)
        if not base.exists():
            return {"ok": False, "error": f"path not found: {path}"}
        if not base.is_dir():
            return {"ok": False, "error": f"not a directory: {path}"}
        q = (query or "").strip()
        if not q:
            return {"ok": False, "error": "query must not be empty"}

        limit = max(1, min(int(max_matches), 1000))
        rel_base = self._to_workspace_rel(base)

        if shutil.which("rg"):
            per_file_limit = max(1, min(limit, 20))
            cmd = [
                "rg",
                "--json",
                "--line-number",
                "--color",
                "never",
                "--no-messages",
                "--max-count",
                str(per_file_limit),
                "--max-filesize",
                "1M",
            ]
            if not case_sensitive:
                cmd.append("-i")
            if not regex:
                cmd.append("-F")
            if glob and glob != "**/*":
                cmd.extend(["-g", glob])
            cmd.extend([q, str(base)])
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                matches: list[dict[str, Any]] = []
                if proc.stdout is not None:
                    for raw in proc.stdout:
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        if item.get("type") != "match":
                            continue
                        data = item.get("data", {})
                        path_text = str(data.get("path", {}).get("text", "")).strip()
                        line_num = int(data.get("line_number", 0) or 0)
                        text_data = data.get("lines", {}).get("text", "")
                        text_val = str(text_data).rstrip("\n")
                        if not path_text:
                            continue
                        try:
                            rel = self._to_workspace_rel(Path(path_text))
                        except Exception:
                            rel = path_text
                        matches.append({"path": rel, "line": line_num, "text": text_val})
                        if len(matches) >= limit:
                            proc.terminate()
                            break
                try:
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=1)
                    except Exception:
                        pass

                if proc.returncode not in (0, 1, None):
                    err_excerpt = ""
                    if proc.stderr is not None:
                        try:
                            err_excerpt = (proc.stderr.read() or "")[:600]
                        except Exception:
                            err_excerpt = ""
                    return {
                        "ok": False,
                        "engine": "rg",
                        "query": q,
                        "path": rel_base,
                        "error": f"rg failed with exit code {proc.returncode}",
                        "stderr": err_excerpt,
                    }

                return {
                    "ok": True,
                    "engine": "rg",
                    "query": q,
                    "path": rel_base,
                    "count": len(matches),
                    "matches": matches,
                }
            except Exception:
                pass

        # Python fallback search
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(q, flags=flags) if regex else None
        matches: list[dict[str, Any]] = []
        max_fallback_file_bytes = 1_000_000
        for p in base.glob(glob.strip() or "**/*"):
            if len(matches) >= limit:
                break
            if not p.is_file():
                continue
            try:
                if p.stat().st_size > max_fallback_file_bytes:
                    continue
            except Exception:
                continue
            try:
                raw = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if "\x00" in raw:
                continue
            for i, line in enumerate(raw.splitlines(), start=1):
                hit = (
                    bool(pattern.search(line))
                    if pattern
                    else (q in line if case_sensitive else q.lower() in line.lower())
                )
                if not hit:
                    continue
                matches.append(
                    {"path": self._to_workspace_rel(p), "line": i, "text": line[:400]}
                )
                if len(matches) >= limit:
                    break

        return {
            "ok": True,
            "engine": "python",
            "query": q,
            "path": rel_base,
            "count": len(matches),
            "matches": matches,
        }

    @staticmethod
    def _code_language(path: Path) -> str:
        ext = path.suffix.lower()
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".hpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".cs": "csharp",
        }
        return mapping.get(ext, "text")

    def _extract_symbols(
        self, path: Path, text: str, max_symbols_per_file: int = 200
    ) -> list[dict[str, Any]]:
        language = self._code_language(path)
        patterns: list[tuple[str, str]]
        if language == "python":
            patterns = [
                ("class", r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
                ("function", r"^\s*(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
            ]
        elif language in {"javascript", "typescript"}:
            patterns = [
                ("class", r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
                ("function", r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
                (
                    "function",
                    r"^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\(",
                ),
            ]
        elif language == "go":
            patterns = [
                ("function", r"^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
                (
                    "method",
                    r"^\s*func\s*\([^)]*\)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                ),
                ("type", r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+struct\b"),
            ]
        elif language == "rust":
            patterns = [
                ("struct", r"^\s*(?:pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
                ("enum", r"^\s*(?:pub\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
                ("trait", r"^\s*(?:pub\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
                ("function", r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
            ]
        elif language in {"java", "csharp"}:
            patterns = [
                (
                    "class",
                    r"^\s*(?:public|private|protected)?\s*(?:final\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                ),
                (
                    "interface",
                    r"^\s*(?:public|private|protected)?\s*interface\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                ),
                (
                    "method",
                    r"^\s*(?:public|private|protected)?\s*(?:static\s+)?[A-Za-z0-9_<>\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                ),
            ]
        else:
            patterns = [
                (
                    "symbol",
                    r"^\s*(?:def|function|class|struct|enum|trait|fn)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                )
            ]

        symbols: list[dict[str, Any]] = []
        for i, line in enumerate(text.splitlines(), start=1):
            if len(symbols) >= max_symbols_per_file:
                break
            for sym_type, pat in patterns:
                m = re.search(pat, line)
                if not m:
                    continue
                symbols.append(
                    {
                        "name": m.group(1),
                        "type": sym_type,
                        "line": i,
                        "language": language,
                    }
                )
                break
        return symbols

    def index_symbols(
        self,
        path: str = ".",
        glob: str = "**/*",
        max_files: int = 300,
        max_symbols: int = 5000,
    ) -> dict[str, Any]:
        base = self._resolve(path)
        if not base.exists():
            return {"ok": False, "error": f"path not found: {path}"}
        if not base.is_dir():
            return {"ok": False, "error": f"not a directory: {path}"}

        code_ext = {
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".go",
            ".rs",
            ".java",
            ".c",
            ".h",
            ".cpp",
            ".hpp",
            ".cc",
            ".cxx",
            ".cs",
        }
        file_limit = max(1, min(int(max_files), 5000))
        symbol_limit = max(1, min(int(max_symbols), 20000))
        pattern = glob.strip() or "**/*"

        indexed_files = 0
        symbols: list[dict[str, Any]] = []
        for p in base.glob(pattern):
            if indexed_files >= file_limit or len(symbols) >= symbol_limit:
                break
            if not p.is_file() or p.suffix.lower() not in code_ext:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if "\x00" in text:
                continue
            indexed_files += 1
            for sym in self._extract_symbols(p, text):
                if len(symbols) >= symbol_limit:
                    break
                item = dict(sym)
                item["path"] = self._to_workspace_rel(p)
                symbols.append(item)

        return {
            "ok": True,
            "path": self._to_workspace_rel(base),
            "indexed_files": indexed_files,
            "count": len(symbols),
            "symbols": symbols,
        }

    def lookup_symbol(
        self,
        symbol: str,
        path: str = ".",
        glob: str = "**/*",
        exact: bool = False,
        max_results: int = 30,
    ) -> dict[str, Any]:
        query = (symbol or "").strip()
        if not query:
            return {"ok": False, "error": "symbol must not be empty"}

        index = self.index_symbols(path=path, glob=glob, max_files=400, max_symbols=8000)
        if not index.get("ok", False):
            return index
        symbols = index.get("symbols", [])
        if not isinstance(symbols, list):
            symbols = []

        q = query.lower()
        exact_match = bool(exact)
        results: list[dict[str, Any]] = []
        limit = max(1, min(int(max_results), 500))
        for item in symbols:
            if len(results) >= limit:
                break
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            if not name:
                continue
            hit = name.lower() == q if exact_match else q in name.lower()
            if not hit:
                continue
            results.append(item)

        return {
            "ok": True,
            "symbol": query,
            "exact": exact_match,
            "count": len(results),
            "matches": results,
        }

    def summarize_file(self, path: str, max_symbols: int = 20) -> dict[str, Any]:
        file_path = self._resolve(path)
        if not file_path.exists():
            return {"ok": False, "error": f"file not found: {path}"}
        if not file_path.is_file():
            return {"ok": False, "error": f"not a file: {path}"}

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"ok": False, "error": f"failed to read file: {e}"}

        lines = text.splitlines()
        language = self._code_language(file_path)
        symbols = self._extract_symbols(file_path, text, max_symbols_per_file=max_symbols)

        imports: list[str] = []
        for line in lines[:400]:
            s = line.strip()
            if (
                s.startswith("import ")
                or s.startswith("from ")
                or s.startswith("#include ")
                or s.startswith("use ")
            ):
                imports.append(s)
            if len(imports) >= 20:
                break

        symbol_names = [f"{x.get('type')}:{x.get('name')}" for x in symbols[:max_symbols]]
        summary_parts = [
            f"path={self._to_workspace_rel(file_path)}",
            f"language={language}",
            f"lines={len(lines)}",
            f"imports={len(imports)}",
            f"symbols={len(symbols)}",
        ]
        if symbol_names:
            summary_parts.append(f"top_symbols={', '.join(symbol_names[:10])}")
        summary = " | ".join(summary_parts)

        return {
            "ok": True,
            "path": self._to_workspace_rel(file_path),
            "language": language,
            "line_count": len(lines),
            "imports": imports,
            "symbols": symbols,
            "summary": summary,
        }

    def detect_project_context(
        self, path: str = ".", include_runtime: bool = True
    ) -> dict[str, Any]:
        base = self._resolve(path)
        if not base.exists() or not base.is_dir():
            return {"ok": False, "error": f"not a directory: {path}"}

        top_files = {p.name for p in base.iterdir() if p.is_file()}
        all_files: list[Path] = []
        for p in base.glob("**/*"):
            if p.is_file():
                all_files.append(p)
            if len(all_files) >= 6000:
                break

        lang_counts: dict[str, int] = {}
        for p in all_files:
            ext = p.suffix.lower() or "<none>"
            lang_counts[ext] = lang_counts.get(ext, 0) + 1

        framework = "unknown"
        test_runner = "unknown"
        entry_points: list[str] = []

        req_text = ""
        req_path = base / "requirements.txt"
        if req_path.exists():
            try:
                req_text = req_path.read_text(encoding="utf-8", errors="replace").lower()
            except Exception:
                req_text = ""

        pyproject_text = ""
        pyproject = base / "pyproject.toml"
        if pyproject.exists():
            try:
                pyproject_text = pyproject.read_text(
                    encoding="utf-8", errors="replace"
                ).lower()
            except Exception:
                pyproject_text = ""

        package_json_text = ""
        package_json = base / "package.json"
        if package_json.exists():
            try:
                package_json_text = package_json.read_text(
                    encoding="utf-8", errors="replace"
                ).lower()
            except Exception:
                package_json_text = ""

        if "fastapi" in req_text or "fastapi" in pyproject_text:
            framework = "FastAPI"
            test_runner = "pytest"
        elif "flask" in req_text or "flask" in pyproject_text:
            framework = "Flask"
            test_runner = "pytest"
        elif "django" in req_text or "django" in pyproject_text:
            framework = "Django"
            test_runner = "pytest"
        elif package_json.exists():
            framework = "Node.js"
            test_runner = "npm test"
            if "next" in package_json_text:
                framework = "Next.js"
            elif "react" in package_json_text:
                framework = "React"
            elif "express" in package_json_text:
                framework = "Express"
        elif (base / "go.mod").exists():
            framework = "Go"
            test_runner = "go test ./..."
        elif (base / "Cargo.toml").exists():
            framework = "Rust"
            test_runner = "cargo test"
        elif any(p.suffix.lower() == ".py" for p in all_files):
            framework = "Python"
            test_runner = "pytest"

        candidates = [
            "main.py",
            "app.py",
            "server.py",
            "manage.py",
            "src/main.py",
            "src/app.py",
            "index.js",
            "server.js",
            "main.go",
        ]
        for rel in candidates:
            p = base / rel
            if p.exists() and p.is_file():
                entry_points.append(self._to_workspace_rel(p))

        if not entry_points:
            for p in all_files[:250]:
                name = p.name.lower()
                if name in {"main.py", "app.py", "server.py", "index.js", "main.go"}:
                    entry_points.append(self._to_workspace_rel(p))
                    if len(entry_points) >= 8:
                        break

        runtime: dict[str, Any] = {}
        if include_runtime:
            try:
                git_proc = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(base),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                runtime["git_dirty_files"] = len(
                    [ln for ln in git_proc.stdout.splitlines() if ln.strip()]
                )
            except Exception:
                runtime["git_dirty_files"] = None
            try:
                py_proc = subprocess.run(
                    ["python3", "--version"],
                    cwd=str(base),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                runtime["python_version"] = (
                    py_proc.stdout.strip() or py_proc.stderr.strip()
                )
            except Exception:
                runtime["python_version"] = ""

        return {
            "ok": True,
            "path": self._to_workspace_rel(base),
            "framework": framework,
            "test_runner": test_runner,
            "entry_points": entry_points,
            "top_files": sorted(top_files)[:60],
            "language_extensions": dict(
                sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            ),
            "runtime": runtime,
        }

    def execute_command(
        self,
        cmd: str,
        path: str = ".",
        timeout: int = 120,
        max_output_chars: int = 12000,
    ) -> dict[str, Any]:
        base = self._resolve(path)
        if not base.exists() or not base.is_dir():
            return {"ok": False, "error": f"not a directory: {path}"}
        if not isinstance(cmd, str) or not cmd.strip():
            return {"ok": False, "error": "cmd must not be empty"}

        started = time.time()
        try:
            proc = subprocess.run(
                # Use non-login shell for deterministic behavior inside the workspace.
                # Login shells may source user profiles and introduce aliases/path hacks
                # that make autonomous runs operate on unexpected locations.
                ["/bin/bash", "-c", cmd],
                cwd=str(base),
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout)),
                check=False,
            )
            duration_ms = int((time.time() - started) * 1000)
            out = (proc.stdout or "")[:max_output_chars]
            err = (proc.stderr or "")[:max_output_chars]
            return {
                "ok": proc.returncode == 0,
                "cmd": cmd,
                "path": self._to_workspace_rel(base),
                "exit_code": int(proc.returncode),
                "duration_ms": duration_ms,
                "stdout": out,
                "stderr": err,
                "truncated": (
                    len(proc.stdout or "") > max_output_chars
                    or len(proc.stderr or "") > max_output_chars
                ),
            }
        except subprocess.TimeoutExpired as e:
            duration_ms = int((time.time() - started) * 1000)
            stdout = (e.stdout or "")[:max_output_chars] if isinstance(e.stdout, str) else ""
            stderr = (e.stderr or "")[:max_output_chars] if isinstance(e.stderr, str) else ""
            return {
                "ok": False,
                "cmd": cmd,
                "path": self._to_workspace_rel(base),
                "exit_code": None,
                "duration_ms": duration_ms,
                "timeout": True,
                "stdout": stdout,
                "stderr": stderr,
                "error": f"command timed out after {timeout}s",
            }
        except Exception as e:
            return {"ok": False, "error": f"failed to execute command: {e}", "cmd": cmd}

    def run_tests(
        self,
        path: str = ".",
        runner: str = "auto",
        args: str = "",
        timeout: int = 300,
    ) -> dict[str, Any]:
        base = self._resolve(path)
        if not base.exists() or not base.is_dir():
            return {"ok": False, "error": f"not a directory: {path}"}

        selected = (runner or "auto").strip().lower()
        if selected == "auto":
            ctx = self.detect_project_context(path=path, include_runtime=False)
            selected = str(ctx.get("test_runner", "pytest")).strip().lower()
            if not selected or selected == "unknown":
                selected = "pytest"

        if selected in {"pytest", "py"}:
            cmd = f"pytest -q {args}".strip()
        elif selected in {"npm", "npm test", "node"}:
            cmd = f"npm test -- {args}".strip()
        elif selected in {"go", "go test"}:
            cmd = f"go test ./... {args}".strip()
        elif selected in {"cargo", "cargo test", "rust"}:
            cmd = f"cargo test {args}".strip()
        else:
            cmd = f"{selected} {args}".strip()

        result = self.execute_command(
            cmd=cmd, path=path, timeout=timeout, max_output_chars=18000
        )
        stdout = str(result.get("stdout", ""))
        stderr = str(result.get("stderr", ""))
        text = f"{stdout}\n{stderr}"
        parsed = self._parse_test_output(text, exit_code=result.get("exit_code"))
        exit_code = result.get("exit_code")
        completed = exit_code is not None and not bool(result.get("timeout", False))
        tests_passed = bool(
            completed
            and int(exit_code or 0) == 0
            and int(parsed.get("failed", 0) or 0) == 0
            and int(parsed.get("errors", 0) or 0) == 0
        )
        if bool(result.get("timeout", False)):
            failure_mode = "timeout"
        elif not completed:
            failure_mode = "runner_error"
        elif tests_passed:
            failure_mode = "none"
        elif int(parsed.get("collection_errors", 0) or 0) > 0:
            failure_mode = "collection_error"
        elif int(parsed.get("failed", 0) or 0) > 0:
            failure_mode = "assertion_failures"
        elif bool(parsed.get("unparsed_nonzero_exit", False)):
            failure_mode = "unparsed_nonzero_exit"
        else:
            failure_mode = "runner_error"
        runner_failed = failure_mode in {
            "timeout",
            "runner_error",
            "collection_error",
            "unparsed_nonzero_exit",
        }

        return {
            "ok": completed,
            "completed": completed,
            "runner": selected,
            "command": cmd,
            "path": self._to_workspace_rel(base),
            "exit_code": exit_code,
            "duration_ms": result.get("duration_ms"),
            "passed": int(parsed.get("passed", 0) or 0),
            "failed": int(parsed.get("failed", 0) or 0),
            "errors": int(parsed.get("errors", 0) or 0),
            "skipped": int(parsed.get("skipped", 0) or 0),
            "collection_errors": int(parsed.get("collection_errors", 0) or 0),
            "test_failures": parsed.get("test_failures", []),
            "tests_passed": tests_passed,
            "runner_failed": runner_failed,
            "failure_mode": failure_mode,
            "summary_line": str(parsed.get("summary_line", "")).strip(),
            "unparsed_nonzero_exit": bool(parsed.get("unparsed_nonzero_exit", False)),
            "stdout": stdout,
            "stderr": stderr,
            "raw": result,
        }

    @staticmethod
    def _parse_test_output(
        text: str, exit_code: int | None = None, max_failures: int = 5
    ) -> dict[str, Any]:
        failed = 0
        passed = 0
        errors = 0
        skipped = 0
        collection_errors = 0
        summary_line = ""
        failed_counts = re.findall(r"(?<!\S)(\d+)\s+failed\b", text, flags=re.IGNORECASE)
        passed_counts = re.findall(r"(?<!\S)(\d+)\s+passed\b", text, flags=re.IGNORECASE)
        error_counts = re.findall(r"(?<!\S)(\d+)\s+errors?\b", text, flags=re.IGNORECASE)
        skipped_counts = re.findall(r"(?<!\S)(\d+)\s+skipped\b", text, flags=re.IGNORECASE)
        if failed_counts:
            failed = int(failed_counts[-1])
        if passed_counts:
            passed = int(passed_counts[-1])
        if error_counts:
            errors = int(error_counts[-1])
        if skipped_counts:
            skipped = int(skipped_counts[-1])

        failures: list[dict[str, str]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lowered = line.lower()
            if (
                not summary_line
                and any(token in lowered for token in (" failed", " passed", " error", " skipped"))
                and ("=" in line or "collected" in lowered)
            ):
                summary_line = line[:320]
            if not line.startswith("FAILED "):
                if line.startswith("ERROR collecting "):
                    collection_errors += 1
                    failures.append(
                        {
                            "nodeid": line[len("ERROR collecting ") :].strip(),
                            "message": "collection error",
                            "summary": line[:320],
                            "kind": "collection_error",
                        }
                    )
                elif line.startswith("ERROR "):
                    body = line[len("ERROR ") :].strip()
                    nodeid = body
                    message = ""
                    if " - " in body:
                        nodeid, message = body.split(" - ", 1)
                    failures.append(
                        {
                            "nodeid": nodeid.strip(),
                            "message": message.strip()[:240],
                            "summary": line[:320],
                            "kind": "error",
                        }
                    )
                if len(failures) >= max(1, max_failures):
                    break
                continue
            body = line[len("FAILED ") :].strip()
            nodeid = body
            message = ""
            if " - " in body:
                nodeid, message = body.split(" - ", 1)
            failures.append(
                {
                    "nodeid": nodeid.strip(),
                    "message": message.strip()[:240],
                    "summary": line[:320],
                    "kind": "failed",
                }
            )
            if len(failures) >= max(1, max_failures):
                break

        if failed == 0:
            failed = sum(1 for item in failures if item.get("kind") == "failed")
        if errors == 0:
            errors = (
                collection_errors
                + sum(1 for item in failures if item.get("kind") == "error")
            )

        if not failures:
            # Fallback for runners that don't emit pytest-style FAILED lines.
            excerpt_lines: list[str] = []
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                lowered = line.lower()
                if any(token in lowered for token in ("assert", "traceback", "error", "exception")):
                    excerpt_lines.append(line[:240])
                if len(excerpt_lines) >= max(1, max_failures):
                    break
            failures = [
                {"nodeid": "", "message": line, "summary": line, "kind": "error"}
                for line in excerpt_lines
            ]
            if failures and errors == 0:
                errors = len(failures)

        unparsed_nonzero_exit = False
        if exit_code not in (None, 0) and failed == 0 and errors == 0:
            unparsed_nonzero_exit = True
            errors = 1
            excerpt = ""
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if line:
                    excerpt = line[:240]
                    break
            failures = [
                {
                    "nodeid": "",
                    "message": excerpt or f"non-zero exit code: {exit_code}",
                    "summary": "pytest exited non-zero before structured failures were parsed",
                    "kind": "runner_error",
                }
            ]

        if not summary_line:
            for raw_line in reversed(text.splitlines()):
                line = raw_line.strip()
                if not line:
                    continue
                lowered = line.lower()
                if any(token in lowered for token in ("failed", "passed", "error", "skipped")):
                    summary_line = line[:320]
                    break

        return {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "collection_errors": collection_errors,
            "summary_line": summary_line,
            "unparsed_nonzero_exit": unparsed_nonzero_exit,
            "test_failures": failures,
        }

    @staticmethod
    def _diff_stats(diff_text: str) -> dict[str, Any]:
        files: set[str] = set()
        additions = 0
        deletions = 0
        hunks = 0
        for line in diff_text.splitlines():
            if line.startswith("@@"):
                hunks += 1
                continue
            if line.startswith("+++ b/") or line.startswith("--- a/"):
                candidate = line[6:].strip()
                if candidate and candidate != "/dev/null":
                    files.add(candidate)
                continue
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1

        return {
            "changed_file_count": len(files),
            "changed_files": sorted(files)[:200],
            "additions": additions,
            "deletions": deletions,
            "hunks": hunks,
            "has_diff": bool(files or additions or deletions),
        }

    @staticmethod
    def _normalize_workspace_path(path: str) -> str:
        return Path(str(path or "")).as_posix().lstrip("./")

    def _run_git_command(
        self, args: list[str], *, path: str = ".", timeout: int = 30
    ) -> dict[str, Any]:
        base = self._resolve(path)
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(base),
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout)),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "timeout": True, "stdout": "", "stderr": ""}
        except Exception as e:
            return {"ok": False, "error": f"git command failed: {e}", "stdout": "", "stderr": ""}
        return {
            "ok": proc.returncode == 0,
            "exit_code": int(proc.returncode),
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
        }

    def _git_status_entries(self, path: str = ".") -> list[dict[str, Any]]:
        result = self._run_git_command(
            ["status", "--porcelain", "--untracked-files=all"],
            path=path,
            timeout=15,
        )
        exit_code = result.get("exit_code")
        if not isinstance(exit_code, int) or exit_code != 0:
            return []
        entries: list[dict[str, Any]] = []
        for raw_line in str(result.get("stdout", "")).splitlines():
            line = raw_line.rstrip()
            if len(line) < 3:
                continue
            status = line[:2]
            payload = line[3:].strip()
            if not payload:
                continue
            if " -> " in payload:
                _, payload = payload.split(" -> ", 1)
            entries.append(
                {
                    "path": self._normalize_workspace_path(payload),
                    "staged_status": status[0],
                    "unstaged_status": status[1],
                    "untracked": status == "??",
                }
            )
        return entries

    def _git_numstat(self, path: str = ".", staged: bool = False) -> dict[str, tuple[int, int]]:
        args = ["diff", "--numstat"]
        if staged:
            args.append("--staged")
        result = self._run_git_command(args, path=path, timeout=20)
        exit_code = result.get("exit_code")
        if not isinstance(exit_code, int) or exit_code != 0:
            return {}
        stats: dict[str, tuple[int, int]] = {}
        for raw_line in str(result.get("stdout", "")).splitlines():
            parts = raw_line.split("\t")
            if len(parts) < 3:
                continue
            add_raw, del_raw, file_raw = parts[0], parts[1], parts[2]
            try:
                additions = int(add_raw)
            except Exception:
                additions = 0
            try:
                deletions = int(del_raw)
            except Exception:
                deletions = 0
            stats[self._normalize_workspace_path(file_raw)] = (additions, deletions)
        return stats

    def _workspace_change_stats(
        self,
        path: str = ".",
        focus_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        workspace_files: set[str] = set()
        untracked_files: set[str] = set()
        for entry in self._git_status_entries(path=path):
            file_path = str(entry.get("path", "")).strip()
            if not file_path:
                continue
            workspace_files.add(file_path)
            if bool(entry.get("untracked", False)):
                untracked_files.add(file_path)

        unstaged_numstat = self._git_numstat(path=path, staged=False)
        staged_numstat = self._git_numstat(path=path, staged=True)
        workspace_files.update(unstaged_numstat.keys())
        workspace_files.update(staged_numstat.keys())

        focus_set = {
            self._normalize_workspace_path(item)
            for item in (focus_paths or [])
            if str(item).strip()
        }
        if focus_set:
            focused_files = sorted(path for path in workspace_files if path in focus_set)
        else:
            focused_files = sorted(workspace_files)

        workspace_additions = 0
        workspace_deletions = 0
        for additions, deletions in list(unstaged_numstat.values()) + list(staged_numstat.values()):
            workspace_additions += additions
            workspace_deletions += deletions

        focused_additions = 0
        focused_deletions = 0
        for file_path in focused_files:
            for mapping in (unstaged_numstat, staged_numstat):
                additions, deletions = mapping.get(file_path, (0, 0))
                focused_additions += additions
                focused_deletions += deletions

        return {
            "workspace_changed_file_count": len(workspace_files),
            "workspace_changed_files": sorted(workspace_files)[:200],
            "focused_changed_file_count": len(focused_files),
            "focused_changed_files": focused_files[:200],
            "workspace_additions": workspace_additions,
            "workspace_deletions": workspace_deletions,
            "focused_additions": focused_additions,
            "focused_deletions": focused_deletions,
            "has_diff": bool(workspace_files),
            "focus_applied": bool(focus_set),
            "focus_paths": sorted(focus_set)[:200],
            "untracked_files": sorted(untracked_files)[:200],
        }

    def get_git_diff(
        self,
        path: str = ".",
        staged: bool = False,
        max_chars: int = 12000,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        base = self._resolve(path)
        if not base.exists() or not base.is_dir():
            return {"ok": False, "error": f"not a directory: {path}"}
        cmd_parts = ["git", "diff"]
        if staged:
            cmd_parts.append("--staged")
        normalized_paths = [
            self._normalize_workspace_path(item)
            for item in (paths or [])
            if str(item).strip()
        ]
        if normalized_paths:
            cmd_parts.append("--")
            cmd_parts.extend(normalized_paths)
        started = time.time()
        try:
            proc = subprocess.run(
                cmd_parts,
                cwd=str(base),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            duration_ms = int((time.time() - started) * 1000)
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "path": self._to_workspace_rel(base),
                "staged": bool(staged),
                "diff": "",
                "error": "git diff timed out",
                "exit_code": None,
                "paths": normalized_paths,
            }
        except Exception as e:
            return {
                "ok": False,
                "path": self._to_workspace_rel(base),
                "staged": bool(staged),
                "diff": "",
                "error": f"git diff failed: {e}",
                "exit_code": None,
                "paths": normalized_paths,
            }
        return {
            "ok": int(proc.returncode) == 0,
            "path": self._to_workspace_rel(base),
            "staged": bool(staged),
            "diff": stdout[:max_chars],
            "error": stderr[:1000],
            "exit_code": int(proc.returncode),
            "truncated": len(stdout) > max_chars,
            "duration_ms": duration_ms,
            "paths": normalized_paths,
        }

    def validate_workspace_changes(
        self,
        path: str = ".",
        test_runner: str = "auto",
        test_args: str = "",
        timeout: int = 300,
        focus_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        diff_data = self.get_git_diff(
            path=path,
            staged=False,
            max_chars=4000,
            paths=focus_paths,
        )
        change_stats = self._workspace_change_stats(path=path, focus_paths=focus_paths)
        test_data = self.run_tests(
            path=path,
            runner=test_runner,
            args=test_args,
            timeout=timeout,
        )
        diff_text = str(diff_data.get("diff", ""))
        focused_count = int(change_stats.get("focused_changed_file_count", 0) or 0)
        workspace_count = int(change_stats.get("workspace_changed_file_count", 0) or 0)
        return {
            "ok": bool(test_data.get("completed", False)),
            "path": path,
            "tests_passed": bool(test_data.get("tests_passed", False)),
            "changed_file_count": focused_count if focus_paths else workspace_count,
            "changed_files": (
                change_stats.get("focused_changed_files", [])
                if focus_paths
                else change_stats.get("workspace_changed_files", [])
            ),
            "focused_changed_file_count": focused_count,
            "focused_changed_files": change_stats.get("focused_changed_files", []),
            "workspace_changed_file_count": workspace_count,
            "workspace_changed_files": change_stats.get("workspace_changed_files", []),
            "diff_stats": {
                "changed_file_count": focused_count if focus_paths else workspace_count,
                "changed_files": (
                    change_stats.get("focused_changed_files", [])
                    if focus_paths
                    else change_stats.get("workspace_changed_files", [])
                ),
                "workspace_changed_file_count": workspace_count,
                "workspace_changed_files": change_stats.get("workspace_changed_files", []),
                "focused_changed_file_count": focused_count,
                "focused_changed_files": change_stats.get("focused_changed_files", []),
                "additions": int(
                    change_stats.get("focused_additions" if focus_paths else "workspace_additions", 0)
                    or 0
                ),
                "deletions": int(
                    change_stats.get("focused_deletions" if focus_paths else "workspace_deletions", 0)
                    or 0
                ),
                "has_diff": bool(change_stats.get("has_diff", False)),
                "focus_applied": bool(change_stats.get("focus_applied", False)),
                "untracked_files": change_stats.get("untracked_files", []),
            },
            "tests": test_data,
            "diff_excerpt": diff_text[:4000],
            "validation_signals": {
                "validation_completed": bool(test_data.get("completed", False)),
                "tests_passed": bool(test_data.get("tests_passed", False)),
                "test_exit_code": test_data.get("exit_code"),
                "failed_tests": int(test_data.get("failed", 0) or 0),
                "test_errors": int(test_data.get("errors", 0) or 0),
                "collection_errors": int(test_data.get("collection_errors", 0) or 0),
                "test_failures": test_data.get("test_failures", []),
                "failure_mode": str(test_data.get("failure_mode", "")).strip(),
                "summary_line": str(test_data.get("summary_line", "")).strip(),
                "unparsed_nonzero_exit": bool(
                    test_data.get("unparsed_nonzero_exit", False)
                ),
                "runner_failed": bool(test_data.get("runner_failed", False)),
                "has_diff": bool(change_stats.get("has_diff", False)),
                "changed_file_count": focused_count if focus_paths else workspace_count,
                "focused_changed_file_count": focused_count,
                "workspace_changed_file_count": workspace_count,
                "diff_additions": int(
                    change_stats.get("focused_additions" if focus_paths else "workspace_additions", 0)
                    or 0
                ),
                "diff_deletions": int(
                    change_stats.get("focused_deletions" if focus_paths else "workspace_deletions", 0)
                    or 0
                ),
                "focus_applied": bool(change_stats.get("focus_applied", False)),
                "focused_changed_files": change_stats.get("focused_changed_files", []),
                "workspace_changed_files": change_stats.get("workspace_changed_files", []),
                "untracked_files": change_stats.get("untracked_files", []),
            },
        }

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
            return {
                "ok": False,
                "error": f"plan not found: {plan_id}",
                "hint": "Call create_plan first, or list_plans to get a valid plan_id.",
            }
        return {"ok": True, "plan": plan}

    def add_todo(self, plan_id: str, text: str) -> dict[str, Any]:
        plan = self._load_plan(plan_id)
        if not plan:
            return {
                "ok": False,
                "error": f"plan not found: {plan_id}",
                "hint": "Call create_plan first, then add_todo using the returned plan_id.",
            }

        todos = plan.get("todos", [])
        if not isinstance(todos, list):
            todos = []
        next_id = (
            max([int(t.get("id", 0)) for t in todos if isinstance(t, dict)] + [0]) + 1
        )
        now = utc_now_iso()
        todo = {
            "id": next_id,
            "text": text.strip(),
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        }
        todos.append(todo)
        plan["todos"] = todos
        plan["updated_at"] = now
        write_json(self._plan_path(plan_id), plan)
        return {"ok": True, "plan_id": plan_id, "todo": todo}

    def update_todo(self, plan_id: str, todo_id: int, status: str) -> dict[str, Any]:
        plan = self._load_plan(plan_id)
        if not plan:
            return {
                "ok": False,
                "error": f"plan not found: {plan_id}",
                "hint": "Call create_plan first, then update_todo using the returned plan_id.",
            }

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
