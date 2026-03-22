import shutil
import subprocess
import time
import unittest
from pathlib import Path

from assistant.workspace_tools import WorkspaceTools


class TestWorkspaceAwareness(unittest.TestCase):
    def setUp(self):
        self.root = Path("test_workspace_awareness")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "main.py").write_text(
            "from fastapi import FastAPI\napp = FastAPI()\n",
            encoding="utf-8",
        )
        (self.root / "requirements.txt").write_text("fastapi\npytest\n", encoding="utf-8")
        self.tools = WorkspaceTools(self.root)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def _init_git_repo(self):
        commands = [
            ["git", "init"],
            ["git", "config", "user.email", "tests@example.com"],
            ["git", "config", "user.name", "Tests"],
            ["git", "add", "main.py", "requirements.txt"],
            ["git", "commit", "-m", "init"],
        ]
        for cmd in commands:
            subprocess.run(
                cmd,
                cwd=self.root,
                check=True,
                capture_output=True,
                text=True,
            )

    def test_detect_project_context(self):
        ctx = self.tools.detect_project_context(path=".")
        self.assertTrue(ctx["ok"])
        self.assertEqual(ctx["framework"], "FastAPI")
        self.assertIn("main.py", ctx["entry_points"][0])
        self.assertEqual(ctx["test_runner"], "pytest")

    def test_execute_command_structured(self):
        result = self.tools.execute_command("echo hello", path=".")
        self.assertTrue(result["ok"])
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("hello", result["stdout"])

    def test_edit_file_fallback_normalized_line_match(self):
        (self.root / "calc.py").write_text(
            "def calc(a, b):\n    return a \u2212 b\n",
            encoding="utf-8",
        )
        result = self.tools.edit_file(
            path="calc.py",
            find_text="return a - b",
            replace_text="return a + b",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result.get("match_mode"), "normalized_line")
        updated = (self.root / "calc.py").read_text(encoding="utf-8")
        self.assertIn("return a + b", updated)

    def test_edit_file_trailing_newline_in_find_text(self):
        (self.root / "ops.py").write_text(
            "def add(a, b):\n    return a - b  # WRONG (intentional demo bug)\n",
            encoding="utf-8",
        )
        result = self.tools.edit_file(
            path="ops.py",
            find_text="return a - b  # WRONG (intentional demo bug)\n",
            replace_text="return a + b",
        )
        self.assertTrue(result["ok"])
        updated = (self.root / "ops.py").read_text(encoding="utf-8")
        self.assertIn("    return a + b", updated)

    def test_update_todo_missing_plan_returns_hint(self):
        result = self.tools.update_todo(plan_id="1", todo_id=1, status="done")
        self.assertFalse(result["ok"])
        self.assertIn("plan not found", result.get("error", ""))
        self.assertIn("create_plan", result.get("hint", ""))

    def test_run_terminal_bounds_buffered_output(self):
        self.tools._terminal_buffer_lines = 5
        started = self.tools.run_terminal(action="start", session_id="burst")
        self.assertTrue(started["ok"])
        try:
            sent = self.tools.run_terminal(
                action="send",
                session_id="burst",
                cmd="for i in $(seq 1 40); do echo line-$i; done",
            )
            self.assertTrue(sent["ok"])
            time.sleep(0.2)
            read = self.tools.run_terminal(action="read", session_id="burst")
            self.assertTrue(read["ok"])
            self.assertTrue(read["truncated"])
            self.assertGreater(read["dropped_lines"], 0)
            self.assertIn("line-40", read["output"])
        finally:
            self.tools.run_terminal(action="close", session_id="burst")

    def test_parse_test_output_parses_error_collecting(self):
        parsed = self.tools._parse_test_output(
            """
============================= test session starts ==============================
ERROR collecting tests/test_app.py
ImportError while importing test module '/tmp/tests/test_app.py'.
collected 0 items / 1 error
=========================== short test summary info ============================
ERROR tests/test_app.py
""",
            exit_code=2,
        )
        self.assertEqual(parsed["collection_errors"], 1)
        self.assertGreaterEqual(parsed["errors"], 1)
        self.assertEqual(parsed["test_failures"][0]["kind"], "collection_error")

    def test_parse_test_output_marks_unparsed_nonzero_exit(self):
        parsed = self.tools._parse_test_output(
            "pytest crashed before printing a summary",
            exit_code=5,
        )
        self.assertTrue(parsed["unparsed_nonzero_exit"])
        self.assertEqual(parsed["errors"], 1)
        self.assertIn("non-zero", parsed["test_failures"][0]["summary"])

    def test_validate_workspace_changes_includes_untracked_files(self):
        self._init_git_repo()
        (self.root / "scratch.py").write_text("print('draft')\n", encoding="utf-8")
        result = self.tools.validate_workspace_changes(path=".", focus_paths=["scratch.py"])
        self.assertTrue(result["ok"])
        self.assertIn("scratch.py", result["workspace_changed_files"])
        self.assertIn("scratch.py", result["focused_changed_files"])
        self.assertIn("scratch.py", result["validation_signals"]["untracked_files"])

    def test_validate_workspace_changes_distinguishes_validation_success_from_test_failure(self):
        self._init_git_repo()
        (self.root / "test_sample.py").write_text(
            "def test_fail():\n    assert 1 == 2\n",
            encoding="utf-8",
        )
        result = self.tools.validate_workspace_changes(path=".")
        self.assertTrue(result["ok"])
        self.assertFalse(result["tests_passed"])
        self.assertTrue(result["validation_signals"]["validation_completed"])
        self.assertEqual(result["validation_signals"]["failure_mode"], "assertion_failures")

    def test_validate_workspace_changes_can_filter_to_focus_paths(self):
        self._init_git_repo()
        (self.root / "main.py").write_text(
            "from fastapi import FastAPI\napp = FastAPI(title='changed')\n",
            encoding="utf-8",
        )
        (self.root / "requirements.txt").write_text(
            "fastapi\npytest\nuvicorn\n",
            encoding="utf-8",
        )
        result = self.tools.validate_workspace_changes(
            path=".",
            focus_paths=["main.py"],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["focused_changed_files"], ["main.py"])
        self.assertIn("requirements.txt", result["workspace_changed_files"])


if __name__ == "__main__":
    unittest.main()
