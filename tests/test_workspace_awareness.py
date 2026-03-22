import shutil
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


if __name__ == "__main__":
    unittest.main()
