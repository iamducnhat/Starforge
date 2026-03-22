import shutil
import unittest
from pathlib import Path

from assistant.memory import MemoryStore


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.root = Path("test_memory_workspace")
        self.test_dir = self.root / "memory_blocks"
        self.store = MemoryStore(blocks_dir=self.test_dir)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_create_and_find_block(self):
        self.store.create_block(
            name="Test Block",
            topic="Testing",
            keywords=["unittest", "python"],
            knowledge="This is a test knowledge block.",
            source=["http://example.com"],
        )

        # Search by keyword
        results = self.store.find_in_memory(["unittest"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Test Block")
        self.assertEqual(results[0]["knowledge"], "This is a test knowledge block.")

    def test_find_no_match(self):
        self.store.create_block(
            name="Test Block",
            topic="Testing",
            keywords=["unittest"],
            knowledge="...",
            source=[],
        )
        results = self.store.find_in_memory(["nonexistent"])
        self.assertEqual(len(results), 0)

    def test_multiple_keywords_scoring(self):
        self.store.create_block(
            name="Block A",
            topic="T1",
            keywords=["apple", "banana"],
            knowledge="...",
            source=[],
        )
        self.store.create_block(
            name="Block B",
            topic="T2",
            keywords=["apple", "cherry"],
            knowledge="...",
            source=[],
        )

        results = self.store.find_in_memory(["apple", "banana"])
        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[0]["name"], "Block A"
        )  # Higher score due to more keyword matches

    def test_semantic_search(self):
        self.store.create_block(
            name="Reliability Notes",
            topic="Tooling",
            keywords=["network", "stability"],
            knowledge="Use retry and fallback strategy when tool calls fail with timeout or rate limit.",
            source=[],
        )
        results = self.store.semantic_search("retry fallback for timeout failures", limit=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Reliability Notes")
        self.assertEqual(results[0]["match_type"], "semantic")

    def test_find_in_memory_semantic_fallback(self):
        self.store.create_block(
            name="Execution Strategy",
            topic="Agent",
            keywords=["planner"],
            knowledge="Task reflection improves success by reviewing each tool result and retrying when needed.",
            source=[],
        )
        # Query keyword is not in block keywords, but is semantically present in knowledge.
        results = self.store.find_in_memory(["reflection"])
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Execution Strategy")

    def test_memory_feedback_stats(self):
        self.store.create_block(
            name="Bug Fix Pattern",
            topic="Python",
            keywords=["import", "error"],
            knowledge="Fix import path by checking package __init__ and PYTHONPATH.",
            source=[],
        )
        fb = self.store.record_feedback(
            block_name="bug_fix_pattern",
            success=True,
            confidence=0.9,
            source="test",
        )
        self.assertTrue(fb["ok"])
        results = self.store.find_in_memory(["import"])
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("block", results[0])

    def test_record_and_find_strategy(self):
        saved = self.store.record_strategy(
            goal="fix failing tests",
            strategy=[
                {
                    "step_id": 1,
                    "action": "run_tests",
                    "args": {"path": ".", "runner": "pytest"},
                    "depends_on": [],
                    "expected_output": "failing tests identified",
                },
                {
                    "step_id": 2,
                    "action": "edit_file",
                    "args": {"path": "app.py"},
                    "depends_on": [1],
                    "expected_output": "targeted patch applied",
                },
            ],
            success=True,
            notes="Validated with pytest",
            context={"framework": "pytest"},
        )
        self.assertTrue(saved["ok"])

        matches = self.store.find_strategies("fix tests", limit=3)
        self.assertGreaterEqual(len(matches), 1)
        self.assertEqual(matches[0]["goal"], "fix failing tests")
        self.assertGreater(matches[0]["score"], 0.0)
        self.assertEqual(matches[0]["strategy"][1]["depends_on"], [1])
        self.assertEqual(matches[0]["context"].get("framework"), "pytest")

    def test_find_strategies_success_rate_is_per_item(self):
        self.store.record_strategy(
            goal="stable flow",
            strategy=[{"step_id": 1, "action": "run_tests", "args": {}, "depends_on": []}],
            success=True,
        )
        self.store.record_strategy(
            goal="risky flow",
            strategy=[{"step_id": 1, "action": "execute_command", "args": {"cmd": "npm test"}, "depends_on": []}],
            success=False,
        )
        matches = self.store.find_strategies("flow", limit=5)
        rates = {str(item.get("goal", "")): float(item.get("success_rate", -1.0)) for item in matches}
        self.assertIn("stable flow", rates)
        self.assertIn("risky flow", rates)
        self.assertEqual(rates["stable flow"], 1.0)
        self.assertEqual(rates["risky flow"], 0.0)

    def test_find_root_cause_interpolates_fix_template(self):
        root_file = self.store.root_causes_dir / "import_errors.json"
        root_file.write_text(
            (
                "[\n"
                "  {\n"
                '    "pattern": "ModuleNotFoundError: ${module}",\n'
                '    "context": {"language": "python"},\n'
                '    "fix_template": [{"tool": "run_terminal", "args": {"cmd": "pip install ${module}"}}],\n'
                '    "success_count": 2,\n'
                '    "fail_count": 0,\n'
                '    "confidence": 0.9\n'
                "  }\n"
                "]\n"
            ),
            encoding="utf-8",
        )
        self.store._load_root_causes()
        matches = self.store.find_root_causes(
            "ModuleNotFoundError: requests",
            context={"language": "python"},
            limit=1,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(
            matches[0]["fix_template"][0]["args"]["cmd"],
            "pip install requests",
        )

    def test_record_root_cause_feedback_updates_counts(self):
        root_file = self.store.root_causes_dir / "test_failures.json"
        root_file.write_text(
            (
                "[\n"
                '  {"id":"rc_1","pattern":"AssertionError","context":{},"fix_template":[],"success_count":0,"fail_count":0,"confidence":0.5}\n'
                "]\n"
            ),
            encoding="utf-8",
        )
        self.store._load_root_causes()
        result = self.store.record_root_cause_feedback("rc_1", success=True, confidence=1.0)
        self.assertTrue(result["ok"])
        matches = self.store.find_root_causes("AssertionError: bad", context={}, limit=1)
        self.assertEqual(matches[0]["success_count"], 1)

    def test_upsert_root_cause_creates_and_merges_entries(self):
        created = self.store.upsert_root_cause(
            pattern="ModuleNotFoundError: ${module}",
            context={"language": "python"},
            fix_template=[{"tool": "execute_command", "args": {"cmd": "pip install ${module}"}}],
            success=True,
            confidence=0.8,
            source="test",
        )
        self.assertTrue(created["ok"])
        self.assertTrue(created["created"])
        self.assertEqual(created["bucket"], "import_errors.json")

        merged = self.store.upsert_root_cause(
            pattern="ModuleNotFoundError: ${module}",
            context={"language": "python"},
            fix_template=[{"tool": "execute_command", "args": {"cmd": "pip install ${module}"}}],
            success=False,
            confidence=0.2,
            source="test",
        )
        self.assertTrue(merged["ok"])
        self.assertFalse(merged["created"])
        self.assertEqual(int(merged["entry"]["success_count"]), 1)
        self.assertEqual(int(merged["entry"]["fail_count"]), 1)

        import_path = self.store.root_causes_dir / "import_errors.json"
        payload = import_path.read_text(encoding="utf-8")
        self.assertEqual(payload.count("ModuleNotFoundError: ${module}"), 1)

    def test_upsert_root_cause_bucket_classifier(self):
        self.store.upsert_root_cause(
            pattern="missing dependency for package",
            context={"language": "python"},
            fix_template=[{"tool": "execute_command", "args": {"cmd": "pip install requests"}}],
            success=True,
            confidence=0.7,
            source="test",
        )
        self.store.upsert_root_cause(
            pattern="AssertionError",
            context={"framework": "pytest"},
            fix_template=[{"tool": "run_tests", "args": {"path": ".", "runner": "pytest"}}],
            success=True,
            confidence=0.7,
            source="test",
        )
        imports = (self.store.root_causes_dir / "import_errors.json").read_text(encoding="utf-8")
        tests = (self.store.root_causes_dir / "test_failures.json").read_text(encoding="utf-8")
        self.assertIn("missing dependency for package", imports)
        self.assertIn("AssertionError", tests)


if __name__ == "__main__":
    unittest.main()
