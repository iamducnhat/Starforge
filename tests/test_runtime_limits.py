import shutil
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import assistant.model as model_module
from assistant.chat_engine import ChatEngine
from assistant.functions_registry import FunctionRegistry
from assistant.memory import MemoryStore


class _DummyModel:
    def generate(self, messages):
        return "{}"

    def stream_generate(self, messages):
        yield "{}"


class _CleanupCounter:
    def __init__(self, value=0):
        self.calls = 0
        self.value = value

    def evict_cold_state(self):
        self.calls += 1
        return self.value

    def close_idle_terminals(self, max_idle_s=None):
        self.calls += 1
        return self.value


class _GuardTools:
    def __init__(self):
        self.memory_store = _CleanupCounter({"knowledge_evicted": 1})
        self.function_registry = _CleanupCounter(2)
        self.workspace_tools = _CleanupCounter(3)


class TestRuntimeLimits(unittest.TestCase):
    def setUp(self):
        self.root = Path("test_runtime_limits")
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        model_module._dns_cache.clear()
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_strategy_cache_is_trimmed(self):
        store = MemoryStore(blocks_dir=self.root / "memory_blocks")
        store.max_hot_strategies = 2
        for idx in range(3):
            saved = store.record_strategy(
                goal=f"goal-{idx}",
                strategy=[{"step_id": 1, "action": f"step-{idx}", "args": {}, "depends_on": []}],
                success=True,
            )
            self.assertTrue(saved["ok"])
        self.assertLessEqual(len(store._strategy_cache), 2)

    def test_root_cause_feedback_can_reload_trimmed_entries(self):
        store = MemoryStore(blocks_dir=self.root / "memory_blocks")
        store.max_hot_root_causes_per_bucket = 2
        created_ids = []
        for idx, confidence in enumerate((0.1, 0.2, 0.3), start=1):
            saved = store.upsert_root_cause(
                pattern=f"ModuleNotFoundError: package_{idx}",
                context={"language": "python"},
                fix_template=[{"tool": "execute_command", "args": {"cmd": f"pip install package_{idx}"}}],
                success=True,
                confidence=confidence,
                source="test",
                bucket_hint="import_errors",
            )
            self.assertTrue(saved["ok"])
            created_ids.append(str(saved["entry"]["id"]))
        self.assertLessEqual(
            len(store._root_cause_cache.get("import_errors.json", [])),
            2,
        )
        result = store.record_root_cause_feedback(created_ids[0], success=False, confidence=0.0)
        self.assertTrue(result["ok"])

    def test_skill_cache_is_trimmed(self):
        registry = FunctionRegistry(self.root / "functions")
        registry.max_hot_skills = 2
        for idx in range(3):
            created = registry.create_skill(
                name=f"skill-{idx}",
                description="demo",
                keywords=[f"k{idx}"],
                tool_name="get_current_datetime",
            )
            self.assertTrue(created["ok"])
        self.assertLessEqual(len(registry._skills_cache), 2)

    def test_dns_cache_is_bounded(self):
        original_limit = model_module._dns_cache_max_entries
        try:
            model_module._dns_cache.clear()
            model_module._dns_cache_max_entries = 2
            model_module._remember_dns("one.example", "1.1.1.1")
            model_module._remember_dns("two.example", "2.2.2.2")
            model_module._remember_dns("three.example", "3.3.3.3")
            self.assertEqual(len(model_module._dns_cache), 2)
            self.assertNotIn("one.example", model_module._dns_cache)
        finally:
            model_module._dns_cache_max_entries = original_limit

    def test_memory_guard_runs_cleanup(self):
        engine = ChatEngine(model=_DummyModel(), tools=_GuardTools(), system_prompt="test")
        engine.memory_soft_limit_mb = 1
        engine.memory_hard_limit_mb = 10
        readings = iter([2 * 1024 * 1024, 512 * 1024])
        engine._memory_usage_bytes = lambda: next(readings)
        with patch("assistant.chat_engine.clear_dns_cache", return_value=4) as cleared:
            engine._maybe_enforce_memory_limits(force=True, context="test")
        self.assertEqual(engine.tools.workspace_tools.calls, 1)
        self.assertEqual(engine.tools.memory_store.calls, 1)
        self.assertEqual(engine.tools.function_registry.calls, 1)
        cleared.assert_called_once()

    def test_memory_guard_raises_at_hard_limit(self):
        engine = ChatEngine(model=_DummyModel(), tools=_GuardTools(), system_prompt="test")
        engine.memory_soft_limit_mb = 1
        engine.memory_hard_limit_mb = 1
        engine._memory_usage_bytes = lambda: 2 * 1024 * 1024
        with self.assertRaises(MemoryError):
            engine._maybe_enforce_memory_limits(force=True, context="hard_limit")

    def test_learned_signature_cache_is_bounded(self):
        engine = ChatEngine(model=_DummyModel(), tools=_GuardTools(), system_prompt="test")
        engine.max_learned_signatures = 2
        learned = OrderedDict()
        self.assertTrue(engine._remember_learned_signature(learned, "sig-1"))
        self.assertTrue(engine._remember_learned_signature(learned, "sig-2"))
        self.assertTrue(engine._remember_learned_signature(learned, "sig-3"))
        self.assertEqual(list(learned.keys()), ["sig-2", "sig-3"])
        self.assertFalse(engine._remember_learned_signature(learned, "sig-3"))


if __name__ == "__main__":
    unittest.main()
