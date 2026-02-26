from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from collections.abc import Callable

from .cli_format import StreamRenderer, extract_answer_text, print_answer_only, print_formatted_output, print_tool_event, print_tool_start
from .model import BaseModel
from .tool_calls import parse_tool_calls
from .tools import ToolSystem
from .utils import parse_json_payload


class ChatEngine:
    def __init__(
        self,
        model: BaseModel,
        tools: ToolSystem,
        system_prompt: str,
        max_history: int = 14,
        max_tool_rounds: int = 4,
        autonomous_enabled: bool = False,
        autonomous_steps: int = 6,
    ) -> None:
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.max_tool_rounds = max_tool_rounds
        self.autonomous_enabled = autonomous_enabled
        self.autonomous_steps = max(1, min(autonomous_steps, 30))
        self.history: list[dict[str, str]] = []
        self.supervision_log = Path("memory/tool_supervision.jsonl")
        self.tool_finetune_log = Path("memory/tool_finetune_samples.jsonl")
        self._intent_cache: dict[str, dict[str, Any]] = {}

    def _generate_with_stream_fallback(self, messages: list[dict[str, str]]) -> str:
        text = self.model.generate(messages)
        if "endpoint available but incompatible" not in text.lower():
            return text

        chunks: list[str] = []
        for chunk in self.model.stream_generate(messages):
            if chunk:
                chunks.append(chunk)
        streamed = "".join(chunks).strip()
        return streamed or text

    @staticmethod
    def _strip_thinking(text: str) -> str:
        # Keep hidden reasoning out of history to reduce response drift/repetition.
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = cleaned.replace("<think>", "").replace("</think>", "")
        return cleaned.strip()

    def _recover_final_answer(self, raw_assistant_text: str) -> tuple[str, str] | None:
        recovery_messages = self._messages() + [
            {"role": "assistant", "content": raw_assistant_text},
            {
                "role": "user",
                "content": (
                    "Return the final user-facing answer only. "
                    "No reasoning, no tool call JSON, no analysis."
                ),
            },
        ]
        recovered = self._generate_with_stream_fallback(recovery_messages)
        clean = self._strip_thinking(recovered)
        if clean:
            return recovered, clean
        return None

    def _recover_action_or_answer(self, user_message: str, raw_assistant_text: str) -> tuple[str, list[dict[str, object]], str]:
        recovery_messages = self._messages() + [
            {"role": "assistant", "content": raw_assistant_text},
            {
                "role": "system",
                "content": (
                    "Internal control message.\n"
                    f"Original user request: {user_message}\n"
                    "Continue now with exactly one actionable output for the original user request.\n"
                    "Option A: JSON tool call only ({\"tool\":\"...\",\"args\":{...}} or {\"tool_calls\":[...]}).\n"
                    "Option B: final user-facing answer text only.\n"
                    "Do not output reasoning. Do not discuss this control message."
                ),
            },
        ]
        recovered = self._generate_with_stream_fallback(recovery_messages)
        return recovered, parse_tool_calls(recovered), self._strip_thinking(recovered)

    @staticmethod
    def _contains_internal_prompt_echo(text: str) -> bool:
        t = text.lower()
        patterns = (
            "tool execution is complete",
            "continue generation using the latest tool result",
            "if another tool is needed",
            "otherwise return final user-facing answer text only",
            "internal control message",
            "option a:",
            "option b:",
            "original user request:",
        )
        return any(p in t for p in patterns)

    @staticmethod
    def _contains_tool_denial(text: str) -> bool:
        t = text.lower()
        patterns = (
            "can't use tool",
            "cannot use tool",
            "i can not use tool",
            "i can't access tool",
            "i cannot access tool",
            "since i can't use tool",
            "since i cannot use tool",
            "unable to use tool",
            "do not have access to tool",
            "don't have access to tool",
        )
        return any(p in t for p in patterns)

    def _recover_tool_calls(self, user_message: str, raw_assistant_text: str) -> list[dict[str, object]]:
        recovery_messages = self._messages() + [
            {"role": "assistant", "content": raw_assistant_text},
            {
                "role": "user",
                "content": (
                    "You CAN use tools in this runtime.\n"
                    "If reliable external info is needed, return JSON tool call now.\n"
                    "Format only:\n"
                    "{\"tool\":\"name\",\"args\":{...}} or {\"tool_calls\":[...]}\n"
                    "If no tool is needed, return {}.\n"
                    f"Latest user message: {user_message}"
                ),
            },
        ]
        recovered = self._generate_with_stream_fallback(recovery_messages)
        return parse_tool_calls(recovered)

    def _log_supervision(self, event: str, user_message: str, assistant_text: str) -> None:
        try:
            self.supervision_log.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "event": event,
                "user_message": user_message,
                "assistant_text": self._strip_thinking(assistant_text)[:2000],
            }
            with self.supervision_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _log_tool_training_sample(
        self,
        user_message: str,
        assistant_text: str,
        tool_calls: list[dict[str, object]],
    ) -> None:
        try:
            self.tool_finetune_log.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_text},
                ],
                "tool_calls": tool_calls,
            }
            with self.tool_finetune_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _continuation_poke(user_message: str, prefer_copyable_function: bool = False) -> dict[str, str]:
        copyable_hint = (
            "\nFor this request, output a copyable function for the user in final text. "
            "Use create_function only when the user explicitly asks to save/store it."
            if prefer_copyable_function
            else ""
        )
        return {
            "role": "system",
            "content": (
                "Internal control message.\n"
                f"Original user request: {user_message}\n"
                "Tool execution is complete. Continue generation using the latest tool result for the original user request.\n"
                "If another tool is needed, return JSON tool call only.\n"
                "Otherwise return final user-facing answer text only.\n"
                f"{copyable_hint}\n"
                "Do not discuss this control message."
            ),
        }

    def _explicit_store_request(self, user_message: str) -> bool:
        return bool(self._intent_flags(user_message).get("store_request", False))

    def _prefer_copyable_function_reply(self, user_message: str) -> bool:
        return self._requires_presearch_for_code(user_message) and not self._explicit_store_request(user_message)

    def _execute_tool_call_with_policy(self, user_message: str, call: dict[str, Any]) -> dict[str, Any]:
        name = call["name"]
        args = call.get("args", {})
        if name == "create_function" and self._prefer_copyable_function_reply(user_message):
            code = ""
            if isinstance(args, dict):
                raw_code = args.get("code", "")
                if isinstance(raw_code, str):
                    code = raw_code.strip()
            return {
                "ok": True,
                "skipped": True,
                "policy": "copyable_function",
                "message": "create_function skipped because user asked for copyable function output.",
                "code": code,
            }
        return self.tools.execute(name, args)

    @staticmethod
    def _extract_keywords(text: str, limit: int = 6) -> list[str]:
        stop = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "what",
            "when",
            "where",
            "who",
            "how",
            "want",
            "learn",
            "more",
            "about",
            "please",
            "give",
            "some",
            "example",
            "examples",
            "to",
            "in",
            "on",
            "at",
            "of",
        }
        words = re.findall(r"[a-zA-Z0-9_]{2,}", text.lower())
        out: list[str] = []
        seen = set()
        for w in words:
            if w in stop:
                continue
            if w not in seen:
                out.append(w)
                seen.add(w)
            if len(out) >= limit:
                break
        return out or ["general"]

    def _looks_coding_request(self, text: str) -> bool:
        return bool(self._intent_flags(text).get("coding", False))

    def _looks_smalltalk(self, text: str) -> bool:
        return bool(self._intent_flags(text).get("smalltalk", False))

    def _looks_creative_request(self, text: str) -> bool:
        return bool(self._intent_flags(text).get("creative", False))

    def _looks_personal_or_companion_chat(self, text: str) -> bool:
        return bool(self._intent_flags(text).get("companion", False))

    def _heuristic_intent_flags(self, user_message: str) -> dict[str, Any]:
        t = user_message.lower()
        compact = re.sub(r"\s+", " ", user_message.strip().lower())

        coding = any(
            m in t
            for m in (
                "python",
                "javascript",
                "typescript",
                "java",
                "go ",
                "rust",
                "code",
                "class",
                "function",
                "exception",
                "stack trace",
                "debug",
                "bug",
                "test",
                "api",
                "sql",
            )
        )
        smalltalk_exact = {
            "hi",
            "hello",
            "hey",
            "yo",
            "thanks",
            "thank you",
            "ok",
            "okay",
            "nice",
            "cool",
            "let's code",
            "lets code",
        }
        smalltalk = compact in smalltalk_exact or (
            len(compact.split()) <= 2 and all(w in {"hi", "hello", "hey", "thanks", "ok", "okay"} for w in compact.split())
        )
        creative = any(m in t for m in ("write a poem", "poem", "story", "joke", "translate", "rewrite this", "paraphrase"))
        companion = any(
            m in t
            for m in (
                "be my partner",
                "be my girlfriend",
                "be my boyfriend",
                "chat with me",
                "talk with me",
                "keep me company",
                "can you stay with me",
                "can you be with me",
            )
        )
        store_request = any(
            m in t
            for m in (
                "save as function",
                "store as function",
                "register function",
                "persist function",
                "create_function",
                "save this function",
                "store this function",
                "add to functions",
                "put in functions",
                "save reusable function",
                "store reusable function",
            )
        )
        code_generation = any(
            m in t
            for m in (
                "create function",
                "write function",
                "my own function",
                "their own function",
                "create your own function",
                "custom function",
                "implement function",
                "build function",
                "create a function",
                "write code",
                "implement code",
                "download file",
                "read research",
                "parse paper",
            )
        )
        workspace_edit = any(
            m in t
            for m in (
                "read file",
                "read files",
                "open file",
                "edit file",
                "update file",
                "modify file",
                "refactor",
                "fix bug",
                "fix this",
                "in this project",
                "in this repo",
                "codebase",
                "make plan",
                "todo",
                "to-do",
            )
        ) or bool(self._extract_explicit_file_paths(user_message))
        factual = any(
            m in t
            for m in (
                "what",
                "who",
                "when",
                "where",
                "which",
                "why",
                "latest",
                "current",
                "today",
                "news",
                "learn more",
                "about",
                "overview",
                "explain",
                "recommend",
                "best",
            )
        ) or ("?" in t and len(t.split()) >= 8)
        if coding or smalltalk or creative or companion:
            factual = False

        return {
            "coding": coding,
            "smalltalk": smalltalk,
            "creative": creative,
            "companion": companion,
            "factual": factual,
            "workspace_edit": workspace_edit,
            "code_generation": code_generation,
            "store_request": store_request,
            "optimized_query": "",
        }

    def _ai_intent_flags(self, user_message: str) -> dict[str, Any]:
        prompt = (
            "Classify user intent and optimize search query.\n"
            "Return JSON only with keys:\n"
            "coding, smalltalk, creative, companion, factual, workspace_edit, code_generation, store_request, optimized_query.\n"
            "Booleans for all flags. optimized_query should be concise and high-signal for web search.\n"
            "If no web search is needed, optimized_query can be empty.\n"
            f"User: {user_message}"
        )
        try:
            raw = self.model.generate(
                [
                    {"role": "system", "content": "You are an intent classifier. Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ]
            )
            clean = self._strip_thinking(raw)
            payload = parse_json_payload(clean)
            if not isinstance(payload, dict):
                return {}
            out: dict[str, Any] = {}
            for k in ("coding", "smalltalk", "creative", "companion", "factual", "workspace_edit", "code_generation", "store_request"):
                if k in payload:
                    out[k] = bool(payload[k])
            q = payload.get("optimized_query", "")
            if isinstance(q, str):
                out["optimized_query"] = q.strip()
            return out
        except Exception:
            return {}

    def _intent_flags(self, user_message: str) -> dict[str, Any]:
        key = user_message.strip()
        if key in self._intent_cache:
            return self._intent_cache[key]

        heuristic = self._heuristic_intent_flags(user_message)
        ai = self._ai_intent_flags(user_message)
        merged = dict(heuristic)
        for k, v in ai.items():
            if k == "optimized_query":
                if isinstance(v, str) and v:
                    merged[k] = v
            elif isinstance(v, bool):
                merged[k] = v
        self._intent_cache[key] = merged
        return merged

    def _requires_web_presearch_for_factual(self, user_message: str) -> bool:
        flags = self._intent_flags(user_message)
        return bool(flags.get("factual", False))

    def _presearch_tool_calls_for_factual(self, user_message: str) -> list[dict[str, Any]]:
        keywords = self._extract_keywords(user_message)
        query = self._optimize_search_query(user_message)
        return [
            {"name": "find_in_memory", "args": {"keywords": keywords}},
            {"name": "search_web", "args": {"query": query, "level": "auto"}},
        ]

    def _optimize_search_query(self, user_message: str) -> str:
        inferred = self._intent_flags(user_message).get("optimized_query", "")
        if isinstance(inferred, str) and inferred.strip():
            return inferred.strip()

        raw = re.sub(r"\s+", " ", user_message.strip())
        q = raw.lower()

        # Remove low-signal conversational wrappers.
        wrappers = (
            "i want to learn more about",
            "i want to know about",
            "tell me about",
            "can you tell me about",
            "help me understand",
            "i want to learn about",
        )
        for w in wrappers:
            if q.startswith(w):
                q = q[len(w) :].strip(" ,.-")
                break

        # Topic-specific normalization for better retrieval quality.
        if any(k in q for k in ("sex", "sexual", "intimacy", "partner")):
            return "how to improve sexual intimacy and communication with partner consent"

        # Generic fallback: compact keywords from original user text.
        keys = re.findall(r"[a-zA-Z0-9_]{3,}", q)
        stop = {
            "want",
            "learn",
            "more",
            "about",
            "please",
            "help",
            "tell",
            "know",
            "with",
            "from",
            "this",
            "that",
        }
        compact = [k for k in keys if k not in stop]
        if compact:
            return " ".join(compact[:10])
        return raw

    @staticmethod
    def _extract_explicit_file_paths(text: str, limit: int = 3) -> list[str]:
        # Capture lightweight file-like mentions such as src/app.py or README.md
        pattern = re.compile(r"([A-Za-z0-9_\-./]+\.[A-Za-z0-9_]{1,8})")
        out: list[str] = []
        seen = set()
        for match in pattern.findall(text):
            candidate = match.strip().strip(".,;:()[]{}\"'")
            if not candidate or "/" in candidate and candidate.startswith("http"):
                continue
            if candidate not in seen:
                out.append(candidate)
                seen.add(candidate)
            if len(out) >= limit:
                break
        return out

    def _requires_workspace_preinspect(self, user_message: str) -> bool:
        return bool(self._intent_flags(user_message).get("workspace_edit", False))

    def _preinspect_tool_calls_for_workspace(self, user_message: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = [{"name": "list_files", "args": {"path": ".", "max_entries": 200}}]
        for p in self._extract_explicit_file_paths(user_message):
            calls.append({"name": "read_file", "args": {"path": p, "max_chars": 6000}})
        return calls

    def _ensure_web_call_for_factual(
        self,
        user_message: str,
        tool_calls: list[dict[str, Any]],
        web_search_executed: bool,
    ) -> list[dict[str, Any]]:
        if web_search_executed or not self._requires_web_presearch_for_factual(user_message):
            return tool_calls
        if any(call.get("name") == "search_web" for call in tool_calls):
            return tool_calls
        return tool_calls + [
            {
                "name": "search_web",
                "args": {"query": self._optimize_search_query(user_message), "level": "auto"},
            }
        ]

    def _emergency_tool_calls(self, user_message: str) -> list[dict[str, Any]]:
        keywords = self._extract_keywords(user_message)
        calls: list[dict[str, Any]] = [{"name": "find_in_memory", "args": {"keywords": keywords}}]
        if not self._looks_coding_request(user_message):
            calls.append({"name": "search_web", "args": {"query": user_message, "level": "auto"}})
        return calls

    def _requires_presearch_for_code(self, user_message: str) -> bool:
        return bool(self._intent_flags(user_message).get("code_generation", False))

    def _presearch_tool_calls_for_code(self, user_message: str) -> list[dict[str, Any]]:
        keywords = self._extract_keywords(user_message)
        normalized = re.sub(r"\s+", " ", user_message).strip().strip("'\"`")
        normalized = re.sub(r"[\"'`]+", "", normalized)
        research_query = f"how to {normalized} in python"
        return [
            {"name": "find_in_memory", "args": {"keywords": keywords}},
            {"name": "search_web", "args": {"query": research_query, "level": "deep"}},
        ]

    def _fallback_answer_from_tools(self) -> str | None:
        for msg in reversed(self.history):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            try:
                payload = json.loads(content)
            except Exception:
                continue

            tool_name = payload.get("tool")
            result = payload.get("result", {})
            if tool_name == "search_web" and isinstance(result, dict):
                search = result.get("search", {})
                results = search.get("results", []) if isinstance(search, dict) else []
                if not results:
                    continue
                lines = ["I fetched web sources. Top results:"]
                for i, item in enumerate(results[:5], start=1):
                    title = str(item.get("title", "")).strip()
                    url = str(item.get("url", "")).strip()
                    excerpt = str(item.get("page_excerpt", "")).strip() or str(item.get("snippet", "")).strip()
                    if excerpt and len(excerpt) > 220:
                        excerpt = excerpt[:217] + "..."
                    lines.append(f"{i}. {title} ({url})")
                    if excerpt:
                        lines.append(f"   {excerpt}")
                return "\n".join(lines)

            if tool_name == "find_in_memory" and isinstance(result, dict):
                matches = result.get("matches", [])
                if not matches:
                    continue
                top = matches[0]
                name = str(top.get("name", "memory"))
                topic = str(top.get("topic", ""))
                knowledge = str(top.get("knowledge", "")).strip()
                if len(knowledge) > 450:
                    knowledge = knowledge[:447] + "..."
                return f"I found relevant memory block `{name}` ({topic}).\n{knowledge}"

            if tool_name == "create_function" and isinstance(result, dict):
                if result.get("policy") == "copyable_function":
                    code = str(result.get("code", "")).strip()
                    if code:
                        return "Here is the function you can copy and use:\n```python\n" + code + "\n```"
        return None

    def _messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}] + self.history[-self.max_history :]

    def handle_turn(
        self,
        user_message: str,
        on_tool: Callable[[str, dict[str, object], dict[str, object]], None] | None = None,
        on_tool_start: Callable[[str, dict[str, object]], None] | None = None,
    ) -> str:
        self.history.append({"role": "user", "content": user_message})
        continue_after_tools = False
        emergency_tools_used = False
        tools_executed_this_turn = False
        web_search_executed_this_turn = False

        for _ in range(self.max_tool_rounds):
            messages = self._messages()
            if continue_after_tools:
                messages = messages + [
                    self._continuation_poke(
                        user_message,
                        prefer_copyable_function=self._prefer_copyable_function_reply(user_message),
                    )
                ]
            assistant_text = self.model.generate(messages)
            tool_calls = parse_tool_calls(assistant_text)
            if not tool_calls and self._contains_tool_denial(self._strip_thinking(assistant_text)):
                self._log_supervision("tool_denial_detected", user_message, assistant_text)
                tool_calls = self._recover_tool_calls(user_message=user_message, raw_assistant_text=assistant_text)
            tool_calls = self._ensure_web_call_for_factual(
                user_message=user_message,
                tool_calls=tool_calls,
                web_search_executed=web_search_executed_this_turn,
            )

            if not tool_calls:
                clean = self._strip_thinking(assistant_text)
                if clean and self._contains_internal_prompt_echo(clean):
                    clean = ""
                if not clean:
                    recovered_text, recovered_calls, recovered_clean = self._recover_action_or_answer(user_message, assistant_text)
                    if recovered_calls:
                        assistant_text = recovered_text
                        tool_calls = recovered_calls
                    elif recovered_clean:
                        assistant_text = recovered_text
                        clean = recovered_clean
                    else:
                        recovered = self._recover_final_answer(assistant_text)
                        if recovered:
                            assistant_text, clean = recovered
                        else:
                            assistant_text = ""
                            clean = ""
                if not tool_calls and not clean and not emergency_tools_used:
                    emergency = self._emergency_tool_calls(user_message)
                    if emergency:
                        emergency_tools_used = True
                        tool_calls = emergency
                        assistant_text = json.dumps(
                            {
                                "tool_calls": [
                                    {"tool": call["name"], "args": call.get("args", {})} for call in emergency
                                ]
                            },
                            ensure_ascii=False,
                        )
                        self._log_supervision("emergency_tool_calls", user_message, assistant_text)
                if not tool_calls:
                    if not clean:
                        fallback_from_tools = self._fallback_answer_from_tools()
                        if fallback_from_tools:
                            assistant_text = fallback_from_tools
                            clean = assistant_text
                        else:
                            assistant_text = (
                                "I could not generate a final answer text from the model output. "
                                "Please try again or use a model/config that emits final content."
                            )
                            clean = assistant_text
                    presearch: list[dict[str, Any]] = []
                    event_name = ""
                    if clean and self._requires_workspace_preinspect(user_message) and not tools_executed_this_turn:
                        presearch = self._preinspect_tool_calls_for_workspace(user_message)
                        event_name = "workspace_preinspect"
                    elif clean and self._requires_presearch_for_code(user_message) and not tools_executed_this_turn:
                        presearch = self._presearch_tool_calls_for_code(user_message)
                        event_name = "presearch_for_code"
                    elif clean and self._requires_web_presearch_for_factual(user_message) and not web_search_executed_this_turn:
                        presearch = self._presearch_tool_calls_for_factual(user_message)
                        event_name = "presearch_for_factual"
                    if presearch:
                        tool_calls = presearch
                        assistant_text = json.dumps(
                            {
                                "tool_calls": [
                                    {"tool": call["name"], "args": call.get("args", {})} for call in presearch
                                ]
                            },
                            ensure_ascii=False,
                        )
                        self._log_supervision(event_name, user_message, assistant_text)
                        clean = ""
                    if tool_calls:
                        pass
                    else:
                        self.history.append({"role": "assistant", "content": clean})
                        return assistant_text

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history[-1]["content"] = self._strip_thinking(self.history[-1]["content"])
            self._log_tool_training_sample(user_message=user_message, assistant_text=assistant_text, tool_calls=tool_calls)

            for call in tool_calls:
                if on_tool_start:
                    on_tool_start(call["name"], call.get("args", {}))
                result = self._execute_tool_call_with_policy(user_message, call)
                if on_tool:
                    on_tool(call["name"], call.get("args", {}), result)
                if call["name"] == "search_web" and isinstance(result, dict) and result.get("ok", False):
                    web_search_executed_this_turn = True
                tool_payload = {
                    "tool": call["name"],
                    "args": call.get("args", {}),
                    "result": result,
                }
                self.history.append(
                    {
                        "role": "tool",
                        "content": json.dumps(tool_payload, ensure_ascii=False),
                    }
                )
            tools_executed_this_turn = True
            continue_after_tools = True

        final_text = self._fallback_answer_from_tools() or "Tool-call loop limit reached. Return direct answer."
        self.history.append({"role": "assistant", "content": final_text})
        return final_text

    def handle_turn_stream(
        self,
        user_message: str,
        on_chunk: Callable[[str], None],
        on_tool: Callable[[str, dict[str, object], dict[str, object]], None] | None = None,
        on_tool_start: Callable[[str, dict[str, object]], None] | None = None,
        on_tool_phase: Callable[[], None] | None = None,
    ) -> str:
        self.history.append({"role": "user", "content": user_message})
        continue_after_tools = False
        emergency_tools_used = False
        tools_executed_this_turn = False
        web_search_executed_this_turn = False

        for _ in range(self.max_tool_rounds):
            assistant_text = ""
            in_think_stream = False
            pending_nonthink_chunks: list[str] = []

            messages = self._messages()
            if continue_after_tools:
                messages = messages + [
                    self._continuation_poke(
                        user_message,
                        prefer_copyable_function=self._prefer_copyable_function_reply(user_message),
                    )
                ]

            for chunk in self.model.stream_generate(messages):
                if not chunk:
                    continue

                assistant_text += chunk

                if chunk == "<think>":
                    in_think_stream = True
                    on_chunk(chunk)
                    continue
                if chunk == "</think>":
                    in_think_stream = False
                    on_chunk(chunk)
                    continue

                if in_think_stream:
                    on_chunk(chunk)
                else:
                    pending_nonthink_chunks.append(chunk)

            tool_calls = parse_tool_calls(assistant_text)
            if not tool_calls and self._contains_tool_denial(self._strip_thinking(assistant_text)):
                self._log_supervision("tool_denial_detected", user_message, assistant_text)
                tool_calls = self._recover_tool_calls(user_message=user_message, raw_assistant_text=assistant_text)
            tool_calls = self._ensure_web_call_for_factual(
                user_message=user_message,
                tool_calls=tool_calls,
                web_search_executed=web_search_executed_this_turn,
            )
            if not tool_calls:
                pending_nonthink = "".join(pending_nonthink_chunks)
                clean = self._strip_thinking(assistant_text)
                if clean and self._contains_internal_prompt_echo(clean):
                    clean = ""
                if not clean:
                    recovered_text, recovered_calls, recovered_clean = self._recover_action_or_answer(user_message, assistant_text)
                    if recovered_calls:
                        assistant_text = recovered_text
                        tool_calls = recovered_calls
                    elif recovered_clean:
                        assistant_text = recovered_text
                        clean = recovered_clean
                    else:
                        recovered = self._recover_final_answer(assistant_text)
                        if recovered:
                            assistant_text, clean = recovered
                        else:
                            assistant_text = ""
                            clean = ""
                if not tool_calls and not clean and not emergency_tools_used:
                    emergency = self._emergency_tool_calls(user_message)
                    if emergency:
                        emergency_tools_used = True
                        tool_calls = emergency
                        assistant_text = json.dumps(
                            {
                                "tool_calls": [
                                    {"tool": call["name"], "args": call.get("args", {})} for call in emergency
                                ]
                            },
                            ensure_ascii=False,
                        )
                        self._log_supervision("emergency_tool_calls", user_message, assistant_text)
                if not tool_calls:
                    if not clean:
                        fallback_from_tools = self._fallback_answer_from_tools()
                        if fallback_from_tools:
                            assistant_text = fallback_from_tools
                            clean = assistant_text
                        else:
                            assistant_text = (
                                "I could not generate a final answer text from the model output. "
                                "Please try again or use a model/config that emits final content."
                            )
                            clean = assistant_text
                    presearch: list[dict[str, Any]] = []
                    event_name = ""
                    if clean and self._requires_workspace_preinspect(user_message) and not tools_executed_this_turn:
                        presearch = self._preinspect_tool_calls_for_workspace(user_message)
                        event_name = "workspace_preinspect"
                    elif clean and self._requires_presearch_for_code(user_message) and not tools_executed_this_turn:
                        presearch = self._presearch_tool_calls_for_code(user_message)
                        event_name = "presearch_for_code"
                    elif clean and self._requires_web_presearch_for_factual(user_message) and not web_search_executed_this_turn:
                        presearch = self._presearch_tool_calls_for_factual(user_message)
                        event_name = "presearch_for_factual"
                    if presearch:
                        tool_calls = presearch
                        assistant_text = json.dumps(
                            {
                                "tool_calls": [
                                    {"tool": call["name"], "args": call.get("args", {})} for call in presearch
                                ]
                            },
                            ensure_ascii=False,
                        )
                        self._log_supervision(event_name, user_message, assistant_text)
                        clean = ""
                    if tool_calls:
                        pass
                    else:
                        if pending_nonthink:
                            on_chunk(pending_nonthink)
                        self.history.append({"role": "assistant", "content": clean})
                        return assistant_text

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history[-1]["content"] = self._strip_thinking(self.history[-1]["content"])
            self._log_tool_training_sample(user_message=user_message, assistant_text=assistant_text, tool_calls=tool_calls)
            if on_tool_phase:
                on_tool_phase()

            for call in tool_calls:
                if on_tool_start:
                    on_tool_start(call["name"], call.get("args", {}))
                result = self._execute_tool_call_with_policy(user_message, call)
                if on_tool:
                    on_tool(call["name"], call.get("args", {}), result)
                if call["name"] == "search_web" and isinstance(result, dict) and result.get("ok", False):
                    web_search_executed_this_turn = True
                tool_payload = {
                    "tool": call["name"],
                    "args": call.get("args", {}),
                    "result": result,
                }
                self.history.append(
                    {
                        "role": "tool",
                        "content": json.dumps(tool_payload, ensure_ascii=False),
                    }
                )
            tools_executed_this_turn = True
            continue_after_tools = True

        final_text = self._fallback_answer_from_tools() or "Tool-call loop limit reached. Return direct answer."
        self.history.append({"role": "assistant", "content": final_text})
        return final_text

    def run_cli(self) -> None:
        print("Local Coding Assistant")
        print("Type 'exit' or 'quit' to stop.")
        print("Commands: /reset, /maxout <n>, /maxout, /stream <auto|native|chunk>, /stream")
        print("Autonomous: /auto, /auto on [steps], /auto off")

        while True:
            try:
                user_input = input("\nyou> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nbye")
                return

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("bye")
                return
            if user_input.lower() in {"/reset", "reset"}:
                self.history.clear()
                print("context reset")
                continue
            if user_input.startswith("/"):
                parts = user_input.strip().split()
                cmd = parts[0].lower()
                if cmd in {"/maxout", "/maxoutput", "/max_output"}:
                    if len(parts) == 1:
                        getter = getattr(self.model, "get_max_output_tokens", None)
                        current = getter() if callable(getter) else None
                        if current is None:
                            print("max output tokens: unavailable for this model")
                        else:
                            print(f"max output tokens: {current}")
                        continue
                    try:
                        value = int(parts[1])
                    except ValueError:
                        print("usage: /maxout <positive_integer>")
                        continue
                    setter = getattr(self.model, "set_max_output_tokens", None)
                    if not callable(setter):
                        print("this model does not support changing max output tokens")
                        continue
                    ok, msg = setter(value)
                    print(msg)
                    continue
                if cmd == "/stream":
                    if len(parts) == 1:
                        getter = getattr(self.model, "get_stream_mode", None)
                        mode = getter() if callable(getter) else "unknown"
                        print(f"stream mode: {mode}")
                        continue
                    setter = getattr(self.model, "set_stream_mode", None)
                    if not callable(setter):
                        print("this model does not support stream mode changes")
                        continue
                    ok, msg = setter(parts[1])
                    print(msg)
                    continue
                if cmd in {"/help", "help"}:
                    print("Commands:")
                    print("- /reset")
                    print("- /maxout <n>   set max output tokens")
                    print("- /maxout       show current max output tokens")
                    print("- /stream <auto|native|chunk>   set stream mode")
                    print("- /stream       show current stream mode")
                    print("- /auto         show autonomous status")
                    print("- /auto on [steps] / /auto off")
                    continue
                if cmd in {"/auto", "/autonomous"}:
                    if len(parts) == 1:
                        mode = "on" if self.autonomous_enabled else "off"
                        print(f"autonomous: {mode} (steps={self.autonomous_steps})")
                        continue
                    sub = parts[1].lower()
                    if sub == "off":
                        self.autonomous_enabled = False
                        print("autonomous: off")
                        continue
                    if sub == "on":
                        if len(parts) >= 3:
                            try:
                                self.autonomous_steps = max(1, min(int(parts[2]), 30))
                            except ValueError:
                                print("usage: /auto on [steps]")
                                continue
                        self.autonomous_enabled = True
                        print(f"autonomous: on (steps={self.autonomous_steps})")
                        continue
                    print("usage: /auto | /auto on [steps] | /auto off")
                    continue

            if self.autonomous_enabled:
                self.run_autonomous(user_input, self.autonomous_steps)
                continue

            renderer = StreamRenderer()
            response = self.handle_turn_stream(
                user_input,
                renderer.feed,
                print_tool_event,
                print_tool_start,
                renderer.prepare_tool_output,
            )
            renderer.finish()
            if not renderer.has_output:
                print_formatted_output(response=response)
            elif not renderer.has_answer_output:
                answer = extract_answer_text(response)
                if answer:
                    print_answer_only(answer)

    def run_autonomous(self, objective: str, steps: int) -> None:
        max_steps = max(1, min(steps, 30))
        print(f"autonomous run: objective='{objective}' | steps={max_steps}")
        print("scope: workspace root only")

        for i in range(1, max_steps + 1):
            if i == 1:
                prompt = (
                    "Autonomous mode enabled.\n"
                    f"Objective: {objective}\n"
                    "You may plan, research, read/edit files, and improve the project inside workspace root only.\n"
                    "If finished, include token AUTONOMOUS_DONE in final answer."
                )
            else:
                prompt = (
                    "Autonomous continue.\n"
                    f"Objective: {objective}\n"
                    f"Step: {i}/{max_steps}\n"
                    "Choose next best action yourself. If finished, include token AUTONOMOUS_DONE."
                )

            print(f"\n[auto step {i}/{max_steps}]")
            renderer = StreamRenderer()
            response = self.handle_turn_stream(
                prompt,
                renderer.feed,
                print_tool_event,
                print_tool_start,
                renderer.prepare_tool_output,
            )
            renderer.finish()
            final_text = extract_answer_text(response).strip()
            if re.search(r"\bAUTONOMOUS_DONE\b", final_text, flags=re.IGNORECASE):
                print("[auto] done")
                return
        print("[auto] step limit reached")
