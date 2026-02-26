from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content or "")


def _pick_openrouter_model(
    available: list[str],
    requested: str,
    env_fallback: str = "",
) -> tuple[str, str]:
    if not available:
        return requested, "model list unavailable, keep requested model"
    requested_clean = (requested or "").strip()
    if requested_clean in available:
        return requested_clean, "requested model is available"

    candidates = [env_fallback.strip(), "arcee-ai/trinity-large-preview:free", "openrouter/auto"]
    for cand in candidates:
        if cand and cand in available:
            return cand, f"requested model not found, fallback to {cand}"

    free_models = [m for m in available if m.endswith(":free")]
    if free_models:
        return free_models[0], f"requested model not found, fallback to {free_models[0]}"

    return available[0], f"requested model not found, fallback to {available[0]}"


class BaseModel:
    provider = "unknown"
    model_name = ""
    endpoint = ""
    native_streaming = False
    connect_log: list[str] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    def set_max_output_tokens(self, value: int) -> tuple[bool, str]:
        return False, "model does not support dynamic max output tokens"

    def get_max_output_tokens(self) -> int | None:
        return None

    def set_stream_mode(self, mode: str) -> tuple[bool, str]:
        return False, "model does not support stream mode changes"

    def get_stream_mode(self) -> str:
        return "chunk"

    @staticmethod
    def _stream_text(text: str, chunk_size: int = 28) -> Iterator[str]:
        if not text:
            return
        i = 0
        n = len(text)
        while i < n:
            j = min(n, i + chunk_size)
            # Prefer splitting on whitespace for readability.
            if j < n:
                pivot = text.rfind(" ", i, j)
                if pivot > i:
                    j = pivot + 1
            yield text[i:j]
            i = j

    def info(self) -> dict[str, Any]:
        details: dict[str, Any] = {}
        for key in ("temperature", "max_tokens", "timeout", "stream_mode", "stream_timeout"):
            if hasattr(self, key):
                details[key] = getattr(self, key)
        if hasattr(self, "options") and isinstance(getattr(self, "options"), dict):
            details["options"] = dict(getattr(self, "options"))
        return {
            "provider": getattr(self, "provider", "unknown"),
            "model": getattr(self, "model_name", ""),
            "endpoint": getattr(self, "endpoint", ""),
            "native_streaming": bool(getattr(self, "native_streaming", False)),
            "connect_log": list(getattr(self, "connect_log", []) or []),
            "details": details,
        }

    def stream_generate(self, messages: list[dict[str, str]]) -> Iterator[str]:
        text = self.generate(messages)
        yield from self._stream_text(text)


class OllamaModel(BaseModel):
    def __init__(self, model_name: str, base_url: str = "http://127.0.0.1:11434", timeout: int = 60) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.endpoint = self.base_url
        self.provider = "ollama"
        self.timeout = timeout
        self.native_streaming = True
        self.connect_log = []
        self.stream_mode = os.getenv("ASSISTANT_STREAM_MODE", "auto").strip().lower()
        if self.stream_mode not in {"auto", "native", "chunk"}:
            self.stream_mode = "auto"
        num_ctx = _env_int("ASSISTANT_NUM_CTX", 8192)
        num_predict = _env_int("ASSISTANT_NUM_PREDICT", 2048)
        temperature = _env_float("ASSISTANT_TEMPERATURE", 0.2)
        self.options = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }

    def set_max_output_tokens(self, value: int) -> tuple[bool, str]:
        if value <= 0:
            return False, "max output tokens must be > 0"
        self.options["num_predict"] = int(value)
        return True, f"ollama num_predict set to {value}"

    def get_max_output_tokens(self) -> int | None:
        raw = self.options.get("num_predict")
        return int(raw) if isinstance(raw, int) else None

    def set_stream_mode(self, mode: str) -> tuple[bool, str]:
        m = (mode or "").strip().lower()
        if m not in {"auto", "native", "chunk"}:
            return False, "stream mode must be one of: auto, native, chunk"
        self.stream_mode = m
        return True, f"ollama stream mode set to {m}"

    def get_stream_mode(self) -> str:
        return self.stream_mode

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout) as resp:  # nosec B310 - localhost endpoint
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def _stream_post_json(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout) as resp:  # nosec B310 - localhost endpoint
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n\n".join(lines)

    @staticmethod
    def _compose_chat_text(data: dict[str, Any]) -> str:
        message = data.get("message", {})
        thinking = message.get("thinking", "")
        content = message.get("content", "")
        out = ""
        if thinking:
            out += f"<think>{thinking}</think>"
        if content:
            out += content
        return out

    @staticmethod
    def _compose_generate_text(data: dict[str, Any]) -> str:
        thinking = data.get("thinking", "")
        response = data.get("response", "")
        out = ""
        if thinking:
            out += f"<think>{thinking}</think>"
        if response:
            out += response
        return out

    @staticmethod
    def _compose_chat_stream_text(data: dict[str, Any]) -> tuple[str, str]:
        message = data.get("message", {})
        return message.get("thinking", ""), message.get("content", "")

    @staticmethod
    def _compose_generate_stream_text(data: dict[str, Any]) -> tuple[str, str]:
        return data.get("thinking", ""), data.get("response", "")

    def is_available(self) -> bool:
        try:
            req = Request(f"{self.base_url}/api/tags", method="GET")
            with urlopen(req, timeout=5):  # nosec B310 - localhost endpoint
                return True
        except URLError:
            return False
        except Exception:
            return False

    def generate(self, messages: list[dict[str, str]]) -> str:
        chat_payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": self.options,
        }
        try:
            data = self._post_json("/api/chat", chat_payload)
            text = self._compose_chat_text(data)
            if text:
                return text
            return data.get("message", {}).get("content", "")
        except HTTPError as e:
            if e.code != 404:
                raise
        except Exception:
            pass

        generate_payload = {
            "model": self.model_name,
            "prompt": self._messages_to_prompt(messages),
            "stream": False,
            "options": self.options,
        }
        try:
            data = self._post_json("/api/generate", generate_payload)
            text = self._compose_generate_text(data)
            if text:
                return text
            return data.get("response", "")
        except Exception:
            return (
                "Local model endpoint available but incompatible. "
                "Set --ollama-url to a valid Ollama server."
            )

    def stream_generate(self, messages: list[dict[str, str]]) -> Iterator[str]:
        if self.stream_mode == "chunk":
            yield from self._stream_text(self.generate(messages))
            return

        chat_payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": self.options,
        }
        in_think = False
        try:
            for data in self._stream_post_json("/api/chat", chat_payload):
                thinking, content = self._compose_chat_stream_text(data)
                if thinking:
                    if not in_think:
                        yield "<think>"
                        in_think = True
                    yield thinking
                if content:
                    if in_think:
                        yield "</think>"
                        in_think = False
                    yield content
            if in_think:
                yield "</think>"
            return
        except HTTPError as e:
            if e.code != 404:
                raise
        except Exception:
            pass

        generate_payload = {
            "model": self.model_name,
            "prompt": self._messages_to_prompt(messages),
            "stream": True,
            "options": self.options,
        }
        in_think = False
        try:
            for data in self._stream_post_json("/api/generate", generate_payload):
                thinking, response = self._compose_generate_stream_text(data)
                if thinking:
                    if not in_think:
                        yield "<think>"
                        in_think = True
                    yield thinking
                if response:
                    if in_think:
                        yield "</think>"
                        in_think = False
                    yield response
            if in_think:
                yield "</think>"
            return
        except Exception:
            fallback_text = self.generate(messages)
            yield from self._stream_text(fallback_text)


class OpenRouterModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 120,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.endpoint = self.base_url
        self.provider = "openrouter"
        self.timeout = timeout
        self.native_streaming = True
        self.connect_log = []
        self.stream_mode = os.getenv("ASSISTANT_STREAM_MODE", "auto").strip().lower()
        if self.stream_mode not in {"auto", "native", "chunk"}:
            self.stream_mode = "auto"
        self.stream_timeout = _env_int("ASSISTANT_STREAM_TIMEOUT", 35)
        self.temperature = _env_float("ASSISTANT_TEMPERATURE", 0.2)
        self.max_tokens = _env_int("ASSISTANT_NUM_PREDICT", 2048)
        self.http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost").strip()
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "Local Coding Assistant").strip()
        self.fallback_model = os.getenv("OPENROUTER_FALLBACK_MODEL", "arcee-ai/trinity-large-preview:free").strip()

    def set_max_output_tokens(self, value: int) -> tuple[bool, str]:
        if value <= 0:
            return False, "max output tokens must be > 0"
        self.max_tokens = int(value)
        return True, f"openrouter max_tokens set to {value}"

    def get_max_output_tokens(self) -> int | None:
        return int(self.max_tokens) if isinstance(self.max_tokens, int) else None

    def set_stream_mode(self, mode: str) -> tuple[bool, str]:
        m = (mode or "").strip().lower()
        if m not in {"auto", "native", "chunk"}:
            return False, "stream mode must be one of: auto, native, chunk"
        self.stream_mode = m
        return True, f"openrouter stream mode set to {m}"

    def get_stream_mode(self) -> str:
        return self.stream_mode

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        with urlopen(req, timeout=self.timeout) as resp:  # nosec B310 - HTTPS OpenRouter endpoint
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def _stream_post_json(self, path: str, payload: dict[str, Any], timeout: int | None = None) -> Iterator[dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        effective_timeout = timeout if isinstance(timeout, int) and timeout > 0 else self.timeout
        with urlopen(req, timeout=effective_timeout) as resp:  # nosec B310 - HTTPS OpenRouter endpoint
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    continue
                if line.startswith("data:"):
                    data_chunk = line[5:].strip()
                    if data_chunk == "[DONE]":
                        break
                else:
                    # Some proxies strip `data:` prefix, keep parser permissive.
                    data_chunk = line
                try:
                    yield json.loads(data_chunk)
                except json.JSONDecodeError:
                    continue

    def _chat_payload(self, messages: list[dict[str, str]], stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": self.temperature,
        }
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        return payload

    @staticmethod
    def _extract_final_text(data: dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""
        first = choices[0]
        message = first.get("message", {})
        reasoning = _message_content_to_text(message.get("reasoning", ""))
        content = _message_content_to_text(message.get("content", ""))
        out = ""
        if reasoning:
            out += f"<think>{reasoning}</think>"
        if content:
            out += content
        return out

    @staticmethod
    def _extract_stream_delta(data: dict[str, Any]) -> tuple[str, str]:
        choices = data.get("choices", [])
        if not choices:
            return "", ""
        delta = choices[0].get("delta", {})
        reasoning = _message_content_to_text(
            delta.get("reasoning") or delta.get("reasoning_content") or ""
        )
        content = _message_content_to_text(delta.get("content", ""))
        return reasoning, content

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            req = Request(
                f"{self.base_url}/models",
                headers=self._headers(),
                method="GET",
            )
            with urlopen(req, timeout=8):  # nosec B310 - HTTPS OpenRouter endpoint
                return True
        except Exception:
            return False

    def _switch_model_if_missing(self) -> bool:
        available = list_openrouter_models(self.api_key, self.base_url, timeout=12)
        next_model, reason = _pick_openrouter_model(
            available=available,
            requested=self.model_name,
            env_fallback=self.fallback_model,
        )
        if next_model and next_model != self.model_name:
            old = self.model_name
            self.model_name = next_model
            self.connect_log.append(f"[warn] model '{old}' unavailable")
            self.connect_log.append(f"[ok] switched model to '{next_model}' ({reason})")
            return True
        return False

    def generate(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            return "OpenRouter API key missing. Set OPENROUTER_API_KEY or use --provider ollama."
        payload = self._chat_payload(messages, stream=False)
        try:
            data = self._post_json("/chat/completions", payload)
            text = self._extract_final_text(data)
            return text or "OpenRouter returned an empty response."
        except HTTPError as e:
            if e.code == 404 and self._switch_model_if_missing():
                try:
                    retry = self._post_json("/chat/completions", self._chat_payload(messages, stream=False))
                    retry_text = self._extract_final_text(retry)
                    return retry_text or "OpenRouter returned an empty response."
                except Exception as e2:
                    return f"OpenRouter request failed after model fallback: {e2}"
            return f"OpenRouter request failed: {e}"
        except Exception as e:
            return f"OpenRouter request failed: {e}"

    def stream_generate(self, messages: list[dict[str, str]]) -> Iterator[str]:
        if not self.api_key:
            yield "OpenRouter API key missing. Set OPENROUTER_API_KEY or use --provider ollama."
            return

        if self.stream_mode == "chunk":
            yield from self._stream_text(self.generate(messages))
            return

        payload = self._chat_payload(messages, stream=True)
        in_think = False
        emitted = False
        try:
            for data in self._stream_post_json("/chat/completions", payload, timeout=self.stream_timeout):
                thinking, content = self._extract_stream_delta(data)
                if thinking:
                    if not in_think:
                        yield "<think>"
                        in_think = True
                    yield thinking
                    emitted = True
                if content:
                    if in_think:
                        yield "</think>"
                        in_think = False
                    yield content
                    emitted = True
            if in_think:
                yield "</think>"
            if not emitted:
                yield from self._stream_text(self.generate(messages))
        except HTTPError as e:
            if in_think:
                yield "</think>"
            if e.code == 404 and self._switch_model_if_missing():
                yield from self._stream_text(self.generate(messages))
                return
            fallback_text = f"OpenRouter stream failed: {e}"
            yield from self._stream_text(fallback_text)
        except Exception as e:
            if in_think:
                yield "</think>"
            # Stream fallback for models/providers that fail SSE.
            fallback_text = self.generate(messages)
            if "OpenRouter request failed" in fallback_text:
                fallback_text = f"OpenRouter stream failed: {e}. {fallback_text}"
            yield from self._stream_text(fallback_text)


class FallbackModel(BaseModel):
    CONTINUATION_PREFIX = "Tool execution is complete."

    def __init__(self, reason: str = "") -> None:
        self.reason = reason.strip()
        self.provider = "fallback"
        self.model_name = "unavailable"
        self.endpoint = ""
        self.native_streaming = False
        self.connect_log = []
        self.stream_mode = "chunk"

    def get_stream_mode(self) -> str:
        return self.stream_mode

    def generate(self, messages: list[dict[str, str]]) -> str:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if content.startswith(self.CONTINUATION_PREFIX):
                    continue
                last_user = content
                break
        reason_text = f" Reason: {self.reason}." if self.reason else ""
        return (
            "Model backend not available."
            f"{reason_text} "
            "Start Ollama or configure OpenRouter and retry. "
            "No tool calls were executed. "
            f"Last user message: {last_user[:300]}"
        )

    def stream_generate(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield from self._stream_text(self.generate(messages))


def list_openrouter_models(
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout: int = 20,
) -> list[str]:
    key = api_key.strip()
    if not key:
        return []
    req = Request(
        f"{base_url.rstrip('/')}/models",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:  # nosec B310 - HTTPS OpenRouter endpoint
            raw = resp.read().decode("utf-8")
        payload = json.loads(raw)
        data = payload.get("data", [])
        names: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id.strip():
                names.append(model_id.strip())
        return sorted(set(names))
    except Exception:
        return []


def build_model(
    model_name: str,
    provider: str = "auto",
    ollama_url: str = "http://127.0.0.1:11434",
    openrouter_url: str = "https://openrouter.ai/api/v1",
    openrouter_api_key: str | None = None,
) -> BaseModel:
    provider_key = (provider or "auto").strip().lower()
    if provider_key not in {"auto", "ollama", "openrouter"}:
        provider_key = "auto"

    api_key = (openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
    connect_log: list[str] = []

    if provider_key == "ollama":
        connect_log.append(f"[try] ollama @ {ollama_url}")
        ollama = OllamaModel(model_name=model_name, base_url=ollama_url)
        if ollama.is_available():
            connect_log.append(f"[ok] ollama connected, model={model_name}")
            ollama.connect_log = connect_log
            return ollama
        connect_log.append("[fail] ollama unavailable")
        fb = FallbackModel(reason="Ollama unavailable")
        fb.connect_log = connect_log
        return fb

    if provider_key == "openrouter":
        if not api_key:
            connect_log.append("[fail] openrouter api key missing")
            fb = FallbackModel(reason="OPENROUTER_API_KEY missing")
            fb.connect_log = connect_log
            return fb
        connect_log.append(f"[try] openrouter @ {openrouter_url}")
        available_models = list_openrouter_models(api_key=api_key, base_url=openrouter_url, timeout=12)
        resolved_model, model_note = _pick_openrouter_model(
            available=available_models,
            requested=model_name,
            env_fallback=os.getenv("OPENROUTER_FALLBACK_MODEL", "arcee-ai/trinity-large-preview:free"),
        )
        if resolved_model != model_name:
            connect_log.append(f"[warn] requested model '{model_name}' unavailable")
            connect_log.append(f"[ok] using '{resolved_model}' ({model_note})")
        else:
            connect_log.append(f"[ok] using requested model '{resolved_model}'")
        openrouter = OpenRouterModel(
            model_name=resolved_model,
            api_key=api_key,
            base_url=openrouter_url,
        )
        if openrouter.is_available():
            connect_log.append(f"[ok] openrouter connected, model={resolved_model}")
            openrouter.connect_log = connect_log
            return openrouter
        connect_log.append("[fail] openrouter unavailable or unauthorized")
        fb = FallbackModel(reason="OpenRouter unavailable or unauthorized")
        fb.connect_log = connect_log
        return fb

    # auto: prefer local first, then OpenRouter.
    connect_log.append(f"[try] auto->ollama @ {ollama_url}")
    ollama = OllamaModel(model_name=model_name, base_url=ollama_url)
    if ollama.is_available():
        connect_log.append(f"[ok] auto selected ollama, model={model_name}")
        ollama.connect_log = connect_log
        return ollama
    connect_log.append("[fail] auto ollama unavailable")
    if api_key:
        connect_log.append(f"[try] auto->openrouter @ {openrouter_url}")
        available_models = list_openrouter_models(api_key=api_key, base_url=openrouter_url, timeout=12)
        resolved_model, model_note = _pick_openrouter_model(
            available=available_models,
            requested=model_name,
            env_fallback=os.getenv("OPENROUTER_FALLBACK_MODEL", "arcee-ai/trinity-large-preview:free"),
        )
        if resolved_model != model_name:
            connect_log.append(f"[warn] requested model '{model_name}' unavailable")
            connect_log.append(f"[ok] using '{resolved_model}' ({model_note})")
        else:
            connect_log.append(f"[ok] using requested model '{resolved_model}'")
        openrouter = OpenRouterModel(
            model_name=resolved_model,
            api_key=api_key,
            base_url=openrouter_url,
        )
        if openrouter.is_available():
            connect_log.append(f"[ok] auto selected openrouter, model={resolved_model}")
            openrouter.connect_log = connect_log
            return openrouter
        connect_log.append("[fail] auto openrouter unavailable or unauthorized")
    else:
        connect_log.append("[skip] auto openrouter (api key missing)")
    fb = FallbackModel(reason="No available backend (Ollama/OpenRouter)")
    fb.connect_log = connect_log
    return fb
