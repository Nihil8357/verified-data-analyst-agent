"""Groq connection for CrewAI.

CrewAI accepts any class that implements BaseLLM.call(). GroqLLM talks to Groq's free API and
switches to another free model when one model's daily token allowance is used up
(Groq retires and rate-limits models, so a fixed model name is fragile)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from crewai.llms.base_llm import BaseLLM
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]


class ModelUnavailable(RuntimeError):
    pass


class BadKey(RuntimeError):
    pass


def clean_key(key: str) -> str:
    """Groq keys contain only letters, digits and '_'. Pasting often adds spaces, quotes or a stray '\\'."""
    import re
    return re.sub(r"[^A-Za-z0-9_]", "", key or "")


def _daily(e: Exception) -> bool:
    t = str(e).lower()
    return "tokens per day" in t or "(tpd)" in t or "requests per day" in t


def _minute(e: Exception) -> bool:
    t = str(e).lower()
    return ("429" in t or "rate limit" in t) and not _daily(e)


def _gone(e: Exception) -> bool:
    t = str(e).lower()
    return "model_not_found" in t or "decommissioned" in t or "does not exist" in t or "404" in t


class GroqLLM(BaseLLM):
    """CrewAI-compatible LLM backed by Groq with automatic model fallback."""

    def __init__(self, api_key: str, models: list[str] | None = None, temperature: float = 0.1, **kw):
        preferred = os.getenv("GROQ_MODEL", "").strip()
        order = ([preferred] if preferred else []) + [m for m in (models or MODELS) if m != preferred]
        super().__init__(model=order[0], temperature=temperature, **kw)
        from groq import Groq
        self._client = Groq(api_key=api_key, max_retries=2, timeout=90)
        self._models = order
        self.calls = 0
        self.tokens = 0

    @property
    def current(self) -> str:
        return self._models[0]

    def supports_function_calling(self) -> bool:
        return False                    # our agents return text/JSON; code is executed by our own sandbox

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 32000

    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None,
             from_agent=None, response_model=None):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        msgs = [{"role": m["role"], "content": str(m["content"])} for m in messages]
        kwargs = {}
        if response_model is not None:
            schema = json.dumps(response_model.model_json_schema())
            msgs.append({"role": "user", "content": f"Return ONLY a JSON object matching this schema: {schema}"})
            kwargs["response_format"] = {"type": "json_object"}
        while self._models:
            model = self._models[0]
            extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
            for attempt in range(3):
                try:
                    r = self._client.chat.completions.create(model=model, messages=msgs, temperature=self.temperature,
                                                             max_tokens=4000, **extra, **kwargs)
                    self.calls += 1
                    self.tokens += getattr(r.usage, "total_tokens", 0) or 0
                    return r.choices[0].message.content or ""
                except Exception as e:
                    if "invalid_api_key" in str(e).lower() or "401" in str(e):
                        raise BadKey("Groq rejected the API key. Create or copy it again at console.groq.com/keys "
                                     "(it starts with gsk_) and paste it with the 'Change key' button.") from e
                    if _daily(e) or _gone(e):
                        break
                    if _minute(e) and attempt < 2:
                        time.sleep(20)
                        continue
                    raise
            self._models.pop(0)                         # this model is out for today: try the next one
            if self._models:
                self.model = self._models[0]
        raise ModelUnavailable("All free Groq models are unavailable or out of daily tokens. Try again later "
                               "or use Demo mode.")


def get_llm():
    if os.getenv("AGENT_OFFLINE") == "1":
        return None
    key = clean_key(os.getenv("GROQ_API_KEY", ""))
    return GroqLLM(api_key=key) if key else None


def describe(llm) -> str:
    if llm is None:
        return "Demo mode (no AI model: built-in statistical templates)"
    return f"Groq {getattr(llm, 'current', getattr(llm, 'model', '?'))}"
