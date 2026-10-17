"""A scripted stand-in for the LLM, so the real CrewAI crews, sandbox and verifier can be tested offline."""
import json
import re

from crewai.llms.base_llm import BaseLLM


class ScriptedLLM(BaseLLM):
    """Answers each kind of task from a script. The script values may be lists (one item per call)."""

    def __init__(self, script: dict, **kw):
        super().__init__(model="scripted", **kw)
        self._script = {k: (list(v) if isinstance(v, list) else v) for k, v in script.items()}
        self.log = []

    def supports_function_calling(self):
        return False

    def supports_stop_words(self):
        return False

    def get_context_window_size(self):
        return 32000

    def _next(self, key):
        v = self._script[key]
        if isinstance(v, list):
            return v.pop(0) if len(v) > 1 else v[0]
        return v

    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None,
             from_agent=None, response_model=None):
        text = messages if isinstance(messages, str) else "\n".join(str(m.get("content", "")) for m in messages)
        if "analysis questions" in text:
            key = "questions"
        elif "Draft answer:" in text:
            key = "critique"
        elif "Write ONE Python script" in text:
            key = "code"
        elif "Raw output of the executed analysis code" in text:
            key = "answer"
        else:
            key = "other"
        self.log.append(key)
        out = self._next(key) if key in self._script else "ok"
        return json.dumps(out) if isinstance(out, (dict, list)) else out
