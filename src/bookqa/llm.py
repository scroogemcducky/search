"""Chat model client with a response cache.

All generation goes through ``LLM.complete_many``: a list of prompts in, a
list of replies out, in the same order. Callers build their whole batch of
work first, then hand it over in one go. That lets the client send requests
in parallel and lets the cache skip work already done.

The cache matters for an overnight run: claim extraction makes hundreds of
calls, and if the run stops at call 300, re-running it replays the first 300
from disk in seconds and continues from there.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bookqa.config import LLMConfig
from bookqa.httpjson import auth_headers, post_json

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class ResponseCache:
    """A key-value store of prompt hash to reply, in SQLite (safe across threads)."""

    def __init__(self, path: Path | None):
        self.lock = threading.Lock()
        self.db = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path, check_same_thread=False)
            self.db.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, reply TEXT)")

    def get(self, key: str) -> str | None:
        if self.db is None:
            return None
        with self.lock:
            row = self.db.execute("SELECT reply FROM replies WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, reply: str) -> None:
        if self.db is None:
            return
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO replies VALUES (?, ?)", (key, reply))
            self.db.commit()


class LLM:
    """Base class: subclasses implement ``_call`` for one prompt."""

    def __init__(self, cache: ResponseCache | None = None, parallel: int = 1):
        self.cache = cache or ResponseCache(None)
        self.parallel = max(1, parallel)

    def identity(self) -> str:
        return type(self).__name__

    def _call(self, system: str, user: str, json_mode: bool) -> str:
        raise NotImplementedError

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        return self.complete_many(system, [user], json_mode)[0]

    def complete_many(
        self,
        system: str,
        users: list[str],
        json_mode: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[str]:
        keys = [
            hashlib.sha256(json.dumps([self.identity(), system, u, json_mode]).encode()).hexdigest()
            for u in users
        ]
        replies: list[str | None] = [self.cache.get(k) for k in keys]
        todo = [i for i, r in enumerate(replies) if r is None]
        done = len(users) - len(todo)

        def run(i: int) -> tuple[int, str]:
            reply = _THINK.sub("", self._call(system, users[i], json_mode)).strip()
            self.cache.put(keys[i], reply)
            return i, reply

        with ThreadPoolExecutor(self.parallel) as pool:
            for i, reply in pool.map(run, todo):
                replies[i] = reply
                done += 1
                if progress:
                    progress(done, len(users))
        return [r or "" for r in replies]


class OpenAIChat(LLM):
    """Any server with an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(self, config: LLMConfig, cache: ResponseCache | None = None):
        super().__init__(cache, config.parallel)
        self.config = config
        self.url = config.base_url.rstrip("/") + "/chat/completions"
        self.headers = auth_headers(config.api_key_env)

    def identity(self) -> str:
        c = self.config
        return f"openai:{c.base_url}:{c.model}:{c.temperature}:{c.prompt_suffix}"

    def _call(self, system: str, user: str, json_mode: bool) -> str:
        payload: dict = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user + self.config.prompt_suffix},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        reply = post_json(self.url, payload, self.headers, self.config.timeout_s)
        return reply["choices"][0]["message"]["content"] or ""


class ScriptedLLM(LLM):
    """Replies with a Python function of the prompt. For tests and dry runs."""

    def __init__(self, respond: Callable[[str, str], str], cache: ResponseCache | None = None):
        super().__init__(cache, parallel=1)
        self.respond = respond
        self.calls = 0

    def _call(self, system: str, user: str, json_mode: bool) -> str:
        self.calls += 1
        return self.respond(system, user)


def make_llm(config: LLMConfig, cache_path: Path | None = None) -> LLM:
    if config.backend == "openai":
        return OpenAIChat(config, ResponseCache(cache_path))
    raise ValueError(f"unknown llm backend: {config.backend}")


def parse_json_object(text: str) -> dict:
    """Pull the first JSON object out of a model reply.

    Models sometimes wrap JSON in code fences or add a sentence before it,
    even when asked not to. This finds the outermost braces and parses those.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start : end + 1])
