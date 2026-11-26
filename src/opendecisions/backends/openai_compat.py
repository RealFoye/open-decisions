"""OpenAI-compatible Chat Completions backend.

Works with OpenAI and with any server that speaks the same API: vLLM, SGLang,
llama.cpp server, Ollama, LM Studio, OpenRouter, Together, DeepSeek and others.

Probabilities come from ``logprobs`` / ``top_logprobs`` of the first answer token.
Not every model returns them (OpenAI's reasoning models, for example, do not). When a
response has no log-probabilities the backend falls back to sampling ``n`` answers
in one request and uses smoothed frequencies; those answers are marked
``estimated=True, method="sampling"`` so callers can tell the difference.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import httpx

from ..aggregate import TaskScore, logsumexp
from ..prompts import SYSTEM_PROMPT, ScoringTask
from ..schema import Usage
from .base import Backend, BackendResult
from .remote_common import counts_to_log_scores, match_label, user_parts

MAX_TOP_LOGPROBS = 20


class OpenAICompatBackend(Backend):
    name = "openai"
    exact = True  # when the server returns logprobs; per-answer `estimated` says otherwise

    def __init__(self, model: str, *, base_url: str | None = None, api_key: str | None = None,
                 concurrency: int = 16, fallback_samples: int = 8, timeout: float = 60.0,
                 client: httpx.Client | None = None, extra_body: dict | None = None):
        self.model_id = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.client = client or httpx.Client(timeout=timeout, headers=headers)
        if client is not None and key:
            self.client.headers.update(headers)
        self.concurrency = concurrency
        self.fallback_samples = fallback_samples
        self.extra_body = extra_body or {}

    def _messages(self, input_text: str, task: ScoringTask) -> list[dict]:
        shared, q = user_parts(input_text, task)
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": shared + q}]

    def _post(self, body: dict) -> dict:
        r = self.client.post(f"{self.base_url}/chat/completions", json={**body, **self.extra_body})
        r.raise_for_status()
        return r.json()

    def _score_one(self, input_text: str, task: ScoringTask) -> tuple[TaskScore, Usage]:
        usage = Usage()
        base = {"model": self.model_id, "messages": self._messages(input_text, task), "max_tokens": 1}
        data = self._post({**base, "temperature": 0, "logprobs": True, "top_logprobs": MAX_TOP_LOGPROBS})
        usage.remote_calls += 1
        self._add_usage(usage, data)
        content = ((data.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
        if content and content[0].get("top_logprobs"):
            return TaskScore(self._from_top_logprobs(content[0]["top_logprobs"], task.labels)), usage

        # no log-probabilities: estimate by sampling n answers in one request
        body = {**base, "max_tokens": 8, "temperature": 1.0, "n": self.fallback_samples}
        data = self._post(body)
        usage.remote_calls += 1
        self._add_usage(usage, data)
        answers = [(c.get("message") or {}).get("content") for c in data.get("choices", [])]
        return TaskScore(counts_to_log_scores(answers, task.labels), method="sampling", estimated=True), usage

    @staticmethod
    def _from_top_logprobs(top: list[dict], labels: list[str]) -> list[float]:
        buckets: list[list[float]] = [[] for _ in labels]
        for entry in top:
            i = match_label(entry.get("token", ""), labels)
            if i is not None:
                buckets[i].append(float(entry["logprob"]))
        seen = [lp for b in buckets for lp in b]
        # labels outside the top-k get less mass than the least likely token returned
        floor = (min(float(e["logprob"]) for e in top) if top else 0.0) - math.log(2)
        if not seen:
            floor = 0.0  # nothing matched: uniform
        return [logsumexp(b) if b else floor for b in buckets]

    @staticmethod
    def _add_usage(usage: Usage, data: dict) -> None:
        u = data.get("usage") or {}
        usage.total_input_tokens += int(u.get("prompt_tokens", 0))
        usage.total_output_tokens += int(u.get("completion_tokens", 0))
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        usage.prefix_tokens_reused += int(cached)
        usage.total_tokens = usage.total_input_tokens + usage.total_output_tokens

    def score(self, input_text: str, tasks: list[ScoringTask]) -> BackendResult:
        if not tasks:
            return BackendResult([], Usage())
        # the first call writes the provider's prefix cache; the rest can then read it
        results = [self._score_one(input_text, tasks[0])]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            results += list(pool.map(lambda t: self._score_one(input_text, t), tasks[1:]))
        total = Usage()
        for _, u in results:
            total.total_input_tokens += u.total_input_tokens
            total.total_output_tokens += u.total_output_tokens
            total.prefix_tokens_reused += u.prefix_tokens_reused
            total.remote_calls += u.remote_calls
        total.total_tokens = total.total_input_tokens + total.total_output_tokens
        total.prefix_cache_hit = total.prefix_tokens_reused > 0
        return BackendResult([s for s, _ in results], total)
