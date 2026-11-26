"""Claude backend (Anthropic Messages API, official ``anthropic`` SDK).

The Messages API does not return token log-probabilities, so exact label
probabilities are impossible here. Two honest estimators are offered, and every
answer from this backend carries ``estimated=True``:

``sampling`` (default)
    Ask the same question ``samples`` times. The answer is constrained to the label
    set with structured outputs (a JSON schema with an ``enum``), so every sample is
    a valid label. Probabilities are smoothed frequencies (Dirichlet(0.5) prior).
    Cost grows with ``samples``; resolution is about ``1 / samples``.

``verbalized``
    One call per question that returns a probability for every label (again via a
    JSON schema). Cheap, but verbalized probabilities are known to be poorly
    calibrated, so fit a calibrator (``opendecisions calibrate``) before trusting
    the numbers.

The shared input is sent as its own content block with ``cache_control``, so after
the first question the rest read it from Anthropic's prompt cache (only for inputs
above the model's minimum cacheable length). Cache reads are reported in
``usage.prefix_tokens_reused``.

For large offline jobs the Message Batches API is cheaper; it is not wired in here.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from ..aggregate import TaskScore
from ..prompts import SYSTEM_PROMPT, ScoringTask
from ..schema import Usage
from .base import Backend, BackendResult
from .remote_common import counts_to_log_scores, probs_to_log_scores, user_parts

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicBackend(Backend):
    name = "anthropic"
    exact = False

    def __init__(self, model: str = DEFAULT_MODEL, *, method: str = "sampling", samples: int = 8,
                 effort: str | None = "low", max_tokens: int = 4096, concurrency: int = 8,
                 client=None):
        if method not in ("sampling", "verbalized"):
            raise ValueError("method must be 'sampling' or 'verbalized'")
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model_id = model
        self.method = method
        self.samples = samples
        self.effort = effort
        self.max_tokens = max_tokens
        self.concurrency = concurrency

    # ------------------------------------------------------------------ requests

    def _request(self, input_text: str, task: ScoringTask, schema: dict) -> dict:
        shared, q = user_parts(input_text, task)
        output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort
        return dict(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": shared, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": q},
                ],
            }],
            output_config=output_config,
        )

    @staticmethod
    def _answer_schema(labels: list[str]) -> dict:
        return {"type": "object", "properties": {"answer": {"type": "string", "enum": labels}},
                "required": ["answer"], "additionalProperties": False}

    @staticmethod
    def _probs_schema(labels: list[str]) -> dict:
        props = {lab: {"type": "number"} for lab in labels}
        inner = {"type": "object", "properties": props, "required": labels, "additionalProperties": False}
        return {"type": "object", "properties": {"probabilities": inner},
                "required": ["probabilities"], "additionalProperties": False}

    def _call(self, kwargs: dict, usage: Usage) -> dict | None:
        msg = self.client.messages.create(**kwargs)
        usage.remote_calls += 1
        u = getattr(msg, "usage", None)
        if u is not None:
            usage.total_input_tokens += (getattr(u, "input_tokens", 0) or 0) \
                + (getattr(u, "cache_read_input_tokens", 0) or 0) \
                + (getattr(u, "cache_creation_input_tokens", 0) or 0)
            usage.prefix_tokens_reused += getattr(u, "cache_read_input_tokens", 0) or 0
            usage.total_output_tokens += getattr(u, "output_tokens", 0) or 0
        if getattr(msg, "stop_reason", None) == "refusal":
            return None
        text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _score_one(self, input_text: str, task: ScoringTask) -> tuple[TaskScore, Usage]:
        usage = Usage()
        if self.method == "verbalized":
            data = self._call(self._request(input_text, task, self._probs_schema(task.labels)), usage) or {}
            scores = probs_to_log_scores(data.get("probabilities") or {}, task.labels)
            return TaskScore(scores, method="verbalized", estimated=True), usage
        kwargs = self._request(input_text, task, self._answer_schema(task.labels))
        answers = []
        for _ in range(self.samples):
            data = self._call(kwargs, usage)
            answers.append(data.get("answer") if data else None)
        return TaskScore(counts_to_log_scores(answers, task.labels), method="sampling", estimated=True), usage

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
