"""Orchestrator: combine several decision backends.

Google's Decisions API is one closed model. Here a "decision" can be routed through a
chain of open-source models and API models:

``cascade`` (default)
    Ask the first stage every question. Any answer whose (calibrated) confidence is
    below that stage's ``min_confidence`` is sent to the next stage, and so on. A
    small local model answers the easy 90% in one shared-prefix pass; only the hard
    questions reach a bigger local model, GPT or Claude. Each answer records which
    backend produced it and whether it was escalated.

``ensemble``
    Ask every stage every question and average the label distributions (optionally
    weighted). Useful as a stronger, more expensive judge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .aggregate import TaskScore, to_answer
from .prompts import build_tasks
from .schema import DecisionRequest, DecisionResponse, Usage
from .service import DecisionService


@dataclass
class Stage:
    service: DecisionService
    min_confidence: float = 0.0   # answers below this go to the next stage
    weight: float = 1.0           # ensemble weight


def _add_usage(total: Usage, u: Usage) -> None:
    total.total_input_tokens += u.total_input_tokens
    total.total_output_tokens += u.total_output_tokens
    total.total_tokens += u.total_tokens
    total.prefix_tokens = max(total.prefix_tokens, u.prefix_tokens)
    total.prefix_tokens_reused += u.prefix_tokens_reused
    total.remote_calls += u.remote_calls


class Orchestrator:
    def __init__(self, stages: list[Stage], strategy: str = "cascade", label_style: str = "letters"):
        if not stages:
            raise ValueError("need at least one stage")
        if strategy not in ("cascade", "ensemble"):
            raise ValueError("strategy must be 'cascade' or 'ensemble'")
        self.stages = stages
        self.strategy = strategy
        self.label_style = label_style

    @property
    def model_name(self) -> str:
        return f"{self.strategy}(" + ", ".join(s.service.model_name for s in self.stages) + ")"

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        t0 = time.perf_counter()
        text = request.input_text()
        tasks = build_tasks(request.questions, self.label_style)
        usage = Usage()
        answers = {}

        if self.strategy == "ensemble":
            acc = [np.zeros(len(t.labels)) for t in tasks]
            wsum = 0.0
            estimated = [False] * len(tasks)
            for stage in self.stages:
                probs, res = stage.service.score_tasks(text, tasks)
                _add_usage(usage, res.usage)
                wsum += stage.weight
                for i, p in enumerate(probs):
                    acc[i] += stage.weight * p
                    estimated[i] |= res.scores[i].estimated
            for i, t in enumerate(tasks):
                score = TaskScore([], method="logprob" if not estimated[i] else "sampling", estimated=estimated[i])
                answers[t.key] = to_answer(t, acc[i] / wsum, score, "ensemble")
        else:
            pending = list(tasks)
            for n, stage in enumerate(self.stages):
                probs, res = stage.service.score_tasks(text, pending)
                _add_usage(usage, res.usage)
                last = n == len(self.stages) - 1
                still = []
                for t, p, s in zip(pending, probs, res.scores):
                    a = to_answer(t, p, s, stage.service.backend.name)
                    a.escalated = n > 0
                    answers[t.key] = a
                    if not last and a.confidence < stage.min_confidence:
                        still.append(t)
                pending = still
                if not pending:
                    break

        ordered = {t.key: answers[t.key] for t in tasks}
        return DecisionResponse(model=self.model_name, answers=ordered, usage=usage,
                                latency_ms=round((time.perf_counter() - t0) * 1000, 2))
