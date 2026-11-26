from __future__ import annotations

import time

import numpy as np

from .aggregate import to_answer
from .backends.base import Backend, BackendResult
from .calibration import Calibrator
from .prompts import ScoringTask, build_tasks
from .schema import DecisionRequest, DecisionResponse

CONTENT_FREE_INPUT = "N/A"


class DecisionService:
    """One backend + optional calibration -> Decisions-style responses."""

    def __init__(self, backend: Backend, calibrator: Calibrator | None = None, *,
                 contextual: bool = False, label_style: str = "letters"):
        if contextual and not backend.exact:
            raise ValueError("contextual calibration needs a backend with exact log-probabilities")
        self.backend = backend
        self.calibrator = calibrator or Calibrator()
        self.contextual = contextual
        self.label_style = label_style

    @property
    def model_name(self) -> str:
        return self.backend.model_name

    def score_tasks(self, input_text: str, tasks: list[ScoringTask]) -> tuple[list[np.ndarray], BackendResult]:
        """Calibrated label distributions for each task."""
        res = self.backend.score(input_text, tasks)
        probs = [self.calibrator.apply(t.question.type, s.log_scores) for t, s in zip(tasks, res.scores)]
        if self.contextual:
            cf = self.backend.score(CONTENT_FREE_INPUT, tasks)
            res.usage.total_input_tokens += cf.usage.total_input_tokens
            res.usage.total_tokens += cf.usage.total_tokens
            for i, (t, s) in enumerate(zip(tasks, cf.scores)):
                prior = self.calibrator.apply(t.question.type, s.log_scores)
                p = probs[i] / np.maximum(prior, 1e-9)
                probs[i] = p / p.sum()
        return probs, res

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        t0 = time.perf_counter()
        tasks = build_tasks(request.questions, self.label_style)
        probs, res = self.score_tasks(request.input_text(), tasks)
        answers = {t.key: to_answer(t, p, s, self.backend.name) for t, p, s in zip(tasks, probs, res.scores)}
        return DecisionResponse(
            model=self.model_name, answers=answers, usage=res.usage,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
