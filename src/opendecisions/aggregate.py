"""Backend-independent post-processing: log-scores -> calibrated typed answers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .prompts import ScoringTask
from .schema import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer, ScoreQuestion, NoulQuestion

NOUL_CLAMP = (0.01, 0.99)


@dataclass
class TaskScore:
    """What a backend returns for one task: one log-score per label (unnormalised)."""

    log_scores: list[float]
    method: str = "logprob"
    estimated: bool = False


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def to_answer(task: ScoringTask, probs: np.ndarray, score: TaskScore, backend: str | None) -> Answer:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    common = dict(method=score.method, estimated=score.estimated, backend=backend)
    q = task.question

    if isinstance(q, NoulQuestion):
        p_true = float(probs[0])
        lo, hi = NOUL_CLAMP
        p_true = min(max(p_true, lo), hi)
        return NoulAnswer(noul=round(p_true, 6), confidence=round(max(p_true, 1 - p_true), 6), **common)

    probs_map = {k: round(float(p), 6) for k, p in zip(task.answer_keys, probs)}
    top = int(np.argmax(probs))
    confidence = round(float(probs[top]), 6)

    if isinstance(q, ScoreQuestion):
        expected = float(np.dot(np.arange(len(probs)), probs))
        legend = {str(i): d for i, d in enumerate(q.levels)}
        return ScoreAnswer(score=round(expected, 6), legend=legend, probabilities=probs_map,
                           confidence=confidence, **common)

    return ChoiceAnswer(choice=task.answer_keys[top], probabilities=probs_map, confidence=confidence, **common)


def answer_confidence(answer: Answer) -> float:
    return answer.confidence


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))
