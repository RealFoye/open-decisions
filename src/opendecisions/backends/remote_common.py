"""Helpers shared by the API backends (OpenAI-compatible, Anthropic)."""

from __future__ import annotations

import math
from collections import Counter

from ..prompts import ScoringTask, input_block


def user_parts(input_text: str, task: ScoringTask) -> tuple[str, str]:
    """(shared part, per-question part) of the user message.

    The shared part goes first and is byte-identical for every question of a request,
    so provider-side prefix caches (OpenAI automatic caching, vLLM/SGLang prefix
    caching, Anthropic ``cache_control``) can reuse it.
    """
    return input_block(input_text), f"### Question\n{task.text}\n\nReply with the label only."


def match_label(token: str, labels: list[str]) -> int | None:
    t = token.strip().strip(".:)\"'").strip()
    if not t:
        return None
    for i, lab in enumerate(labels):
        if t == lab:
            return i
    low = t.lower()
    for i, lab in enumerate(labels):
        if low == lab.lower():
            return i
    return None


def counts_to_log_scores(answers: list[str | None], labels: list[str], alpha: float = 0.5) -> list[float]:
    """Empirical label frequencies with a symmetric Dirichlet(alpha) prior, in log space."""
    idx = [match_label(a, labels) if a is not None else None for a in answers]
    c = Counter(i for i in idx if i is not None)
    total = sum(c.values()) + alpha * len(labels)
    return [math.log((c.get(i, 0) + alpha) / total) for i in range(len(labels))]


def probs_to_log_scores(probs: dict[str, float], labels: list[str], floor: float = 1e-4) -> list[float]:
    vals = [max(float(probs.get(lab, 0.0)), 0.0) for lab in labels]
    s = sum(vals)
    if s <= 0:
        vals = [1.0] * len(labels)
        s = float(len(labels))
    return [math.log(max(v / s, floor)) for v in vals]
