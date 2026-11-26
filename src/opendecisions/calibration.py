"""Post-hoc calibration of label distributions.

Raw LLM label probabilities are usually over-confident. Two standard fixes are
provided, fitted per question type on a small labelled set:

* **temperature scaling** (Guo et al., 2017): ``p ∝ exp(log_score / T)``, one scalar
  ``T`` per question type. Works for any number of options and never changes the
  arg-max, so accuracy is untouched.
* **Platt scaling** for yes/no: ``P(true) = sigmoid(a · (s_yes - s_no) + b)``. The bias
  ``b`` also corrects a model's habit of saying "Yes" too often.

Optionally the service can also apply *contextual calibration* (Zhao et al., 2021):
divide by the label distribution the model gives for a content-free input. That one
needs no labels at all; see :class:`opendecisions.service.DecisionService`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .aggregate import softmax

_TYPES = ("noul", "choice", "score")


def _nll(logits: list[np.ndarray], labels: list[int], t: float) -> float:
    total = 0.0
    for z, y in zip(logits, labels):
        p = softmax(z, t)
        total -= np.log(max(p[y], 1e-12))
    return total / max(len(labels), 1)


def fit_temperature(logits: list[list[float]], labels: list[int], lo: float = 0.05, hi: float = 20.0) -> float:
    """Minimise NLL over T with golden-section search on log T (NLL is unimodal in T)."""
    zs = [np.asarray(z, dtype=np.float64) for z in logits]
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = _nll(zs, labels, np.exp(c)), _nll(zs, labels, np.exp(d))
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = _nll(zs, labels, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = _nll(zs, labels, np.exp(d))
    return float(np.exp((a + b) / 2))


def fit_platt(margins: list[float], y: list[int], iters: int = 200) -> tuple[float, float]:
    """Logistic regression of y on the yes-no margin, by Newton's method."""
    x = np.asarray(margins, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    # Platt's smoothed targets avoid infinite weights on separable data
    n_pos, n_neg = t.sum(), len(t) - t.sum()
    t = np.where(t > 0.5, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    w = np.zeros(2)
    X = np.stack([x, np.ones_like(x)], axis=1)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        grad = X.T @ (p - t)
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, grad)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def expected_calibration_error(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    conf = np.asarray(confidences, dtype=np.float64)
    acc = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


@dataclass
class Calibrator:
    """Per-question-type calibration parameters."""

    temperature: dict[str, float] = field(default_factory=dict)
    platt: tuple[float, float] | None = None   # for noul only, overrides temperature

    def apply(self, qtype: str, log_scores: list[float]) -> np.ndarray:
        z = np.asarray(log_scores, dtype=np.float64)
        if qtype == "noul" and self.platt is not None:
            a, b = self.platt
            p_true = 1 / (1 + np.exp(-(a * (z[0] - z[1]) + b)))
            return np.array([p_true, 1 - p_true])
        return softmax(z, self.temperature.get(qtype, 1.0))

    # -------------------------------------------------------------- fitting / io

    @classmethod
    def fit(cls, examples: list[tuple[str, list[float], int]], platt_for_noul: bool = True) -> "Calibrator":
        """``examples``: (question type, raw label log-scores, index of the correct label)."""
        cal = cls()
        for qt in _TYPES:
            rows = [(z, y) for t, z, y in examples if t == qt]
            if len(rows) >= 2:
                cal.temperature[qt] = fit_temperature([z for z, _ in rows], [y for _, y in rows])
        noul = [(z, y) for t, z, y in examples if t == "noul"]
        if platt_for_noul and len(noul) >= 4 and len({y for _, y in noul}) == 2:
            cal.platt = fit_platt([z[0] - z[1] for z, _ in noul], [1 if y == 0 else 0 for _, y in noul])
        return cal

    def to_dict(self) -> dict:
        return {"temperature": self.temperature, "platt": list(self.platt) if self.platt else None}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(temperature=dict(d.get("temperature", {})),
                   platt=tuple(d["platt"]) if d.get("platt") else None)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        return cls.from_dict(json.loads(Path(path).read_text()))
