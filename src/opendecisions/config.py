"""Build a service or an orchestrator from a JSON config.

Single backend::

    {"backend": "local", "model": "Qwen/Qwen2.5-1.5B-Instruct", "calibration": "calib.json"}

Orchestrator (cascade: local model first, Claude for the questions it is unsure about)::

    {
      "strategy": "cascade",
      "stages": [
        {"backend": "local", "model": "Qwen/Qwen2.5-1.5B-Instruct", "min_confidence": 0.8},
        {"backend": "openai", "model": "gpt-4.1-mini", "min_confidence": 0.8},
        {"backend": "anthropic", "model": "claude-opus-5-5", "method": "sampling", "samples": 8}
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path

from .backends.base import Backend
from .calibration import Calibrator
from .orchestrator import Orchestrator, Stage
from .service import DecisionService

_STAGE_KEYS = {"min_confidence", "weight", "calibration", "contextual"}


def build_backend(spec: dict) -> Backend:
    spec = dict(spec)
    kind = spec.pop("backend", "local")
    for k in _STAGE_KEYS:
        spec.pop(k, None)
    if kind == "local":
        from .backends.local import LocalBackend

        model = spec.pop("model")
        if model == "tiny-random":
            from .testing import tiny_backend

            return tiny_backend(**spec)
        return LocalBackend.from_pretrained(model, **spec)
    if kind == "openai":
        from .backends.openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(spec.pop("model"), **spec)
    if kind == "anthropic":
        from .backends.anthropic_backend import AnthropicBackend

        return AnthropicBackend(**spec)
    raise ValueError(f"unknown backend {kind!r} (expected local, openai or anthropic)")


def build_service(spec: dict) -> DecisionService:
    cal = Calibrator.load(spec["calibration"]) if spec.get("calibration") else None
    return DecisionService(build_backend(spec), cal, contextual=bool(spec.get("contextual", False)))


def build(spec: dict) -> DecisionService | Orchestrator:
    if "stages" in spec:
        stages = [Stage(build_service(s), float(s.get("min_confidence", 0.0)), float(s.get("weight", 1.0)))
                  for s in spec["stages"]]
        return Orchestrator(stages, strategy=spec.get("strategy", "cascade"))
    return build_service(spec)


def load(path: str | Path) -> DecisionService | Orchestrator:
    return build(json.loads(Path(path).read_text()))
