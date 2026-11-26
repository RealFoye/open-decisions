from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..aggregate import TaskScore
from ..prompts import ScoringTask
from ..schema import Usage


@dataclass
class BackendResult:
    scores: list[TaskScore]
    usage: Usage = field(default_factory=Usage)


class Backend(ABC):
    """Scores a batch of tasks that share one input.

    ``exact`` backends return true model log-probabilities over the labels; others
    estimate them (sampling, verbalized probabilities) and mark answers as estimated.
    """

    name: str = "backend"
    exact: bool = True

    @property
    def model_name(self) -> str:
        return getattr(self, "model_id", self.name)

    @abstractmethod
    def score(self, input_text: str, tasks: list[ScoringTask]) -> BackendResult: ...
