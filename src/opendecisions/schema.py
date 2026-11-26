"""Request / response schema.

The shape follows the public Gemini Decisions API closely (a shared ``input`` plus a
map of ``questions``, typed ``answers`` with calibrated probabilities and a
``confidence``), so code written against one is easy to port to the other. It is not
a byte-for-byte clone: options and levels are given explicitly as ``options`` /
``levels`` instead of a free-form ``criteria`` blob, and only text input is supported.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_QUESTIONS = 1000
MAX_OPTIONS = 255
KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- questions


class NoulQuestion(_Base):
    """A yes/no question. ``noul`` in the answer is P(statement is true)."""

    type: Literal["noul"] = "noul"
    instructions: str = Field(min_length=1)
    true_description: str | None = None
    false_description: str | None = None


class ChoiceQuestion(_Base):
    """Pick exactly one option. ``options`` maps option key -> description."""

    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1)
    options: dict[str, str]

    @field_validator("options")
    @classmethod
    def _check_options(cls, v: dict[str, str]) -> dict[str, str]:
        if not 2 <= len(v) <= MAX_OPTIONS:
            raise ValueError(f"choice questions need 2..{MAX_OPTIONS} options, got {len(v)}")
        for k in v:
            if not 1 <= len(k) <= 128:
                raise ValueError(f"option key {k!r} must be 1..128 characters")
        return v


class ScoreQuestion(_Base):
    """Rate on an ordered scale. ``levels[i]`` describes level ``i`` (0-indexed)."""

    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1)
    levels: list[str]

    @field_validator("levels")
    @classmethod
    def _check_levels(cls, v: list[str]) -> list[str]:
        if not 2 <= len(v) <= MAX_OPTIONS:
            raise ValueError(f"score questions need 2..{MAX_OPTIONS} levels, got {len(v)}")
        return v


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class TextPart(_Base):
    text: str


class DecisionRequest(_Base):
    model: str | None = None
    input: str | list[TextPart]
    questions: dict[str, Question]

    @field_validator("questions")
    @classmethod
    def _check_questions(cls, v: dict[str, Question]) -> dict[str, Question]:
        if not 1 <= len(v) <= MAX_QUESTIONS:
            raise ValueError(f"need 1..{MAX_QUESTIONS} questions, got {len(v)}")
        for k in v:
            if not KEY_RE.match(k):
                raise ValueError(f"question key {k!r} must match [A-Za-z0-9_-]{{1,64}}")
        return v

    def input_text(self) -> str:
        if isinstance(self.input, str):
            return self.input
        return "\n\n".join(p.text for p in self.input)


# --------------------------------------------------------------------------- answers

Method = Literal["logprob", "sampling", "verbalized"]


class _AnswerBase(_Base):
    confidence: float = Field(description="Estimated probability that the top answer is right.")
    method: Method = "logprob"
    estimated: bool = Field(
        default=False,
        description="True when probabilities were estimated (sampling / verbalized) "
        "because the backend exposes no token log-probabilities.",
    )
    backend: str | None = None
    escalated: bool = False


class NoulAnswer(_AnswerBase):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(_AnswerBase):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]


class ScoreAnswer(_AnswerBase):
    type: Literal["score"] = "score"
    score: float = Field(description="Probability-weighted mean level index.")
    legend: dict[str, str]
    probabilities: dict[str, float]


Answer = Annotated[Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")]


class Usage(_Base):
    total_input_tokens: int = 0
    prefix_tokens: int = Field(0, description="Tokens in the shared prefix.")
    prefix_tokens_reused: int = Field(
        0, description="Prefix tokens that did not have to be recomputed (shared across "
        "questions, or served from the cross-request prefix cache)."
    )
    prefix_cache_hit: bool = False
    total_output_tokens: int = 0
    total_tokens: int = 0
    remote_calls: int = 0

    @model_validator(mode="after")
    def _total(self) -> "Usage":
        if not self.total_tokens:
            self.total_tokens = self.total_input_tokens + self.total_output_tokens
        return self


class DecisionResponse(_Base):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    latency_ms: float
