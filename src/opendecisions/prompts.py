"""Turn typed questions into scoring tasks.

Every question becomes a *scoring task*: a question block that follows the shared
input, plus a closed set of answer labels. A backend only has to return one
log-score per label; turning those into typed answers is backend-independent
(see :mod:`opendecisions.aggregate`).

The prompt is split so that everything question-independent comes first::

    [SYSTEM + INPUT]  [QUESTION BLOCK]  [" " + LABEL]
     shared prefix     per question      continuation

That ordering is what lets the local engine prefill the prefix once and what lets
remote providers hit their automatic prompt caches.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

from .schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

SYSTEM_PROMPT = (
    "You are a careful, impartial judge. Read the input, then answer the question "
    "with exactly one of the allowed option labels and nothing else."
)

NOUL_KEYS = ("true", "false")


@dataclass
class ScoringTask:
    key: str
    question: Question
    text: str                      # question block (without the shared input)
    labels: list[str]              # what the model is asked to output, e.g. "A", "Yes", "3"
    answer_keys: list[str]         # what each label means in the answer, e.g. option keys
    meta: dict = field(default_factory=dict)


def _letter_labels(n: int) -> list[str]:
    if n <= 26:
        return list(string.ascii_uppercase[:n])
    return [str(i + 1) for i in range(n)]


def build_task(key: str, q: Question, label_style: str = "letters") -> ScoringTask:
    if isinstance(q, NoulQuestion):
        lines = [f"Statement: {q.instructions}"]
        if q.true_description:
            lines.append(f"Answer Yes if: {q.true_description}")
        if q.false_description:
            lines.append(f"Answer No if: {q.false_description}")
        lines.append("Is the statement true for the input? Answer Yes or No.")
        return ScoringTask(key, q, "\n".join(lines), ["Yes", "No"], list(NOUL_KEYS))

    if isinstance(q, ChoiceQuestion):
        # Same convention as Gemini Decisions: options sorted by UTF-8 bytes.
        keys = sorted(q.options, key=lambda k: k.encode("utf-8"))
        labels = keys if label_style == "keys" else _letter_labels(len(keys))
        lines = [f"Question: {q.instructions}", "Options:"]
        for lab, k in zip(labels, keys):
            desc = q.options[k]
            if label_style == "keys":
                lines.append(f"{lab}: {desc}" if desc else lab)
            else:
                lines.append(f"{lab}) {k}: {desc}" if desc else f"{lab}) {k}")
        lines.append("Answer with the option label only.")
        return ScoringTask(key, q, "\n".join(lines), labels, keys)

    if isinstance(q, ScoreQuestion):
        labels = [str(i) for i in range(len(q.levels))]
        lines = [f"Question: {q.instructions}", "Rating scale:"]
        lines += [f"{i}: {desc}" for i, desc in enumerate(q.levels)]
        lines.append("Answer with the level number only.")
        return ScoringTask(key, q, "\n".join(lines), labels, labels)

    raise TypeError(f"unsupported question type: {type(q).__name__}")


def build_tasks(questions: dict[str, Question], label_style: str = "letters") -> list[ScoringTask]:
    return [build_task(k, q, label_style) for k, q in questions.items()]


# ---- plain-text layout (used by base models, and as the remote user message) ----


def input_block(input_text: str) -> str:
    return f"### Input\n{input_text}\n\n"


def question_block(task: ScoringTask) -> str:
    return f"### Question\n{task.text}\n\n### Answer:"


def plain_prefix(input_text: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n{input_block(input_text)}"
