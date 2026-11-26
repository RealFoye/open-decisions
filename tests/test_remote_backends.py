import json
import math
from types import SimpleNamespace

import httpx
import pytest

from opendecisions.backends.anthropic_backend import AnthropicBackend
from opendecisions.backends.openai_compat import OpenAICompatBackend
from opendecisions.schema import DecisionRequest
from opendecisions.service import DecisionService

QS = {
    "spam": {"type": "noul", "instructions": "The message is spam."},
    "lang": {"type": "choice", "instructions": "Language?", "options": {"en": "", "zh": "", "fr": ""}},
}


def _openai_transport(with_logprobs: bool):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        user = body["messages"][1]["content"]
        assert user.startswith("### Input\nhello world\n\n")  # shared part first, for prefix caching
        usage = {"prompt_tokens": 50, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 32}}
        if body.get("logprobs") and with_logprobs:
            if "Yes or No" in user:
                top = [{"token": "No", "logprob": math.log(0.7)}, {"token": "Yes", "logprob": math.log(0.2)},
                       {"token": " yes", "logprob": math.log(0.05)}]
            else:
                top = [{"token": "B", "logprob": math.log(0.9)}, {"token": "A", "logprob": math.log(0.06)}]
            choice = {"message": {"content": top[0]["token"]}, "logprobs": {"content": [{"token": top[0]["token"], "top_logprobs": top}]}}
            return httpx.Response(200, json={"choices": [choice], "usage": usage})
        if body.get("logprobs"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "A"}, "logprobs": None}], "usage": usage})
        n = body["n"]
        answers = ["A"] * (n - 2) + ["B", "nonsense"]
        return httpx.Response(200, json={"choices": [{"message": {"content": a}} for a in answers], "usage": usage})

    return httpx.MockTransport(handler), calls


def test_openai_logprobs():
    transport, calls = _openai_transport(True)
    b = OpenAICompatBackend("m", base_url="http://x/v1", api_key="k", client=httpx.Client(transport=transport))
    r = DecisionService(b).decide(DecisionRequest(input="hello world", questions=QS))
    spam = r.answers["spam"]
    assert spam.noul == pytest.approx(0.25 / 0.95, abs=1e-4)  # " yes" merged into "Yes"
    lang = r.answers["lang"]
    assert lang.choice == "fr"  # options sorted: en=A, fr=B, zh=C
    assert lang.probabilities["fr"] > 0.9 and lang.probabilities["zh"] < lang.probabilities["en"]
    assert not spam.estimated and spam.method == "logprob"
    assert r.usage.remote_calls == 2 and r.usage.prefix_tokens_reused == 64
    assert all(c["max_tokens"] == 1 and c["top_logprobs"] == 20 for c in calls)


def test_openai_falls_back_to_sampling_without_logprobs():
    transport, _ = _openai_transport(False)
    b = OpenAICompatBackend("m", base_url="http://x/v1", api_key="", fallback_samples=10,
                            client=httpx.Client(transport=transport))
    r = DecisionService(b).decide(DecisionRequest(input="hello world", questions=QS))
    lang = r.answers["lang"]
    assert lang.estimated and lang.method == "sampling"
    # 8 x A, 1 x B, 1 invalid; Dirichlet(0.5): (8.5, 1.5, 0.5) / 10.5
    assert lang.probabilities["en"] == pytest.approx(8.5 / 10.5, abs=1e-4)
    assert r.usage.remote_calls == 4


class _FakeMessages:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        text = self.replies(kw, len(self.calls))
        usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=40,
                                cache_creation_input_tokens=0)
        return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=text)],
                               usage=usage, stop_reason="end_turn")


def test_anthropic_sampling():
    def replies(kw, n):
        labels = kw["output_config"]["format"]["schema"]["properties"]["answer"]["enum"]
        return json.dumps({"answer": labels[0] if n % 4 else labels[1]})

    fake = _FakeMessages(replies)
    b = AnthropicBackend(samples=4, client=SimpleNamespace(messages=fake), concurrency=1)
    r = DecisionService(b).decide(DecisionRequest(input="hello world", questions=QS))
    assert all(a.estimated and a.method == "sampling" for a in r.answers.values())
    first = fake.calls[0]
    assert first["model"] == "claude-opus-5-5"
    blocks = first["messages"][0]["content"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"} and blocks[0]["text"].startswith("### Input")
    assert "cache_control" not in blocks[1]
    assert "temperature" not in first
    assert len(fake.calls) == 8 and r.usage.prefix_tokens_reused == 8 * 40
    assert 0 < r.answers["spam"].noul < 1


def test_anthropic_verbalized():
    def replies(kw, n):
        labels = kw["output_config"]["format"]["schema"]["properties"]["probabilities"]["required"]
        return json.dumps({"probabilities": {lab: (3.0 if i == 0 else 1.0) for i, lab in enumerate(labels)}})

    fake = _FakeMessages(replies)
    b = AnthropicBackend(method="verbalized", client=SimpleNamespace(messages=fake))
    r = DecisionService(b).decide(DecisionRequest(input="hello world", questions=QS))
    assert r.answers["lang"].probabilities["en"] == pytest.approx(0.6, abs=1e-4)
    assert r.answers["spam"].noul == pytest.approx(0.75, abs=1e-4)
    assert len(fake.calls) == 2 and all(a.method == "verbalized" for a in r.answers.values())
