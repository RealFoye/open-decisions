import numpy as np
import pytest

from opendecisions.backends.local import LocalBackend
from opendecisions.prompts import build_tasks
from opendecisions.schema import DecisionRequest
from opendecisions.service import DecisionService

from conftest import TICKET


def _scores(backend, tasks):
    return [np.array(s.log_scores) for s in backend.score(TICKET, tasks).scores]


@pytest.mark.parametrize("label_style", ["letters", "keys"])
@pytest.mark.parametrize("batch_size", [1, 3, 64])
def test_shared_prefix_matches_naive(model, tok, questions, label_style, batch_size):
    req = DecisionRequest(input=TICKET, questions=questions)
    tasks = build_tasks(req.questions, label_style)
    shared = LocalBackend(model, tok, prompt_format="plain", batch_size=batch_size)
    naive = LocalBackend(model, tok, prompt_format="plain", share_prefix=False, batch_size=batch_size)
    for a, b in zip(_scores(shared, tasks), _scores(naive, tasks)):
        np.testing.assert_allclose(a, b, atol=1e-4)


def test_multi_token_labels_are_scored(model, tok, questions):
    req = DecisionRequest(input=TICKET, questions=questions)
    tasks = build_tasks(req.questions, "keys")
    backend = LocalBackend(model, tok, prompt_format="plain")
    assert any(len(backend._label_ids(lab)) > 1 for t in tasks for lab in t.labels)
    for s in backend.score(TICKET, tasks).scores:
        assert all(np.isfinite(s.log_scores))


def test_prefix_cache_reused_across_requests(model, tok, questions):
    req = DecisionRequest(input=TICKET, questions=questions)
    tasks = build_tasks(req.questions)
    backend = LocalBackend(model, tok, prompt_format="plain")
    first = backend.score(TICKET, tasks).usage
    second = backend.score(TICKET, tasks).usage
    assert not first.prefix_cache_hit and second.prefix_cache_hit
    assert second.total_input_tokens < first.total_input_tokens
    assert first.prefix_tokens_reused == first.prefix_tokens * (len(tasks) - 1)
    assert second.prefix_tokens_reused == second.prefix_tokens * len(tasks)


def test_shared_prefix_computes_fewer_tokens(model, tok, questions):
    req = DecisionRequest(input=TICKET, questions=questions)
    tasks = build_tasks(req.questions)
    shared = LocalBackend(model, tok, prompt_format="plain").score(TICKET, tasks).usage
    naive = LocalBackend(model, tok, prompt_format="plain", share_prefix=False).score(TICKET, tasks).usage
    assert shared.total_input_tokens < naive.total_input_tokens / 2


def test_service_answers(model, tok, questions):
    svc = DecisionService(LocalBackend(model, tok, prompt_format="plain"))
    r = svc.decide(DecisionRequest(input=TICKET, questions=questions))
    assert list(r.answers) == list(questions)
    a = r.answers["unhappy"]
    assert a.type == "noul" and 0.01 <= a.noul <= 0.99 and a.confidence == pytest.approx(max(a.noul, 1 - a.noul))
    c = r.answers["category"]
    assert c.choice in questions["category"]["options"]
    assert sum(c.probabilities.values()) == pytest.approx(1, abs=1e-4)
    assert c.confidence == pytest.approx(max(c.probabilities.values()), abs=1e-6)
    s = r.answers["severity"]
    assert 0 <= s.score <= 11 and len(s.probabilities) == 12 and s.legend["0"] == "level 0"
    assert s.score == pytest.approx(sum(int(k) * v for k, v in s.probabilities.items()), abs=1e-3)
    assert not any(x.estimated for x in r.answers.values())


def test_contextual_calibration_runs(model, tok, questions):
    svc = DecisionService(LocalBackend(model, tok, prompt_format="plain"), contextual=True)
    r = svc.decide(DecisionRequest(input=TICKET, questions=questions))
    assert sum(r.answers["category"].probabilities.values()) == pytest.approx(1, abs=1e-4)


def test_chat_template_prompt(model, tok, questions):
    tok_chat = tok.__class__(tokenizer_object=tok.backend_tokenizer, bos_token="<s>", eos_token="</s>",
                             pad_token="<pad>", unk_token="<unk>")
    tok_chat.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}</s>{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    b = LocalBackend(model, tok_chat)  # auto -> chat
    assert b.prompt_format == "chat"
    head, tail = b._split_prompt("hello")
    assert head.endswith("hello\n\n") and tail.startswith("</s><|assistant|>")
    req = DecisionRequest(input=TICKET, questions=questions)
    tasks = build_tasks(req.questions)
    naive = LocalBackend(model, tok_chat, share_prefix=False)
    for x, y in zip(_scores(b, tasks), _scores(naive, tasks)):
        np.testing.assert_allclose(x, y, atol=1e-4)
