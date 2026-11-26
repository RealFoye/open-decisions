import httpx
import pytest
from fastapi.testclient import TestClient

from opendecisions import DecisionsClient, Orchestrator, Stage
from opendecisions.aggregate import TaskScore
from opendecisions.backends.base import Backend, BackendResult
from opendecisions.config import build
from opendecisions.schema import DecisionRequest, Usage
from opendecisions.server import create_app
from opendecisions.service import DecisionService

from conftest import TICKET


class FixedBackend(Backend):
    """Returns preset log-scores per question key; records which keys it saw."""

    def __init__(self, name, table, estimated=False):
        self.name, self.table, self.estimated, self.seen = name, table, estimated, []

    def score(self, input_text, tasks):
        self.seen.append([t.key for t in tasks])
        return BackendResult([TaskScore(self.table[t.key], estimated=self.estimated) for t in tasks],
                             Usage(total_input_tokens=10 * len(tasks), remote_calls=len(tasks)))


QS = {"easy": {"type": "noul", "instructions": "a"}, "hard": {"type": "noul", "instructions": "b"}}


def test_cascade_escalates_only_unsure_answers():
    small = FixedBackend("small", {"easy": [3.0, 0.0], "hard": [0.1, 0.0]})
    big = FixedBackend("big", {"easy": [0.0, 5.0], "hard": [0.0, 4.0]}, estimated=True)
    orch = Orchestrator([Stage(DecisionService(small), min_confidence=0.8), Stage(DecisionService(big))])
    r = orch.decide(DecisionRequest(input="x", questions=QS))
    assert small.seen == [["easy", "hard"]] and big.seen == [["hard"]]
    assert r.answers["easy"].backend == "small" and not r.answers["easy"].escalated
    assert r.answers["hard"].backend == "big" and r.answers["hard"].escalated and r.answers["hard"].estimated
    assert r.answers["hard"].noul < 0.05
    assert r.usage.remote_calls == 3


def test_ensemble_averages():
    a = FixedBackend("a", {"easy": [2.0, 0.0], "hard": [0.0, 0.0]})
    b = FixedBackend("b", {"easy": [0.0, 2.0], "hard": [0.0, 0.0]})
    orch = Orchestrator([Stage(DecisionService(a)), Stage(DecisionService(b))], strategy="ensemble")
    r = orch.decide(DecisionRequest(input="x", questions=QS))
    assert r.answers["easy"].noul == pytest.approx(0.5) and r.answers["easy"].backend == "ensemble"


@pytest.fixture(scope="module")
def http():
    engine = build({"backend": "local", "model": "tiny-random"})
    return TestClient(create_app(engine))


def test_server_and_client(http, questions):
    assert http.get("/healthz").json()["status"] == "ok"
    client = DecisionsClient(client=http)
    r = client.decide(TICKET, questions)
    assert set(r.answers) == set(questions)
    assert r.answers["category"].type == "choice"
    raw = http.post("/v1beta/decisions", json={"input": [{"text": "hi"}], "questions": QS}).json()
    assert set(raw["answers"]) == {"easy", "hard"} and raw["usage"]["total_output_tokens"] == 0


def test_server_rejects_bad_requests(http):
    bad_key = http.post("/v1/decisions", json={"input": "hi", "questions": {"no spaces!": QS["easy"]}})
    assert bad_key.status_code == 422
    one_option = http.post("/v1/decisions", json={"input": "hi", "questions": {
        "q": {"type": "choice", "instructions": "x", "options": {"only": ""}}}})
    assert one_option.status_code == 422


def test_client_surfaces_errors():
    transport = httpx.MockTransport(lambda req: httpx.Response(500, text="boom"))
    c = DecisionsClient(client=httpx.Client(base_url="http://x", transport=transport))
    with pytest.raises(RuntimeError, match="500"):
        c.decide("hi", QS)
