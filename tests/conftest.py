import pytest

from opendecisions.testing import tiny_model, tiny_tokenizer


@pytest.fixture(scope="session")
def tok():
    return tiny_tokenizer()


@pytest.fixture(scope="session")
def model(tok):
    return tiny_model(len(tok))


@pytest.fixture
def questions():
    return {
        "unhappy": {"type": "noul", "instructions": "The customer is unhappy.",
                    "true_description": "complains or asks for a refund"},
        "category": {"type": "choice", "instructions": "Which queue should handle this ticket?",
                     "options": {"refund_request": "wants money back", "shipping_delay": "parcel is late",
                                 "billing_question": "question about an invoice", "other": ""}},
        "severity": {"type": "score", "instructions": "How urgent is this?",
                     "levels": [f"level {i}" for i in range(12)]},  # 10 and 11 are multi-token
    }


TICKET = ("Hello, I ordered a lamp two weeks ago. The parcel arrived late and the lamp was "
          "broken. I want a refund as soon as possible. ") * 8
