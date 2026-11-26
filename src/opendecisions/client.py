"""Small Python client for an open-decisions server.

    >>> from opendecisions import DecisionsClient
    >>> c = DecisionsClient("http://localhost:8000")
    >>> r = c.decide("Ticket: my parcel never arrived", {
    ...     "is_complaint": {"type": "noul", "instructions": "The customer is complaining."},
    ... })
    >>> r.answers["is_complaint"].noul
"""

from __future__ import annotations

from typing import Any

import httpx

from .schema import DecisionRequest, DecisionResponse


class DecisionsClient:
    def __init__(self, base_url: str = "http://localhost:8000", *, timeout: float = 120.0,
                 client: httpx.Client | None = None, path: str = "/v1/decisions"):
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)
        self.path = path

    def decide(self, input: str | list[dict], questions: dict[str, Any], model: str | None = None) -> DecisionResponse:
        req = DecisionRequest.model_validate({"input": input, "questions": questions, "model": model})
        r = self._client.post(self.path, json=req.model_dump(exclude_none=True))
        if r.status_code >= 400:
            raise RuntimeError(f"decisions request failed ({r.status_code}): {r.text}")
        return DecisionResponse.model_validate(r.json())

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "DecisionsClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
