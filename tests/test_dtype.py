import argparse
import json

import pytest
import torch

from opendecisions.backends.local import LocalBackend, resolve_dtype
from opendecisions.cli import _spec_from_args, main
from opendecisions.config import build_service
from opendecisions.testing import tiny_model


@pytest.mark.parametrize("dtype, device, expected", [
    (None, None, torch.float32),
    (None, "cpu", torch.float32),
    (None, torch.device("cpu"), torch.float32),
    (None, "cuda", "auto"),
    (None, "cuda:1", "auto"),
    (None, "mps", "auto"),
    ("auto", None, "auto"),
    ("bfloat16", None, torch.bfloat16),
    ("float16", "cuda", torch.float16),
])
def test_resolve_dtype(dtype, device, expected):
    assert resolve_dtype(dtype, device) == expected


@pytest.mark.parametrize("bad", ["fp32", "Tensor", "nonsense"])
def test_resolve_dtype_rejects_unknown(bad):
    with pytest.raises(ValueError, match="unknown dtype"):
        resolve_dtype(bad, None)


@pytest.fixture(scope="module")
def bf16_checkpoint(tmp_path_factory, tok):
    """A tiny model saved in bf16, like most Hub checkpoints (e.g. Qwen2.5)."""
    path = tmp_path_factory.mktemp("bf16-model")
    tiny_model(len(tok)).to(torch.bfloat16).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


@pytest.mark.parametrize("dtype, expected", [
    (None, torch.float32),          # CPU default: float32, not the checkpoint's bf16
    ("auto", torch.bfloat16),       # explicit: keep the checkpoint dtype
    ("bfloat16", torch.bfloat16),
    ("float32", torch.float32),
])
def test_from_pretrained_dtype_on_cpu(bf16_checkpoint, dtype, expected):
    backend = LocalBackend.from_pretrained(bf16_checkpoint, dtype=dtype, prompt_format="plain")
    assert backend.model.dtype == expected
    assert backend.device.type == "cpu"


def test_cli_passes_dtype_to_spec():
    ns = argparse.Namespace(config=None, backend="local", model="m", device=None, dtype="bfloat16",
                            calibration=None)
    assert _spec_from_args(ns) == {"backend": "local", "model": "m", "dtype": "bfloat16"}
    ns.dtype = None
    assert "dtype" not in _spec_from_args(ns)


def test_config_dtype_for_tiny_backend():
    assert build_service({"backend": "local", "model": "tiny-random"}).backend.model.dtype == torch.float32
    svc = build_service({"backend": "local", "model": "tiny-random", "dtype": "bfloat16"})
    assert svc.backend.model.dtype == torch.bfloat16


def test_calibrate_command_accepts_dtype(tmp_path):
    rows = [{"input": f"Ticket {i}", "question": {"type": "noul", "instructions": "Late."}, "label": bool(i % 2)}
            for i in range(4)]
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    main(["calibrate", "--model", "tiny-random", "--dtype", "float32", "--data", str(data),
          "-o", str(tmp_path / "cal.json")])
    assert (tmp_path / "cal.json").exists()
