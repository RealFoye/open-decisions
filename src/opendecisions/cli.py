"""Command line: ``opendecisions serve`` and ``opendecisions calibrate``."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _spec_from_args(args) -> dict:
    if args.config:
        return json.loads(Path(args.config).read_text())
    spec = {"backend": args.backend, "model": args.model}
    if getattr(args, "device", None):
        spec["device"] = args.device
    if getattr(args, "dtype", None):
        spec["dtype"] = args.dtype
    if getattr(args, "calibration", None):
        spec["calibration"] = args.calibration
    return spec


def cmd_serve(args) -> None:
    import uvicorn

    from .config import build
    from .server import create_app

    engine = build(_spec_from_args(args))
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


def _label_index(task, label) -> int:
    if task.question.type == "noul":
        if isinstance(label, bool):
            label = "true" if label else "false"
        label = str(label).lower()
    return task.answer_keys.index(str(label))


def collect_examples(service, rows: list[dict]) -> list[tuple[str, list[float], int]]:
    """Raw (uncalibrated) label log-scores for labelled rows, grouped by shared input."""
    from .prompts import build_task
    from .schema import DecisionRequest

    by_input: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_input[r["input"]].append(r)
    out = []
    for text, group in by_input.items():
        req = DecisionRequest.model_validate(
            {"input": text, "questions": {f"q{i}": g["question"] for i, g in enumerate(group)}})
        tasks = [build_task(k, q, service.label_style) for k, q in req.questions.items()]
        res = service.backend.score(text, tasks)
        for t, s, g in zip(tasks, res.scores, group):
            out.append((t.question.type, s.log_scores, _label_index(t, g["label"])))
    return out


def _report(name: str, cal, examples) -> str:
    from .calibration import expected_calibration_error

    conf, correct, nll = [], [], []
    for qt, z, y in examples:
        p = cal.apply(qt, z)
        conf.append(float(p.max()))
        correct.append(int(np.argmax(p)) == y)
        nll.append(-np.log(max(float(p[y]), 1e-12)))
    return (f"{name}: n={len(examples)} acc={np.mean(correct):.3f} "
            f"nll={np.mean(nll):.4f} ece={expected_calibration_error(conf, correct):.4f}")


def cmd_calibrate(args) -> None:
    from .calibration import Calibrator
    from .config import build_service

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    service = build_service(_spec_from_args(args))
    examples = collect_examples(service, rows)
    random.Random(0).shuffle(examples)
    n_hold = int(len(examples) * args.holdout) if len(examples) >= 10 else 0
    held, train = examples[:n_hold], examples[n_hold:]
    cal = Calibrator.fit(train, platt_for_noul=not args.no_platt)
    cal.save(args.output)
    print(f"saved {args.output}: {json.dumps(cal.to_dict())}")
    for name, ex in (("train", train), ("held-out", held)):
        if ex:
            print(_report(f"{name} before", Calibrator(), ex))
            print(_report(f"{name} after ", cal, ex))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="opendecisions")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", help="JSON config (single backend or orchestrator)")
        sp.add_argument("--backend", default="local", choices=["local", "openai", "anthropic"])
        sp.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
        sp.add_argument("--device", default=None, help="local backend: cpu, cuda, mps, ...")
        sp.add_argument("--dtype", default=None,
                        help="local backend: float32, bfloat16, float16 or auto (checkpoint dtype); "
                             "default float32 on CPU, auto otherwise")

    s = sub.add_parser("serve", help="run the HTTP server")
    common(s)
    s.add_argument("--calibration", default=None)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    c = sub.add_parser("calibrate", help="fit temperature / Platt scaling on labelled JSONL")
    common(c)
    c.add_argument("--data", required=True, help='JSONL rows: {"input", "question", "label"}')
    c.add_argument("--output", "-o", default="calibration.json")
    c.add_argument("--holdout", type=float, default=0.3)
    c.add_argument("--no-platt", action="store_true")
    c.set_defaults(fn=cmd_calibrate)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
