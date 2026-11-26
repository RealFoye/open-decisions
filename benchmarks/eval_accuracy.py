"""Accuracy and calibration of a local model on public labelled datasets.

    pip install -e ".[local]" pyarrow
    python benchmarks/eval_accuracy.py --model Qwen/Qwen2.5-0.5B-Instruct --n 400

One task per question type, each sampled from a public Hugging Face dataset:

    noul    SST-2 (validation)         "The movie review is positive."
    choice  AG News (test)             topic: business / sci_tech / sports / world
    score   Yelp Review Full (test)    star rating, levels 0-4 = 1-5 stars

For each task ``--n`` examples are sampled (seed 0) and split in half: the first half
fits the calibrator (temperature scaling; Platt scaling for noul), the second half is
the test set. On the test half the script prints accuracy, NLL and ECE for the raw
probabilities, for label-free contextual calibration and for the fitted calibrator,
next to a majority-class baseline; for score also the mean absolute error of the
expected level. ``--save-rows`` writes the sampled rows as calibration JSONL, usable
with ``opendecisions calibrate --data``.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from opendecisions.calibration import Calibrator, expected_calibration_error
from opendecisions.cli import collect_examples
from opendecisions.config import build_service
from opendecisions.service import CONTENT_FREE_INPUT

AG_TOPICS = ["world", "sports", "business", "sci_tech"]  # AG News label ids 0..3


def _load(repo: str, path: str, text_col: str = "text") -> list[dict]:
    table = pq.read_table(hf_hub_download(repo, path, repo_type="dataset"), columns=[text_col, "label"])
    return table.to_pylist()


def sst2_rows(n: int, rng: random.Random) -> list[dict]:
    data = _load("stanfordnlp/sst2", "data/validation-00000-of-00001.parquet", "sentence")
    q = {"type": "noul", "instructions": "The movie review is positive."}
    return [{"input": r["sentence"], "question": q, "label": bool(r["label"])} for r in rng.sample(data, n)]


def agnews_rows(n: int, rng: random.Random) -> list[dict]:
    data = _load("fancyzhx/ag_news", "data/test-00000-of-00001.parquet")
    q = {"type": "choice", "instructions": "What is the topic of this news article?",
         "options": {"world": "world news and politics", "sports": "sports",
                     "business": "business and the economy", "sci_tech": "science and technology"}}
    return [{"input": r["text"], "question": q, "label": AG_TOPICS[r["label"]]} for r in rng.sample(data, n)]


def yelp_rows(n: int, rng: random.Random) -> list[dict]:
    data = _load("Yelp/yelp_review_full", "yelp_review_full/test-00000-of-00001.parquet")
    q = {"type": "score", "instructions": "How many stars did the reviewer give?",
         "levels": ["1 star (terrible)", "2 stars (poor)", "3 stars (average)",
                    "4 stars (good)", "5 stars (excellent)"]}
    return [{"input": r["text"], "question": q, "label": str(r["label"])} for r in rng.sample(data, n)]


TASKS = [("noul", "SST-2", sst2_rows), ("choice", "AG News", agnews_rows), ("score", "Yelp stars", yelp_rows)]


def metrics(cal: Calibrator, examples) -> dict:
    conf, correct, nll, err = [], [], [], []
    for qt, z, y in examples:
        p = cal.apply(qt, z)
        k = int(np.argmax(p))
        conf.append(float(p[k]))
        correct.append(k == y)
        nll.append(-np.log(max(float(p[y]), 1e-12)))
        if qt == "score":
            err.append(abs(float(p @ np.arange(len(p))) - y))
    return {"acc": float(np.mean(correct)), "nll": float(np.mean(nll)),
            "ece": expected_calibration_error(conf, correct), "mae": float(np.mean(err)) if err else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=400, help="examples per task (half calibrate, half test)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-rows", default=None, help="write the sampled rows as calibration JSONL")
    args = ap.parse_args()

    service = build_service({"backend": "local", "model": args.model, "device": args.device, "dtype": args.dtype})
    print(f"model: {args.model} ({service.backend.model.dtype}, {service.backend.device}); "
          f"{args.n} examples per task, seed {args.seed}\n")

    print("| type | dataset | test n | majority baseline | method | accuracy | NLL | ECE | MAE (score) |")
    print("|---|---|---:|---:|---|---:|---:|---:|---:|")
    all_rows, fitted, timing = [], {}, {}
    for qtype, dataset, make in TASKS:
        rows = make(args.n, random.Random(args.seed))
        all_rows += rows
        t0 = time.perf_counter()
        ex = collect_examples(service, rows)
        timing[qtype] = (time.perf_counter() - t0) / len(ex)
        half = len(ex) // 2
        fit, test = ex[:half], ex[half:]
        fitted[qtype] = cal = Calibrator.fit(fit)
        # Contextual calibration (label-free): divide by the answer distribution the model
        # gives for the same question on a content-free input, as DecisionService(contextual=True).
        (_, prior, _), = collect_examples(service, [dict(rows[0], input=CONTENT_FREE_INPUT)])
        log_prior = np.log(Calibrator().apply(qtype, prior))
        contextual = [(qt, list(np.asarray(z) - log_prior), y) for qt, z, y in test]
        counts = np.bincount([y for _, _, y in test])
        methods = [("raw", Calibrator(), test),
                   ("contextual (no labels)", Calibrator(), contextual),
                   (f"fitted on {len(fit)} labels", cal, test)]
        for name, c, data in methods:
            m = metrics(c, data)
            mae = f"{m['mae']:.3f}" if m["mae"] is not None else ""
            print(f"| `{qtype}` | {dataset} | {len(test)} | {counts.max() / counts.sum():.3f} | {name} | "
                  f"{m['acc']:.3f} | {m['nll']:.3f} | {m['ece']:.3f} | {mae} |", flush=True)

    print("\nms per example (one question per input):",
          ", ".join(f"{k} {v * 1e3:.0f}" for k, v in timing.items()))
    print("fitted calibrators:", json.dumps({k: v.to_dict() for k, v in fitted.items()}))

    if args.save_rows:
        with open(args.save_rows, "w") as f:
            for r in all_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"saved {len(all_rows)} rows to {args.save_rows}")


if __name__ == "__main__":
    main()
