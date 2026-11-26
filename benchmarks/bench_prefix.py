"""Naive per-question scoring vs shared-prefix batched scoring.

    python benchmarks/bench_prefix.py                      # offline, random-weight model
    python benchmarks/bench_prefix.py --model Qwen/Qwen2.5-0.5B-Instruct --questions 10 50

"naive" is what you get by sending every question as its own full prompt (one forward
pass over input + question each). "shared" prefills the input once and scores the
questions in batches on top of its KV cache. "shared (warm)" repeats the request, so
the prefix comes from the cross-request prefix cache.

Speed does not depend on the weights, so the default random-weight model gives
representative timings for its size without downloading anything.
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from opendecisions.backends.local import LocalBackend
from opendecisions.prompts import build_tasks
from opendecisions.schema import DecisionRequest

WORDS = ("order parcel refund late broken invoice account password delivery support "
         "customer warehouse address payment card shipped damaged lamp chair replace").split()


def make_input(n_words: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def make_questions(n: int) -> dict:
    qs = {}
    for i in range(n):
        kind = i % 3
        if kind == 0:
            qs[f"q{i}"] = {"type": "noul", "instructions": f"The text mentions problem number {i}."}
        elif kind == 1:
            qs[f"q{i}"] = {"type": "choice", "instructions": f"Which team should handle aspect {i}?",
                           "options": {"billing": "", "shipping": "", "returns": "", "other": ""}}
        else:
            qs[f"q{i}"] = {"type": "score", "instructions": f"Rate urgency of aspect {i}.",
                           "levels": ["none", "low", "medium", "high", "critical"]}
    return qs


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF model id; default: offline random Llama")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--input-words", type=int, default=1500)
    ap.add_argument("--questions", type=int, nargs="+", default=[10, 50, 100])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.model:
        shared = LocalBackend.from_pretrained(args.model, batch_size=args.batch_size)
        model, tok, name = shared.model, shared.tokenizer, args.model
    else:
        from opendecisions.testing import tiny_model, tiny_tokenizer

        tok = tiny_tokenizer()
        model = tiny_model(len(tok), hidden=args.hidden, layers=args.layers, heads=8, kv_heads=4)
        name = f"random Llama ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)"

    fmt = "auto" if args.model else "plain"
    text = make_input(args.input_words)
    print(f"model: {name}; torch threads: {torch.get_num_threads()}")
    print(f"input: {len(tok(text)['input_ids'])} tokens\n")
    print("| questions | naive (1 call / question) | shared prefix, cold | shared prefix, warm | speed-up (cold) | tokens computed naive → shared | max |Δ log p| |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for n in args.questions:
        tasks = build_tasks(DecisionRequest(input=text, questions=make_questions(n)).questions)
        naive = LocalBackend(model, tok, prompt_format=fmt, share_prefix=False, batch_size=1)
        shared = LocalBackend(model, tok, prompt_format=fmt, batch_size=args.batch_size)
        rn, tn = timed(lambda: naive.score(text, tasks))
        rc, tc = timed(lambda: shared.score(text, tasks))
        rw, tw = timed(lambda: shared.score(text, tasks))
        diff = max(float(np.max(np.abs(np.array(a.log_scores) - np.array(b.log_scores))))
                   for a, b in zip(rn.scores, rc.scores))
        print(f"| {n} | {tn:.2f} s | {tc:.2f} s | {tw:.2f} s | {tn / tc:.1f}× | "
              f"{rn.usage.total_input_tokens:,} → {rc.usage.total_input_tokens:,} | {diff:.1e} |")


if __name__ == "__main__":
    main()
