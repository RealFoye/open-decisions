"""Local Hugging Face backend with shared-prefix KV reuse.

How one request is scored::

    1. prefill  [system + input]                    -> KV cache P   (once per request,
                                                                     or zero times on a
                                                                     prefix-cache hit)
    2. pass 1   P ⊕ [question_i]  for all i, batched -> next-token distribution at the
                                                         end of each question; this
                                                         scores every single-token label
    3. pass 2   P ⊕ [question_i ⊕ label_ij] for the few multi-token labels only, batched,
                reading log p(label tokens) by teacher forcing

The prefix KV is broadcast across the batch with ``expand`` (no copy). Questions are
left-padded *between* prefix and question; padding is masked out and position ids
are set explicitly, so every row sees exactly the tokens it would see alone. The
result is numerically equivalent to scoring each question with its own full prompt
(``share_prefix=False``), which the test-suite checks.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from collections import OrderedDict
from dataclasses import dataclass

import torch

from ..aggregate import TaskScore
from ..prompts import SYSTEM_PROMPT, ScoringTask, input_block, plain_prefix, question_block
from ..schema import Usage
from .base import Backend, BackendResult

_SENTINEL = "<<<OPENDECISIONS_QUESTION>>>"


@dataclass
class _Prefix:
    kv: list[tuple[torch.Tensor, torch.Tensor]]
    length: int


class LocalBackend(Backend):
    name = "local"
    exact = True

    def __init__(
        self,
        model,
        tokenizer,
        *,
        model_id: str = "local",
        device: str | torch.device | None = None,
        prompt_format: str = "auto",
        batch_size: int = 16,
        share_prefix: bool = True,
        prefix_cache_size: int = 4,
        option_scoring: str = "sum",
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = torch.device(device) if device else next(model.parameters()).device
        self.model.to(self.device)
        if prompt_format == "auto":
            prompt_format = "chat" if getattr(tokenizer, "chat_template", None) else "plain"
        if prompt_format not in ("plain", "chat"):
            raise ValueError("prompt_format must be 'auto', 'plain' or 'chat'")
        if option_scoring not in ("sum", "mean"):
            raise ValueError("option_scoring must be 'sum' or 'mean'")
        self.prompt_format = prompt_format
        self.batch_size = batch_size
        self.share_prefix = share_prefix
        self.option_scoring = option_scoring
        self._prefix_cache: OrderedDict[str, _Prefix] = OrderedDict()
        self._prefix_cache_size = prefix_cache_size
        fwd = inspect.signature(self.model.forward).parameters
        self._cache_kw = "cache" if "cache" in fwd else "past_key_values"
        self._has_logits_to_keep = "logits_to_keep" in fwd

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_pretrained(cls, model_id: str, *, device: str | None = None, dtype: str | None = None,
                        **kwargs) -> "LocalBackend":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        torch_dtype = getattr(torch, dtype) if dtype else None
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
        return cls(model, tok, model_id=model_id, device=device, **kwargs)

    # ------------------------------------------------------------------ prompt text

    def _split_prompt(self, input_text: str) -> tuple[str, str]:
        """Return (prefix, template_tail). The question goes between them."""
        if self.prompt_format == "plain":
            return plain_prefix(input_text), ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_block(input_text) + _SENTINEL},
        ]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        head, tail = rendered.split(_SENTINEL, 1)
        return head, tail

    def _encode(self, text: str, special: bool = False) -> list[int]:
        return self.tokenizer(text, add_special_tokens=special)["input_ids"]

    def _question_ids(self, task: ScoringTask, tail: str) -> list[int]:
        if self.prompt_format == "plain":
            return self._encode(question_block(task))
        # In chat mode the question sits in the user turn; the answer is the assistant turn.
        return self._encode(f"### Question\n{task.text}" + tail + "Answer:")

    def _label_ids(self, label: str) -> list[int]:
        ids = self._encode(" " + label)
        if not ids:
            raise ValueError(f"label {label!r} tokenises to nothing")
        return ids

    # ------------------------------------------------------------------ model calls

    def _prefill(self, ids: list[int]) -> _Prefix:
        x = torch.tensor([ids], device=self.device)
        out = self.model(input_ids=x, use_cache=True, **({"logits_to_keep": 1} if self._has_logits_to_keep else {}))
        cache = getattr(out, "cache", None) or out.past_key_values
        if hasattr(cache, "layers"):
            kv = [(layer.keys, layer.values) for layer in cache.layers]
        else:  # legacy tuple format
            kv = [(k, v) for k, v in cache]
        return _Prefix(kv=kv, length=len(ids))

    def _get_prefix(self, ids: list[int]) -> tuple[_Prefix, bool]:
        key = hashlib.sha1(str(ids).encode()).hexdigest()
        if key in self._prefix_cache:
            self._prefix_cache.move_to_end(key)
            return self._prefix_cache[key], True
        p = self._prefill(ids)
        if self._prefix_cache_size > 0:
            self._prefix_cache[key] = p
            while len(self._prefix_cache) > self._prefix_cache_size:
                self._prefix_cache.popitem(last=False)
        return p, False

    def _make_cache(self, prefix: _Prefix, batch: int):
        from transformers import DynamicCache

        data = [(k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1)) for k, v in prefix.kv]
        try:
            return DynamicCache(ddp_cache_data=data, config=self.model.config)
        except TypeError:
            return DynamicCache.from_legacy_cache(tuple(data))

    def _forward_rows(self, prefix: _Prefix | None, rows: list[list[int]], keep: int) -> torch.Tensor:
        """Run rows (left-padded) after an optional shared prefix.

        Returns log-softmax over the vocabulary for the last ``keep`` positions of
        every row: tensor [len(rows), keep, vocab].
        """
        width = max(len(r) for r in rows)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        plen = prefix.length if prefix else 0
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), plen + width), dtype=torch.long)
        pos = torch.zeros((len(rows), width), dtype=torch.long)
        mask[:, :plen] = 1
        for i, r in enumerate(rows):
            off = width - len(r)
            ids[i, off:] = torch.tensor(r)
            mask[i, plen + off:] = 1
            pos[i, off:] = torch.arange(plen, plen + len(r))
            pos[i, :off] = plen  # padded slots: any valid position, they are masked out
        kwargs = dict(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                      position_ids=pos.to(self.device), use_cache=prefix is not None)
        if prefix is not None:
            kwargs[self._cache_kw] = self._make_cache(prefix, len(rows))
        if self._has_logits_to_keep:
            kwargs["logits_to_keep"] = keep
        logits = self.model(**kwargs).logits[:, -keep:, :]
        return torch.log_softmax(logits.float(), dim=-1)

    # ------------------------------------------------------------------ scoring

    @torch.inference_mode()
    def score(self, input_text: str, tasks: list[ScoringTask]) -> BackendResult:
        head, tail = self._split_prompt(input_text)
        prefix_ids = self._encode(head, special=True)
        q_ids = [self._question_ids(t, tail) for t in tasks]
        label_ids = [[self._label_ids(lab) for lab in t.labels] for t in tasks]
        scores: list[list[float]] = [[-math.inf] * len(t.labels) for t in tasks]

        usage = Usage(prefix_tokens=len(prefix_ids))
        if self.share_prefix:
            prefix, hit = self._get_prefix(prefix_ids)
            usage.prefix_cache_hit = hit
            computed_prefix = 0 if hit else len(prefix_ids)
            usage.prefix_tokens_reused = len(prefix_ids) * len(tasks) - computed_prefix
            run = lambda rows, keep: self._forward_rows(prefix, rows, keep)  # noqa: E731
            base = []
        else:
            computed_prefix = 0
            run = lambda rows, keep: self._forward_rows(None, rows, keep)  # noqa: E731
            base = prefix_ids
        computed = computed_prefix

        # pass 1: one row per question -> first-token distribution after the question
        order = sorted(range(len(tasks)), key=lambda i: len(q_ids[i]))
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            rows = [base + q_ids[i] for i in idx]
            computed += sum(len(r) for r in rows)
            lp = run(rows, 1)[:, -1, :]
            for b, i in enumerate(idx):
                for j, lab in enumerate(label_ids[i]):
                    if len(lab) == 1:
                        scores[i][j] = float(lp[b, lab[0]])

        # pass 2: teacher-force the multi-token labels only
        pairs = [(i, j) for i in range(len(tasks)) for j, lab in enumerate(label_ids[i]) if len(lab) > 1]
        pairs.sort(key=lambda p: len(q_ids[p[0]]) + len(label_ids[p[0]][p[1]]))
        for s in range(0, len(pairs), self.batch_size):
            chunk = pairs[s:s + self.batch_size]
            rows = [base + q_ids[i] + label_ids[i][j] for i, j in chunk]
            computed += sum(len(r) for r in rows)
            keep = max(len(label_ids[i][j]) for i, j in chunk) + 1
            lp = run(rows, keep)
            for b, (i, j) in enumerate(chunk):
                lab = label_ids[i][j]
                n = len(lab)
                # positions -(n+1) .. -2 predict label tokens 0 .. n-1
                total = sum(float(lp[b, keep - 1 - n + t, tok]) for t, tok in enumerate(lab))
                scores[i][j] = total / n if self.option_scoring == "mean" else total

        usage.total_input_tokens = computed
        usage.total_tokens = computed
        return BackendResult([TaskScore(s) for s in scores], usage)
