"""Offline tiny model + tokenizer, for tests and CPU benchmarks.

The weights are random, so probabilities are meaningless; what these are good for is
checking that shared-prefix scoring matches naive scoring exactly, and measuring
speed (which does not depend on the weights).
"""

from __future__ import annotations

import string

_CORPUS = [
    "You are a careful, impartial judge. Read the input, then answer the question.",
    "### Input ### Question ### Answer: Statement: Options: Rating scale: Yes No",
    "Answer with the option label only. Answer with the level number only.",
    "The customer asked for a refund because the parcel arrived damaged and late.",
    "Is the statement true for the input? Answer Yes or No.",
    " ".join(string.ascii_uppercase) + " " + " ".join(str(i) for i in range(100)),
    "refund_request billing_question shipping_delay account_access other",
]


def tiny_tokenizer(vocab_size: int = 600):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(_CORPUS * 20, trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>"
    )


def tiny_model(vocab_size: int, *, hidden: int = 64, layers: int = 2, heads: int = 4,
               kv_heads: int = 2, seed: int = 0):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab_size, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv_heads,
        max_position_embeddings=16384, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = LlamaForCausalLM(cfg)
    # random init gives near-uniform outputs; sharpen them so tests compare real numbers
    with torch.no_grad():
        model.lm_head.weight.mul_(20.0)
    return model.float().eval()


def tiny_backend(**kwargs):
    from .backends.local import LocalBackend

    tok = tiny_tokenizer()
    model = tiny_model(len(tok), **{k: kwargs.pop(k) for k in ("hidden", "layers", "heads", "kv_heads", "seed") if k in kwargs})
    kwargs.setdefault("prompt_format", "plain")
    return LocalBackend(model, tok, model_id="tiny-random-llama", **kwargs)
