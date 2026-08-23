"""
Data: a fixed ~1M-token slice of BabyLM-2026-Strict-Small.

Design notes
------------
* The slice is taken deterministically (first N characters of the concatenated
  train split) so every arm sees byte-identical data. No shuffling at the
  corpus level.
* The tokenizer is trained ONCE on the train slice and cached. Retraining it
  per-arm would make arms incomparable.
* Everything is cached to `--cache-dir` so repeated Kaggle runs skip the
  download and the BPE fit.

Budget accounting
-----------------
`n_tokens` is the number of *unique* corpus tokens. Every arm is given the same
corpus and the same number of token-visits (steps x batch x seq_len). Arms that
reuse tokens (replay) are therefore making an epoch-allocation choice, not
consuming extra data. This is the control that matters.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

DATASET = "BabyLM-community/BabyLM-2026-Strict-Small"
SPECIALS = ["<unk>", "<mask>"]


@dataclass
class DataCfg:
    n_tokens: int = 1_000_000     # target train tokens
    val_tokens: int = 100_000
    vocab_size: int = 4096
    seq_len: int = 256
    cache_dir: str = "./cache"
    dataset: str = DATASET


def _raw_text(cfg: DataCfg, chars_needed: int):
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset)
    splits = list(ds.keys())
    tr = "train" if "train" in splits else splits[0]
    col = next(k for k in ["text", "content", "document", "raw"] if k in ds[tr][0])

    buf, total = [], 0
    for t in ds[tr][col]:
        if not t or not t.strip():
            continue
        buf.append(t)
        total += len(t) + 1
        if total >= chars_needed:
            break
    return "\n".join(buf)


def synthetic(cfg: DataCfg, device="cpu"):
    """Deterministic Markov-ish stream. For loop validation only, never results."""
    rng = np.random.default_rng(0)
    n = cfg.n_tokens
    tr = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        tr[i] = (tr[i - 1] * 7 + rng.integers(0, 5)) % cfg.vocab_size
    va = tr[: cfg.val_tokens].copy()
    return Corpus(cfg, tr, va, None, {"vocab_size": cfg.vocab_size}, device)


def build_or_load(cfg: DataCfg, device="cpu", use_synthetic=False):
    if use_synthetic:
        return synthetic(cfg, device)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    tag = f"v{cfg.vocab_size}_n{cfg.n_tokens}_m{cfg.val_tokens}"
    ids_path = os.path.join(cfg.cache_dir, f"ids_{tag}.npz")
    tok_path = os.path.join(cfg.cache_dir, f"tok_{tag}.json")
    meta_path = os.path.join(cfg.cache_dir, f"meta_{tag}.json")

    if os.path.exists(ids_path) and os.path.exists(tok_path):
        z = np.load(ids_path)
        train_ids, val_ids = z["train"], z["val"]
        meta = json.load(open(meta_path))
    else:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel as BLPre
        from tokenizers.decoders import ByteLevel as BLDec

        # ~4.2 chars/token for byte-level BPE at this vocab; pad generously.
        need = int((cfg.n_tokens + cfg.val_tokens) * 6.0)
        text = _raw_text(cfg, need)

        # Split BEFORE fitting the tokenizer: no val text in the BPE merges.
        split_at = int(len(text) * cfg.n_tokens / (cfg.n_tokens + cfg.val_tokens))
        train_text, val_text = text[:split_at], text[split_at:]

        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = BLPre()
        tok.decoder = BLDec()

        def chunks(s, n=10_000):
            for i in range(0, len(s), n):
                yield s[i : i + n]

        tok.train_from_iterator(
            chunks(train_text),
            BpeTrainer(
                vocab_size=cfg.vocab_size,
                special_tokens=SPECIALS,
                initial_alphabet=BLPre.alphabet(),
                show_progress=False,
            ),
        )
        tok.save(tok_path)

        train_ids = np.array(tok.encode(train_text).ids, dtype=np.int32)[: cfg.n_tokens]
        val_ids = np.array(tok.encode(val_text).ids, dtype=np.int32)[: cfg.val_tokens]
        np.savez_compressed(ids_path, train=train_ids, val=val_ids)
        meta = {"vocab_size": tok.get_vocab_size(), "dataset": cfg.dataset}
        json.dump(meta, open(meta_path, "w"))

    return Corpus(cfg, train_ids, val_ids, tok_path, meta, device)


class Corpus:
    def __init__(self, cfg, train_ids, val_ids, tok_path, meta, device):
        self.cfg = cfg
        self.device = device
        self.vocab_size = meta["vocab_size"]
        self.tok_path = tok_path
        self.train = torch.from_numpy(train_ids.astype(np.int64)).to(device)
        self.val = torch.from_numpy(val_ids.astype(np.int64)).to(device)
        # Non-overlapping spans, used as the unit of replay accounting.
        self.span_len = cfg.seq_len
        self.n_spans = (self.train.numel() - 1) // self.span_len
        self.mask_id = SPECIALS.index("<mask>")

    # --------------------------------------------------------- tokenizer
    def tokenizer(self):
        from tokenizers import Tokenizer

        return Tokenizer.from_file(self.tok_path)

    # --------------------------------------------------------- batching
    def spans_to_batch(self, span_ids):
        """Deterministic: span i -> tokens [i*L, (i+1)*L + 1)."""
        L = self.span_len
        starts = span_ids.to(self.train.device) * L
        off = torch.arange(L + 1, device=self.train.device)
        seq = self.train[starts[:, None] + off[None, :]]
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous(), starts

    def random_spans(self, batch_size, generator=None):
        return torch.randint(
            0, self.n_spans, (batch_size,), device=self.train.device, generator=generator
        )

    def val_batch(self, batch_size, generator=None):
        L = self.span_len
        n = self.val.numel() - L - 1
        si = torch.randint(0, n, (batch_size,), device=self.val.device, generator=generator)
        off = torch.arange(L + 1, device=self.val.device)
        seq = self.val[si[:, None] + off[None, :]]
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

    def val_stream(self, batch_size):
        """Deterministic, non-overlapping sweep of the whole val set."""
        L = self.span_len
        n = (self.val.numel() - 1) // L
        for i in range(0, n, batch_size):
            js = torch.arange(i, min(i + batch_size, n), device=self.val.device)
            off = torch.arange(L + 1, device=self.val.device)
            seq = self.val[js[:, None] * L + off[None, :]]
            yield seq[:, :-1].contiguous(), seq[:, 1:].contiguous()
