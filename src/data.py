"""
Data: a fixed ~1M-token slice of BabyLM-2026-Strict-Small.

WHY THE SPLIT LOOKS LIKE THIS
-----------------------------
The first version of this file read rows until it had enough characters, then
took the last 10% as validation. BabyLM concatenates six source files in order
(bnc_spoken, childes, gutenberg, open_subtitles, simple_wiki, switchboard), so
that produced train = adult conversation transcripts and val = pure
child-directed speech, with four of the six domains never read at all. Every
arm then scored at or ABOVE the uniform-random ceiling on val, and the harness
happily ranked twelve arms against each other on the resulting noise.

The fix is a BLOCK-INTERLEAVED split: the whole corpus is read, cut into blocks
of `block_chars`, the blocks are permuted with a fixed seed, and train/val are
drawn from the same permuted stream. Train and val are then guaranteed to carry
the same domain mixture, while blocks stay large enough to preserve local
coherence (default 8192 chars is roughly 8 sequences at seq_len=256).

Two gates run at startup and print loudly:
  * unigram KL(val || train) -- catches any remaining distribution mismatch
  * a trigram anchor fit on train, scored on val -- an absolute floor that a
    competent model must beat. If the neural model cannot beat it, the run is
    broken and no comparison between arms is meaningful.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

DATASET = "BabyLM-community/BabyLM-2026-Strict-Small"
SPECIALS = ["<unk>", "<mask>"]
CHARS_PER_TOKEN = 4.2   # byte-level BPE at vocab 2-8k on this corpus


@dataclass
class DataCfg:
    n_tokens: int = 1_000_000
    val_tokens: int = 100_000
    vocab_size: int = 2048
    seq_len: int = 256
    cache_dir: str = "./cache"
    dataset: str = DATASET
    block_chars: int = 8192       # unit of the train/val split
    split_seed: int = 0


def _all_text(cfg: DataCfg):
    """Read the WHOLE train split. Never stop early -- stopping early is what
    silently reduced this corpus to two of its six domains."""
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset)
    splits = list(ds.keys())
    tr = "train" if "train" in splits else splits[0]
    col = next(k for k in ["text", "content", "document", "raw"] if k in ds[tr][0])
    return "\n".join(t for t in ds[tr][col] if t and t.strip())


def _block_split(text, cfg: DataCfg):
    """Permute fixed-size blocks, then draw train and val from the same stream."""
    B = cfg.block_chars
    n_blocks = len(text) // B
    if n_blocks < 32:
        raise RuntimeError(f"corpus too small for block splitting ({n_blocks} blocks)")

    order = np.random.default_rng(cfg.split_seed).permutation(n_blocks)

    need_train = int(cfg.n_tokens * CHARS_PER_TOKEN * 1.25)
    need_val = int(cfg.val_tokens * CHARS_PER_TOKEN * 1.25)
    n_val_blocks = max(1, int(np.ceil(need_val / B)))
    n_train_blocks = max(1, int(np.ceil(need_train / B)))
    if n_val_blocks + n_train_blocks > n_blocks:
        raise RuntimeError(
            f"asked for {n_train_blocks + n_val_blocks} blocks, corpus has {n_blocks}. "
            f"Reduce --n-tokens or --val-tokens."
        )

    val_idx = order[:n_val_blocks]
    train_idx = order[n_val_blocks : n_val_blocks + n_train_blocks]
    grab = lambda ix: "".join(text[i * B : (i + 1) * B] for i in ix)
    return grab(train_idx), grab(val_idx), dict(
        n_blocks_total=int(n_blocks),
        n_blocks_train=int(n_train_blocks),
        n_blocks_val=int(n_val_blocks),
        block_chars=B,
        corpus_chars=len(text),
    )


def unigram_kl(train_ids, val_ids, V):
    """KL(val || train) over unigram token distributions, in nats.

    A domain split shows up here as a large value. Same-distribution splits on
    this corpus land around 0.01-0.05.
    """
    tc = np.bincount(train_ids, minlength=V).astype(np.float64) + 1.0
    vc = np.bincount(val_ids, minlength=V).astype(np.float64) + 1.0
    p, q = vc / vc.sum(), tc / tc.sum()
    return float((p * np.log(p / q)).sum())


def synthetic(cfg: DataCfg, device="cpu"):
    """Deterministic Markov-ish stream. Validates the loop offline. Never a result."""
    rng = np.random.default_rng(0)
    n = cfg.n_tokens
    tr = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        tr[i] = (tr[i - 1] * 7 + rng.integers(0, 5)) % cfg.vocab_size
    va = tr[: cfg.val_tokens].copy()
    return Corpus(cfg, tr, va, None, {"vocab_size": cfg.vocab_size, "synthetic": True}, device)


def build_or_load(cfg: DataCfg, device="cpu", use_synthetic=False):
    if use_synthetic:
        return synthetic(cfg, device)

    os.makedirs(cfg.cache_dir, exist_ok=True)
    tag = f"v{cfg.vocab_size}_n{cfg.n_tokens}_m{cfg.val_tokens}_b{cfg.block_chars}_s{cfg.split_seed}"
    ids_path = os.path.join(cfg.cache_dir, f"ids_{tag}.npz")
    tok_path = os.path.join(cfg.cache_dir, f"tok_{tag}.json")
    meta_path = os.path.join(cfg.cache_dir, f"meta_{tag}.json")

    if os.path.exists(ids_path) and os.path.exists(tok_path) and os.path.exists(meta_path):
        z = np.load(ids_path)
        train_ids, val_ids = z["train"], z["val"]
        meta = json.load(open(meta_path))
    else:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel as BLPre
        from tokenizers.decoders import ByteLevel as BLDec

        text = _all_text(cfg)
        train_text, val_text, split_meta = _block_split(text, cfg)
        del text

        # tokenizer fit on TRAIN ONLY -- no val text in the merges
        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = BLPre()
        tok.decoder = BLDec()

        def chunks(s, n=100_000):
            for i in range(0, len(s), n):
                yield s[i : i + n]

        tok.train_from_iterator(
            chunks(train_text),
            BpeTrainer(vocab_size=cfg.vocab_size, special_tokens=SPECIALS,
                       initial_alphabet=BLPre.alphabet(), show_progress=False),
        )
        tok.save(tok_path)

        train_ids = np.array(tok.encode(train_text).ids, dtype=np.int32)[: cfg.n_tokens]
        val_ids = np.array(tok.encode(val_text).ids, dtype=np.int32)[: cfg.val_tokens]
        if len(train_ids) < cfg.n_tokens * 0.95 or len(val_ids) < cfg.val_tokens * 0.95:
            raise RuntimeError(
                f"short tokenisation: got {len(train_ids)}/{cfg.n_tokens} train, "
                f"{len(val_ids)}/{cfg.val_tokens} val. Raise CHARS_PER_TOKEN."
            )

        V = tok.get_vocab_size()
        meta = dict(vocab_size=V, dataset=cfg.dataset, **split_meta)
        meta["unigram_kl_val_train"] = unigram_kl(train_ids.astype(np.int64),
                                                  val_ids.astype(np.int64), V)
        np.savez_compressed(ids_path, train=train_ids, val=val_ids)
        json.dump(meta, open(meta_path, "w"))

    return Corpus(cfg, train_ids, val_ids, tok_path, meta, device)


class Corpus:
    def __init__(self, cfg, train_ids, val_ids, tok_path, meta, device):
        self.cfg = cfg
        self.device = device
        self.meta = meta
        self.vocab_size = meta["vocab_size"]
        self.tok_path = tok_path
        self.train_np = np.asarray(train_ids, dtype=np.int64)
        self.val_np = np.asarray(val_ids, dtype=np.int64)
        self.train = torch.from_numpy(self.train_np).to(device)
        self.val = torch.from_numpy(self.val_np).to(device)
        self.span_len = cfg.seq_len
        self.n_spans = (self.train.numel() - 1) // self.span_len
        self.mask_id = SPECIALS.index("<mask>")

    # --------------------------------------------------------- sanity gates
    def anchor(self, cache_dir=None):
        """Trigram fit on train, scored on val. nats/token. The absolute floor."""
        from .ngram import anchor_nll

        path = None
        if cache_dir:
            path = os.path.join(cache_dir, f"anchor_{self.vocab_size}_{self.train.numel()}.npy")
        return anchor_nll(self.train_np, self.val_np, cache_path=path,
                          vocab_size=self.vocab_size)

    def report(self):
        import math

        kl = self.meta.get("unigram_kl_val_train", float("nan"))
        lines = [
            f"  corpus     {self.train.numel():,} train / {self.val.numel():,} val tokens"
            f" | vocab {self.vocab_size}",
            f"  split      block-interleaved ({self.meta.get('n_blocks_train','?')} train /"
            f" {self.meta.get('n_blocks_val','?')} val blocks of"
            f" {self.meta.get('block_chars','?')} chars, from"
            f" {self.meta.get('n_blocks_total','?')} available)",
            f"  unigram KL(val||train) {kl:.4f} nats"
            f"   [{'OK' if kl < 0.15 else 'WARN: possible domain mismatch'}]",
            f"  uniform-random ceiling  {math.log(self.vocab_size):.3f} nats"
            f" (ppl {self.vocab_size})",
        ]
        return lines

    # --------------------------------------------------------- tokenizer
    def tokenizer(self):
        from tokenizers import Tokenizer

        return Tokenizer.from_file(self.tok_path)

    # --------------------------------------------------------- batching
    def spans_to_batch(self, span_ids):
        L = self.span_len
        starts = span_ids.to(self.train.device) * L
        off = torch.arange(L + 1, device=self.train.device)
        seq = self.train[starts[:, None] + off[None, :]]
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous(), starts

    def random_spans(self, batch_size, generator=None):
        return torch.randint(0, self.n_spans, (batch_size,),
                             device=self.train.device, generator=generator)

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
