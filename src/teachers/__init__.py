"""Teacher registry.

A teacher is named by a small spec string so arms stay declarative:

    trigram                 fixed order-2 exact match
    varorder                adaptive-order exact match (infini-gram style)
    cache                   recency / burstiness
    class                   distributional abstraction
    embed                   soft context generalisation
    mix:varorder+cache      equal-weight mixture
    mix:varorder+class+cache
    shuffled:varorder       matched-shape control for any rung
    topm_uniform:varorder   matched-support control for any rung
    unigram / uniform       unstructured controls
"""

from .base import Teacher
from .cache import CacheTeacher
from .classes import ClassTeacher, EmbedTeacher
from .controls import (MixtureTeacher, ShuffledTeacher, TopMUniformTeacher,
                       UniformTeacher, UnigramTeacher)
from .ngram import NgramTeacher, VarOrderTeacher

BASE = {
    "trigram": NgramTeacher,
    "varorder": VarOrderTeacher,
    "cache": CacheTeacher,
    "class": ClassTeacher,
    "embed": EmbedTeacher,
    "unigram": UnigramTeacher,
    "uniform": UniformTeacher,
}

# ordered by how much information the teacher holds that the student cannot get
LADDER = ["unigram", "trigram", "varorder", "cache", "embed", "class"]


def _kw(name, args):
    if args is None:
        return {}
    g = lambda k, d: getattr(args, k, d)
    if name == "trigram":
        return {"order": g("teacher_order", 2)}
    if name == "varorder":
        return {"max_order": g("teacher_max_order", 6)}
    if name == "cache":
        return {"half_life": g("cache_half_life", 512.0), "window": g("cache_window", 4096)}
    if name == "class":
        return {"n_classes": g("teacher_n_classes", 128), "emb_dim": g("teacher_emb_dim", 64),
                "top_classes": g("teacher_top_classes", 4)}
    if name == "embed":
        return {"emb_dim": g("teacher_emb_dim", 64), "n_neighbours": g("teacher_neighbours", 8)}
    return {}


def build_teacher(spec, ids, vocab_size, m=8, min_count=3, args=None):
    spec = (spec or "trigram").strip()
    common = dict(m=m, min_count=min_count, args=args)

    if spec.startswith("mix:"):
        names = spec[4:].split("+")
        parts = [build_teacher(n, ids, vocab_size, m, min_count, args) for n in names]
        ws = getattr(args, "mixture_weights", None)
        weights = [float(x) for x in ws.split(",")] if ws else None
        if weights and len(weights) != len(parts):
            raise ValueError("--mixture-weights length must match the mixture spec")
        return MixtureTeacher(ids, vocab_size, parts=parts, weights=weights, **common)

    for wrap, cls in (("shuffled:", ShuffledTeacher), ("topm_uniform:", TopMUniformTeacher)):
        if spec.startswith(wrap):
            inner = build_teacher(spec[len(wrap):], ids, vocab_size, m, min_count, args)
            kw = dict(common)
            if cls is ShuffledTeacher:
                kw["seed"] = getattr(args, "seed", 0)
            return cls(ids, vocab_size, inner=inner, **kw)

    if spec not in BASE:
        raise ValueError(f"unknown teacher {spec!r}; base={sorted(BASE)} "
                         f"or mix:/shuffled:/topm_uniform: prefixes")
    return BASE[spec](ids, vocab_size, **common, **_kw(spec, args))


__all__ = ["build_teacher", "Teacher", "BASE", "LADDER"]
