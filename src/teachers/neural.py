"""
Neural teacher: a larger model trained on the SAME corpus, cached as top-k.

THE CLAIM BEING TESTED
----------------------
Run 4 found that `topm_uniform` -- the teacher's candidate set with the
probabilities thrown away and flattened -- matched or beat the full teacher.
The identity of the plausible tokens carried the signal; the probability
structure carried nothing measurable.

That was established for a trigram teacher. Whether it holds for a NEURAL
teacher is both untested and consequential.

Classical distillation transfers the teacher's full soft distribution, which is
what makes it expensive: you either keep the teacher resident during student
training, or you cache a full distribution per token -- 128k floats per position
at a modern vocabulary size. If only the SUPPORT matters, you cache 8 int16 per
token instead. That is roughly four orders of magnitude less, and it turns
"distill from a large model" from an infrastructure project into a file.

So this teacher caches the top-k with probabilities, and the arms differ only in
whether the probabilities are used:

    teach_neural        support only, flattened   <- the novel claim
    teach_neural_probs  top-k with probabilities  <- classical distillation
    shuffled:neural     matched shape, wrong context

Precedent worth knowing: Baby Llama (Timiryasov & Tastet 2023) distilled an
ensemble into a 58M student on 10M words and the student beat both teachers;
BabyLlama-2 confirmed this with a hyperparameter sweep. Distillation at this
scale is the largest single effect in the BabyLM literature. What nobody has
asked is how much of the teacher's distribution you actually need.

The table is produced by scripts/train_teacher.py, which trains the teacher and
then caches predictions with strided windows so every position has real left
context rather than being at the start of a span.
"""

import os

import numpy as np

from .base import Teacher


class NeuralTeacher(Teacher):
    name = "neural"
    lacks = "a larger model's learned conditional, cached as top-k support"

    def __init__(self, *a, table_path=None, **kw):
        super().__init__(*a, **kw)
        self.table_path = table_path
        self._meta = {}

    def signature(self):
        return {"table": os.path.basename(self.table_path or "none")}

    def _compute(self):
        if not self.table_path or not os.path.exists(self.table_path):
            raise FileNotFoundError(
                f"neural teacher table not found: {self.table_path!r}\n"
                f"Build one first:\n"
                f"  python -m scripts.train_teacher --teacher-size base "
                f"--cache-dir <dir>"
            )
        z = np.load(self.table_path)
        idx, prob = z["idx"], z["prob"]
        if idx.shape[0] != len(self.ids):
            raise ValueError(
                f"teacher table has {idx.shape[0]} rows but corpus has "
                f"{len(self.ids)} tokens -- the table was built for a different "
                f"corpus. Rebuild it."
            )
        self._meta = {
            "teacher_val_nats": float(z["val_nats"]) if "val_nats" in z else float("nan"),
            "teacher_params_m": float(z["params_m"]) if "params_m" in z else float("nan"),
            "cached_m": int(idx.shape[1]),
        }
        n = len(self.ids)
        count = np.full(n, 1e6, dtype=np.float32)      # no evidence notion
        mass = prob.sum(1).astype(np.float32)
        return idx.astype(np.int32), prob.astype(np.float32), count, mass

    def report(self, *a):
        r = super().report(*a)
        r.update(self._meta)
        return r
