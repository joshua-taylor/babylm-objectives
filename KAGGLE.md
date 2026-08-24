# Running on Kaggle (T4)

Paste each block into its own notebook cell. Enable GPU T4 and Internet in
notebook settings.

## Cell 1 — setup and smoke test

```python
!pip -q install datasets tokenizers 2>/dev/null
!rm -rf /kaggle/working/babylm-objectives
!git clone -q https://github.com/joshua-taylor/babylm-objectives.git /kaggle/working/babylm-objectives
%cd /kaggle/working/babylm-objectives
!python -m scripts.smoke
```

## Cell 1b — data sanity gate (run this before anything else, ~3 min)

```python
%cd /kaggle/working/babylm-objectives
!python run.py --arm baseline --n-steps 300 --eval-every 100 \
    --size small --cache-dir /kaggle/working/cache \
    --out-dir /kaggle/working/results --registry /tmp/throwaway.csv
```

Check the header. It must show `unigram KL ... [OK]` and a trigram anchor well
below `log(vocab)`, and val ppl must be falling. If val loss sits at or above the
uniform-random ceiling, stop: the split is broken and nothing downstream means
anything.

## Cell 2 — noise floor (do this first, ~20 min)

```python
%cd /kaggle/working/babylm-objectives
!python run.py --stage noisefloor \
    --size small --n-steps 1500 --batch-size 32 --seq-len 256 --dropout 0.1 \
    --cache-dir /kaggle/working/cache \
    --out-dir /kaggle/working/results \
    --registry /kaggle/working/results/experiments_objectives.csv
```

## Cell 3 — screen all arms (~1.5–2 h)

```python
%cd /kaggle/working/babylm-objectives
!python run.py --stage screen --seeds 3 \
    --size small --n-steps 1500 --batch-size 32 --seq-len 256 --dropout 0.1 \
    --cache-dir /kaggle/working/cache \
    --out-dir /kaggle/working/results \
    --registry /kaggle/working/results/experiments_objectives.csv
```

## Cell 4 — apply the decision rule

```python
!python run.py --summarise --registry /kaggle/working/results/experiments_objectives.csv
```

## Cell 5 — confirm a survivor at 3 seeds

```python
!python run.py --arm selective --seeds 3 \
    --n-steps 2000 --cache-dir /kaggle/working/cache \
    --out-dir /kaggle/working/results \
    --registry /kaggle/working/results/experiments_objectives.csv
```

## Notes

* The tokenizer and tokenised corpus are cached in `--cache-dir`, so only the
  first run pays the download and BPE fit. Keep the cache dir on
  `/kaggle/working` and the whole screen reuses it.
* Runs take ~1.3 min each on a T4 (measured), not the 7 min originally estimated,
  so the whole screen at 3 seeds is well under an hour. Use the seeds.
* `--n-steps 1500` at batch 32, seq 256 is 12.3M token-visits over a 1M-token
  corpus, i.e. ~12 epochs.
* If train ppl still drops below ~30 while val rises, drop to `--size tiny` or
  raise `--dropout` to 0.2. Watch the `best@` column: if every arm's best is at
  the first eval, the arms are being compared while undertrained.
* If a session is going to time out, run cells 2 and 3 in separate sessions and
  keep the same `--registry` path (append-only). Download it between sessions.
* `--lr-schedule constant` when you want the achievable floor rather than a
  fixed-budget number.


---

# Run 5: support-only distillation + the self-teacher control

## Cell A — setup

```python
!pip -q install datasets tokenizers 2>/dev/null
!rm -rf /kaggle/working/babylm-objectives
!git clone -q https://github.com/joshua-taylor/babylm-objectives.git /kaggle/working/babylm-objectives
%cd /kaggle/working/babylm-objectives
!python -m scripts.smoke
```

## Cell B — the cheap control FIRST (~25 min)

If the model's own EMA top-k matches the n-gram teacher, the whole ladder is
unnecessary. Find out before spending anything else.

```python
!python run.py --stage screen --seeds 5 \
    --arms baseline,teach_self,teach_self_probs,teach_varorder,teach_topm_uniform \
    --n-steps 9000 --lr-schedule constant \
    --cache-dir /kaggle/working/cache --out-dir /kaggle/working/results \
    --registry /kaggle/working/results/exp.csv

!python run.py --summarise --registry /kaggle/working/results/exp.csv --fit-at 3.10
```

## Cell C — train the teacher, then the three-way distillation test (~45 min)

```python
!python -m scripts.train_teacher --teacher-size base --teacher-k 64 \
    --n-steps 9000 --lr-schedule constant \
    --cache-dir /kaggle/working/cache --out-dir /kaggle/working/results \
    --registry /kaggle/working/results/exp.csv
```

Note the printed table path, then:

```python
TABLE = "/kaggle/working/cache/teacher_neural_base_v2048_n1000000.npz"
!python run.py --stage screen --seeds 5 \
    --arms teach_neural,teach_neural_probs,teach_neural_shuffled \
    --teacher-table {TABLE} --n-steps 9000 --lr-schedule constant \
    --cache-dir /kaggle/working/cache --out-dir /kaggle/working/results \
    --registry /kaggle/working/results/exp.csv

!python run.py --summarise --registry /kaggle/working/results/exp.csv --fit-at 3.10
```

**The headline number is `teach_neural` minus `teach_neural_probs`.** If |t| < 2,
the teacher's probabilities are unnecessary and support-only distillation works.

## Cell D — support-size sweep (free, the table is cached at K=64)

```python
!python run.py --stage sweep --arm teach_neural --seeds 3 \
    --sweep-param soft-top-m --sweep-values 1,2,4,8,16,32 \
    --teacher-table {TABLE} --n-steps 9000 --lr-schedule constant \
    --cache-dir /kaggle/working/cache --out-dir /kaggle/working/results \
    --registry /kaggle/working/results/exp.csv
```

Support size is now the central variable and has only ever been tested at m=8.

## Reading the output

* `t` is the difference over its own standard error (Welch). |t| > 2 is the bar.
* `fit` is val loss at matched TRAIN loss — separates "learns better" from
  "regularises". Prefer it to the raw delta.
* If every arm shows `best at LAST eval`, raise `--n-steps` — the table is
  ranking convergence speed, not sample efficiency.
