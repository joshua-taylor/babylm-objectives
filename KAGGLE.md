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

## Cell 2 — noise floor (do this first, ~20 min)

```python
%cd /kaggle/working/babylm-objectives
!python run.py --stage noisefloor \
    --n-steps 2000 --batch-size 32 --seq-len 256 \
    --cache-dir /kaggle/working/cache \
    --out-dir /kaggle/working/results \
    --registry /kaggle/working/results/experiments_objectives.csv
```

## Cell 3 — screen all arms (~1.5–2 h)

```python
%cd /kaggle/working/babylm-objectives
!python run.py --stage screen \
    --n-steps 2000 --batch-size 32 --seq-len 256 \
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
* `--n-steps 2000` at batch 32, seq 256 is 16.4M token-visits over a 1M-token
  corpus, i.e. ~16 epochs. Raise it to find a true floor; lower it to iterate.
* If a session is going to time out, run cells 2 and 3 in separate sessions and
  keep the same `--registry` path (append-only). Download it between sessions.
* `--lr-schedule constant` when you want the achievable floor rather than a
  fixed-budget number.
