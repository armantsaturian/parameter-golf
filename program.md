# parameter-golf autoresearch

Autonomous experiment runner for the OpenAI Parameter Golf competition.
Train the best language model that fits in a 16MB artifact on 8xH100s in 10 minutes.

## Setup

Work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — competition rules, leaderboard, submission process.
   - `train_gpt.py` — the file you modify. Architecture, optimizer, quantization, compression, training loop, evaluation. This is your entire canvas.
   - `prepare.py` — read-only evaluation harness. Do NOT modify. `train_gpt.py` is self-contained; `prepare.py` is just a reference.
4. **Copy the SOTA baseline**: The current best is `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py` (1.1428 bpb, ~15.9MB). Copy it to the repo root as your starting `train_gpt.py`. Don't start from the naive baseline.
5. **Verify data exists**: Check `./data/datasets/fineweb10B_sp1024/` has `fineweb_train_*.bin` and `fineweb_val_*.bin`, and `./data/tokenizers/fineweb_1024_bpe.model` exists. If not, tell the human to run `python3 data/cached_challenge_fineweb.py --variant sp1024`.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row.
7. **Confirm and go**.

## The dual objective

This is a CONSTRAINED optimization problem:

- **Minimize** `val_bpb` (bits per byte) — lower is better.
- **Subject to** total artifact size <= 16,000,000 bytes (decimal 16MB, not 16 MiB).

Artifact = compressed model bytes + code bytes (`train_gpt.py` UTF-8 size). A run that improves bpb but exceeds 16MB is **invalid**. Track both metrics on every run. The `train_gpt.py` file itself counts toward the 16MB — keep it tight (~53KB currently).

**Competitive landscape** (from open PRs, Mar 2026):
- Our starting baseline: **1.1428** (10L, int5/int6, BigramHash, SWA)
- Best without val-TTT: **1.1236** (11L, EMA, GPTQ-lite, XSA4, Partial RoPE, LN Scale)
- Best with val-TTT: **1.1027** (AdamW TTT 10ep on already-evaluated val tokens — this is competition-legal)
- Val-TTT is legal per competition rules: you may test-time train on validation tokens you've already evaluated.

## Experimentation

Each experiment trains on 8xH100 GPUs with a fixed 10-minute wallclock cap.

**What you CAN do:**
- Modify `train_gpt.py` — the only file you edit. Everything is fair game: architecture, optimizer, hyperparameters, quantization, compression, training loop, batch size, model size, evaluation.

**What you CANNOT do:**
- Install new packages beyond `requirements.txt`.
- Access validation data during training.
- Change data/tokenizer files in `./data/`.
- Run for longer than 10 minutes of training wall time.

**Where to find ideas:** Read `levers.md` for a quick reference of competitive techniques. Study the SOTA submissions in `records/` for full implementations. When stuck, think harder — try combining near-misses, try radical changes, re-read the records for new angles.

## The experiment loop

LOOP FOREVER:

1. Look at git state and the best val_bpb so far.
2. Pick an experiment idea. One change at a time — isolate variables.
3. Edit `train_gpt.py`.
4. git commit (so you can revert cleanly).
5. Run: `timeout 1200 torchrun --standalone --nproc_per_node=8 train_gpt.py > run.log 2>&1`
6. Parse: `grep "final_int8_zlib_roundtrip_exact\|Total submission size" run.log`
7. If grep is empty, it crashed. `tail -n 50 run.log` to diagnose. Typo/import → fix and re-run. OOM/fundamental → log as crash, revert, move on. Give up after 2-3 fix attempts.
8. Record results in `results.tsv` (do NOT commit it).
9. **Decision:**
   - bpb improved AND artifact <= 16MB → **keep**, advance the branch.
   - bpb improved but artifact > 16MB → **invalid**, `git reset --hard HEAD~1`.
   - bpb equal or worse → **discard**, `git reset --hard HEAD~1`.
   - crashed → **crash**, revert and move on.

Each run takes ~12-15 min (10 min training + 2-5 min eval). If a run exceeds 20 min, kill it and treat as crash.

## Logging results

Log to `results.tsv` (tab-separated). Header and 5 columns:

```
commit	val_bpb	artifact_mb	status	description
```

- commit: short hash (7 chars)
- val_bpb: from `final_int8_zlib_roundtrip_exact` line (0.000000 for crashes)
- artifact_mb: total submission size / 1,000,000, rounded to .2f (0.00 for crashes)
- status: `keep`, `discard`, `crash`, or `invalid`
- description: short text of what was tried

## NEVER STOP

Once the loop begins, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be away and expects you to continue working **indefinitely** until manually stopped. You are autonomous. The loop runs until the human interrupts you, period.

Each experiment takes ~12-15 minutes, so ~4 per hour, ~96 in 24 hours. Make every run count.
