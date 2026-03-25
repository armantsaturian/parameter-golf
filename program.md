# parameter-golf autoresearch

Autonomous experiment runner for the OpenAI Parameter Golf competition.
Train the best language model that fits in a 16MB artifact on 8xH100s in 10 minutes.

## Setup

Work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar24`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — competition rules, leaderboard, submission process.
   - `train_gpt.py` — the file you modify. Architecture, optimizer, quantization, compression, training loop, evaluation. This is your entire canvas.
   - `prepare.py` — read-only evaluation harness. Do NOT modify. `train_gpt.py` is self-contained; `prepare.py` is just a reference.
4. **Copy the SOTA baseline**: The current best is `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py` (1.1194 bpb, ~15.95MB). Copy it to the repo root as your starting `train_gpt.py`. Don't start from the naive baseline.
5. **Verify data exists**: Check `./data/datasets/fineweb10B_sp1024/` has `fineweb_train_*.bin` and `fineweb_val_*.bin`, and `./data/tokenizers/fineweb_1024_bpe.model` exists. If not, tell the human to run `python3 data/cached_challenge_fineweb.py --variant sp1024`.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row.
7. **Confirm and go**.

## The dual objective

This is a CONSTRAINED optimization problem:

- **Minimize** `val_bpb` (bits per byte) — lower is better.
- **Subject to** total artifact size <= 16,000,000 bytes (decimal 16MB, not 16 MiB).

Artifact = compressed model bytes + code bytes (`train_gpt.py` UTF-8 size). A run that improves bpb but exceeds 16MB is **invalid**. Track both metrics on every run. The `train_gpt.py` file itself counts toward the 16MB — keep it tight.

**Competitive landscape** (Mar 25, 2026 evening):
- Our current best: **0.9884** (Hedge Mixer + backed-off 2-5gram cache + XSA-all, commit 6791de1)
- Best open PR: **0.9850** (Cosine TTT + Multi-Order N-gram Cache, PR #741)
- Gap to close: **0.0034 bpb**
- The mixer blend=1.000 means the n-gram cache is doing ALL the work. Focus on improving cache/mixing quality.
- PR #611 (0.5601, LoRA TTT) was **REJECTED** — min-NLL epoch selection ruled as training on val set. Do NOT use min-NLL epoch selection.
- PR #762 (0.7139, multi-epoch LoRA TTT) is also likely invalid — same information leakage principle.
- **DO NOT** implement multi-pass replay (score all tokens, then re-score with populated cache). This is information leakage.
- Val-TTT is legal per competition rules: you may test-time train on validation tokens you've already evaluated. But you must score tokens BEFORE weight updates. Single-pass only.

## Hardware modes

### 8xH100 (target, competition hardware)
- `torchrun --standalone --nproc_per_node=8 train_gpt.py`
- `ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600`
- Uses FlashAttention 3, bfloat16, Parallel Muon
- Each run: ~12-15 min (10 min train + 2-5 min eval)

### 8xV100 (development, p3.16xlarge)
- `torchrun --standalone --nproc_per_node=8 train_gpt.py`
- `ITERATIONS=7200 MAX_WALLCLOCK_SECONDS=0`
- **Required patches** before first run:
  - Replace `from flash_attn_interface import flash_attn_func as flash_attn_3_func` with PyTorch SDPA fallback
  - Change all `torch.bfloat16` / `bfloat16` autocast to `torch.float16` / `float16`
  - V100 is sm_70: no FA3, no bf16
- Step count 7200 matches ~what H100 does in 10 min for this model
- Each run: ~60-80 min on 8xV100
- **Results transfer**: architecture/technique improvements are hardware-independent. Absolute bpb may differ slightly (fp16 vs bf16) but relative deltas are valid.

## Experimentation

**What you CAN do:**
- Modify `train_gpt.py` — the only file you edit. Everything is fair game: architecture, optimizer, hyperparameters, quantization, compression, training loop, batch size, model size, evaluation.

**What you CANNOT do:**
- Install new packages beyond `requirements.txt`.
- Access validation data during training.
- Change data/tokenizer files in `./data/`.
- Run for longer than 10 minutes of training wall time (on H100; V100 mode ignores wallclock).

**Where to find ideas:** Read `levers.md` for a reference of what competitors are doing — use it as inspiration, not a checklist. You are free to try any technique you believe could improve bpb. Don't limit yourself to what's listed there. Think about what's widely known to improve LLMs in the broader ML community (new activations, attention variants, training tricks, quantization methods, etc.) — if it's proven elsewhere but hasn't been tried in this competition yet, it's a great candidate. Also actively try **combining** multiple techniques together — individual levers might give small gains, but stacking them is how you reach sub-1.0. Study the SOTA submissions in `records/` for full implementations when you need concrete code to reference.

## EVAL-ONLY mode (critical for fast iteration)

If your experiment ONLY changes eval/TTT/cache/mixing code and does NOT change the model architecture, training loop, optimizer, or quantization, you do NOT need to retrain. This saves ~10 minutes per experiment.

**How it works:**
1. Add an `EVAL_ONLY` env var check near the top of `train_gpt.py`'s main block.
2. When `EVAL_ONLY=1`, skip training entirely — load the existing `final_model.int6.ptz` artifact directly.
3. Jump straight to the eval/TTT/cache code path.
4. The saved artifact from the last kept training run must exist as `final_model.int6.ptz` in the repo root.

**When to use EVAL_ONLY:**
- Cache/mixing changes (PAQ-style mixing, PPM exclusion, extended n-grams, SSE, temperature calibration)
- TTT hyperparameter changes (LR, epochs, chunk size)
- Blending/interpolation formula changes
- Any change that only touches code AFTER the model is loaded from the artifact

**When NOT to use EVAL_ONLY:**
- Architecture changes (layers, dims, activations, attention)
- Training hyperparameters (LR, optimizer, batch size, warmdown)
- Quantization changes (int5, CROWN-Q, Soft-Round QAT)
- Any change that affects the trained weights

**Run command for eval-only:**
```
EVAL_ONLY=1 timeout 3000 torchrun --standalone --nproc_per_node=8 train_gpt.py > run.log 2>&1
```

Each eval-only run takes ~5-8 min instead of ~15 min. This doubles your experiment throughput for eval-time techniques.

## LEGALITY WARNING: No multi-pass replay

**DO NOT** implement a "phase 2 cache replay" where you score all tokens in phase 1 (building an n-gram cache), then replay and re-score all tokens with the populated cache. This is **information leakage** — the cache in phase 2 contains statistics from tokens that haven't been scored yet in that pass. This is the same principle that got PR #611 rejected. Any result from a replay approach is INVALID.

Legal approach: single-pass, score-before-update. When scoring token X, the cache must only contain statistics from tokens scored BEFORE X in this pass.

## The experiment loop

LOOP FOREVER:

1. Look at git state and the best val_bpb so far.
2. Pick an experiment idea. One change at a time — isolate variables.
3. Edit `train_gpt.py`.
4. git commit (so you can revert cleanly), then `git push origin HEAD` to back up to GitHub. **Always push after every commit** — if the instance dies, unpushed work is lost. Commit message format: `exp: <description> [bpb=X.XXXX, size=XX.XXmb, STATUS]` — e.g. `exp: add n-gram cache at eval [bpb=1.0508, size=15.92mb, keep]`. For pre-run commits use `exp: <description> [pending]`.
5. Run: `timeout 6000 torchrun --standalone --nproc_per_node=8 train_gpt.py > run.log 2>&1` — or use `EVAL_ONLY=1` with `timeout 3000` for eval-only changes. **Context efficiency**: don't poll the log every few seconds. You may check `tail -5 run.log` once or twice during a run to confirm it's progressing, but sleep 4+ minutes between checks. The run takes ~12-15 min on H100 (full) or ~5-8 min (eval-only) — be patient and conserve context. **Early stopping**: if a mid-run validation (e.g. step 4000) is significantly worse than your current best (say >0.05 bpb behind), consider killing the run (`pkill -f torchrun`) and moving on — but use your judgment, some techniques recover during TTT/eval.
6. Parse: `grep "final_int8_zlib_roundtrip_exact\|Total submission size" run.log`
7. If grep is empty, it crashed. `tail -n 50 run.log` to diagnose. Typo/import -> fix and re-run. OOM/fundamental -> log as crash, revert, move on. Give up after 2-3 fix attempts.
8. Record results in `results.tsv`. **Commit and push results.tsv after every experiment** — if the instance dies, the experiment history must survive.
9. **Decision:**
   - bpb improved AND artifact <= 16MB -> **keep**, advance the branch.
   - bpb improved but artifact > 16MB -> **revert** code, but log as `invalid` in results.tsv.
   - bpb equal or worse -> **revert** code, but log as `discard` in results.tsv.
   - early-stopped (killed mid-run) -> **revert** code, but log as `early-stop` in results.tsv with the partial metric if available.
   - crashed -> **revert** code, log as `crash`.

   **Important**: always log every experiment in results.tsv before reverting, so you never retry the same idea. Check results.tsv before starting a new experiment to avoid duplicates.

On V100, each run takes ~60-80 min. On H100, ~12-15 min (full) or ~5-8 min (eval-only). If a run exceeds 2x expected time, kill it and treat as crash.

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

On V100: ~1 experiment per hour, ~10-12 in an overnight session. Make every run count.
On H100: ~4 experiments per hour, ~96 in 24 hours. Make every run count.
