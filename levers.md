# Competitive Levers

Quick reference of techniques used by top leaderboard entries. Updated Mar 25, 2026 (evening).

Current leaderboard SOTA: **1.1194** (#549). Open PR frontier: **0.9850** (#741, Cosine TTT + N-gram Cache). Our best: **0.9884** (Hedge Mixer + 2-5gram cache + XSA-all).

> **PR #611 (0.5601, LoRA TTT) was REJECTED** — min-NLL epoch selection ruled as training on val set. That path is dead. The new frontier is eval-time n-gram caching + improved TTT.

## URGENT: Highest-Impact New Techniques (Mar 25)

These are from the latest open PRs. Implement in priority order.

### 1. Multi-Order N-gram Cache (PR #741 — 0.9850 bpb)
Eval-time technique. After model scores a chunk, blend predictions with 2-5gram statistics:
- Sliding window over eval tokens, build n-gram frequency tables
- Entropy-adaptive interpolation: when model is uncertain, lean more on n-gram cache
- Weight: `alpha * model_prob + (1-alpha) * ngram_prob`, alpha adapts per-token
- Two-phase eval: Phase 1 = Cosine TTT ~330s, Phase 2 = n-gram cache ~150s
- **Legal**: eval-time only, no training data access
- Worth **-0.13 bpb** over baseline TTT. This is the single biggest lever right now.

### 2. XSA on ALL Layers (PRs #745, #740 — confirmed by multiple teams)
Extend cross-sequence attention from last-4 to all 11 layers. Multiple open PRs validate this works.
- Free **-0.004 bpb** with no artifact size increase.
- Already in our priority list but now confirmed across teams.

### 3. Hedge Mixer TTT (PR #745 — 1.0222 bpb)
Online mixing of 5 experts during TTT eval via Hedge algorithm:
- Expert 1: neural model predictions
- Expert 2-4: unigram/bigram/trigram caches
- Expert 5: entropy-based predictor
- Hedge algorithm updates mixture weights per-token based on each expert's loss
- Worth **-0.10 bpb** over single-expert TTT

### 4. Depth Recurrence (PR #745 — now working)
Reuse layers 4-5 to create 13 virtual layers from 11 physical layers.
- Earlier attempts (#319, #386) got 1.27-1.41 — not competitive.
- PR #745 makes it work as part of a combined approach (1.0222 bpb).
- Free extra depth within same artifact budget.

### 5. CROWN-Q Quantization (PR #745)
Curvature-weighted quantization penalty during warmdown phase.
- Replaces naive STE-based QAT with Hessian-informed rounding decisions.
- Complementary to Full Hessian GPTQ.

### 6. PAQ-Style Log-Odds Context Mixing (from Hutter Prize compressors)
Replace entropy-adaptive linear interpolation with **online logistic regression mixing in log-odds domain**. This is how PAQ/cmix (the world's best compressors) blend predictions:
- Convert all model predictions to log-odds: `ln(p/(1-p))`
- Combine linearly with **online-learned weights** (logistic regression, updated every token)
- Convert back to probabilities
- Weights converge to provably optimal mixture
- Mixing in log-odds space is mathematically superior to mixing in probability space
- Our current `alpha * p_model + (1-alpha) * p_ngram` is a known-inferior approach
- Worth **-0.005 to -0.010 bpb** over entropy-adaptive interpolation. This is the single most likely technique to close the gap to PR #741.

### 7. Extended N-gram Models (from PAQ/cmix/Hutter Prize)
Our cache stops at 5-gram. Top compressors use much richer context models:
- **6-8gram contexts** — more specificity for repetitive text patterns
- **Skip-grams** — non-contiguous positions, e.g. tokens at -1, -3, -5 jointly predicting next. Captures patterns that contiguous n-grams miss.
- **Longest match model** — find the longest previous substring match in scored text, predict from the token after the match. Very strong for repetitive/templated web text.
- **Word-level contexts** — hash word boundaries separately from token boundaries
- Each of these becomes an additional expert in the mixer
- Worth **-0.005 to -0.015 bpb** combined.

### 8. PPM Exclusion in N-gram Backoff
Prediction by Partial Matching with exclusion: when a longer n-gram confidently predicts token X, shorter n-grams **exclude X from their probability mass** and redistribute to other tokens. Current approach blends all orders independently, wasting probability mass on redundant predictions.
- Well-studied in compression literature (PPMC, PPMD variants)
- Nearly free to implement on top of existing n-gram cache
- Worth **-0.003 to -0.008 bpb**.

### 9. SSE (Secondary Symbol Estimation) Post-Processing
After mixing all predictions (neural + n-gram), pass through a small **adaptive lookup table** indexed by (short context hash + binned prediction). The table outputs a refined probability and adapts online after each token. This is PAQ's final refinement stage.
- Nearly free in compute
- Catches systematic biases in the mixer output
- Worth **-0.002 to -0.005 bpb**.

### 10. Temperature Calibration at Eval
Grid-search a temperature scaling factor on the already-scored prefix before final scoring. If model probabilities are slightly miscalibrated (too sharp or too flat), temperature scaling is free bpb.
- Legal: uses only already-scored tokens for calibration
- Can be done as a quick sweep during the n-gram cache phase
- Worth **-0.001 to -0.003 bpb**.

---

## Table Stakes (already in SOTA baseline 1.1194)

All of these are in the current best submission. Don't re-implement — they're your starting point.

- **11 Layers** (512d, 8H, 4KV GQA)
- **MLP 3x expansion** with **LeakyReLU(0.5)²** activation
- **XSA4** — Cross-Sequence Attention on last 4 layers
- **Partial RoPE** — 16/64 dims (25%)
- **LN Scale** — 1/sqrt(layer_idx+1)
- **EMA(0.997)** + **Tight SWA(every 50)**
- **GPTQ-lite** — 5 clip percentiles per row, min MSE
- **Late QAT** — STE int6 fake-quantization at LR scale < 0.15
- **Warmdown 3500 steps**
- **SmearGate** + **BigramHash** (2048 buckets in SOTA, some PRs use 10240+)
- **U-Net skip connections**
- **Tied FP16 embeddings** + logit softcap=30.0
- **OrthoInit** + muP-scaled output projections
- **Parallel Muon optimizer** — batched Newton-Schulz, async reduce-scatter/all-gather
- **Value Embedding (VE128)** — shared dim=128 on layers 9-10
- **Int6 per-row** (MLP + attn weights), **int8 per-row** (embeddings)
- **zstd-22** compression
- **Legal TTT** — score-first, SGD(lr=0.002, mom=0.9, 3ep), all blocks unfrozen, cosine decay

## Tier 1: Biggest Remaining Levers

**N-gram Cache at Eval** — THE dominant new lever. PR #741 gets **0.9850 bpb**. Blend model predictions with 2-5gram statistics during evaluation. See URGENT section above for details.

**Hedge Mixer / Multi-Expert TTT** — PR #745 gets **1.0222 bpb**. Online mixing of neural + n-gram experts. See URGENT section above.

**LoRA TTT (CAUTION)** — Still powerful but **min-NLL epoch selection is ILLEGAL** (PR #611 rejected). Legal LoRA TTT: score tokens BEFORE weight updates, single-pass. PRs #596 (0.64), #614 (0.69) may still be valid if they don't use min-NLL.

**Multi-Pass Streaming TTT** — PR #573 gets **1.0523 bpb** without LoRA. 3 independent adaptation trajectories with shifted data orderings, take min(NLL) per token. Simpler than LoRA TTT. Worth -0.07 bpb over single-pass TTT.

**SwiGLU FFN** — Gated FFN variant. Interacts strongly with TTT — AdamW TTT gives 3x more improvement on SwiGLU than relu². Worth -0.01 to -0.02 bpb. See PR #462.

**Full Hessian GPTQ** — Second-order quantization with Cholesky error compensation and column reordering. Dramatically reduces quant error. Multiple top PRs use this (#593, #606, #609). Worth -0.002 to -0.005 bpb.

**int5 Quantization + Soft-Round QAT** — 31 levels stored as int8 with zstd. Fits **33.6M params in 16MB** (PR #606, 1.1162 bpb). Soft-Round: differentiable tanh-based rounding replacing STE during QAT.

**XSA on all 11 layers** — Extend cross-sequence attention from last-4 to all layers. PR #609 gets **1.1154** (no TTT). Worth -0.004 bpb over XSA-4.

## Tier 2: Promising Directions

**BigramHash tuning** — SOTA uses 2048 buckets. PR #593 uses 3072x80 (narrow embeddings, more buckets, fewer collisions). Worth sweeping 3072/4096/8192 with varying dims.

**Larger model via int5** — int5 GPTQ lets you fit 33.6M params (vs ~25M with int6). Worth -0.005+ bpb purely from more capacity.

**TTT optimizer switch** — The SOTA uses SGD for TTT. AdamW with higher LR might work better on LoRA TTT. Cost: more memory during eval.

**TTT chunk size** — SOTA uses 32K tokens. Smaller chunks = more frequent adaptation. Larger chunks = more context. Worth ablating.

**More layers (12L)** — If byte budget allows with int5 compression. Each layer costs ~1MB compressed.

## Tier 3: Speculative / Negative Results

- **Depth recurrence** — Reuse layer weights for multiple passes. PRs #319, #386 got 1.27-1.41. Not competitive yet.
- **Mixture of Experts** — Sparse routing. PR #250, unclear results.
- **Low-rank factorization** — Factor weights as UV. PR #316.
- **BitNet b1.58** — Ternary weights. PR #367 got 1.177, interesting but behind.
- **Error Correction Table** — Post-quantization error correction. PR #232.
- **Progressive training** — Start small, grow. Untested.
- **LR rewinding** — Short fine-tune after SWA at very low LR.

## Priority Order for Autoresearch

Starting from 0.9884 (current best, backed-off 2-5gram cache + hedge mixer + XSA-all):

**Eval-time only (no retraining needed):**
1. **PAQ-style log-odds context mixing** — Replace entropy-adaptive blending with online logistic regression in log-odds domain. THE most likely technique to close the gap to PR #741 (-0.005 to -0.010 bpb). Study PAQ/cmix source code.
2. **PPM exclusion in n-gram backoff** — When longer n-gram is confident, exclude its predictions from shorter n-gram probability mass. Free -0.003 to -0.008 bpb.
3. **Extended n-gram models** — Add 6-7gram, skip-grams (positions -1,-3,-5), and longest-match model as additional mixer experts. -0.005 to -0.015 bpb combined.
4. **SSE post-processing** — Adaptive lookup table after mixing to refine final probabilities. -0.002 to -0.005 bpb.
5. **Temperature calibration** — Grid-search temperature on scored prefix. Free -0.001 to -0.003 bpb.
6. **Tune cache knobs** — Sweep backoff_beta, alpha_high, ent_thresh on the existing cache. The mixer_final blend=1.000 means cache is carrying everything — optimize it.

**Requires retraining:**
7. **SwiGLU FFN** — Gated FFN, interacts 3x better with TTT. -0.01 to -0.02 bpb. Study PR #462.
8. **Full Hessian GPTQ + Soft-Round QAT** — Better quantization. -0.003 to -0.008 bpb. Study PR #593, #606.
9. **CROWN-Q quantization** — Curvature-weighted quant penalty during warmdown. Study PR #745.
10. **int5 quantization + larger model (12L)** — 33.6M params in 16MB. -0.005 to -0.010 bpb.
11. **Depth Recurrence** — Reuse layers 4-5 for virtual depth. Study PR #745.
12. **Combine best of above** — PAQ mixing + extended n-grams + PPM exclusion + SSE is the path to sub-0.97.

## V100 Compatibility Notes

The SOTA uses FlashAttention 3 (Hopper-only, sm_90). On V100 (sm_70):
- Replace `flash_attn_3_func` with PyTorch SDPA (`F.scaled_dot_product_attention`)
- Change `torch.bfloat16` autocast to `torch.float16`
- Parallel Muon async comms should still work with NCCL on V100
- Results should be directionally equivalent; absolute bpb may differ slightly due to fp16 vs bf16
