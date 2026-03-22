# Competitive Levers

Quick reference of techniques used by top leaderboard entries. Derived from scanning 300 open PRs (Mar 2026) and the `records/` directory.

Current top scores: 1.0672 (#462), 1.1027 (#442), 1.1213 (#398), 1.1227 (#417), 1.1233 (#414).

## Tier 1: Biggest Impact

**TTT (Test-Time Training)** — THE dominant lever. Fine-tune the model on validation data at eval time. Causal: only train on already-scored tokens, so it's legal. AdamW(lr=5e-4, wd=0, 10ep) beats SGD(lr=0.008, mom=0.9, 20ep). freeze=0 (unfreeze all layers) is critical. Worth -0.02 to -0.05 bpb. Costs ~160s eval time. See PRs #442, #398, #462.

**11 Layers** — Up from 10. Nearly universal in top entries. Funded by int5/int6 byte savings. ~-0.005 bpb.

**XSA4 (Cross-Sequence Attention)** — On the last 4 layers, tokens attend across sequence boundaries within a batch. Zero extra parameters. ~-0.005 to -0.01 bpb. See PR #315.

**SwiGLU FFN** — Gated FFN variant (Star-ReLU activation). More expressive per parameter. Interacts strongly with TTT — AdamW TTT gives 3x more improvement on SwiGLU than standard architecture. See PR #462.

## Tier 2: Free or Cheap Wins

**EMA (Exponential Moving Average)** — Shadow weights: `w_ema = 0.997 * w_ema + 0.003 * w`. Updated every step. Smoother than SWA, can coexist with it. ~-0.001 to -0.003 bpb. See PR #414.

**Partial RoPE** — Apply RoPE to only 16/64 head dims (25%). Remaining dims do position-free content-based attention. Zero params. ~-0.001 bpb. See PR #315.

**LN Scale** — Scale RMSNorm output by `1/sqrt(layer_idx+1)`. Dampens deeper layers, stabilizes training. Zero params. ~-0.001 bpb. See PR #315.

**GPTQ-lite** — Post-training quantization refinement. Try 5 clip percentiles per row (0.999, 0.9995, 0.9999, 0.99999, 1.0), pick min MSE. Zero training cost. ~-0.0006 bpb. See PR #414.

**Extended warmdown (3500 steps)** — Up from 3000. ~-0.0002 bpb. See PR #414.

**Late QAT threshold 0.15** — Delay QAT activation (up from 0.1). ~-0.0001 bpb. See PR #414.

## Tier 3: Architecture (already in SOTA baseline)

- **MLP 3x expansion** — Biggest single architecture contributor.
- **SmearGate** — Blends adjacent token embeddings. ~512 params.
- **BigramHash** — `(prev*31 + curr) % buckets` -> learned embedding. SOTA: 10240 buckets. Some PRs use 12288.
- **U-Net skip connections** — Layer i -> layer n-i. Cheap, helps gradient flow.
- **Tied FP16 embeddings** — Quantization-sensitive, kept in FP16.
- **Orthogonal init** — muP-scaled output projections by 1/sqrt(2*n_layers).

## Tier 3: Quantization & Compression (already in SOTA baseline)

- **Int5 MLP / Int6 attention** — Mixed precision. MLP tolerates aggressive quantization.
- **QAT (STE)** — Train with simulated quantization noise.
- **zstd-22** — ~5% better than zlib-9.
- **Magnitude pruning 3%** — Set small weights to zero for better compression.

## Tier 4: Unexplored / Speculative

- **Depth recurrence** — Reuse layer weights for multiple passes. Some PRs tried it (PR #319, #386) but results are mixed (1.27-1.41 bpb). Needs more work.
- **Mixture of Experts** — Sparse routing. PR #250 tried it, unclear results.
- **Low-rank factorization** — Factor weight matrices as UV. PR #316 tried it.
- **TrigramHash** — Extension of BigramHash to trigrams. PR #440 tried it.
- **BitNet b1.58** — Ternary weights. PR #367 got 1.177, interesting but not competitive yet.
- **NTK-RoPE** — NTK-aware scaling for longer eval context. PR #369.
- **Error Correction Table** — Post-quantization error correction. PR #232.
- **Progressive training** — Start small, grow.
- **LR rewinding** — Short fine-tune after SWA at very low LR.

## Priority Order for Autoresearch

1. Implement TTT (AdamW, 10ep) — this alone should get us from 1.1428 to ~1.12
2. Add 11th layer — nearly free with current byte budget
3. Add XSA4 on last 4 layers — zero params
4. Replace SWA with EMA (or add EMA alongside SWA)
5. Add Partial RoPE (16/64 dims) + LN Scale — zero params
6. Add GPTQ-lite post-training — zero cost
7. Try SwiGLU FFN — architecture change, interacts with TTT
8. Tune: warmdown 3500, late QAT 0.15, BigramHash 12288
