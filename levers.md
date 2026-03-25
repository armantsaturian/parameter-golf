# Competitive Levers

Quick reference of techniques used by top leaderboard entries. Updated Mar 24, 2026.

Current leaderboard SOTA: **1.1194** (#549). Open PR frontier: **0.5601** (#611, LoRA TTT).

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

**LoRA TTT** — THE dominant lever. PR #611 gets **0.5601 bpb** using LoRA adapters during test-time training. Key details:
- Add LoRA to K projections (not just Q/V) with 0.3x LR multiplier
- **Min-NLL epoch selection**: track min NLL per document across TTT epochs, use best epoch's scores (prevents late-epoch overfitting)
- 8+ epochs of LoRA adaptation per chunk
- Worth **-0.20 to -0.55 bpb** over no-TTT baseline
- See PRs #611 (0.56), #596 (0.64), #614 (0.69)

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

Starting from 1.1194 SOTA baseline:

1. **LoRA TTT** — Add LoRA adapters to Q/K/V projections. Min-NLL epoch selection. This is the path to sub-1.0 bpb. Study PRs #611, #596.
2. **Multi-pass streaming TTT** — 3 trajectories, min NLL/token. Simpler than LoRA, gets ~1.05. Study PR #573.
3. **XSA all layers** — Extend from last-4 to all 11. Free -0.004 bpb. Study PR #609.
4. **Full Hessian GPTQ** — Replace naive int6 rounding. Free -0.002 to -0.005 bpb. Study PR #593.
5. **SwiGLU FFN** — Biggest base-model architecture lever remaining. ~-0.01 bpb.
6. **int5 quantization** — More params in same bytes. Enables 33.6M params.
7. **BigramHash 3072x80** — Narrow + wide table. ~-0.001 bpb.
8. **Combine best of above** — LoRA TTT + SwiGLU + full GPTQ + int5 is likely the path to 0.5x.

## V100 Compatibility Notes

The SOTA uses FlashAttention 3 (Hopper-only, sm_90). On V100 (sm_70):
- Replace `flash_attn_3_func` with PyTorch SDPA (`F.scaled_dot_product_attention`)
- Change `torch.bfloat16` autocast to `torch.float16`
- Parallel Muon async comms should still work with NCCL on V100
- Results should be directionally equivalent; absolute bpb may differ slightly due to fp16 vs bf16
