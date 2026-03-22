#!/usr/bin/env bash
# setup_gcp.sh — Idempotent setup for parameter-golf on a GCP H100 instance.
# Run this after SSH-ing into the instance.
# Usage: bash setup_gcp.sh

set -euo pipefail

REPO_URL="https://github.com/openai/parameter-golf.git"
REPO_DIR="$HOME/parameter-golf"
VARIANT="sp1024"

echo "=== Parameter Golf GCP Setup ==="

# ---------------------------------------------------------------
# 1. Clone repo (skip if already present)
# ---------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/5] Repo already cloned at $REPO_DIR, pulling latest..."
    cd "$REPO_DIR"
    git pull --ff-only || true
else
    echo "[1/5] Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# ---------------------------------------------------------------
# 2. Install Python dependencies
# ---------------------------------------------------------------
echo "[2/5] Installing Python dependencies..."
pip install -q -r requirements.txt
pip install -q zstandard

# ---------------------------------------------------------------
# 3. Download data
# ---------------------------------------------------------------
DATA_DIR="$REPO_DIR/data/datasets/fineweb10B_$VARIANT"
if [ -d "$DATA_DIR" ] && ls "$DATA_DIR"/fineweb_train_*.bin 1>/dev/null 2>&1; then
    SHARD_COUNT=$(ls "$DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
    echo "[3/5] Data already downloaded ($SHARD_COUNT training shards found), skipping."
else
    echo "[3/5] Downloading data (variant=$VARIANT, full 80 shards)..."
    python3 data/cached_challenge_fineweb.py --variant "$VARIANT"
fi

# ---------------------------------------------------------------
# 4. Verify GPU setup
# ---------------------------------------------------------------
echo "[4/5] Checking GPUs..."
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo "  Found $GPU_COUNT GPU(s):"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo "  WARNING: nvidia-smi not found. NVIDIA drivers may not be installed."
    echo "  On Deep Learning VM images, wait a few minutes after first boot for driver installation."
fi

# ---------------------------------------------------------------
# 5. Smoke test (single GPU, short run)
# ---------------------------------------------------------------
echo "[5/5] Running smoke test (1 GPU, 200 iterations)..."
RUN_ID=smoke_test \
DATA_PATH="$REPO_DIR/data/datasets/fineweb10B_$VARIANT/" \
TOKENIZER_PATH="$REPO_DIR/data/tokenizers/fineweb_1024_bpe.model" \
VOCAB_SIZE=1024 \
ITERATIONS=200 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=60 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run the full 8xH100 baseline:"
echo ""
echo "  cd $REPO_DIR"
echo "  RUN_ID=baseline_sp1024 \\"
echo "  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \\"
echo "  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\"
echo "  VOCAB_SIZE=1024 \\"
echo "  torchrun --standalone --nproc_per_node=8 train_gpt.py"
echo ""
echo "Tip: Run inside tmux so SSH disconnects don't kill your session."
