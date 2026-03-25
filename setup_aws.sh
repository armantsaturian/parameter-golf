#!/usr/bin/env bash
# setup_aws.sh — Idempotent setup for parameter-golf on an AWS p5.48xlarge instance.
# Run this after SSH-ing into the instance.
# Usage: bash setup_aws.sh

set -euo pipefail

REPO_URL="https://github.com/openai/parameter-golf.git"
REPO_DIR="$HOME/parameter-golf"
VARIANT="sp1024"

echo "=== Parameter Golf AWS Setup ==="

# ---------------------------------------------------------------
# 1. Mount NVMe instance storage (p5.48xlarge has 3.8TB NVMe)
# ---------------------------------------------------------------
NVME_DEV="/dev/nvme1n1"
NVME_MOUNT="/mnt/nvme"
if lsblk "$NVME_DEV" &>/dev/null && ! mountpoint -q "$NVME_MOUNT" 2>/dev/null; then
    echo "[1/6] Mounting NVMe instance storage at $NVME_MOUNT..."
    sudo mkfs.ext4 -q "$NVME_DEV" 2>/dev/null || true
    sudo mkdir -p "$NVME_MOUNT"
    sudo mount "$NVME_DEV" "$NVME_MOUNT"
    sudo chown "$(whoami):$(whoami)" "$NVME_MOUNT"
    echo "  Mounted $(lsblk -no SIZE "$NVME_DEV" | head -1) at $NVME_MOUNT"
else
    echo "[1/6] NVMe storage already mounted or not available, skipping."
fi

# ---------------------------------------------------------------
# 2. Clone repo (skip if already present)
# ---------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
    echo "[2/6] Repo already cloned at $REPO_DIR, pulling latest..."
    cd "$REPO_DIR"
    git pull --ff-only || true
else
    echo "[2/6] Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# ---------------------------------------------------------------
# 3. Install Python dependencies
# ---------------------------------------------------------------
echo "[3/6] Installing Python dependencies..."
pip install -q -r requirements.txt
pip install -q zstandard

# ---------------------------------------------------------------
# 4. Download data
# ---------------------------------------------------------------
DATA_DIR="$REPO_DIR/data/datasets/fineweb10B_$VARIANT"
if [ -d "$DATA_DIR" ] && ls "$DATA_DIR"/fineweb_train_*.bin 1>/dev/null 2>&1; then
    SHARD_COUNT=$(ls "$DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
    echo "[4/6] Data already downloaded ($SHARD_COUNT training shards found), skipping."
else
    echo "[4/6] Downloading data (variant=$VARIANT, full 80 shards)..."
    python3 data/cached_challenge_fineweb.py --variant "$VARIANT"
fi

# ---------------------------------------------------------------
# 5. Verify GPU setup
# ---------------------------------------------------------------
echo "[5/6] Checking GPUs..."
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo "  Found $GPU_COUNT GPU(s):"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    # Quick NCCL check
    if [ "$GPU_COUNT" -ge 8 ]; then
        echo "  8 GPUs detected — ready for full autoresearch runs."
    else
        echo "  WARNING: Expected 8 GPUs for p5.48xlarge, found $GPU_COUNT."
    fi
else
    echo "  WARNING: nvidia-smi not found. NVIDIA drivers may not be installed."
    echo "  On Deep Learning AMIs, drivers should be pre-installed."
fi

# ---------------------------------------------------------------
# 6. Smoke test (single GPU, short run)
# ---------------------------------------------------------------
echo "[6/6] Running smoke test (1 GPU, 200 iterations)..."
cd "$REPO_DIR"
RUN_ID=smoke_test \
DATA_PATH="$REPO_DIR/data/datasets/fineweb10B_$VARIANT/" \
TOKENIZER_PATH="$REPO_DIR/data/tokenizers/fineweb_1024_bpe.model" \
VOCAB_SIZE=1024 \
ITERATIONS=200 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=120 \
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
echo "To start autoresearch:"
echo ""
echo "  tmux new -s autoresearch"
echo "  cd $REPO_DIR && claude"
echo ""
echo "Tip: Run inside tmux so SSH disconnects don't kill your session."
