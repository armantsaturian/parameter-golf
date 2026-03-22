# Parameter Golf on GCP: 8xH100 Spot Instance Setup

## Pre-requisites

- GCP account with GPU quota for `a3-highgpu-8g` in your target zone
- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- SSH key configured (`gcloud compute config-ssh`)

To check your H100 quota:

```bash
gcloud compute regions describe us-central1 \
  --format="table(quotas.filter(metric:NVIDIA_H100_GPUS))"
```

If quota is 0, request an increase at https://console.cloud.google.com/iam-admin/quotas (filter for "NVIDIA H100", select your region, request 8).

## Creating the Instance

**Machine:** a3-highgpu-8g (8x H100 SXM, 208 vCPUs, 1.8TB RAM, 6TB local SSD)
**Spot price:** ~$29.71/hr (vs ~$98.37/hr on-demand)
**Regions with H100s:** us-central1-a, us-east4-a, europe-west4-a (availability varies)

```bash
gcloud compute instances create parameter-golf \
  --zone=us-central1-a \
  --machine-type=a3-highgpu-8g \
  --provisioning-model=SPOT \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

If us-central1-a has no capacity, try:

```bash
# us-east4-a
gcloud compute instances create parameter-golf \
  --zone=us-east4-a \
  --machine-type=a3-highgpu-8g \
  --provisioning-model=SPOT \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

SSH in:

```bash
gcloud compute ssh parameter-golf --zone=us-central1-a
```

## After SSH-ing In

Run the setup script (does everything):

```bash
curl -sL https://raw.githubusercontent.com/YOUR_FORK/parameter-golf/master/setup_gcp.sh | bash
```

Or manually:

```bash
# 1. Clone the repo
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# 2. Install dependencies
pip install -r requirements.txt
pip install zstandard  # better compression for model artifacts

# 3. Download data (sp1024 vocab, full 80 training shards)
python3 data/cached_challenge_fineweb.py --variant sp1024

# 4. Verify GPUs
nvidia-smi

# 5. Run baseline (8x H100, ~10 min)
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Running Autoresearch

Start a tmux session (survives SSH disconnects):

```bash
tmux new -s autoresearch
```

Launch your AI agent inside the repo directory. For Claude Code:

```bash
cd ~/parameter-golf
claude
```

Then tell it to read `program.md` (from autoresearch) adapted for parameter-golf, or give it your own research instructions. The agent will iterate: modify `train_gpt.py`, train for ~10 min, check val_bpb, keep or discard, repeat.

To detach from tmux: `Ctrl+B` then `D`. To reattach: `tmux attach -t autoresearch`.

## Cost Tracking

| Item | Cost |
|------|------|
| Spot price | ~$29.71/hr |
| Per experiment (~10 min training + eval) | ~$5/run |
| Overnight (8 hrs, ~48 experiments) | ~$237 |
| Full day (24 hrs, ~144 experiments) | ~$713 |

**Stop the instance when done:**

```bash
gcloud compute instances stop parameter-golf --zone=us-central1-a
```

**Delete it entirely:**

```bash
gcloud compute instances delete parameter-golf --zone=us-central1-a
```

## Spot Instance Considerations

- The instance can be preempted at any time with 30s notice
- For 10-min training runs, the preemption risk per individual run is low
- If preempted mid-run, you lose that run's progress but not prior commits
- Autoresearch handles crashes gracefully (logs crash, reverts, moves on)
- Always use `tmux` or `screen` so SSH disconnects don't kill the agent
- Consider setting up a startup script or snapshot to recover quickly after preemption

## Tips

- **Smoke test first.** Run a 1-GPU quick test before committing to 8-GPU runs to catch bugs fast.
- **Monitor costs.** Set a budget alert in GCP Billing to avoid surprises.
- **Check spot availability.** If one zone is full, try another. H100 spot capacity fluctuates.
- **Local SSD.** The a3-highgpu-8g comes with 6TB local SSD. For faster data loading, copy shards to `/mnt/disks/` if the local SSDs are mounted there.
