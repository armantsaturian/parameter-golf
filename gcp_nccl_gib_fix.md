# GCP H100 NCCL Launch Fix

This note documents a distributed launch failure we hit on a Google Cloud `a3-highgpu-8g` instance and how it was resolved.

## Symptom

The machine had healthy GPUs at the host level:

- `nvidia-smi` saw 8x H100 80GB GPUs
- `torch.cuda.is_available()` was `True`
- `torch.cuda.device_count()` returned `8`

Single-GPU training worked, but multi-GPU `torchrun` failed before training started.

The key error from a minimal NCCL smoke test was:

```text
torch.distributed.DistBackendError: NCCL error ...
ncclInvalidUsage: This usually reflects invalid usage of NCCL library.
Last error:
Error: network gIB not found.
```

## Root Cause

This GCP image was automatically injecting `nccl-gib` environment settings at shell startup:

- `/etc/profile.d/nccl_env.sh`
- `/usr/local/gib/scripts/set_nccl_env.sh`

Those scripts exported settings such as:

```bash
export NCCL_NET=gIB
export NCCL_TUNER_CONFIG_PATH=/usr/local/gib/configs/tuner_config_a3u.txtpb
```

On this instance, the `gIB` network path was not actually available to NCCL, so `dist.init_process_group(backend="nccl")` failed during startup.

## Resolution

Instead of changing the system image, the fix was applied at launch time.

The wrapper script `launch_train_gcp_h100.sh` does four things:

1. Unsets the injected `gIB`-specific NCCL environment variables.
2. Removes `/usr/local/gib/lib64` from `LD_LIBRARY_PATH`.
3. Forces NCCL off the broken IB/gIB path with `NCCL_IB_DISABLE=1`.
4. Sets `NCCL_SOCKET_IFNAME` to the active non-loopback interface and launches `torchrun`.

One verified one-off debugging launch on this VM was:

```bash
env \
  -u NCCL_NET \
  -u NCCL_TUNER_CONFIG_PATH \
  -u NCCL_CROSS_NIC \
  -u NCCL_NET_GDR_LEVEL \
  -u NCCL_P2P_NET_CHUNKSIZE \
  -u NCCL_NVLS_CHUNKSIZE \
  -u NCCL_IB_ADAPTIVE_ROUTING \
  -u NCCL_IB_QPS_PER_CONNECTION \
  -u NCCL_IB_TC \
  -u NCCL_IB_FIFO_TC \
  LD_LIBRARY_PATH= \
  NCCL_IB_DISABLE=1 \
  NCCL_SOCKET_IFNAME=enp0s12 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

In practice, use the wrapper instead of hard-coding the interface:

```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
./launch_train_gcp_h100.sh
```

## Verification

The fix was verified in two stages:

1. A minimal 8-rank NCCL all-reduce smoke test succeeded.
2. `train_gpt.py` launched on all 8 GPUs, logged `world_size:8`, and completed a real training step.

The local verification run launched `train_gpt.py` at `world_size:8` and completed a real training step across all 8 GPUs.

## Notes

- The instance had 8 GPUs, not 7.
- The problem was not CUDA visibility, drivers, or dataset availability.
- The failure happened before model code, inside NCCL distributed initialization.
- If a future GCP image ships with a working `gIB` path on this machine family, this workaround may no longer be necessary.
