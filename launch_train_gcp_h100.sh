#!/usr/bin/env bash
set -euo pipefail

detect_iface() {
    local iface
    iface="$(ip route show default 2>/dev/null | awk '/default/ {print $5; exit}')"
    if [[ -z "${iface}" ]]; then
        iface="$(ip -o link show up 2>/dev/null | awk -F': ' '$2 != "lo" {print $2; exit}')"
    fi
    if [[ -z "${iface}" ]]; then
        echo "Could not determine a non-loopback network interface for NCCL_SOCKET_IFNAME." >&2
        exit 1
    fi
    printf '%s\n' "${iface}"
}

strip_gib_from_ld_library_path() {
    local filtered=""
    local entry
    IFS=':' read -r -a entries <<< "${LD_LIBRARY_PATH-}"
    for entry in "${entries[@]}"; do
        if [[ -z "${entry}" || "${entry}" == "/usr/local/gib/lib64" ]]; then
            continue
        fi
        if [[ -n "${filtered}" ]]; then
            filtered+=":"
        fi
        filtered+="${entry}"
    done
    printf '%s\n' "${filtered}"
}

# Some GCP Deep Learning VM images inject nccl-gib environment by default.
# On single-node H100 A3 instances that can break plain torchrun with:
#   "ncclInvalidUsage ... Error: network gIB not found"
# Clear the gIB-specific overrides and force the local socket path instead.
unset NCCL_NET
unset NCCL_TUNER_CONFIG_PATH
unset NCCL_CROSS_NIC
unset NCCL_NET_GDR_LEVEL
unset NCCL_P2P_NET_CHUNKSIZE
unset NCCL_NVLS_CHUNKSIZE
unset NCCL_IB_ADAPTIVE_ROUTING
unset NCCL_IB_QPS_PER_CONNECTION
unset NCCL_IB_TC
unset NCCL_IB_FIFO_TC

export LD_LIBRARY_PATH="$(strip_gib_from_ld_library_path)"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$(detect_iface)}"

if [[ $# -eq 0 ]]; then
    set -- train_gpt.py
fi

exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" "$@"
