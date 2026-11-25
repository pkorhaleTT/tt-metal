"""
Markidis-3 emulated fp32 matmul on Tenstorrent FPU (matrix engine).

Idea:
  fp32 x = bf16(hi) + bf16(lo)  where lo = round_bf16(x - round_bf16(x))

  A @ B  (fp32)  \u2248  A_hi @ B_hi  +  A_hi @ B_lo  +  A_lo @ B_hi
                   -------- 3 bf16 matmul passes on the FPU --------

This file exposes:
    markidis3_matmul(A_fp32_torch, B_fp32_torch, device) -> torch.Tensor (fp32)

It uses only ttnn.matmul (bf16) and ttnn.add. No custom kernels.
Intended for Wormhole B0 or Blackhole p300c.

Run:
    pytest -q device/markidis_matmul_device.py --device_id 0
or:
    python device/markidis_matmul_device.py        # self-test, needs a device
"""

import numpy as np
import torch

import ttnn


# ---------------------------------------------------------------------------- #
# bf16 split helpers (host-side)                                               #
# ---------------------------------------------------------------------------- #

def _round_to_bf16_torch(x_fp32: torch.Tensor) -> torch.Tensor:
    """Round fp32 -> bf16, return as fp32 tensor (so arithmetic stays fp32)."""
    # bf16 truncation with round-to-nearest-even
    bits = x_fp32.contiguous().view(torch.int32)
    # add 0x7FFF + ((bits >> 16) & 1) for RNE, then mask low 16 bits
    rnd = (bits + 0x7FFF + ((bits >> 16) & 1)) & ~0xFFFF
    return rnd.view(torch.float32).clone()


def split_fp32_to_bf16_pair(x_fp32: torch.Tensor):
    """x = hi + lo, both representable in bf16. Returns (hi_fp32, lo_fp32)."""
    hi = _round_to_bf16_torch(x_fp32)
    lo = _round_to_bf16_torch(x_fp32 - hi)
    return hi, lo


# ---------------------------------------------------------------------------- #
# ttnn upload helpers                                                          #
# ---------------------------------------------------------------------------- #

def _upload_bf16(t_fp32: torch.Tensor, device) -> "ttnn.Tensor":
    """Upload an fp32 tensor as bf16 on device, TILE layout (required for matmul)."""
    return ttnn.from_torch(
        t_fp32,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ---------------------------------------------------------------------------- #
# Core Markidis-3 matmul                                                       #
# ---------------------------------------------------------------------------- #

def markidis3_matmul(
    A_fp32: torch.Tensor,
    B_fp32: torch.Tensor,
    device,
    accum_dtype=ttnn.bfloat16,
) -> torch.Tensor:
    """
    Compute A @ B at ~fp32 accuracy via 3 bf16 matmul passes on the FPU.
    A: (M, K) fp32 torch tensor on host.
    B: (K, N) fp32 torch tensor on host.
    Returns: (M, N) fp32 torch tensor on host.

    accum_dtype is the storage for each partial matmul output. bf16 is enough
    because the three partials are added in fp32 on host (future: do the add
    on-device in bf16 and only down-cast at the end).
    """
    assert A_fp32.dtype == torch.float32 and B_fp32.dtype == torch.float32

    A_hi_t, A_lo_t = split_fp32_to_bf16_pair(A_fp32)
    B_hi_t, B_lo_t = split_fp32_to_bf16_pair(B_fp32)

    A_hi = _upload_bf16(A_hi_t, device)
    A_lo = _upload_bf16(A_lo_t, device)
    B_hi = _upload_bf16(B_hi_t, device)
    B_lo = _upload_bf16(B_lo_t, device)

    # 3 bf16 matmul passes on FPU
    P1 = ttnn.matmul(A_hi, B_hi, dtype=accum_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    P2 = ttnn.matmul(A_hi, B_lo, dtype=accum_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    P3 = ttnn.matmul(A_lo, B_hi, dtype=accum_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Accumulate on device (two bf16 adds). This keeps precision bf16 at this step;
    # upgrading the add to fp32 is a cheap future improvement.
    S = ttnn.add(P1, P2)
    S = ttnn.add(S, P3)

    out = ttnn.to_torch(S).to(torch.float32)

    # Free device tensors we no longer need
    for t in (A_hi, A_lo, B_hi, B_lo, P1, P2, P3, S):
        ttnn.deallocate(t)

    return out


def bf16_matmul_baseline(A_fp32: torch.Tensor, B_fp32: torch.Tensor, device) -> torch.Tensor:
    """1-mul bf16 matmul baseline (what the cryo-EM team is doing today)."""
    A = _upload_bf16(_round_to_bf16_torch(A_fp32), device)
    B = _upload_bf16(_round_to_bf16_torch(B_fp32), device)
    C = ttnn.matmul(A, B, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.to_torch(C).to(torch.float32)
    for t in (A, B, C):
        ttnn.deallocate(t)
    return out


# ---------------------------------------------------------------------------- #
# Self-test                                                                    #
# ---------------------------------------------------------------------------- #

def _rel_l2(a, b):
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp_min(1e-12)).item()


def self_test(device):
    print("=" * 70)
    print("Markidis-3 matmul self-test on device")
    print("=" * 70)
    torch.manual_seed(0)
    print(f"{'M':>6} {'K':>6} {'N':>6} {'bf16 err':>12} {'Markidis-3 err':>16}")
    for (M, K, N) in [(64, 64, 64), (256, 256, 256), (1024, 256, 256), (1024, 1024, 1024)]:
        A = torch.randn(M, K, dtype=torch.float32)
        B = torch.randn(K, N, dtype=torch.float32)
        ref = (A.double() @ B.double()).float()

        C_bf16 = bf16_matmul_baseline(A, B, device)
        C_m3 = markidis3_matmul(A, B, device)

        print(f"{M:>6d} {K:>6d} {N:>6d} "
              f"{_rel_l2(C_bf16, ref):>12.2e} "
              f"{_rel_l2(C_m3, ref):>16.2e}")


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    try:
        self_test(dev)
    finally:
        ttnn.close_device(dev)
