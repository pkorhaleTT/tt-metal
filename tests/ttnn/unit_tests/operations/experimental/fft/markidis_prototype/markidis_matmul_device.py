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
# ttnn dtype resolver (works across ttnn versions)                             #
# ---------------------------------------------------------------------------- #

def _bf16_dtype():
    """Return ttnn bfloat16 dtype, tolerating different ttnn API versions."""
    if hasattr(ttnn, "bfloat16"):
        return ttnn.bfloat16
    if hasattr(ttnn, "DataType"):
        return ttnn.DataType.BFLOAT16
    raise RuntimeError("Cannot locate ttnn bfloat16 dtype in this ttnn build")


def _fp32_dtype():
    if hasattr(ttnn, "float32"):
        return ttnn.float32
    if hasattr(ttnn, "DataType"):
        return ttnn.DataType.FLOAT32
    raise RuntimeError("Cannot locate ttnn float32 dtype in this ttnn build")


_CACHED_HIFI4_CONFIG = None


def _hifi4_compute_config(device=None):
    """
    Build a compute-kernel config that forces HiFi4 math + fp32 destination
    accumulation. CRITICAL for Markidis: without fp32_dest_acc_en=True the
    internal matmul accumulator rounds back to bf16 between tiles and the
    precision benefit of split-bf16 is destroyed.
    """
    global _CACHED_HIFI4_CONFIG
    if _CACHED_HIFI4_CONFIG is not None:
        return _CACHED_HIFI4_CONFIG

    # Preferred: arch-aware helper (works for WH and BH)
    if device is not None and hasattr(ttnn, "init_device_compute_kernel_config"):
        try:
            _CACHED_HIFI4_CONFIG = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            return _CACHED_HIFI4_CONFIG
        except Exception:
            pass

    # Fallback: direct constructor (WH = BH alias in current ttnn)
    for cls_name in ("WormholeComputeKernelConfig", "BlackholeComputeKernelConfig"):
        cls = getattr(ttnn, cls_name, None)
        if cls is not None:
            try:
                _CACHED_HIFI4_CONFIG = cls(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                )
                return _CACHED_HIFI4_CONFIG
            except Exception:
                continue

    # Last resort: return None and let ttnn.matmul use defaults
    return None


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
        dtype=_bf16_dtype(),
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
    accum_dtype=None,
) -> torch.Tensor:
    """
    Compute A @ B at ~fp32 accuracy via 3 bf16 matmul passes on the FPU.
    A: (M, K) fp32 torch tensor on host.
    B: (K, N) fp32 torch tensor on host.
    Returns: (M, N) fp32 torch tensor on host.

    CRITICAL: accum_dtype MUST be fp32. The partial products P2 = A_hi @ B_lo
    and P3 = A_lo @ B_hi are small corrections (~1e-3 of P1's magnitude).
    If stored in bf16, bf16's rounding completely erases those corrections
    and Markidis degenerates back to a plain 1-mul bf16 matmul.
    bf16 inputs + bf16 compute + fp32 output = the correct configuration.
    """
    assert A_fp32.dtype == torch.float32 and B_fp32.dtype == torch.float32
    if accum_dtype is None:
        accum_dtype = _fp32_dtype()  # fp32 storage is REQUIRED for precision recovery

    A_hi_t, A_lo_t = split_fp32_to_bf16_pair(A_fp32)
    B_hi_t, B_lo_t = split_fp32_to_bf16_pair(B_fp32)

    A_hi = _upload_bf16(A_hi_t, device)
    A_lo = _upload_bf16(A_lo_t, device)
    B_hi = _upload_bf16(B_hi_t, device)
    B_lo = _upload_bf16(B_lo_t, device)

    # Force HiFi4 + fp32 destination accumulation on every matmul.
    # Without this, matmul defaults to LoFi + bf16 dest and the Markidis
    # precision recovery collapses back to plain bf16 accuracy.
    kcfg = _hifi4_compute_config(device)
    matmul_kw = dict(dtype=accum_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if kcfg is not None:
        matmul_kw["compute_kernel_config"] = kcfg

    # 3 bf16-input matmul passes on FPU, fp32-output storage
    P1 = ttnn.matmul(A_hi, B_hi, **matmul_kw)
    P2 = ttnn.matmul(A_hi, B_lo, **matmul_kw)
    P3 = ttnn.matmul(A_lo, B_hi, **matmul_kw)

    # fp32 + fp32 adds to preserve precision of the small correction terms
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
    kcfg = _hifi4_compute_config(device)
    mm_kw = dict(dtype=_bf16_dtype(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if kcfg is not None:
        mm_kw["compute_kernel_config"] = kcfg
    C = ttnn.matmul(A, B, **mm_kw)
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
