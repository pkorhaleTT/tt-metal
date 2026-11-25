"""
Markidis-style fp32 emulation precision study.

Validates the bf16-split decomposition on host (NumPy/PyTorch) before any
device work. Measures the accuracy ceiling for 1-mul, 3-mul, and 4-mul
variants across a range of matrix sizes.

No device needed. Run anywhere with Python + NumPy + PyTorch.
"""
import numpy as np
import torch
import struct


def fp32_to_bf16(x_fp32):
    """Round an fp32 tensor to bf16 by truncating the low 16 bits."""
    # bf16 is upper 16 bits of fp32: sign + 8 exp + top 7 mantissa bits
    # Round-to-nearest-even would be more faithful, but truncation matches
    # what most hardware does by default. For Markidis the split works
    # either way as long as we're consistent.
    x_int = x_fp32.view(np.uint32)
    bf16_int = x_int & 0xFFFF0000  # zero the low 16 bits
    return bf16_int.view(np.float32)


def split_fp32_to_bf16_pair(x):
    """Split x into (hi, lo) where hi = bf16(x) and lo = bf16(x - hi)."""
    hi = fp32_to_bf16(x)
    lo = fp32_to_bf16(x - hi)
    return hi, lo


def markidis_matmul(A, B, n_terms=3):
    """
    Emulated fp32 matmul via bf16-split decomposition.

    n_terms: 1 (baseline bf16), 3 (recommended), or 4 (full).
    All intermediate products are computed in fp32 (simulating fp32-accurate
    accumulation; on real hardware the accumulator is fp32).
    """
    A_hi, A_lo = split_fp32_to_bf16_pair(A)
    B_hi, B_lo = split_fp32_to_bf16_pair(B)

    if n_terms == 1:
        return A_hi @ B_hi
    elif n_terms == 3:
        return A_hi @ B_hi + A_hi @ B_lo + A_lo @ B_hi
    elif n_terms == 4:
        return A_hi @ B_hi + A_hi @ B_lo + A_lo @ B_hi + A_lo @ B_lo
    else:
        raise ValueError(f"n_terms must be 1, 3, or 4, got {n_terms}")


def rel_error(y_hat, y_ref):
    return np.linalg.norm(y_hat - y_ref) / np.linalg.norm(y_ref)


def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("Markidis fp32 emulation precision study")
    print("=" * 70)
    print(f"{'M':>6} {'K':>6} {'N':>6}   {'1-mul':>10}   {'3-mul':>10}   "
          f"{'4-mul':>10}   {'fp64 ref ratio':>14}")

    # Test various shapes relevant to FFT matmul decomposition
    # For N-point FFT as matmul: shape is (B, N) @ (N, N) with N in {64, 128, 256, 512, 1024}
    configs = [
        (64, 64, 64),
        (256, 64, 64),
        (1024, 64, 64),
        (4096, 64, 64),
        (256, 256, 256),
        (1024, 256, 256),
        (1024, 1024, 1024),
        (4096, 1024, 1024),
    ]

    for M, K, N in configs:
        # Generate problem with realistic FFT-like magnitude distribution
        # (not just normal-random — FFT twiddles are on the unit circle,
        # inputs can span wide dynamic range after a few passes)
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)

        # Ground truth in fp64
        ref_fp64 = A.astype(np.float64) @ B.astype(np.float64)

        # Native fp32 (what the SFPU gives you)
        native_fp32 = A @ B

        # Emulated variants
        emu_1 = markidis_matmul(A, B, n_terms=1)
        emu_3 = markidis_matmul(A, B, n_terms=3)
        emu_4 = markidis_matmul(A, B, n_terms=4)

        err_1 = rel_error(emu_1, ref_fp64)
        err_3 = rel_error(emu_3, ref_fp64)
        err_4 = rel_error(emu_4, ref_fp64)
        err_native = rel_error(native_fp32, ref_fp64)

        print(f"{M:>6} {K:>6} {N:>6}   {err_1:>10.2e}   {err_3:>10.2e}   "
              f"{err_4:>10.2e}   native fp32: {err_native:.2e}")

    # ------------------------------------------------------------
    # Composed FFT-like scenario: K matmul passes in sequence,
    # each pass's output feeds the next pass's input.
    # Simulates what happens in a multi-pass Cooley-Tukey FFT.
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("Composed pipeline: K sequential matmul passes (FFT-like)")
    print("=" * 70)
    print(f"{'passes':>8}   {'1-mul err':>12}   {'3-mul err':>12}   "
          f"{'4-mul err':>12}   {'native fp32':>12}")

    for n_passes in [1, 2, 3, 5, 10]:
        x_init = rng.standard_normal((256, 256)).astype(np.float32)
        W = rng.standard_normal((256, 256)).astype(np.float32) / np.sqrt(256)

        # Ground truth
        x_ref = x_init.astype(np.float64)
        for _ in range(n_passes):
            x_ref = x_ref @ W.astype(np.float64)

        # Four variants
        results = {}
        for n_terms in [1, 3, 4]:
            x = x_init.copy()
            for _ in range(n_passes):
                x = markidis_matmul(x, W, n_terms=n_terms)
            results[n_terms] = rel_error(x, x_ref)

        x = x_init.copy()
        for _ in range(n_passes):
            x = x @ W
        native_err = rel_error(x, x_ref)

        print(f"{n_passes:>8}   {results[1]:>12.2e}   {results[3]:>12.2e}   "
              f"{results[4]:>12.2e}   {native_err:>12.2e}")


if __name__ == "__main__":
    main()
