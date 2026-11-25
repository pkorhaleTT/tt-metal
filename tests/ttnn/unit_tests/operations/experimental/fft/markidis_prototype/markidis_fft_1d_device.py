"""
Markidis-3 emulated 1-D matmul FFT on Tenstorrent FPU.

Direct-matmul FFT:   X = W_N @ x,   where W_N is the N x N complex DFT matrix.

Complex matmul = 4 real matmuls:
    C_r = A_r B_r - A_i B_i
    C_i = A_r B_i + A_i B_r
With Markidis-3, each real matmul is 3 bf16 matmul passes.
=> 12 bf16 matmul passes per direct-matmul 1-D FFT.

This is O(N^2) and only practical for small-to-medium N (<= 4096 for a 1-D).
For large N, use the two-pass factorization (in the next file).

Accuracy target (host sim): 3-4e-6 vs fp64 reference at N=1024.
"""

import numpy as np
import torch
import ttnn

from markidis_matmul_device import (
    markidis3_matmul,
    bf16_matmul_baseline,
    _rel_l2,
)


# ---------------------------------------------------------------------------- #
# DFT matrix                                                                   #
# ---------------------------------------------------------------------------- #

def dft_matrix_torch(N: int) -> torch.Tensor:
    """Complex fp32 DFT matrix of size N x N."""
    k = torch.arange(N, dtype=torch.float64)
    n = k.view(-1, 1)
    W = torch.exp(-2j * torch.pi * k * n / N)  # complex128
    return W.to(torch.complex64)


# ---------------------------------------------------------------------------- #
# Complex matmul via a real-matmul function                                    #
# ---------------------------------------------------------------------------- #

def complex_matmul_via(
    matmul_fn,
    A_complex: torch.Tensor,
    B_complex: torch.Tensor,
    device,
) -> torch.Tensor:
    """
    Complex matmul using 4 real-matmul calls.
    A, B: complex64 host tensors. Returns complex64.
    matmul_fn(A_fp32, B_fp32, device) -> fp32 result.
    """
    A_r = A_complex.real.contiguous().to(torch.float32)
    A_i = A_complex.imag.contiguous().to(torch.float32)
    B_r = B_complex.real.contiguous().to(torch.float32)
    B_i = B_complex.imag.contiguous().to(torch.float32)

    # 4 real matmul calls
    ArBr = matmul_fn(A_r, B_r, device)
    AiBi = matmul_fn(A_i, B_i, device)
    ArBi = matmul_fn(A_r, B_i, device)
    AiBr = matmul_fn(A_i, B_r, device)

    C_r = ArBr - AiBi
    C_i = ArBi + AiBr
    return torch.complex(C_r, C_i)


# ---------------------------------------------------------------------------- #
# 1-D direct-matmul FFT                                                        #
# ---------------------------------------------------------------------------- #

def fft_1d_markidis(x_complex: torch.Tensor, device) -> torch.Tensor:
    """
    1-D FFT of a complex64 vector x of length N via W_N @ x with Markidis-3.
    Returns complex64 result on host.
    """
    N = x_complex.shape[-1]
    W = dft_matrix_torch(N)                 # (N, N) complex64
    x_col = x_complex.reshape(N, 1)         # (N, 1)
    y = complex_matmul_via(markidis3_matmul, W, x_col, device)
    return y.reshape(-1)


def fft_1d_bf16(x_complex: torch.Tensor, device) -> torch.Tensor:
    """1-D FFT via 1-mul bf16 matmul baseline (cryo-EM team's approach)."""
    N = x_complex.shape[-1]
    W = dft_matrix_torch(N)
    x_col = x_complex.reshape(N, 1)
    y = complex_matmul_via(bf16_matmul_baseline, W, x_col, device)
    return y.reshape(-1)


# ---------------------------------------------------------------------------- #
# Self-test                                                                    #
# ---------------------------------------------------------------------------- #

def self_test(device):
    print("=" * 76)
    print("1-D matmul FFT (direct DFT) on device \u2014 rel L2 vs fp64 numpy.fft")
    print("=" * 76)
    header = f"{'N':>6} {'bf16 (1-mul)':>16} {'Markidis-3':>14} {'native SFPU ttnn.fft':>22}"
    print(header)

    torch.manual_seed(0)
    for N in [64, 256, 1024]:
        x = torch.randn(N, dtype=torch.float32) + 1j * torch.randn(N, dtype=torch.float32)
        x = x.to(torch.complex64)
        ref = torch.fft.fft(x.to(torch.complex128))

        y_bf16 = fft_1d_bf16(x, device)
        y_m3 = fft_1d_markidis(x, device)

        # Also compare against existing ttnn.experimental.fft (SFPU fp32)
        # It takes real input; for a complex input we use the two-input form.
        try:
            tt_re = ttnn.from_torch(
                x.real.reshape(1, N), dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
            )
            tt_im = ttnn.from_torch(
                x.imag.reshape(1, N), dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
            )
            re_out, im_out = ttnn.experimental.fft(tt_re, tt_im)
            y_sfpu = torch.complex(
                ttnn.to_torch(re_out).reshape(-1).to(torch.float32),
                ttnn.to_torch(im_out).reshape(-1).to(torch.float32),
            )
            sfpu_err = _rel_l2(y_sfpu, ref.to(torch.complex64))
            ttnn.deallocate(tt_re); ttnn.deallocate(tt_im)
            ttnn.deallocate(re_out); ttnn.deallocate(im_out)
        except Exception as e:
            sfpu_err = float("nan")

        print(f"{N:>6d} "
              f"{_rel_l2(y_bf16, ref.to(torch.complex64)):>16.2e} "
              f"{_rel_l2(y_m3, ref.to(torch.complex64)):>14.2e} "
              f"{sfpu_err:>22.2e}")


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    try:
        self_test(dev)
    finally:
        ttnn.close_device(dev)
