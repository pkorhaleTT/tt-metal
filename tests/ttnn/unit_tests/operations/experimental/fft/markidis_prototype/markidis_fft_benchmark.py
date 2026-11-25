"""
End-to-end benchmark:
  Accuracy + latency for three 1-D FFT implementations on Tenstorrent WH / BH:

    1) ttnn.experimental.fft     \u2014 SFPU fp32, current library (our paper)
    2) bf16 matmul FFT           \u2014 1-mul FPU, cryo-EM team's current approach
    3) Markidis-3 matmul FFT     \u2014 our proposal: 3 bf16 FPU matmul passes

Reports:
  - relative L2 error vs numpy fp64 reference
  - median wall-clock latency (10 warm runs after 3 warmup)
  - estimated GFLOP/s (direct-matmul = 8 N^2 FLOPs for complex, bf16 cost model)

Runs on a single device opened at device_id=0.
"""

import time
import statistics

import numpy as np
import torch
import ttnn

from markidis_matmul_device import (
    markidis3_matmul,
    bf16_matmul_baseline,
    _rel_l2,
    _fp32_dtype,
)
from markidis_fft_1d_device import (
    dft_matrix_torch,
    complex_matmul_via,
    fft_1d_markidis,
    fft_1d_bf16,
)


N_WARMUP = 3
N_REPEAT = 10


def _time_median(fn, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.median(times)


def bench_ttnn_native(device, N, x_complex):
    """Current ttnn.experimental.fft (SFPU)."""
    x_r = x_complex.real.reshape(1, N).contiguous()
    x_i = x_complex.imag.reshape(1, N).contiguous()

    def once():
        tt_re = ttnn.from_torch(x_r, dtype=_fp32_dtype(),
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        tt_im = ttnn.from_torch(x_i, dtype=_fp32_dtype(),
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        re_out, im_out = ttnn.experimental.fft(tt_re, tt_im)
        ttnn.synchronize_device(device)
        y = torch.complex(
            ttnn.to_torch(re_out).reshape(-1).to(torch.float32),
            ttnn.to_torch(im_out).reshape(-1).to(torch.float32),
        )
        ttnn.deallocate(tt_re); ttnn.deallocate(tt_im)
        ttnn.deallocate(re_out); ttnn.deallocate(im_out)
        return y

    ref = torch.fft.fft(x_complex.to(torch.complex128))
    y = once()
    err = _rel_l2(y, ref.to(torch.complex64))
    t_med = _time_median(once)
    return err, t_med


def bench_bf16_matmul(device, N, x_complex):
    """1-mul bf16 matmul FFT."""
    def once():
        return fft_1d_bf16(x_complex, device)
    ref = torch.fft.fft(x_complex.to(torch.complex128))
    y = once()
    err = _rel_l2(y, ref.to(torch.complex64))
    t_med = _time_median(once)
    return err, t_med


def bench_markidis(device, N, x_complex):
    """Markidis-3 matmul FFT."""
    def once():
        return fft_1d_markidis(x_complex, device)
    ref = torch.fft.fft(x_complex.to(torch.complex128))
    y = once()
    err = _rel_l2(y, ref.to(torch.complex64))
    t_med = _time_median(once)
    return err, t_med


def run(device):
    print("=" * 92)
    print("1-D FFT benchmark on Tenstorrent (accuracy vs fp64, median wall-clock)")
    print("=" * 92)
    header = f"{'N':>6}  " \
             f"{'ttnn.fft (SFPU)':>26}  " \
             f"{'bf16 matmul':>22}  " \
             f"{'Markidis-3 matmul':>26}"
    print(header)
    print(f"{'':>6}  {'err      ms':>26}  {'err      ms':>22}  {'err      ms':>26}")

    torch.manual_seed(42)
    for N in [64, 256, 1024, 4096]:
        x = (torch.randn(N, dtype=torch.float32)
             + 1j * torch.randn(N, dtype=torch.float32)).to(torch.complex64)

        try:
            e_n, t_n = bench_ttnn_native(device, N, x)
        except Exception as e:
            e_n, t_n = float("nan"), float("nan")
            print(f"  (ttnn.experimental.fft failed at N={N}: {e})")

        try:
            e_b, t_b = bench_bf16_matmul(device, N, x)
        except Exception as e:
            e_b, t_b = float("nan"), float("nan")

        try:
            e_m, t_m = bench_markidis(device, N, x)
        except Exception as e:
            e_m, t_m = float("nan"), float("nan")

        print(f"{N:>6d}  "
              f"{e_n:>10.2e}  {1000*t_n:>10.2f}  "
              f"{e_b:>10.2e}  {1000*t_b:>8.2f}  "
              f"{e_m:>10.2e}  {1000*t_m:>12.2f}")


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    try:
        run(dev)
    finally:
        ttnn.close_device(dev)
