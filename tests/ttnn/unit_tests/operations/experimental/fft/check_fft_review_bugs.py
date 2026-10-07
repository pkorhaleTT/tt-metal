# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Manual check of the five device findings from the #21412 FFT review.
# Run on a Wormhole or Blackhole machine, from the repo root:
#
#   python tests/ttnn/unit_tests/operations/experimental/fft/check_fft_review_bugs.py

import torch

import ttnn

device = ttnn.open_device(device_id=0)


def to_dev(x, mem=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(x, dtype=ttnn.float32, layout=layout, device=device, memory_config=mem)


def expect_error(label, fn, text):
    try:
        fn()
    except RuntimeError as exc:
        message = str(exc)
        if text in message:
            print(f"PASS  {label}")
            print(f"      {message.splitlines()[0]}")
            return
        print(f"FAIL  {label}")
        print(f"      expected {text!r} in: {message}")
        return
    print(f"FAIL  {label}: call returned instead of raising")


# Bug 1: these used to abort the process. They must raise before the twiddle table is built.
expect_error(
    "bug 1a  batch 3, N=4096",
    lambda: ttnn.experimental.fft(to_dev(torch.randn(3, 4096))),
    "power of two",
)
expect_error(
    "bug 1b  TILE layout, N=4096",
    lambda: ttnn.experimental.fft(to_dev(torch.randn(1, 4096), layout=ttnn.TILE_LAYOUT)),
    "ROW_MAJOR",
)

# Bug 2: a DRAM complex_mul followed by the same shapes with b in L1 used to return NaN.
a_r, a_i, b_r, b_i = (torch.randn(64, 256) for _ in range(4))
ttnn.experimental.complex_mul(to_dev(a_r), to_dev(a_i), to_dev(b_r), to_dev(b_i))
o_r, o_i = ttnn.experimental.complex_mul(
    to_dev(a_r),
    to_dev(a_i),
    to_dev(b_r, ttnn.L1_MEMORY_CONFIG),
    to_dev(b_i, ttnn.L1_MEMORY_CONFIG),
)
got = torch.complex(ttnn.to_torch(o_r).to(torch.float32), ttnn.to_torch(o_i).to(torch.float32))
ref = torch.complex(a_r, a_i) * torch.complex(b_r, b_i)
rel = (torch.linalg.norm(got - ref) / torch.linalg.norm(ref)).item()
if torch.isnan(got).any() or rel > 5e-4:
    print(f"FAIL  bug 2   complex_mul DRAM then L1 b   nan={torch.isnan(got).any().item()} rel={rel:.3e}")
else:
    print(f"PASS  bug 2   complex_mul DRAM then L1 b   rel={rel:.3e}")

def compare_fft(label, host, tol):
    """Device fft versus torch.fft.fft on the CPU. Relative error over the whole tensor."""
    ref = torch.fft.fft(host.to(torch.complex64), dim=-1)
    re, im = ttnn.experimental.fft(to_dev(host))
    got = torch.complex(ttnn.to_torch(re).to(torch.float32), ttnn.to_torch(im).to(torch.float32))
    rel = (torch.linalg.norm(got - ref) / torch.linalg.norm(ref).clamp_min(1e-12)).item()
    shape_ok = tuple(re.shape) == tuple(host.shape) and tuple(im.shape) == tuple(host.shape)
    status = "PASS" if shape_ok and rel < tol else "FAIL"
    print(f"{status}  {label}  shape={tuple(re.shape)}  rel_vs_cpu={rel:.3e}  tol={tol:.0e}")
    # First four bins of the first row, so the numbers can be read next to the CPU result.
    got_row = got.reshape(-1, host.shape[-1])[0, :4]
    ref_row = ref.reshape(-1, host.shape[-1])[0, :4]
    for i in range(4):
        print(f"      bin {i}: device={got_row[i].item(): .6f}  cpu={ref_row[i].item(): .6f}")


# Bug 3, plus a few supported lengths. Each one is compared with torch.fft on the CPU.
torch.manual_seed(0)
compare_fft("cpu  Stockham N=256", torch.randn(1, 256), 5e-4)
compare_fft("cpu  two-pass N=4096", torch.randn(1, 4096), 1e-3)
compare_fft("bug 3 rank-3 (2, 2, 32768)", torch.randn(2, 2, 32768), 1e-3)
compare_fft("cpu  Bluestein N=100", torch.randn(1, 100), 5e-3)

# Bug 4: non-power-of-two batch must say to pad, not to call ttnn.experimental.fft again.
expect_error(
    "bug 4   batch 3, N=64",
    lambda: ttnn.experimental.fft(to_dev(torch.randn(3, 64))),
    "power of two",
)

# Bug 5: N=44100 is past the fp32 padded-row limit.
expect_error(
    "bug 5   N=44100",
    lambda: ttnn.experimental.fft(to_dev(torch.randn(1, 44100))),
    "multiple of 1024",
)

ttnn.close_device(device)
