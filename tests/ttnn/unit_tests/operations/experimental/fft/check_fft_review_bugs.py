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

# Bug 3: rank-3 input used to come back as (4, 32768).
re, im = ttnn.experimental.fft(to_dev(torch.randn(2, 2, 32768)))
if tuple(re.shape) == (2, 2, 32768) and tuple(im.shape) == (2, 2, 32768):
    print(f"PASS  bug 3   shape {tuple(re.shape)}")
else:
    print(f"FAIL  bug 3   real={tuple(re.shape)} imag={tuple(im.shape)}")

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
