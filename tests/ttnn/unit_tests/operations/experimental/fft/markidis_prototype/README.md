# Markidis-3 FFT — Device Prototype

Phase 2 of the Markidis-style fp32 emulation research. These scripts run on a
Wormhole B0 or Blackhole p300c device via `ttnn`.

Phase 1 (host-only precision studies) lives one directory up:

- `../markidis_precision_study.py` — matmul precision (bf16 vs 3-mul vs 4-mul)
- `../markidis_fft_sim.py` — full FFT pipeline precision, including the
  MotionCor3 40-iteration pipeline

## Files

| File | What |
|---|---|
| `markidis_matmul_device.py` | Core primitive: `markidis3_matmul(A_fp32, B_fp32, device)` on top of 3 bf16 `ttnn.matmul` calls on the FPU. Also provides the `bf16_matmul_baseline` (1-mul) for comparison. |
| `markidis_fft_1d_device.py` | 1-D direct-matmul FFT (y = W_N @ x) using Markidis-3 for the complex matmul. Also prints the current `ttnn.experimental.fft` (SFPU fp32) result for reference. |
| `markidis_fft_benchmark.py` | Accuracy + median-wall-clock benchmark across N = 64, 256, 1024, 4096 for all three implementations. |

## Running

```bash
# Set up ttnn env (same as your existing FFT work)
source $TT_METAL_HOME/python_env/bin/activate

# Primitive sanity check
python device/markidis_matmul_device.py

# 1-D FFT accuracy check
python device/markidis_fft_1d_device.py

# Full benchmark
python device/markidis_fft_benchmark.py
```

## What success looks like

Host-sim predictions (from `../markidis_fft_sim.py`) that should carry over:

| N | bf16 (1-mul) | **Markidis-3** | native fp32 ref |
|---|---|---|---|
| 64 | 1.95e-3 | **2.93e-6** | 2.45e-08 |
| 256 | 2.18e-3 | **3.61e-6** | 2.48e-08 |
| 1024 | 2.07e-3 | **3.90e-6** | 2.64e-08 |

If device numbers match within 2× of the host sim, Phase 2 is done.

## Open items before Phase 3

- [ ] Fuse the two `ttnn.add` partial-sum steps into a single `ttnn.sum` or a
      fused epilogue. Right now we pay 2 extra memory round-trips.
- [ ] Do the partial-sum accumulation in fp32 (currently bf16) to not throw
      away the precision we just paid 3× to compute.
- [ ] Batch the 4 real matmuls of a complex matmul into a single `matmul`
      with M doubled, so we amortize launch overhead.
- [ ] Two-pass (Cooley-Tukey) for N > 4096. The host sim's two-pass numbers
      look buggy (1e-2 for both bf16 and Markidis-3) — need to fix output
      ordering before re-running.
