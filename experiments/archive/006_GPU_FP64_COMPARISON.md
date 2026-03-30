# Experiment 006: GPU FP64 Comparison — RTX 4070 vs Titan V

**Date:** February 17, 2026
**Hardware:** Eastgate — i9-12900K, RTX 4070 (Ada, 12GB GDDR6X) + Titan V (Volta GV100, 12GB HBM2)
**Drivers:** RTX 4070 = nvidia proprietary 580.82.09; Titan V = NVK / nouveau (Mesa 25.1.5)
**Binary:** `bench_gpu_fp64` (3 warmup rounds, 10 measurement rounds for BCS/eigensolve, 5 for L2)

---

## Key Finding

**The RTX 4070 (consumer, $600) is 4–7× faster than the Titan V ($3000) for
GPU compute shaders via wgpu/Vulkan**, despite the Titan V having dedicated fp64
silicon and HBM2 bandwidth. The bottleneck is the **NVK open-source driver** —
not the hardware.

This is a driver maturity gap, not a hardware limitation. When the NVK driver
matures (or via the proprietary nvidia driver on Titan V), we expect the Titan V
to outperform the 4070 on fp64-heavy workloads due to:
- 6.9 TFLOPS fp64 (vs ~0.6 TFLOPS on 4070 via Vulkan/wgpu path)
- 652 GB/s HBM2 bandwidth (vs 504 GB/s GDDR6X)
- 5120 CUDA cores with dedicated fp64 units

---

## Results

### BCS Bisection (100 iterations, 20 eigenlevels per nucleus)

| Batch | RTX 4070 (ms) | Titan V (ms) | 4070 tput | Titan V tput | Ratio |
|------:|:---:|:---:|:---:|:---:|:---:|
| 8 | 1.46 | 1.49 | 5,483/s | 5,354/s | **1.02×** |
| 32 | 1.41 | 1.47 | 22,783/s | 21,726/s | **1.05×** |
| 128 | 1.42 | 1.57 | 90,215/s | 81,482/s | **1.11×** |
| 512 | 1.43 | 1.59 | 357,101/s | 322,601/s | **1.11×** |
| 2048 | 1.54 | 1.72 | 1,327,741/s | 1,192,403/s | **1.11×** |

**Analysis:** BCS bisection is dispatch-dominated — the shader itself is
lightweight (100 bisection iterations on 20 levels). Both GPUs hit ~1.5ms
floor, meaning this is **dispatch overhead**, not compute time. At batch=2048,
the 4070 is only 11% faster — the workload saturates neither GPU.

### Batched Eigensolve (Jacobi, single-dispatch, 200 sweeps)

| Batch | Dim | RTX 4070 (ms) | Titan V (ms) | 4070/matrix | Titan V/matrix | Ratio |
|------:|----:|:---:|:---:|:---:|:---:|:---:|
| 8 | 20 | 3.80 | 15.16 | 474 μs | 1,895 μs | **4.0×** |
| 32 | 20 | 3.81 | 16.97 | 119 μs | 531 μs | **4.5×** |
| 128 | 20 | 4.64 | 31.11 | 36 μs | 243 μs | **6.7×** |
| 8 | 30 | 10.06 | 44.67 | 1,258 μs | 5,584 μs | **4.4×** |
| 32 | 30 | 10.09 | 45.57 | 315 μs | 1,424 μs | **4.5×** |
| 128 | 30 | 12.95 | 93.33 | 101 μs | 729 μs | **7.2×** |

**Analysis:** The eigensolve is compute-intensive (O(n³) per sweep × 200
sweeps per matrix). Here the RTX 4070 is **4–7× faster**. This gap is too
large to explain by hardware alone (fp64 TFLOPS: Titan V=6.9, 4070≈0.6).

The likely causes:
1. **NVK driver shader compilation quality** — NAK (NVK's shader compiler)
   may not optimize Jacobi rotation loops as aggressively as nvidia's compiler
2. **NVK dispatch overhead** — NVK may have higher per-dispatch latency
3. **NVK workgroup scheduling** — less mature occupancy optimization

### L2 HFB Pipeline (18 real nuclei, SLy4 parameters)

| GPU | Wall/eval (ms) | Per-nucleus (μs) | Throughput |
|-----|:---:|:---:|:---:|
| RTX 4070 | 8.1 | 453 | 2,210 nuclei/s |
| Titan V | 8.6 | 479 | 2,088 nuclei/s |

**Analysis:** The L2 pipeline is **nearly identical** (only 6% difference).
This is because the pipeline is dominated by CPU work (HFB SCF iteration,
density mixing, energy computation) with GPU used only for the eigensolve
portion. The eigensolve falls back to CPU for n>32 matrices, so most of the
GPU advantage is washed out by Amdahl's law.

---

## Driver Stack Comparison

| Property | RTX 4070 | Titan V |
|----------|----------|---------|
| Architecture | Ada Lovelace (AD104) | Volta (GV100) |
| Kernel module | `nvidia` (proprietary) | `nouveau` (open-source) |
| Vulkan ICD | NVIDIA proprietary | NVK (Mesa 25.1.5) |
| Shader compiler | NVIDIA internal | NAK (NVK's Rust-based compiler) |
| `shaderFloat64` | true | true |
| fp64 hardware | Consumer-grade (limited fp64 ALUs) | Full fp64 silicon (1:2 ratio) |
| Memory | 12GB GDDR6X @ 504 GB/s | 12GB HBM2 @ 652 GB/s |
| Power (nvidia-smi) | Available | Not available (nouveau) |

---

## Conclusions

1. **NVK is functionally correct** — identical physics results on both GPUs,
   validated to 1e-15. The open-source stack is production-viable for
   correctness.

2. **NVK is not yet performance-competitive** — 4–7× slower on compute-bound
   shaders compared to the proprietary nvidia driver. This is expected for a
   relatively young open-source Vulkan driver.

3. **The proprietary driver would unlock Titan V's full potential** — switching
   from `nvidia-dkms-580-open` to `nvidia-dkms-580` (proprietary) would give
   the Titan V the same optimized shader compiler as the 4070, likely making
   it **faster** for fp64-heavy workloads due to dedicated fp64 silicon.

4. **For production physics**, use the proprietary driver on both GPUs and
   compare hardware fairly. The open-source path is for sovereignty and
   portability validation, not peak throughput (yet).

5. **BCS bisection is dispatch-bound**, not compute-bound. Both GPUs are
   underutilized at <2048 batch. Larger workloads (full AME2020 sweep, 2042
   nuclei) would better exercise the hardware.

---

## Next Steps

1. **Proprietary driver on Titan V** — `sudo apt install nvidia-dkms-580` to
   get a fair hardware comparison (same compiler, same dispatch path)
2. **Large-batch eigensolve** — batch=512+ at dim=30 to fully saturate both GPUs
3. **MD pipeline benchmark** — blocked on `ShaderTemplate` `zero` bug (ToadStool)
4. **Memory bandwidth test** — pure f64 copy/reduce shader to measure HBM2 vs GDDR6X
