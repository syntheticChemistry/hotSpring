# RTX 5060 / Blackwell Dispatch Gaps

**Version:** v0.6.32 | **Date:** 2026-05-15 | **Status:** Active

Documents the known gaps between toadStool's current sovereign dispatch
implementation and what is required for RTX 5060 (Blackwell B, SM 120+).

## Generation Profile (from `cylinder::nv::generation::BLACKWELL_B`)

| Property | Value | Difference from Titan V (Volta) |
|----------|-------|---------------------------------|
| SM range | 120–MAX | 70–74 |
| QMD version | V50 (96 words, 384 bytes) | V22 (64 words, 256 bytes) |
| Channel class | 0xC96F | 0xC36F |
| Compute class | 0xCEC0 | 0xC3C0 |
| Launch method | PCAS2 (0x02C0) | PCAS (0x02BC) |
| Completion | SemaphoreFence | GpGetPoll |
| Boot strategy | KmodPromote | AcrSec2 |
| Memory | GDDR7 | HBM2 |
| NCTAID source | DriverCbuf7 | SystemRegister |
| USERD GP_GET | **absent** | present |
| Full-rate FP64 | no (1:32) | yes (1:2) |
| HW FP64 RCP | no | yes |

## Gap 1: SemaphoreFence Completion (Critical)

**Current:** `sync()` in `compute_device.rs` polls `USERD GP_GET` until it
matches `GP_PUT`. This works for all pre-Blackwell GPUs.

**Blackwell:** `userd_gp_get: false` — the GP_GET field does not exist in
USERD. Blackwell uses `CompletionStrategy::SemaphoreFence`: the compute
engine writes a semaphore value to a GPU-visible address on dispatch
completion. The host polls this address instead of USERD.

**Required work:**
- Allocate a DMA buffer for the semaphore (4 bytes, GPU-writable)
- Append a `RELEASE_NAMED_BARRIER + SEMAPHORE_RELEASE` pushbuffer
  sequence after the compute dispatch
- In `sync()`, poll the semaphore buffer instead of USERD GP_GET
- Branch on `profile.completion` to select the right sync path

**Reference:** NVK `nv_push.h` semaphore release pattern; CUDA driver
traces show `RELEASE_NAMED_BARRIER` followed by `MEM_OP_C` for fence.

## Gap 2: PCAS2 Launch Method

**Current:** `PushBuf::compute_dispatch_with_launch` branches on
`LaunchMethod::Pcas` (method 0x02BC) and `LaunchMethod::Pcas2`
(method 0x02C0). The PCAS2 path exists in code.

**Status:** Implemented but **untested on hardware**. The QMD payload
layout for V50 is larger (96 words vs 64 words). `build_qmd` handles
this via `profile.qmd_word_count`.

**Required work:**
- Hardware validation on RTX 5060
- Verify QMD V50 field layout matches CUDA driver traces
- Confirm PCAS2 method ID in pushbuffer is correct for GB202

## Gap 3: QMD V50 (96-word) Layout

**Current:** `build_qmd` dispatches on `QmdVersion` and produces the
correct word count. V50 QMD has additional fields for Blackwell features
(e.g., new scheduling modes, per-CTA resource limits).

**Status:** Layout is partially implemented from CUDA R580 driver traces.

**Required work:**
- Cross-validate V50 QMD against NVK and nouveau Blackwell patches
- Verify driver constant buffer (CBUF7) encoding for `num_workgroups`
- Test with real SM120 hardware

## Gap 4: DriverCbuf7 for num_workgroups

**Current:** Pre-Blackwell GPUs use `S2R NCTAID_X/Y/Z` system registers
to read the grid dimensions inside shaders. The `NCTAID_SOURCE` profile
field tracks this.

**Blackwell:** `S2R NCTAID` is broken/deprecated. Grid dimensions must
be loaded from driver constant buffer 7 (`LDC c[7][0/4/8]`). The
compiler (coralReef) must emit `LDC` instead of `S2R` for `@builtin(num_workgroups)`.

**Required work:**
- `encode_driver_constants` must populate CBUF7 with grid dims on Blackwell
- coralReef must handle the NCTAID source difference per target SM
- Validation binary must confirm `@builtin(num_workgroups)` works

## Gap 5: sm_unknown PCI Device ID Mapping

**Current:** `dispatch/capabilities.rs` maps PCI device IDs to SM
architecture strings for the `compute.dispatch.capabilities` response.
Unknown NVIDIA IDs fall through to `"sm_unknown"`.

**Blackwell consumer:** RTX 5060 (GB206) PCI IDs are not yet in the
match table. Cylinder's BAR0 SM probing works regardless of the PCI ID
table, but the capabilities response will report `sm_unknown` for
Blackwell consumer GPUs.

**Required work:**
- Add GB202/GB203/GB205/GB206/GB207 PCI device IDs to the match table
- Source IDs from PCI ID database or CUDA driver `nv-pci.ids`
- Alternatively: use `profile_for_sm(probed_sm).name` as the arch string

## Gap 6: KmodPromote Boot Strategy

**Current:** Sovereign boot on Volta uses `AcrSec2` (warm handoff from
nouveau preserves FECS firmware in IMEM). Kepler uses `NoAcr` (direct
PIO upload). Both work without kernel module assistance.

**Blackwell:** `KmodPromote` requires the kernel module (`nvidia-drm` or
`nouveau`) to allocate GPU-side GR context buffers via `GPU_PROMOTE_CTX`
because Blackwell's WPR (Write-Protected Region) blocks direct VFIO
buffer allocation in the GR context area.

**Required work:**
- Investigate whether `nouveau` on Blackwell supports the promote path
- If not, sovereign Blackwell may require a minimal kernel helper
- Document the interaction between VFIO passthrough and WPR constraints
- Alternative: use `nvidia-drm` render node for initial context setup,
  then hand off to VFIO for compute dispatch

## Gap 7: FP64 Performance Implications

**Blackwell B consumer (RTX 5060):** FP64 throughput is 1:32 relative
to FP32 (vs Titan V's 1:2). HotQCD physics workloads that use native
f64 will be significantly slower.

**Mitigation options:**
- DF64 (double-float via f32 pair) already supported in barraCuda
- `PrecisionRoutingAdvice` from toadStool can steer f64 workloads to
  Titan V and f32/DF64 workloads to RTX 5060
- coralReef's precision tier selection handles this per-shader

## Priority Order

1. **SemaphoreFence** — blocks all Blackwell dispatch (Gap 1)
2. **PCAS2 + QMD V50** — required for any compute launch (Gaps 2, 3)
3. **DriverCbuf7** — required for correct grid dim reads (Gap 4)
4. **PCI ID mapping** — cosmetic but affects capability reporting (Gap 5)
5. **KmodPromote** — blocks cold boot; warm handoff may still work (Gap 6)
6. **FP64 performance** — routing strategy, not a correctness issue (Gap 7)
