# Exp 126: DRM + Proprietary Tracing Matrix

**Date:** 2026-03-30
**Hardware:** Titan V (GV100) @ 0000:03:00.0, RTX 5060 (GB206) @ 0000:21:00.0, Tesla K80 (GK210) @ 0000:4c:00.0/4d:00.0
**Purpose:** Map all non-VFIO dispatch paths for Titan V. Solve the maze from both sides.
**Status:** PAUSED — deprioritized in favor of VBIOS DEVINIT track (Exp 141-142). DRM tracing remains useful for comparison but is not on the critical path.

## Current Dual-Use Setup

```
RTX 5060 (0000:21:00.0) ─── nvidia proprietary ─── display + UVM dispatch candidate
Titan V  (0000:03:00.0) ─── vfio-pci ──────────── sovereign compute target
K80 #1   (0000:4c:00.0) ─── vfio-pci ──────────── Kepler sovereign (no security)
K80 #2   (0000:4d:00.0) ─── vfio-pci ──────────── Kepler sovereign (no security)
```

## Path B: nvidia-drm + UVM for Titan V

### What Exists (code-complete in coralReef)

`NvUvmComputeDevice` in `crates/coral-driver/src/nv/uvm_compute.rs`:
- Full RM allocation chain: device → subdevice → UVM GPU → VA space → channel group → GPFIFO → compute engine bind
- VOLTA_USERMODE_A doorbell mapped
- NOP GPFIFO smoke test in `open()`
- QMD build + push buffer submission
- GPFIFO completion polling + sync

`NvDrmDevice` in `crates/coral-driver/src/nv/nvidia_drm.rs`:
- DRM render node probe + identity
- Delegates all compute to `NvUvmComputeDevice`

### What's Needed to Test

1. **Load nvidia proprietary for Titan V** — currently blocked by `vfio-pci.ids=10de:1d81` in kernel cmdline
   - Option A: Remove Titan V from vfio-pci.ids, reboot, load nvidia for both 5060+Titan V
   - Option B: Use glowplug to dynamically swap Titan V from vfio → nvidia (if nvidia accepts late-bound GPU)
   - Option C: Boot with Titan V on nvidia, run UVM tests, then swap to vfio for sovereign work

2. **Run UVM dispatch tests**:
   ```bash
   CORALREEF_UVM_BDF=0000:03:00.0 \
     cargo test --test hw_nv_uvm -p coral-driver --features nvidia-drm -- --ignored
   ```

3. **Trace RM initialization sequence** — what ioctls does nvidia RM issue to initialize FECS/GPCCS on Volta?
   - Instrument `NvUvmComputeDevice::open()` with detailed logging
   - Compare register states (CPUCTL, SCTL, MAILBOX0) before/after RM init
   - This tells us exactly what sequence sovereign boot needs

### Value for Sovereign Pipeline

If UVM dispatch works on Titan V:
- **Immediate:** Proves coralReef QMD/GPFIFO code is correct for SM70
- **Learning:** RM init sequence reveals how FECS gets into a usable state
- **Warm handoff variant:** Load nvidia, let RM init FECS, swap to vfio (nvidia warm handoff)
- **Reference:** UVM dispatch timing provides baseline for sovereign VFIO perf comparison

## Path A: VFIO Warm Handoff (Current Frontier)

### Livepatch Strategy (Exp 125)

Kernel livepatch module `livepatch_nvkm_mc_reset.ko` NOPs four nouveau functions:
1. `nvkm_mc_reset` — prevents PMC engine reset
2. `gf100_gr_fini` — prevents GR teardown
3. `nvkm_falcon_fini` — prevents falcon halt
4. `gk104_runl_commit` — prevents empty runlist submission (FECS self-reset)

Dynamic control via `coralctl warm-fecs`:
1. **Disable** livepatch before nouveau loads (all functions run normally during init)
2. **Load** nouveau, wait 12s for GR init (FECS boots via ACR, enters HS mode)
3. **Enable** livepatch (all 4 NOPs active — runlist frozen, teardown blocked)
4. **Swap** to vfio-pci (ember disables reset_method to prevent PCI bus reset)

### What We Need to Verify

- Does FECS remain in HALT (not HRESET) after this sequence?
- Is the FECS method interface responsive (can we issue GR commands)?
- Does GPFIFO submission + doorbell reach PBDMA with FECS alive?

## Path C: nouveau DRM (Blocked)

### Current Blocker

Titan V (GV100) nouveau DRM compute is blocked by missing PMU firmware:
- `CHANNEL_ALLOC` fails without PMU
- NVIDIA does not distribute signed PMU firmware for desktop Volta
- Firmware inventory: ACR (present), GR (present), SEC2 (present), NVDEC (present), PMU (missing)

### Potential Workarounds

1. **Extract PMU firmware from nvidia proprietary driver** — nvidia-535 VM captures in `data/titanv/` may contain PMU blob
2. **GSP firmware path** — only available on Ampere+ (not Volta)
3. **Bypass PMU requirement** — investigate if nouveau can be patched to skip PMU check for compute-only channels

## Cross-Path Register Tracing

### Glowplug Swap Journal

coral-ember records swap events with timing data. After warm-fecs cycle:
- Pre-swap register snapshot (nouveau state)
- Post-swap register snapshot (vfio state)
- Timing: unbind_ms, bind_ms, total_ms

### VM Capture Data (Exp 124)

`data/k80/nvidia470-vm-captures/` and `data/titanv/nvidia535-vm-captures/` contain proprietary driver register traces from VM passthrough runs. Cross-reference with our sovereign pipeline to identify:
- Register initialization order
- FECS boot sequence from RM perspective
- PMU interaction patterns
- WPR2 setup sequence

## Decision Matrix

| Path | Effort | Reward | Risk |
|------|--------|--------|------|
| VFIO warm (livepatch) | Low (ready to test) | High (full sovereignty) | Medium (FECS may not survive) |
| nvidia+UVM trace | Medium (reboot needed) | High (learning + validation) | Low (code-complete) |
| nouveau DRM | High (PMU firmware extraction) | Medium (not sovereign) | High (may not work) |
| NVK/wgpu fallback | Zero (already works) | Low (not sovereign) | Zero |

**Recommended order:** Test VFIO warm first (zero additional setup). If blocked, pivot to nvidia+UVM tracing (requires one reboot to reconfigure kernel cmdline). NVK/wgpu available immediately for physics work regardless.
