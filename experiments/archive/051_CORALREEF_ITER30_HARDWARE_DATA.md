# Experiment 051: coralReef Iter 30 — Hardware Data Capture + FMA Validation

**Date:** March 10, 2026
**hotSpring:** v0.6.28
**coralReef:** Phase 10, Iteration 30 (`472e5b8`) + docs (`c84137c`)
**barraCuda:** `a012076` (v0.3.4)
**toadStool:** S145 (`969341cd`)

---

## Purpose

Two goals:

1. Validate coralReef Iteration 30's sovereign compilation against hotSpring's
   shader corpus, confirming FMA lowering (`lower_fma` pass) and NVVM bypass
   test hardening maintain parity with Iter 29.

2. Document the hardware data capture plan that coralReef needs from our
   Titan V + RTX 3090 test rig. coralReef updated their hardware testing
   guide (`docs/HARDWARE_TESTING.md`, commit `c84137c`) specifically for
   our two-GPU setup.

---

## Part 1: Sovereign Compile Validation (Iter 30)

### Results

**45/46** shaders compile to native SM70 + SM86 SASS. Identical to Iter 29.

| Metric | SM70 | SM86 |
|--------|------|------|
| Compiled | 45 | 45 |
| Failed | 1 | 1 |
| Total bytes | 219,696 | 219,792 |

**`complex_f64`** remains the sole failure — it is a utility include, not a
standalone compute entry point. This is expected and not a regression.

### Iter 30 Specific Evolutions

| Feature | Impact |
|---------|--------|
| `FmaPolicy::Separate` (`lower_fma` pass) | Splits `FFma→FMul+FAdd` / `DFma→DMul+DAdd`. Enables F64Precise through sovereign compilation. Previously F64Precise was WGSL-text only. |
| FMA lowering ordering | Runs BEFORE f64 transcendental lowering — Newton-Raphson sequences retain internal FMA for convergence. |
| `CompileWgslRequest.fma_policy` | Callers can now request `Separate` per-shader for precision-critical domains. |
| Multi-device compile API | `shader.compile.wgsl.multi` now live in `coralreef-core::service`. |
| NVVM bypass test hardening | Additional `nvvm_bypass.rs` integration tests. |

---

## Part 2: Hardware Data Capture Plan

coralReef's hardware testing guide requests data from our Titan V + RTX 3090
test rig. This data is critical for unblocking NVIDIA dispatch on both the
open-source (nouveau/NVK) and proprietary (`nvidia-drm`) paths.

### Test Rig Inventory

| GPU | Architecture | Driver Available | Needed Tests |
|-----|-------------|-----------------|-------------|
| NVIDIA Titan V | GV100 SM70 (Volta) | nouveau (open) | Channel alloc EINVAL debug, firmware probe, E2E dispatch |
| NVIDIA RTX 3090 | GA102 SM86 (Ampere) | nvidia-drm (proprietary) | UVM RM client, buffer mapping, compute dispatch |

### Step 1: Titan V Diagnostics (nouveau)

```bash
cargo test --test hw_nv_probe -p coral-driver -- --ignored --nocapture 2>&1 | tee nouveau_diag.log
```

### Step 2: Environment Capture

```bash
uname -r
cat /proc/version
modinfo nouveau | head -20
ls -la /dev/dri/renderD*
ls -la /dev/nvidia*

for d in /sys/class/drm/renderD*/device; do
  echo "=== $d ==="
  cat "$d/vendor" "$d/device" 2>/dev/null
  cat "$d/driver_override" 2>/dev/null
done

ls -la /lib/firmware/nvidia/gv100/ 2>/dev/null || echo "No gv100 firmware dir"
ls -la /lib/firmware/nvidia/ga102/ 2>/dev/null || echo "No ga102 firmware dir"

dmesg | grep -i 'nouveau\|nvidia\|drm' | tail -50
```

### Step 3: NVIDIA DRM UVM Probing (RTX 3090)

```bash
cargo test --test hw_nv_probe -p coral-driver -- --ignored --nocapture 2>&1 | tee nv_probe.log
cargo test uvm -p coral-driver -- --ignored --nocapture 2>&1 | tee uvm_diag.log
cargo test --test hw_nv_buffers -p coral-driver --features nvidia-drm -- --ignored --nocapture 2>&1 | tee nv_buffers.log
```

### Step 4: Multi-GPU Enumeration

```bash
cargo test --test hw_nv_probe -p coral-driver -- --ignored multi_gpu --nocapture 2>&1 | tee multi_gpu.log
```

### Step 5: Full Parity Suite

```bash
cargo test --test parity_compilation -p coral-reef 2>&1 | tee parity_compile.log
cargo test --test parity_harness -p coral-gpu --features nouveau -- --ignored --nocapture 2>&1 | tee parity_nouveau.log
```

### Data Return Checklist

| File | Priority | Purpose |
|------|----------|---------|
| `nouveau_diag.log` | Critical | EINVAL debugging for Titan V channel allocation |
| Environment data output | Critical | Kernel, driver, sysfs, firmware, dmesg |
| `nv_probe.log` | High | Device detection on both GPUs |
| `uvm_diag.log` | High | UVM RM client status (if proprietary driver present) |
| `nv_buffers.log` | Medium | nvidia-drm buffer mapping (if proprietary driver present) |
| `multi_gpu.log` | Medium | Multi-GPU enumeration across both cards |
| `parity_compile.log` | Low | Compile-only parity (no hardware needed) |
| `parity_nouveau.log` | High | E2E dispatch attempt on Titan V via nouveau |

---

## Part 3: Results (EXECUTED March 10, 2026)

### Environment

| Item | Value |
|------|-------|
| Kernel | 6.17.9-76061709-generic |
| renderD128 | Titan V (GV100, 0x1d81) — **nouveau** driver v1.4 |
| renderD129 | RTX 3090 (GA102, 0x2204) — **nvidia** proprietary driver |
| /dev/nvidia-uvm | Present (UVM loaded) |
| Vulkan (RTX 3090) | NVIDIA proprietary 580.119.02, Vulkan 1.4.312 |
| Vulkan (Titan V) | **NVK** (Mesa 25.1.5), Vulkan 1.3.311 |
| GV100 firmware | **16/16 present** (acr, gr, nvdec, sec2 — all files) |
| GA102 firmware | 7/16 present (expected — proprietary driver, not nouveau) |

### wgpu Adapter Enumeration

Both GPUs dispatch compute through wgpu/Vulkan:

| GPU | SHADER_F64 | Calibration | Notes |
|-----|-----------|-------------|-------|
| Titan V (NVK) | YES | F32=✓ F64=✓ F64Precise=✓ DF64=✓ | Full 4-tier, all domains route |
| RTX 3090 (proprietary) | YES | F32=✓ F64=✓ F64Precise=△ DF64=△ | Arith-only on DF64/Precise (NVVM risk) |

Dual-GPU cooperative dispatch verified: Split BCS, Split HMC, Redundant
validation, PCIe roundtrip (1.2 GB/s) all functional.

### coralReef hw_nv_probe (coral-driver)

| Test | Result | Notes |
|------|--------|-------|
| nvidia_drm_render_node_discovered | PASS | renderD129 nvidia-drm v0.0 |
| nvidia_drm_device_opens_and_queries_driver | PASS | |
| multi_gpu_enumerates_both | FAIL | Expects amdgpu node; we have 2× NVIDIA |

### coralReef UVM Tests (coral-driver)

| Test | Result | Notes |
|------|--------|-------|
| uvm_device_opens | PASS | /dev/nvidia-uvm opens |
| uvm_initialize | PASS | UVM init succeeds |
| rm_client_alloc | PASS | RM client handle: `0xC1D005C7` |
| **rm_client_alloc_device** | **FAIL** | `RM_ALLOC(NV01_DEVICE_0) failed: status=0x0000001F` |
| **rm_client_alloc_subdevice** | **FAIL** | Depends on device alloc (cascading) |

**Key finding**: RM client allocation works, but **device allocation** fails with
status `0x1F`. This is the UVM pipeline blocker for nvidia-drm dispatch.

### coralReef nvidia-drm Buffer Tests

| Test | Result | Notes |
|------|--------|-------|
| device_opens_successfully | PASS | |
| sync_succeeds | PASS | |
| alloc_returns_pending_uvm_error | PASS | Expected — UVM device alloc is the blocker |
| dispatch_returns_pending_uvm_error | PASS | Expected |
| sm86_compilation_independent_of_driver | PASS | 224 bytes, 22 GPRs, 6 instrs |

### coralReef Nouveau Tests (hw_nv_nouveau)

| Test | Result | Notes |
|------|--------|-------|
| nouveau_channel_alloc_hex_dump | PASS | 92-byte struct, class 0xC3C0 |
| nouveau_firmware_probe | PASS | GV100: 16/16, GA102: 7/16 |
| nouveau_diagnose_channel_alloc | PASS | **All 5 channel classes → EINVAL** |
| nouveau_gpu_identity_probe | PASS | device=0x1D81, SM=70 |
| nouveau_gem_alloc_without_channel | PASS | GEM handle=1 allocated |
| nouveau_device_opens | **FAIL** | `drm_ioctl returned 22` (EINVAL) |
| nouveau_alloc_free | **FAIL** | Cascading from device_opens |
| nouveau_upload_readback_roundtrip | **FAIL** | Cascading |
| nouveau_full_dispatch_cycle | **FAIL** | Cascading |
| nouveau_sync_without_dispatch | **FAIL** | Cascading |
| nouveau_multiple_dispatches | **FAIL** | Cascading |

Channel alloc diagnostic (all classes fail):
```
[FAIL] bare channel (nr_subchan=0)                  → EINVAL
[FAIL] compute-only (0xC3C0 VOLTA_COMPUTE_A)        → EINVAL
[FAIL] NVK-style multi-engine (2D + copy + compute)  → EINVAL
[FAIL] compute-only (0xC5C0 TURING_COMPUTE_A)       → EINVAL
[FAIL] compute-only (0xC6C0 AMPERE_COMPUTE_A)       → EINVAL
```

### coralReef Parity Compilation

| Suite | Result |
|-------|--------|
| parity_compilation (coral-reef) | **10/10 PASS** |
| parity_harness (coral-gpu) | **11/11 PASS** (1 ignored) |

### Root Cause Analysis: EINVAL on Channel Alloc

**Critical finding**: coralReef uses the **legacy** `DRM_NOUVEAU_CHANNEL_ALLOC`
(ioctl 0x42 = DRM_COMMAND_BASE + 0x02). On kernel 6.17+ with Volta (GV100),
this ioctl returns EINVAL for ALL channel types.

**NVK (Mesa 25.1.5) uses the new UAPI** introduced in kernel 6.6+:

| Ioctl | Number | Purpose |
|-------|--------|---------|
| `DRM_NOUVEAU_VM_INIT` | 0x10 | Initialize kernel-managed VA space |
| `DRM_NOUVEAU_VM_BIND` | 0x11 | Map/unmap GEM objects to GPU VA |
| `DRM_NOUVEAU_EXEC` | 0x12 | Submit pushbuf for compute dispatch |

The new UAPI structs are present in `/usr/include/drm/nouveau_drm.h`:
- `drm_nouveau_vm_init` — `kernel_managed_addr` + `kernel_managed_size`
- `drm_nouveau_vm_bind` — op array (MAP/UNMAP), async flag, sync objects
- `drm_nouveau_exec` — channel, push array, wait/signal sync objects

NVK uses `kernel_managed_addr = 0x80_0000_0000` (already defined in
coralReef's `NV_KERNEL_MANAGED_ADDR`).

**Recommendation for coralReef**: Migrate from legacy `DRM_NOUVEAU_CHANNEL_ALLOC`
to the new `VM_INIT → GEM_NEW → VM_BIND → EXEC` pipeline. The kernel headers
are available, the VA space constants match, and NVK's working dispatch on this
exact hardware proves the path is viable.

---

## Status

- Sovereign compilation: **45/46** (unchanged from Iter 29)
- FMA lowering: **Validated** (compiles correctly with `FmaPolicy::Separate` in pipeline)
- Hardware data capture: **COMPLETE** (all tests executed)
- `complex_f64` gap: **Known** (utility include, not standalone entry point)
- Nouveau dispatch: **Blocked** on legacy UAPI → needs migration to `VM_INIT`/`VM_BIND`/`EXEC`
- nvidia-drm dispatch: **Blocked** on UVM device alloc (`status=0x1F`)
- **Both GPUs dispatch through wgpu/NVK Vulkan** — the Vulkan path works
