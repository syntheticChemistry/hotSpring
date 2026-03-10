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

## Status

- Sovereign compilation: **45/46** (unchanged from Iter 29)
- FMA lowering: **Validated** (compiles correctly with `FmaPolicy::Separate` in pipeline)
- Hardware data capture: **Pending** (commands documented, awaiting execution on test rig)
- `complex_f64` gap: **Known** (utility include, not standalone entry point)
- NVIDIA dispatch: **Blocked** on DRM maturation (nouveau EINVAL, nvidia-drm UVM)
