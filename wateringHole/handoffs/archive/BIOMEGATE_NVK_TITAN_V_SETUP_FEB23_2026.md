# biomeGate — Titan V NVK Setup Report + Open Issues

**Date:** February 23, 2026
**From:** hotSpring (biomeGate first-boot session)
**To:** ToadStool / BarraCuda core team
**License:** AGPL-3.0-only
**Purpose:** Document the Titan V NVK driver setup on biomeGate, patches required
to build Mesa 25.1.5 NVK from source on Pop!_OS 22.04, validated dual-GPU results,
and remaining issues for the ToadStool team.

---

## Executive Summary

The Titan V (GV100) on biomeGate is now operational via NVK (Mesa 25.1.5, built
from source) alongside the RTX 3090 (nvidia proprietary 580.119). Both GPUs are
visible in Vulkan, both pass `validate_cpu_gpu_parity` 6/6, and `bench_multi_gpu`
confirms cooperative dispatch with 1.36× specialized routing speedup. Three Mesa
build patches were required. Five issues remain open for the ToadStool team.

---

## 1. Hardware Topology

```
biomeGate — Threadripper 3970X (TRX40)
├── 21:00.0  RTX 3090 (GA102, 24 GB, nvidia proprietary 580.119.02)
├── 4b:00.0  TITAN V (GV100, 12 GB, NVK / nouveau / Mesa 25.1.5)
└── Akida AKD1000 NPU (PCIe x1 Gen2)
```

Vulkan enumeration post-setup:

| GPU | Driver | Vulkan | Device ID |
|-----|--------|--------|-----------|
| NVIDIA GeForce RTX 3090 | nvidia proprietary | 1.4.312 | 0x2204 |
| NVIDIA TITAN V (NVK GV100) | NVK / Mesa 25.1.5 | 1.3.311 | 0x1d81 |
| llvmpipe (LLVM 15.0.7) | Mesa software | 1.4.311 | 0x0000 |

---

## 2. What Was Done

### 2.1 Mesa 25.1.5 NVK Build from Source

Pop!_OS 22.04 ships Mesa 25.1.5 via `mesa-vulkan-drivers` but the package is
compiled **without NVK** (`libvulkan_nouveau.so` absent). Built NVK from source:

**Location:** `~/Development/mesa-nvk-build/mesa-25.1.5/`

**Build command:**
```bash
PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH" \
RUSTC=~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/rustc \
LLVM_CONFIG=/usr/bin/llvm-config-15 \
~/.local/bin/meson setup build \
  -Dvulkan-drivers=nouveau \
  -Dgallium-drivers= \
  -Dglx=disabled -Degl=disabled -Dgles1=disabled -Dgles2=disabled \
  -Dplatforms=x11,wayland \
  -Dbuildtype=release \
  -Dprefix=$HOME/Development/mesa-nvk-build/install
ninja -C build -j$(nproc)
```

**Output:** `build/src/nouveau/vulkan/libvulkan_nouveau.so` (20 MB, ELF x86-64)

### 2.2 Three Build Patches Required

**Patch 1 — bindgen `atomic_uint_fast32_t`**

`rust-bindgen` (0.72.1) uses libclang internally but cannot parse C11
`<stdatomic.h>` types. The `nouveau_bo.h` header uses `atomic_uint_fast32_t`
for a refcount field. Bindgen's clang frontend fails regardless of `-std=gnu11`.

**Fix:** Added `__BINDGEN__` preprocessor guard in `src/nouveau/winsys/nouveau_bo.h`:
```c
#elif defined(__BINDGEN__)
#include <stdint.h>
typedef uint_fast32_t atomic_uint_fast32_t;
```
Plus `-D__BINDGEN__` in `c_args` for both `src/nouveau/compiler/meson.build` and
`src/nouveau/nil/meson.build` bindgen invocations.

**Upstream candidate:** Yes — this is a general bindgen-vs-C11-atomics issue.

**Patch 2 — `opencl-c-base.h` not found**

The `mesa_clc` compiler (built against LLVM 15) cannot locate clang's
`opencl-c-base.h` resource header when compiling `nvk_query.cl` to SPIR-V.

**Fix:** Added `-I/usr/lib/llvm-15/lib/clang/15.0.7/include` to the nvkcl.spv
build target in `src/nouveau/vulkan/meson.build`.

**Upstream candidate:** Possibly — depends on whether mesa_clc should embed the
resource path from its build-time LLVM or discover it at runtime.

**Patch 3 — cbindgen version**

System cbindgen (0.20.0 from apt) is below the 0.25 minimum for Mesa's
`src/nouveau/nil/meson.build`.

**Fix:** `cargo install cbindgen` (installed 0.29.2 to `~/.cargo/bin/`).

### 2.3 System Dependencies Installed

```
apt: python3-mako libvulkan-dev glslang-tools libwayland-dev libdrm-dev
     libelf-dev llvm-15 llvm-15-dev libclang-15-dev clang-15
     libllvmspirvlib-15-dev libclc-dev vulkan-tools
pip: meson==1.3.2 (system was 0.61.2, Mesa requires >= 1.3.0)
cargo: cbindgen 0.29.2
```

### 2.4 Vulkan ICD

```bash
~/.config/vulkan/icd.d/nouveau_icd.json
```
```json
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "/home/biomegate/Development/mesa-nvk-build/mesa-25.1.5/build/src/nouveau/vulkan/libvulkan_nouveau.so",
        "api_version": "1.3.311"
    }
}
```

### 2.5 Modprobe Dual-Driver Coexistence

The nvidia proprietary driver blacklists nouveau:
```
# /lib/modprobe.d/nvidia-graphics-drivers.conf (DO NOT EDIT — package-managed)
blacklist nouveau
blacklist lbm-nouveau
alias nouveau off
alias lbm-nouveau off
```

Override:
```
# /etc/modprobe.d/hotspring-nouveau-titanv.conf
install nouveau /sbin/modprobe --ignore-install nouveau
remove nouveau /sbin/modprobe -r --ignore-remove nouveau
alias nouveau nouveau
alias lbm-nouveau off
```

initramfs updated via `update-initramfs -u`.

### 2.6 Module Loading

Nouveau has a deep dependency chain that must be loaded in order:
```
drm_buddy → drm_exec → drm_gpuvm → gpu-sched → cec → drm_display_helper → nouveau
```

The `alias nouveau off` in the nvidia blacklist prevents `modprobe nouveau` from
working. The modules loaded successfully after relog (display manager restart).
Direct `insmod` of the chain also works.

---

## 3. Validated Results

| Test | Adapter | Result |
|------|---------|--------|
| `vulkaninfo --summary` | Both | GPU0: RTX 3090, GPU1: TITAN V (NVK GV100) |
| `validate_cpu_gpu_parity` | Titan V (NVK) | 6/6 passed |
| `validate_cpu_gpu_parity` | RTX 3090 (nvidia) | 6/6 passed |
| `f64_builtin_test` | Titan V | sqrt OK, exp crashes NAK (known) |
| `bench_multi_gpu` | Both | See below |

### Multi-GPU Benchmark (biomeGate)

```
Config: primary="3090", secondary="titan"

Card A: NVIDIA GeForce RTX 3090 (SHADER_F64=true)
Card B: NVIDIA TITAN V (NVK GV100) (SHADER_F64=true)

Phase 1: Comparative BCS
  RTX 3090:  6.298 ms (650,348/s)
  Titan V:   2.837 ms (1,443,982/s)  → 2.2× faster

Phase 2: Cooperative BCS (data split)
  Single card: 7.739 ms  Cooperative: 8.147 ms  → 0.95×

Phase 3: Cooperative Eigensolve
  Single card: 18.38 ms  Cooperative: 51.57 ms  → 0.36×

Phase 4: Specialized Routing
  Sequential: 24.66 ms  Specialized: 18.19 ms  → 1.36×
```

BCS (streaming fp64) favors the Titan V. Eigensolve (dispatch-heavy) favors the
3090's lower-latency nvidia proprietary driver. Specialized routing (each GPU does
what it's best at) yields a 1.36× net speedup.

---

## 4. Open Issues for ToadStool Team

### 4.1 Nouveau Auto-Load at Boot (HIGH)

**Problem:** The `alias nouveau off` in `/lib/modprobe.d/nvidia-graphics-drivers.conf`
overrides our `install` directive. `modprobe nouveau` resolves the alias to "off"
before checking install rules, yielding `could not find module by name='off'`.

**Current workaround:** Nouveau loads on display manager restart (relog) but not
at cold boot. Direct `insmod` of the full dependency chain works.

**Proposed fix:** A systemd unit (`hotspring-nouveau.service`) that runs after
`nvidia-persistenced.service` and `insmod`s the chain:
```bash
for mod in drm_buddy drm_exec drm_gpuvm gpu-sched cec drm_display_helper; do
  insmod /usr/lib/modules/$(uname -r)/kernel/drivers/gpu/drm/$mod*.ko 2>/dev/null || true
done
insmod /usr/lib/modules/$(uname -r)/kernel/drivers/media/cec/core/cec.ko 2>/dev/null || true
insmod /usr/lib/modules/$(uname -r)/kernel/drivers/gpu/drm/nouveau/nouveau.ko
```

Or a udev rule bound to the Titan V's PCI vendor:device (10de:1d81).

### 4.2 NAK exp(f64) / log(f64) Crash (MEDIUM)

**Problem:** Native `exp()` and `log()` on f64 hit an assertion failure in NAK
(`nak/from_nir.rs:430: assertion failed: vec.len() == bits.div_ceil(32)`). This
is a known NVK limitation for Volta (GV100).

**Current workaround:** `ShaderTemplate::for_driver_auto()` replaces exp/log with
polynomial approximations. `GpuDriverProfile` detects NVK and applies
`Workaround::NvkExpF64Crash` / `NvkLogF64Crash` automatically.

**Upstream path:** Tracked in `contrib/mesa-nak/NAK_DEFICIENCIES.md`. The fix
belongs in Mesa's `src/nouveau/compiler/nak/from_nir.rs` — the 128-bit f64
return value handling for transcendental builtins.

### 4.3 bindgen C11 Atomics Patch (LOW)

The `__BINDGEN__` guard added to `nouveau_bo.h` should be upstreamed to Mesa.
The pattern is reusable for any C header using `<stdatomic.h>` types that must
pass through rust-bindgen. The fix is zero-cost (preprocessor only).

### 4.4 NVK Cooperative Dispatch Overhead (INFORMATIONAL)

Data-parallel splitting across NVK + nvidia proprietary shows 0.36×–0.95× due to
synchronization overhead and NVK's higher dispatch latency for iterative kernels.
Task-level routing (specialized) is the correct strategy: 1.36× validated.

### 4.5 biomegate.env Exports (RESOLVED)

Variables in `metalForge/nodes/biomegate.env` were missing the `export` keyword,
causing `source biomegate.env` to set shell-local variables invisible to child
processes. Fixed — all variables now use `export`.

---

## 5. File Inventory

| File | Purpose |
|------|---------|
| `~/.config/vulkan/icd.d/nouveau_icd.json` | NVK Vulkan ICD |
| `/etc/modprobe.d/hotspring-nouveau-titanv.conf` | Nouveau modprobe override |
| `~/Development/mesa-nvk-build/mesa-25.1.5/` | Mesa NVK source + build |
| `metalForge/gpu/nvidia/NVK_SETUP.md` | Generic NVK setup guide (6 steps) |
| `metalForge/nodes/biomegate.env` | biomeGate node profile |
| `barracuda/src/gpu/adapter.rs` | Priority-list adapter selection (3090,titan,auto) |

---

## 6. Comparison: Eastgate vs biomeGate NVK

| Aspect | Eastgate | biomeGate |
|--------|----------|-----------|
| Primary GPU | RTX 4070 (12 GB) | RTX 3090 (24 GB) |
| Secondary GPU | Titan V (12 GB) | Titan V (12 GB) |
| PCIe slots | 01:00.0 / 05:00.0 | 21:00.0 / 4b:00.0 |
| NVK Mesa | 25.1.5 (source) | 25.1.5 (source, 3 patches) |
| Largest lattice (dynamical) | 40⁴ (8.2 GB) | 48⁴ (16.9 GB) |
| CPU threads | 24 (i9-12900K) | 64 (Threadripper 3970X) |
| System RAM | 32 GB DDR5 | 256 GB DDR4 |
| Specialized routing speedup | 1.48× | 1.36× |

The slightly lower routing speedup on biomeGate (1.36× vs 1.48×) is expected:
the 3090 is faster than the 4070 in absolute terms, so the Titan V's relative
contribution is proportionally smaller. The 3090's 24 GB VRAM is the key
advantage — it enables 48⁴ dynamical fermion QCD that no 12 GB card can run.
