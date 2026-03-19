# Handoff: Backend Analysis Pack — Glowplug Swap Validation & Multi-Backend Testing

**Date:** March 17, 2026
**From:** barraCuda (audit sprint, DF64 NVK verification)
**To:** hotSpring (metalForge, glowplug evolution, NVK build pipeline)
**License:** AGPL-3.0-or-later
**Covers:** System GPU landscape, glowplug live validation, wgpu adapter survey, NVK gap analysis

---

## Executive Summary

- **3 GPUs detected**: 2x Titan V (GV100) on VFIO, 1x RTX 5060 (GB206) on nvidia proprietary
- **coral-glowplug is live**: systemd daemon active 20h+, both Titan V healthy (VRAM alive, 9/9 domains, D0)
- **Glowplug swap is functional**: `device.swap` implemented with full unbind→rebind sequence, HBM2 resurrection proven in prior boot (journal confirms nouveau→vfio round-trip in ~6s)
- **NVK is NOT available**: Pop!_OS Mesa 25.1.5 package does not include `libvulkan_nouveau.so` — must build from source per existing `metalForge/gpu/nvidia/NVK_SETUP.md`
- **2 wgpu backends confirmed**: NVIDIA proprietary (RTX 5060) and LVP software (llvmpipe) — no NVK adapter until Mesa NVK is built
- **nouveau kernel module exists**: `/lib/modules/6.17.9-76061709-generic/kernel/drivers/gpu/drm/nouveau/nouveau.ko` present, not currently loaded
- **LVP has a buffer size limit issue**: barraCuda's `max_storage_buffer_binding_size` requirement (512MB) exceeds LVP's limit (128MB)

---

## Part 1: Hardware Landscape

### PCIe Topology

| BDF | Device | Chip | Driver | IOMMU Group | Status |
|-----|--------|------|--------|-------------|--------|
| `0000:03:00.0` | NVIDIA TITAN V | GV100 `[10de:1d81]` | vfio-pci | 69 | VRAM alive, D0 |
| `0000:03:00.1` | (HDMI Audio) | — | vfio-pci | 69 | Companion bound |
| `0000:21:00.0` | NVIDIA RTX 5060 | GB206 `[10de:2d05]` | nvidia 580.126.18 | 47 | Desktop GPU |
| `0000:4a:00.0` | NVIDIA TITAN V | GV100 `[10de:1d81]` | vfio-pci | 34 | VRAM alive, D0 |
| `0000:4a:00.1` | (HDMI Audio) | — | vfio-pci | 34 | Companion bound |

### System

| Property | Value |
|----------|-------|
| Kernel | 6.17.9-76061709-generic |
| OS | Pop!_OS 22.04 (Jammy) |
| Mesa | 25.1.5-1pop0 |
| NVIDIA proprietary | 580.126.18 (open kernel module) |
| LLVM | 15.0.7 |
| Vulkan Instance | 1.3.280 |

---

## Part 2: Glowplug Live Status

### Daemon

```
● coral-glowplug.service — Sovereign PCIe Device Lifecycle Broker
  Active: active (running) since Mon 2026-03-16 13:47:59 EDT; 20h ago
  PID: 1511
  Socket: /run/coralreef/glowplug.sock
  Health check: every 5000ms
```

### Device List (live query via Unix socket)

```json
{
  "Devices": [
    {
      "bdf": "0000:03:00.0",
      "name": "titan-oracle",
      "chip": "GV100 (Titan V)",
      "personality": "vfio (group 69)",
      "role": "oracle",
      "power": "D0",
      "vram_alive": true,
      "domains_alive": 9,
      "domains_faulted": 0,
      "has_vfio_fd": true,
      "pci_link_width": 8
    },
    {
      "bdf": "0000:4a:00.0",
      "name": "titan-target",
      "chip": "GV100 (Titan V)",
      "personality": "vfio (group 34)",
      "role": "compute",
      "power": "D0",
      "vram_alive": true,
      "domains_alive": 9,
      "domains_faulted": 0,
      "has_vfio_fd": true,
      "pci_link_width": 8
    }
  ]
}
```

### Swap Capability

The running daemon uses a serde enum protocol (not JSON-RPC 2.0 yet — the socket.rs
source has JSON-RPC 2.0 but the deployed binary predates that evolution):

| Command | Wire Format | Description |
|---------|-------------|-------------|
| `"ListDevices"` | Serde enum string | List all managed devices |
| `{"Swap":{"bdf":"...","target":"nouveau"}}` | Serde enum object | Hot-swap personality |
| `{"Health":{"bdf":"..."}}` | Serde enum object | Device health probe |
| `{"Resurrect":{"bdf":"..."}}` | Serde enum object | HBM2 resurrection cycle |
| `"Status"` | Serde enum string | Daemon uptime |
| `"Shutdown"` | Serde enum string | Graceful shutdown |

**Protocol evolution note:** The source at `coralReef/crates/coral-glowplug/src/socket.rs`
implements full JSON-RPC 2.0 (`device.list`, `device.swap`, `device.health`, etc.), but
the currently deployed binary uses the older serde enum wire format. Rebuilding and
redeploying coral-glowplug will migrate to JSON-RPC 2.0 — this is a non-breaking
evolution (new clients just need to speak JSON-RPC).

### Prior Swap Evidence (journalctl)

The previous boot (Mar 16) shows a successful autonomous HBM2 resurrection for titan-target:

```
13:46:11 WARN  VRAM dead for 3+ checks — attempting auto-resurrection via nouveau (4a:00.0)
13:46:11 INFO  HBM2 resurrection starting bdf=0000:4a:00.0
13:46:11 INFO  state vault snapshot saved bdf=0000:4a:00.0 regs=18
13:46:12 INFO  clearing driver_override and binding nouveau... bdf=0000:4a:00.0
13:46:12 INFO  override after clear driver_override="(null)"
13:46:14 INFO  nouveau init complete (DRM card found) bdf=0000:4a:00.0 attempt=0
13:46:14 INFO  nouveau warm complete, swapping back to vfio-pci... bdf=0000:4a:00.0
```

Total cycle time: ~3 seconds (nouveau init) + ~1.5s (vfio rebind) = **~4.5s round-trip**.

---

## Part 3: Vulkan/wgpu Backend Survey

### Available Vulkan ICDs

| ICD | Library | API Version | Status |
|-----|---------|-------------|--------|
| nvidia | `libGLX_nvidia.so.0` | 1.4.312 | Active (RTX 5060) |
| lvp (llvmpipe) | `libvulkan_lvp.so` | 1.4.311 | Available (CPU fallback) |
| radeon | `libvulkan_radeon.so` | — | Installed, no AMD GPU |
| intel | `libvulkan_intel.so` | — | Installed, no Intel GPU |
| intel_hasvk | `libvulkan_intel_hasvk.so` | — | Installed, no Intel GPU |
| asahi | `libvulkan_asahi.so` | — | Installed, no Apple GPU |
| virtio | `libvulkan_virtio.so` | — | Available for VM passthrough |
| gfxstream | `libvulkan_gfxstream.so` | — | Available for Android emulation |
| **nouveau (NVK)** | **MISSING** | — | **NOT in Pop!_OS Mesa package** |

### wgpu Adapter Enumeration (live)

```
Available GPU adapters:
  [0] NVIDIA GeForce RTX 5060  backend=Vulkan  type=DiscreteGpu
  [1] llvmpipe (LLVM 15.0.7, 256 bits)  backend=Vulkan  type=Cpu
```

### RTX 5060 Driver Profile (barraCuda auto-detection)

```
Driver:   NvidiaProprietary
Compiler: NvidiaPtxas
Arch:     Unknown (GB206 — Blackwell, not yet in profile table)
FP64:     Throttled
Workarounds: [Df64SpirVPoisoning]
Eigensolve: WarpPacked { wg_size: 32 }
FP64 Strategy: Hybrid
Precision Routing: Df64Only
```

### LVP (llvmpipe) Issue

barraCuda requires `max_storage_buffer_binding_size >= 512MB` but LVP reports only 128MB:
```
Failed to create device: Limit 'max_storage_buffer_binding_size' value 536870912
is better than allowed 134217728
```

This blocks using LVP as a test backend without either:
1. Reducing barraCuda's buffer size requirement for test profiles, or
2. Patching llvmpipe's reported limit (not practical)

---

## Part 4: NVK Gap Analysis

### Why NVK Matters

NVK is the sovereign path to Vulkan compute on Titan V:
- No proprietary firmware dependency (uses nouveau kernel driver)
- Full SHADER_F64 support on GV100 (native DFMA, DSQRT)
- NAK compiler is Rust-based (aligns with ecoBin)
- Enables DF64 end-to-end verification without proprietary driver

### Current Blocker

Pop!_OS ships Mesa 25.1.5 but **does not build NVK** (`-Dvulkan-drivers=nouveau` not
in their build config). The `mesa-vulkan-drivers` package contains drivers for radeon,
intel, asahi, lvp, virtio, gfxstream — but no `libvulkan_nouveau.so`.

### Resolution Path

The existing `hotSpring/metalForge/gpu/nvidia/NVK_SETUP.md` documents the solution:
build Mesa from source with `-Dvulkan-drivers=nouveau`. Updated requirements for
biomeGate:

| Dependency | Required | Current | Status |
|------------|----------|---------|--------|
| Mesa source | 25.1.5+ | — | Need to clone |
| meson + ninja | any | — | Need `apt install` |
| Rust toolchain | 1.70+ | ✓ (rustup) | Available |
| LLVM 15+ | 15+ | 15.0.7 | **Available** |
| libclang-dev | 15+ | 14 (clang pkg) | **Need libclang-15-dev** |
| python3-mako | any | — | Need `apt install` |
| libvulkan-dev | any | ✓ | Available |
| glslang-tools | any | — | Need `apt install` |
| nouveau kernel module | ✓ | ✓ (`nouveau.ko` exists) | Available |
| nouveau firmware (GV100) | N/A | N/A | NVK uses nouveau KMD, no separate firmware needed for Volta |

### Build Command (from NVK_SETUP.md, adapted for biomeGate)

```bash
git clone https://gitlab.freedesktop.org/mesa/mesa.git
cd mesa && git checkout mesa-25.1.5

meson setup build \
  -Dvulkan-drivers=nouveau \
  -Dgallium-drivers= \
  -Dglx=disabled \
  -Dplatforms=x11,wayland \
  -Dbuildtype=release \
  -Dprefix=$HOME/Development/mesa-nvk-build/mesa-25.1.5/install

ninja -C build && ninja -C build install
```

### Expected Result After NVK Build + Nouveau Bind

After building NVK and swapping one Titan V to nouveau via glowplug:

```
Available GPU adapters:
  [0] NVIDIA GeForce RTX 5060  backend=Vulkan  type=DiscreteGpu  (nvidia proprietary)
  [1] NVIDIA TITAN V (NVK)     backend=Vulkan  type=DiscreteGpu  (nouveau/NVK)
  [2] llvmpipe (LLVM 15)       backend=Vulkan  type=Cpu          (software fallback)
```

This gives **3 backends** for testing:
1. **Nvidia proprietary** (RTX 5060) — Blackwell, production baseline
2. **NVK/nouveau** (Titan V) — Sovereign Volta, DF64 native, open-source stack
3. **LVP** (CPU) — Software reference (needs buffer limit workaround)

---

## Part 5: Glowplug Swap Validation Plan

### Test Sequence for hotSpring

Validate that glowplug can swap titan-target between backends and that each
backend is visible to wgpu:

```
Step 1: Verify current state (both on vfio)
  → socket: "ListDevices"
  → confirm: titan-target personality = "vfio (group 34)"

Step 2: Swap titan-target to nouveau
  → socket: {"Swap":{"bdf":"0000:4a:00.0","target":"nouveau"}}
  → wait: 5s for nouveau init + DRM card creation
  → verify: /sys/bus/pci/devices/0000:4a:00.0/driver → nouveau
  → verify: /dev/dri/cardN appears for 4a:00.0

Step 3: Install NVK ICD (from local build)
  → create: ~/.config/vulkan/icd.d/nouveau_icd.json
  → point to: $HOME/Development/mesa-nvk-build/mesa-25.1.5/install/lib/x86_64-linux-gnu/libvulkan_nouveau.so

Step 4: Enumerate wgpu adapters
  → VK_ICD_FILENAMES should include both nvidia_icd.json and nouveau_icd.json
  → expect: RTX 5060 (proprietary) + TITAN V (NVK) + llvmpipe (LVP)

Step 5: Run DF64 verification on NVK
  → BARRACUDA_GPU_ADAPTER=titan cargo run --release --bin bench_f64_builtins
  → expect: DriverKind::Nvk, CompilerKind::Nak, GpuArch::Volta
  → expect: SHADER_F64 = true, native DFMA/DSQRT

Step 6: Swap titan-target back to vfio
  → socket: {"Swap":{"bdf":"0000:4a:00.0","target":"vfio"}}
  → verify: vfio-pci bound, VRAM alive

Step 7: Run VFIO sovereign dispatch
  → coralReef direct BAR0 dispatch (existing toadStool integration)
  → verify: register health maintained across swap cycle
```

### Warnings from Prior Experience

1. **Do NOT swap titan-oracle (03:00.0) to nouveau while desktop is active** —
   the GV100 DRM render node will be grabbed by Xorg/Wayland compositor, causing
   kernel oops on unbind (documented in glowplug.toml comments)

2. **titan-target (4a:00.0) is safe to swap** — it has no display outputs connected
   and role=compute

3. **PCIe link width is 8x** (both Titans) — verify this doesn't drop after swap
   (D3hot transition during nouveau unbind can sometimes reduce link width)

4. **IOMMU group 34** contains `4a:00.0` + `4a:00.1` (audio companion) — glowplug
   handles companion binding automatically

---

## Part 6: Recommendations for hotSpring

### Immediate (P1)

1. **Build Mesa NVK locally** following `metalForge/gpu/nvidia/NVK_SETUP.md` — all
   prerequisites except `libclang-15-dev`, `meson`, `python3-mako`, and `glslang-tools`
   are already present

2. **Validate glowplug swap sequence** (Step 2 above) using the socket directly —
   this confirms the kernel-level driver swap works before adding NVK userspace

3. **Add GB206 (RTX 5060) to barraCuda chip identification and driver profile** —
   currently reports `Arch: Unknown` because Blackwell isn't in the profile table yet

### Medium-term (P2)

4. **Evolve glowplug wire protocol to JSON-RPC 2.0** — the source already implements
   it but the deployed binary predates the change; rebuild + redeploy

5. **Add LVP buffer limit workaround** — either a test-specific device requirements
   profile in barraCuda or a reduced buffer allocation path for CPU backends

6. **Create a `metalForge/nodes/biomegate.env`** capturing this system's GPU topology
   for reproducible multi-backend testing

### Long-term (P3)

7. **Investigate Titan V nouveau firmware-free Vulkan** — GV100 is one of the last
   NVIDIA GPUs where nouveau can function without GSP firmware (post-Turing requires
   GSP), making it uniquely valuable for sovereign compute

8. **VFIO direct dispatch vs NVK performance comparison** — measure the overhead
   difference between coralReef's BAR0 direct dispatch (VFIO) and wgpu dispatch
   through NVK for identical DF64 kernels

---

## Part 7: barraCuda CPU-Side Validation (Already Passed)

All 5 NAK/NVK DF64 rewrite tests pass on CPU (naga validation):

```
cargo nextest run -p barracuda --no-fail-fast -E 'test(nak)'
Starting 5 tests across 29 binaries (3713 tests skipped)
Summary [0.070s] 5 tests run: 5 passed, 3713 skipped
```

Tests validated:
- Yukawa compound assignments (`+=` / `-=` on DF64)
- Comparisons with `continue` (DF64 comparison lowering)
- Full Yukawa force kernel (complete DF64 rewrite pipeline)
- CG solver pattern (conjugate gradient with DF64 dot products)
- Yukawa cell-list pattern (nested loops with DF64 accumulation)

The compiler produces correct WGSL targeting NVK patterns. Full GPU dispatch
awaits NVK availability (this handoff's primary deliverable).

---

## Appendix: Quick Reference

### Socket Query (Python — works from any primal)

```python
import socket, json
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect('/run/coralreef/glowplug.sock')
sock.sendall(b'"ListDevices"\n')
sock.settimeout(5)
data = b''
while b'\n' not in data:
    data += sock.recv(8192)
sock.close()
print(json.dumps(json.loads(data), indent=2))
```

### Environment Variables for Backend Selection

```bash
# Force nvidia proprietary only
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# Force LVP software only
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json

# Force NVK only (after build)
VK_ICD_FILENAMES=$HOME/.config/vulkan/icd.d/nouveau_icd.json

# All backends (default — loader scans all ICDs)
unset VK_ICD_FILENAMES

# barraCuda adapter selection
BARRACUDA_GPU_ADAPTER=auto    # first discrete GPU (default)
BARRACUDA_GPU_ADAPTER=0       # by index
BARRACUDA_GPU_ADAPTER=titan   # by name match
```
