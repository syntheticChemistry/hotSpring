# Experiment 064: GlowPlug as PCIe Device Lifecycle Broker

**Date**: March 16, 2026
**Status**: REALIZED — architecture implemented in coral-glowplug v0.1.0 (Exp 065, 069). This document is the design spec; see Exp 065 for daemon success and Exp 069 for boot persistence + shutdown safety.
**Context**: Evolved from experiment 063. GlowPlug grows from "GPU warm-up" to
"sovereign hardware broker" that owns devices from boot, hot-swaps drivers on
demand, and presents a stable interface to toadStool.

---

## The Evolution

```
Before:  GlowPlug = GPU warm-up utility (temporary, per-session)
Now:     GlowPlug = PCIe device lifecycle broker (persistent, boot-to-shutdown)
Future:  GlowPlug = sovereign hardware layer for any PCIe device
```

The insight: GlowPlug doesn't need to BE a driver. It's the **authority**
that sits between bare hardware and whatever software personality is
currently active. It owns the device at the deepest level and can loan it
to any driver — nouveau, CUDA, amdgpu, VFIO, or our own sovereign path —
while keeping state alive across transitions.

---

## Core Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    GlowPlug Daemon                            │
│                 (boots with system, never exits)              │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ DeviceSlot 0 │  │ DeviceSlot 1 │  │ DeviceSlot 2 │  ...   │
│  │ Titan V      │  │ Titan V      │  │ MI50         │         │
│  │ 0000:03:00.0 │  │ 0000:4a:00.0 │  │ 0000:XX:00.0 │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐        │
│  │ Personality   │  │ Personality   │  │ Personality   │       │
│  │ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │     │
│  │ │ nouveau   │ │  │ │  VFIO     │ │  │ │  amdgpu   │ │     │
│  │ │ (oracle)  │ │  │ │(sovereign)│ │  │ │ (oracle)  │ │     │
│  │ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │     │
│  │ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │     │
│  │ │ CUDA stub │ │  │ │  nouveau  │ │  │ │  VFIO     │ │     │
│  │ │ (avail)   │ │  │ │  (avail)  │ │  │ │ (avail)   │ │     │
│  │ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │     │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
│                                                               │
│  ┌───────────────────────────────────────────────────┐        │
│  │              State Vault                           │        │
│  │  • Register snapshots (per-device, per-transition) │        │
│  │  • Oracle data (warm reference states)             │        │
│  │  • Power state history                             │        │
│  │  • HBM2 training status                            │        │
│  │  • VBIOS cache                                     │        │
│  └───────────────────────────────────────────────────┘        │
│                                                               │
│  ┌───────────────────────────────────────────────────┐        │
│  │              Health Monitor                        │        │
│  │  • Periodic VRAM sentinel test                     │        │
│  │  • PRI bus fault detection                         │        │
│  │  • Power state watchdog                            │        │
│  │  • Temperature/thermal monitoring                  │        │
│  │  • PCIe link health (correctable errors, width)    │        │
│  └───────────────────────────────────────────────────┘        │
├───────────────────────────────────────────────────────────────┤
│                    Unix Socket API                            │
│              /run/coralreef/glowplug.sock                     │
│                                                               │
│  Commands:                                                    │
│    list_devices       → [{bdf, vendor, personality, vram}]    │
│    get_device(bdf)    → fd (SCM_RIGHTS) + metadata            │
│    swap(bdf, target)  → snapshot → unbind → bind → verify     │
│    snapshot(bdf)      → register state captured               │
│    health(bdf)        → {vram, power, pci_link, thermal}      │
│    set_policy(bdf, p) → power/thermal policy update           │
└───────────────┬───────────────────────────────────────────────┘
                │
    ┌───────────┴────────────┐
    │       toadStool         │
    │  (hardware dispatch)    │
    │                         │
    │  Sees: list of warm,    │
    │  ready devices with     │
    │  capabilities.          │
    │                         │
    │  Doesn't care which     │
    │  driver is active.      │
    │  Asks GlowPlug to swap  │
    │  when needed.           │
    └───────────┬─────────────┘
                │
    ┌───────────┴────────────┐
    │       springs           │
    │  hotSpring, wetSpring,  │
    │  neuralSpring, etc.     │
    │                         │
    │  See: compute resources │
    │  Don't know hardware    │
    └─────────────────────────┘
```

---

## The Hot-Swap Protocol

The key innovation: driver transitions that preserve GPU state.

### Swap Sequence

```
toadStool: "swap 0000:4a:00.0 from vfio to nouveau"

GlowPlug:
  1. SNAPSHOT — capture full register state (4000+ registers)
     - PRI_MASTER, PMC, PBUS, PTOP, PFB, FBPA, LTC, PCLOCK
     - PRAMIN window contents (VRAM page table roots)
     - Power state, PMU mailbox values

  2. QUIESCE — ensure no in-flight DMA
     - Drain PFIFO (wait for PBDMA idle)
     - Fence all pending GPU work
     - Notify toadStool clients to release handles

  3. UNBIND — detach current driver
     - echo BDF > /sys/bus/pci/.../driver/unbind
     - Immediately pin power/control=on (prevent D3hot)
     - Verify BOOT0 still reads (device alive)

  4. BIND — attach new driver
     - echo target > /sys/bus/pci/.../driver_override
     - echo BDF > /sys/bus/pci/drivers/target/bind
     - Pin power/control=on again

  5. VERIFY — confirm device health
     - Read BOOT0 (device responding)
     - Check PMC_ENABLE (engines alive)
     - VRAM sentinel test (HBM2 still trained)
     - If VRAM dead: attempt restore from snapshot
     - If restore fails: flag device as "cold" (needs re-POST)

  6. NOTIFY — tell toadStool new personality is active
     - Pass new fd if applicable (VFIO → SCM_RIGHTS)
     - Update device capability advertisement
```

### State Preservation Across Swaps

| Transition | HBM2 Preserved? | Strategy |
|-----------|-----------------|----------|
| vfio → nouveau | Usually yes (if D0 pinned) | Pin D0, verify VRAM after bind |
| nouveau → vfio | Usually yes | Pin D0 before unbind, force D0 after |
| vfio → nvidia | Depends on nvidia driver | Snapshot before, verify after |
| nvidia → vfio | Usually no (nvidia resets hard) | Accept cold, re-warm from oracle |
| any → same | No swap needed | Just re-verify health |
| cold → any | HBM2 dead | Need BIOS POST or sovereign training |

### The State Vault

Every swap creates a snapshot. Snapshots are diff-compressed against the
oracle (baseline warm state) to keep them small:

```rust
pub struct DeviceSnapshot {
    timestamp: SystemTime,
    bdf: String,
    personality: Personality,
    power_state: PciPmState,
    boot0: u32,
    pmc_enable: u32,
    vram_alive: bool,
    /// Only stores registers that differ from oracle baseline
    register_deltas: BTreeMap<usize, (u32, u32)>,  // offset → (oracle, actual)
    /// PRAMIN window contents for page table preservation
    pramin_pages: Vec<(u32, Vec<u8>)>,  // (vram_base, 4K page)
}
```

---

## Personality System

Each driver backend is a "personality" that GlowPlug can activate.

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum Personality {
    /// VFIO passthrough — full sovereign register access from userspace.
    /// Best for: coralReef sovereign dispatch, diagnostic matrix, HBM2 probing.
    Vfio {
        group_id: u32,
        container_fd: Option<RawFd>,
    },

    /// nouveau — open-source Linux GPU driver.
    /// Best for: NVK/Vulkan dispatch (Tier 1), oracle data capture, mmiotrace.
    Nouveau {
        drm_card: Option<String>,     // /dev/dri/cardN
        render_node: Option<String>,  // /dev/dri/renderDN
    },

    /// nvidia proprietary — closed-source driver.
    /// Best for: CUDA dispatch, maximum perf on supported hardware.
    /// Caution: aggressive resets, may destroy state on unbind.
    NvidiaProprietary {
        cuda_device: Option<u32>,
    },

    /// amdgpu — open-source AMD driver.
    /// Best for: AMD GPUs (MI50), Vulkan dispatch, ROCm.
    Amdgpu {
        drm_card: Option<String>,
        render_node: Option<String>,
    },

    /// Unbound — no driver, device in raw PCI state.
    /// GlowPlug maintains direct register access via sysfs resource0.
    Unbound,
}
```

### Capability Advertisement

Each personality exposes different capabilities to toadStool:

```rust
pub struct DeviceCapabilities {
    pub bdf: String,
    pub vendor: GpuVendor,           // Nvidia, Amd, Intel
    pub chip: String,                // "GV100", "Vega20"
    pub vram_bytes: u64,
    pub vram_alive: bool,
    pub personality: Personality,

    // What this personality can do RIGHT NOW:
    pub can_vulkan: bool,            // Vulkan dispatch available
    pub can_cuda: bool,              // CUDA dispatch available
    pub can_sovereign: bool,         // Direct BAR0/VFIO available
    pub can_dma: bool,               // DMA buffer allocation works
    pub can_peer_transfer: bool,     // PCIe P2P to other devices

    // What this device CAN do if we swap personality:
    pub available_personalities: Vec<Personality>,

    // Hardware topology:
    pub pcie_slot: String,           // physical slot
    pub iommu_group: u32,
    pub numa_node: u32,
    pub pcie_width: u8,              // x16, x8, etc.
    pub pcie_speed: String,          // Gen3, Gen4, etc.
}
```

---

## toadStool Integration

toadStool's existing `BackendKind` maps directly to GlowPlug personalities:

```
BackendKind::BarraCudaGpu     → Personality::Nouveau (wgpu/Vulkan/NVK)
                               OR Personality::NvidiaProprietary (Vulkan)
                               OR Personality::Amdgpu (Vulkan)

BackendKind::CoralReefSovereign → Personality::Vfio (direct BAR0)

BackendKind::KokkosCuda        → Personality::NvidiaProprietary (CUDA)
```

### Dynamic Backend Selection

```rust
// toadStool asks GlowPlug what's available
let devices = glowplug.list_devices();

// Find a device that can do Vulkan
let vulkan_gpu = devices.iter()
    .find(|d| d.can_vulkan && d.vram_alive);

// Or find one that can do sovereign dispatch
let sovereign_gpu = devices.iter()
    .find(|d| d.can_sovereign && d.vram_alive);

// Or ask GlowPlug to SWAP a device to the needed personality
if let Some(gpu) = devices.iter().find(|d| d.vendor == GpuVendor::Nvidia) {
    if !gpu.can_sovereign {
        glowplug.swap(&gpu.bdf, Personality::Vfio { .. })?;
        // Now it can do sovereign dispatch
    }
}
```

### The Key Simplification for toadStool

Before GlowPlug broker:
```
toadStool must:
  1. Know which drivers are installed
  2. Know which GPUs are available
  3. Handle driver binding/unbinding
  4. Manage power states
  5. Handle VRAM loss/recovery
  6. Track device health
  → Complex, fragile, hardware-specific
```

After GlowPlug broker:
```
toadStool just asks:
  "Give me a device that can do X"
  → Gets a warm, ready handle
  → If it needs a different backend, asks GlowPlug to swap
  → Never touches sysfs, power management, or driver binding
  → Hardware-agnostic
```

---

## Beyond GPU: PCIe Device Management

GlowPlug's architecture generalizes to any PCIe device:

### Devices GlowPlug Can Manage

| Device Type | Personality Options | Use Case |
|-------------|-------------------|----------|
| NVIDIA GPU | vfio, nouveau, nvidia | Compute, display, sovereign |
| AMD GPU | vfio, amdgpu | Compute, display, sovereign |
| Intel GPU | vfio, i915, xe | Compute, display |
| NPU (Akida) | vfio, akida-driver | Neural inference |
| FPGA | vfio | Custom accelerators |
| NVLink bridge | vfio | GPU↔GPU high-bandwidth |
| PCIe switch | native | Topology management |
| NIC | vfio, driver | RDMA, network compute |

### PCIe Transfer Coordination

When GlowPlug manages multiple devices, it can coordinate transfers:

```
┌──────────┐  PCIe P2P   ┌──────────┐
│ Titan V  │◄───────────►│ Titan V  │
│ (VFIO)   │  (GlowPlug  │ (nouveau)│
│          │   brokers)   │          │
└────┬─────┘             └────┬─────┘
     │                        │
     │    ┌──────────┐       │
     └────┤ GlowPlug ├───────┘
          │  broker   │
          └─────┬─────┘
                │
          ┌─────┴─────┐
          │ toadStool  │
          │ "transfer  │
          │  64MB from │
          │  slot0 to  │
          │  slot1"    │
          └────────────┘
```

GlowPlug handles:
- IOMMU configuration for P2P
- DMA-buf sharing between devices
- BAR1/BAR3 aperture window management for GPU↔GPU transfers
- NVLink topology discovery and routing
- PCIe bandwidth monitoring and backpressure

---

## Implementation Roadmap

### Phase 1a: Persistent Daemon (works today, ~3 days)

The daemon that holds VFIO fds open. Single personality per device.
No hot-swap yet. Just "bind at boot, stay alive forever."

```
coral-glowplug/
  src/
    main.rs           ← daemon entry, systemd notify
    config.rs         ← TOML config parsing
    device.rs         ← DeviceSlot + current personality
    socket.rs         ← Unix socket + SCM_RIGHTS fd passing
    health.rs         ← periodic health monitor
```

### Phase 1b: Hot-Swap Support (~1 week after 1a)

Add the swap protocol. toadStool can request personality changes.
State vault captures snapshots before each transition.

```
  + src/
      swap.rs         ← swap protocol (quiesce→snapshot→unbind→bind→verify)
      snapshot.rs     ← register state capture/compare
      vault.rs        ← snapshot storage, diff compression
      personality.rs  ← Personality enum + capability detection
```

### Phase 1c: toadStool Client Library (~3 days after 1b)

Rust crate that toadStool uses to talk to GlowPlug:

```rust
// In toadStool or any consumer:
let gp = GlowPlugClient::connect("/run/coralreef/glowplug.sock")?;
let devices = gp.list_devices()?;

let gpu = gp.get_device("0000:4a:00.0")?;
// gpu.fd is a VFIO container fd, ready for BAR0 mmap

// Later, swap to nouveau for Vulkan dispatch:
gp.swap("0000:4a:00.0", Personality::Nouveau)?;
// toadStool can now open /dev/dri/renderDN for wgpu
```

### Phase 2: Multi-Device Coordination (~2 weeks)

PCIe topology awareness, P2P transfer brokering, NUMA-aware placement.

### Phase 3: Sovereign HBM2 + Full Independence

GlowPlug can bring a cold device to warm without any vendor driver.
Makes hot-swap truly universal — even if a transition kills HBM2,
GlowPlug can re-train.

---

## Config File Design

```toml
# /etc/coralreef/glowplug.toml

[daemon]
socket = "/run/coralreef/glowplug.sock"
log_level = "info"
health_interval_ms = 5000

# Per-device configuration
[[device]]
bdf = "0000:4a:00.0"
name = "titan-v-target"
boot_personality = "vfio"        # what to bind at startup
power_policy = "always_on"       # never D3hot
role = "compute"                 # hint for toadStool scheduling
oracle_dump = "/etc/coralreef/oracle_gv100.txt"

[[device]]
bdf = "0000:03:00.0"
name = "titan-v-oracle"
boot_personality = "nouveau"     # keep warm via nouveau
power_policy = "on_demand"       # D3hot when idle is OK
role = "oracle"                  # reference card for register capture

[[device]]
bdf = "0000:XX:00.0"
name = "mi50-a"
boot_personality = "amdgpu"
power_policy = "always_on"
role = "compute"

# Personality-specific config
[vfio]
disable_idle_d3 = true
iommu_passthrough = false

[nouveau]
# Allow nouveau to load for oracle cards
allow_module = true

[nvidia]
persistence_mode = true          # nvidia-persistenced compat
```

---

## Why This Matters

### 1. Eliminates the "reboot tax"

Current: every VFIO session close requires a reboot.
With GlowPlug daemon: device stays warm forever. Hot-swap between
drivers without losing HBM2.

### 2. Makes toadStool hardware-agnostic

toadStool talks to GlowPlug, not to drivers. Adding a new GPU vendor
means adding a new Personality variant, not rewriting dispatch logic.

### 3. Enables mixed-mode compute

Run sovereign dispatch on one Titan V (via VFIO) while the other runs
NVK/Vulkan (via nouveau). Switch them mid-experiment if needed.

### 4. Scales to the full hardware fleet

2× Titan V + 2× MI50 + NPU + future cards — all managed by one daemon.
toadStool sees a pool of warm, capable devices.

### 5. PCIe transfer brokering

GPU↔GPU transfers, GPU↔NPU data sharing, DMA-buf passing — all
coordinated through one authority that understands the PCIe topology.

### 6. Foundation for sovereign boot

When sovereign HBM2 training is ready (Phase 3), GlowPlug is already
the boot-time authority. It just adds one more capability: cold-start.

---

## Relationship to Existing barraCuda BackendKind

```
barraCuda::BackendKind     GlowPlug Personality     How toadStool uses it
─────────────────────────  ───────────────────────  ──────────────────────
BarraCudaCpu               (not managed)            CPU-only, no GlowPlug
BarraCudaGpu               Nouveau / Amdgpu         wgpu → Vulkan → driver
CoralReefSovereign         Vfio                     BAR0 direct, coralReef SASS
KokkosCuda                 NvidiaProprietary        CUDA dispatch
PythonSarkas               (not managed)            External process
External                   Any                      GlowPlug brokers the fd
```

GlowPlug doesn't replace barraCuda's backend abstraction — it provides
the **hardware lifecycle** that backends rely on. barraCuda picks which
backend to use; GlowPlug ensures the hardware is ready for that backend.

---

## Open Design Questions

1. **Swap latency**: How fast can we unbind/bind? nouveau takes ~5s to
   probe. Is that acceptable for dynamic swap, or do we pre-bind?

2. **Multi-client**: Can multiple toadStool instances share a device?
   VFIO allows one group owner. nouveau allows multiple DRM clients.
   GlowPlug needs to manage this per-personality.

3. **Display GPU**: The display GPU (21:00.0 nvidia) should probably
   be excluded from GlowPlug management. Config flag: `managed = false`.

4. **Crash recovery**: If GlowPlug daemon crashes, all devices reset.
   Mitigation: watchdog that restarts daemon fast enough to re-grab
   devices before they cool. Or: systemd `Restart=always` + fast D0.

5. **Security boundary**: GlowPlug runs as root. toadStool runs as
   user. The Unix socket + fd passing provides the privilege boundary.
   Should we add device-level ACLs?

6. **nvidia driver compatibility**: The nvidia proprietary driver is
   aggressive about device ownership. Can we hot-swap to/from it?
   May need `nvidia-persistenced` integration or special handling.
