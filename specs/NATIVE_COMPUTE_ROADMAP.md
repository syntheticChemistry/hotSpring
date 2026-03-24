# Native Compute Roadmap

**Status:** Design — late-stage evolution of GlowPlug
**Prerequisite:** Universal Driver Reagent Architecture (Phases 1-3)
**Related:** [UNIVERSAL_DRIVER_REAGENT_ARCHITECTURE.md](UNIVERSAL_DRIVER_REAGENT_ARCHITECTURE.md)

## Problem Statement

Today, GlowPlug only manages GPUs that boot to VFIO. This is the right model for a
dedicated compute lab (biomeGate: 2x Titan V on VFIO, RTX 5060 on display). But it
does not address the broader deployment scenario: a family gaming PC where you want to
borrow spare GPU compute without breaking the game.

The previous attempt to use a Titan's nvidia driver alongside the 5060's nvidia driver
failed because we tried to swap the system driver — treating it as infrastructure. The
native-compute model avoids this entirely: the GPU stays on its kernel driver, and
GlowPlug borrows compute capacity through the driver's own API.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   GlowPlug Daemon                           │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ VFIO Mode    │  │ Shared Mode  │  │ Native-Compute   │  │
│  │ (existing)   │  │ (existing)   │  │ (NEW)            │  │
│  │              │  │              │  │                  │  │
│  │ Boot→VFIO    │  │ Display+CUDA │  │ Display+Borrow   │  │
│  │ Swap drivers │  │ Power budget │  │ Compute budget   │  │
│  │ Trace MMIO   │  │ VRAM budget  │  │ API-level trace  │  │
│  │              │  │              │  │                  │  │
│  │ Titan V #1   │  │ (single-GPU) │  │ Family gaming PC │  │
│  │ Titan V #2   │  │              │  │ Steam + Science  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│         │                 │                   │             │
│         ▼                 ▼                   ▼             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Ember        │  │ nvidia-smi   │  │ NativeCompute    │  │
│  │ (sysfs swap) │  │ (budgets)    │  │ Client (NEW)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Scenarios

### Scenario 1: Dedicated Lab (Current)

```toml
[[device]]
bdf = "0000:21:00.0"
role = "display"           # RTX 5060 — protected

[[device]]
bdf = "0000:03:00.0"
role = "oracle"            # Titan V — managed, VFIO swap

[[device]]
bdf = "0000:4a:00.0"
role = "compute"           # Titan V — managed, VFIO swap
```

All experimentation happens on managed GPUs. Display GPU untouched.

### Scenario 2: Family Gaming PC (Future)

```toml
[[device]]
bdf = "0000:01:00.0"
role = "native-compute"    # Gaming GPU — borrow idle compute
boot_personality = "nvidia"

[quota.native_compute]
max_vram_mib = 4096        # reserve for games
max_power_w = 200          # thermal budget
compute_api = "cuda"       # nvidia: CUDA, amd: ROCm, intel: oneAPI
priority = "low"           # yield to display/gaming workloads
idle_threshold_pct = 30    # only borrow when GPU < 30% utilized
```

The gaming GPU keeps its driver, keeps its DRM, keeps running Steam. GlowPlug watches
utilization and borrows spare compute through the driver's own API when the GPU is idle.

### Scenario 3: Mixed Fleet (Future)

```toml
[[device]]
bdf = "0000:01:00.0"
role = "display"           # Primary display — protected

[[device]]
bdf = "0000:41:00.0"
role = "compute"           # Dedicated compute card — VFIO managed

[[device]]
bdf = "0000:81:00.0"
role = "native-compute"    # Secondary GPU — borrow when idle
```

A workstation with a display GPU, a dedicated compute card on VFIO, and a secondary
GPU that stays on its driver but contributes idle cycles.

## Phase 5: `role=native-compute`

### Config

```rust
pub fn is_native_compute(&self) -> bool {
    self.role.as_deref() == Some("native-compute")
}

pub fn is_protected(&self) -> bool {
    self.is_display() || self.is_shared()
    // native-compute is NOT protected — it contributes compute
    // but it is NOT swappable — it stays on its kernel driver
}

pub fn is_swappable(&self) -> bool {
    !self.is_protected() && !self.is_native_compute()
}
```

Native-compute devices are not protected (they participate in compute), but they are not
swappable (driver stays bound). They have a third path: borrow.

### Commands

```bash
# VFIO-managed GPUs: swap drivers
coralctl swap 0000:03:00.0 nouveau          # existing

# Native-compute GPUs: borrow/release
coralctl borrow 0000:01:00.0                # reserve a compute slice
coralctl release 0000:01:00.0               # release the slice
coralctl status 0000:01:00.0                # shows utilization + borrow state
```

### NativeComputeClient

A new component that talks to the GPU through its kernel driver's compute API:

```rust
pub trait NativeComputeBackend: Send + Sync {
    fn name(&self) -> &str;
    fn is_available(&self, bdf: &str) -> bool;
    fn reserve(&self, bdf: &str, quota: &NativeComputeQuota) -> Result<ComputeSlice, Error>;
    fn release(&self, slice: ComputeSlice) -> Result<(), Error>;
    fn utilization(&self, bdf: &str) -> Result<GpuUtilization, Error>;
}
```

Implementations:
- `CudaComputeBackend` — Uses CUDA runtime/driver API
- `VulkanComputeBackend` — Uses Vulkan compute queues
- `RocmComputeBackend` — Uses ROCm/HIP for AMD GPUs

barraCuda already has CUDA and Vulkan backends. GlowPlug becomes the broker that
selects the right backend based on the device's kernel driver.

### Utilization Monitoring

For idle-threshold-based borrowing:

```rust
pub struct GpuUtilization {
    pub gpu_pct: u32,       // 0-100
    pub vram_used_mib: u64,
    pub vram_total_mib: u64,
    pub power_draw_w: u32,
    pub temperature_c: u32,
}
```

Sources:
- NVIDIA: `nvidia-smi --query-gpu=...` or NVML library
- AMD: `rocm-smi` or sysfs hwmon
- Intel: sysfs i915/xe power/frequency

GlowPlug polls utilization and auto-borrows when the GPU drops below the idle threshold.
It auto-releases when utilization rises (game started).

### Trace in Native-Compute Mode

mmiotrace is not available (the driver stays bound). Instead, trace captures API-level
telemetry:

| Artifact | Description |
|----------|-------------|
| `borrow_log.jsonl` | Timestamped borrow/release events |
| `utilization_series.jsonl` | GPU utilization over time |
| `workload_manifest.json` | What was computed, duration, result |
| `api_trace.jsonl` | CUDA/Vulkan API calls (if enabled) |

This is a different kind of trace than MMIO-level, but it still builds the knowledge
corpus about how the GPU behaves under different workloads.

## Phase 6: DRM Passthrough and Multi-Machine

### Local DRM Passthrough

For native-compute GPUs that also have display outputs (e.g. a gaming GPU):

```
Steam renders to GPU via DRM/KMS → display output
barraCuda computes via CUDA → results to CPU/network
GlowPlug brokers both: display priority > compute priority
```

The key insight: modern GPU drivers already support concurrent display + compute. CUDA
contexts can coexist with Vulkan/OpenGL display contexts. GlowPlug's role is to enforce
**resource budgets** so the compute workload doesn't starve the display.

### Multi-Machine (Household GPU Pool)

The long-term vision: GlowPlug instances across multiple machines form a compute pool.

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ biomeGate Lab    │     │ Gaming PC #1     │     │ Gaming PC #2     │
│ 2x Titan V (VFIO)│    │ RTX 4070 (native)│     │ RTX 3080 (native)│
│ RTX 5060 (display)│    │ Gaming + Science │     │ Gaming + Science │
│                  │     │                  │     │                  │
│ GlowPlug Master │◄───►│ GlowPlug Agent  │◄───►│ GlowPlug Agent  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Household GPU Pool    │
                    │   Total: ~60 TFLOPS f32 │
                    │   barraCuda workloads   │
                    │   distributed across    │
                    │   idle GPU slices       │
                    └─────────────────────────┘
```

Requirements:
- **Network transport:** GlowPlug RPC over TLS (currently Unix socket only)
- **Trust model:** mTLS certificates per machine, household CA
- **Workload scheduling:** barraCuda partitions work across available GPU slices
- **Priority:** Gaming always wins — auto-release on utilization spike
- **Privacy:** Compute data stays local; only results are transmitted

This is a significant engineering effort and is documented here as a roadmap target,
not an immediate implementation plan.

## Implementation Priority

| Phase | Effort | Dependency | When |
|-------|--------|------------|------|
| 5a: `role=native-compute` config | Small | Phase 1-3 | After open targets land |
| 5b: `coralctl borrow/release` | Medium | 5a | After config |
| 5c: NativeComputeBackend trait | Medium | 5b | After commands |
| 5d: CUDA backend integration | Large | 5c + barraCuda | After trait |
| 5e: Utilization monitoring | Medium | 5a | Parallel with 5c |
| 5f: Auto-borrow on idle | Medium | 5e + 5b | After both |
| 6a: Network transport | Large | 5d | After local works |
| 6b: Multi-machine pool | Large | 6a | After network |

## Relationship to barraCuda

barraCuda already has:
- CUDA backend (`coral-driver/src/cuda/`)
- Vulkan compute backend (via wgpu)
- MD simulation kernels that can run on either

GlowPlug becomes the **deployment broker**: it knows which GPUs are available, which
backends they support, and what resource budgets apply. barraCuda submits work to
GlowPlug, which routes it to the best available GPU slice.

This mirrors the existing relationship: hotSpring designs the pattern, coralReef
implements the infrastructure, barraCuda consumes the compute.
