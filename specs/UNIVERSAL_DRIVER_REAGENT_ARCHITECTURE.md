# Universal Driver Reagent Architecture

**Status:** Active — foundational architectural pattern for GlowPlug/Ember
**Origin:** hotSpring GPU cracking pipeline, "5060 lesson", user directive
**Absorbers:** coralReef (GlowPlug/Ember/coralctl), ecoPrimals standards
**Related:** [DRIVER_AS_SOFTWARE.md](DRIVER_AS_SOFTWARE.md), [GPU_CRACKING_GAP_TRACKER.md](GPU_CRACKING_GAP_TRACKER.md)

## Core Thesis

A kernel driver is a **reagent** — a controlled substance you apply to hardware to observe
its effects, then withdraw. It is not infrastructure. The moment you treat a driver as
infrastructure (binding it permanently to the kernel, letting it own device lifecycle),
you lose control. The "5060 incident" proved this: binding a legacy nvidia driver to the
kernel broke the display stack because the driver was treated as a permanent fixture
rather than a temporary experimental tool.

GlowPlug's role is to be the **lifecycle broker** that ensures:
1. Protected GPUs are never touched
2. Managed GPUs can receive **any** driver as a reagent
3. Every reagent application is traced and captured
4. The GPU always returns to its managed state after the experiment

## The 5060 Lesson

On a previous session, a legacy nvidia driver was loaded to the kernel and bound to the
RTX 5060 (the display GPU). This broke the display stack — DRM became unavailable, and
recovery required a reboot. The root cause was treating the driver as infrastructure
("install this driver to use this GPU") rather than as a reagent ("apply this driver
temporarily to this managed GPU, observe the effects, then remove it").

**Rule:** The system-connected driver on a display GPU is sacrosanct. All experimentation
happens on managed GPUs through GlowPlug's swap lifecycle. If you want to test a legacy
nvidia driver, you bind it to a Titan via `coralctl swap`, not to the kernel via modprobe.

## Open Target Acceptance

### Current Model (Restrictive)

```rust
const KNOWN_TARGETS: &[&str] = &[
    "vfio", "vfio-pci", "nouveau", "amdgpu", "nvidia", "nvidia_oracle",
    "xe", "i915", "akida-pcie", "unbound",
];
if !KNOWN_TARGETS.contains(&target) {
    return Err("unknown target driver");
}
```

This prevents experimentation. You cannot bind `amdgpu` to an NVIDIA card, even though
the failure itself is informative. You cannot test a custom research driver without
first adding it to the allowlist.

### New Model (Open)

```rust
const WELL_KNOWN: &[&str] = &[
    "vfio", "vfio-pci", "nouveau", "amdgpu", "nvidia", "nvidia_oracle",
    "xe", "i915", "akida-pcie", "unbound",
];
if !WELL_KNOWN.contains(&target) && !target.starts_with("nvidia_oracle_") {
    tracing::warn!(bdf, target, "target is not well-known — treating as reagent driver");
}
// Validate: must be a valid kernel module name
if !is_valid_module_name(target) {
    return Err(format!("invalid driver name '{target}'"));
}
// Protection layers still enforced:
//   - is_active_display_gpu(bdf) blocks display GPUs
//   - config.is_protected() blocks role=display/shared
//   - managed_bdfs allowlist in Ember
```

The swap proceeds for any valid module name on any managed GPU. Unknown drivers get
`GenericLifecycle` (conservative defaults: pin power, simple bind, 5s settle). The
protection model is **unchanged** — it's role-based and BDF-based, not driver-based.

### Why Nonsensical Combinations Have Value

Binding `amdgpu` to an NVIDIA Titan V will fail at probe time. But the **trace of that
failure** is useful:

- It validates that the protection model correctly allows the attempt on managed hardware
- It captures the kernel's error path (dmesg, probe failure reason)
- It proves the GPU returns cleanly to VFIO after a failed bind
- It exercises the GenericLifecycle error recovery path
- For novel hardware or custom drivers, "will this work?" is a genuine research question

The capability to test failure is inherent to validation.

## Reagent Safety Taxonomy

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU SAFETY TAXONOMY                          │
├─────────────┬───────────────────────────────────────────────────┤
│ PROTECTED   │ role=display                                     │
│             │ - Ember does not hold VFIO fds                   │
│             │ - Swaps forbidden at GlowPlug layer              │
│             │ - Driver stays kernel-connected                  │
│             │ - is_active_display_gpu() as last-resort guard   │
│             │ - Example: RTX 5060 on nvidia 580.126.18         │
├─────────────┼───────────────────────────────────────────────────┤
│ SHARED      │ role=shared                                      │
│             │ - Display GPU that also does compute              │
│             │ - Power/VRAM budgets enforced via SharedQuota     │
│             │ - Never swapped, never loses DRM                 │
│             │ - Compute via driver's native API (CUDA/Vulkan)  │
│             │ - Example: single-GPU gaming PC doing science    │
├─────────────┼───────────────────────────────────────────────────┤
│ MANAGED     │ role=compute, role=oracle                        │
│             │ - Boot to VFIO                                   │
│             │ - ANY driver can be bound as a reagent            │
│             │ - Trace captures every swap (default-on)         │
│             │ - Always returns to VFIO after experiment         │
│             │ - GenericLifecycle handles unknown drivers        │
│             │ - Example: Titan V on vfio-pci, swapped to       │
│             │   nouveau/nvidia/nvidia_oracle/amdgpu/custom     │
├─────────────┼───────────────────────────────────────────────────┤
│ NATIVE      │ role=native-compute (LATE-STAGE)                 │
│ COMPUTE     │ - GPU stays on its kernel driver                 │
│             │ - GlowPlug borrows compute via driver API        │
│             │ - Resource budgets prevent display starvation    │
│             │ - No swap, no DRM interruption                   │
│             │ - Example: family gaming GPU lending idle cycles  │
└─────────────┴───────────────────────────────────────────────────┘
```

### Protection Layers (Defense in Depth)

| Layer | Location | Mechanism |
|-------|----------|-----------|
| 1. Config role | `glowplug.toml` | `is_protected()` checks `role=display\|shared` |
| 2. GlowPlug startup | `main.rs` | Protected slots skip VFIO activation |
| 3. GlowPlug swap | `device/swap.rs` | `swap_traced()` refuses protected devices |
| 4. Ember BDF allowlist | `managed_bdfs` | `require_managed_bdf()` on every RPC |
| 5. Ember display check | `swap.rs` | `is_active_display_gpu()` via DRM connector sysfs |
| 6. DRM isolation | Xorg + udev | `AutoAddGPU=false`, seat tag stripping |

Removing the `KNOWN_TARGETS` allowlist does NOT weaken any of these. The allowlist was
a driver-name check; protection is BDF-and-role-based.

## Trace as Foundation

### Principle

Every swap is a learning opportunity. Trace should be **default-on** for managed GPUs,
not an opt-in afterthought. The system accumulates a corpus of driver init sequences
that the ACR boot solver and hw-learn distiller consume.

### Default Inversion

```
BEFORE: coralctl swap <bdf> <target>            → trace OFF
        coralctl swap <bdf> <target> --trace     → trace ON

AFTER:  coralctl swap <bdf> <target>            → trace ON (from config)
        coralctl swap <bdf> <target> --no-trace  → trace OFF
```

### Configuration

```toml
[daemon]
trace_default = true
trace_data_dir = "/var/lib/coralreef/traces"
```

### Failure Tolerance

The trace path already handles failures gracefully — if mmiotrace cannot be enabled
(e.g. another tracer is active, or debugfs is not mounted), the swap proceeds with a
warning. Making trace default-on does not risk blocking swaps.

### What Gets Captured

| Artifact | Every Swap | Well-Known Only |
|----------|-----------|-----------------|
| mmiotrace_raw.txt | Yes | — |
| mmiotrace_writes.txt | Yes | — |
| Falcon-filtered traces | — | Yes (NVIDIA) |
| ACR/DMA-filtered traces | — | Yes (NVIDIA) |
| BAR0 warm snapshot | Yes | — |
| BAR0 residual snapshot | Yes | — |
| manifest.json (metadata) | Yes | — |
| demmio annotation | — | Yes (if demmio installed) |

Unknown/reagent drivers still get raw trace and BAR0 snapshots. Vendor-specific filtering
is additive (via `VendorLifecycle::trace_filter_ranges()`).

## Generic Personality

When a managed GPU is swapped to an unknown driver, GlowPlug needs a personality variant
to represent the result:

```rust
pub enum Personality {
    Vfio { group_id: u32 },
    Nouveau { drm_card: Option<String> },
    Nvidia { drm_card: Option<String> },
    NvidiaOracle { drm_card: Option<String>, module_name: String },
    Amdgpu { drm_card: Option<String> },
    Xe { drm_card: Option<String> },
    I915 { drm_card: Option<String> },
    Akida,
    Unbound,
    Custom { driver_name: String, drm_card: Option<String> },  // NEW
}
```

`Custom` is the catch-all for reagent drivers that don't have a dedicated variant. It
stores the driver name for display/logging and any DRM card that appeared after bind.

`GenericLifecycle` (already implemented in `vendor_lifecycle.rs`) provides conservative
defaults for unknown vendors. `Custom` personality pairs with `GenericLifecycle` to form
a complete handling path for any driver on any managed GPU.

## Cross-Driver Validation Matrix

The open target model enables a systematic validation matrix:

| Target Driver | On NVIDIA GPU | On AMD GPU | On Intel GPU | Purpose |
|--------------|--------------|-----------|-------------|---------|
| nouveau | Init trace (warm-up path) | Probe fail (validation) | Probe fail | Recipe capture |
| nvidia | Proprietary ACR trace | Probe fail | Probe fail | ACR boot solver |
| nvidia_oracle | Renamed coexistence test | Probe fail | Probe fail | Version-indexed recipes |
| amdgpu | Probe fail (validation) | Init trace | Probe fail | AMD recipe capture |
| xe | Probe fail | Probe fail | Init trace | Intel recipe capture |
| custom_research | Unknown behavior | Unknown | Unknown | Novel driver testing |

Every cell in this matrix is a valid experiment. Failed probes validate error handling.
Successful probes produce recipes. The matrix grows with every new driver version and
hardware generation.

## Evolution Path

1. **Near-term (Phases 1-3):** Open targets, Custom personality, trace-as-default
   - Unblocks GPU cracking experiments on Titans
   - coralReef implementation handoff delivered

2. **Late-stage (Phases 5-6):** Native-compute mode for gaming systems
   - `role=native-compute` borrows GPU compute without swap
   - Family gaming safety (never interrupt DRM)
   - See [NATIVE_COMPUTE_ROADMAP.md](NATIVE_COMPUTE_ROADMAP.md)

3. **Ecosystem:** Multi-machine compute borrowing
   - Network-capable GlowPlug
   - Household GPU pool management
   - barraCuda as portable compute runtime

## Relationship to Other Specs

- **[DRIVER_AS_SOFTWARE.md](DRIVER_AS_SOFTWARE.md):** Established the swap-capture-return
  cycle and recipe distillation. This document extends it with open targets and the
  reagent safety model.

- **[GPU_CRACKING_GAP_TRACKER.md](GPU_CRACKING_GAP_TRACKER.md):** Open target acceptance
  is prerequisite for Gap 6 (nvidia_oracle) and enables novel driver experimentation
  beyond the current allowlist.

- **CORALREEF_TRACE_INTEGRATION_HANDOFF.md:** Specifies the mmiotrace module and
  VendorLifecycle trace hooks. This document adds the default-on policy and the
  Generic/Custom handling for unknown drivers.
