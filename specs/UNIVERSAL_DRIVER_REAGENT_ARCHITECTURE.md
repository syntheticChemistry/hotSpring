# Universal Driver Reagent Architecture

**Status:** Active — foundational architectural pattern for GlowPlug/Ember
**Origin:** hotSpring GPU cracking pipeline, "5060 lesson", user directive
**Absorbers:** coralReef (GlowPlug/Ember/coralctl), ecoPrimals standards
**Related:** [DRIVER_AS_SOFTWARE.md](DRIVER_AS_SOFTWARE.md), [GPU_CRACKING_GAP_TRACKER.md](archive/GPU_CRACKING_GAP_TRACKER.md)

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

## Ember Ring Architecture — Multi-Track Command Transport

### Problem

GPU hardware uses multiple independent command interfaces, each with its own wire format:

| Interface | Location | Format | Purpose |
|-----------|----------|--------|---------|
| SEC2 CMDQ/MSGQ | DMEM ring buffer | Structured queue entries with head/tail | ACR falcon bootstrap |
| FECS MTHD | BAR0 `0x409500-0x409804` | Register write/poll | GR context management |
| GPFIFO | DMA ring buffer | GP entries → PB segments | Compute dispatch |
| MAILBOX | Falcon `MAILBOX0`/`MAILBOX1` | Raw u32 read/write | Simple status exchange |

Our Layer 10 failure demonstrates the consequence of not respecting these boundaries:
writing `BOOTSTRAP_FALCON` to SEC2's `MAILBOX0`/`MAILBOX1` partially bootstraps FECS
but fails for GPCCS because the GV100 ACR firmware reads from its DMEM **command queue**,
not the mailbox registers. FECS then gets stuck (PC=0x0307) waiting for GPCCS
(PC=0x0000) which was never properly loaded.

### Architecture

**GlowPlug is the Mailbox. Ember is the Ring.**

```
                  ┌─────────────────────────────────┐
                  │         GlowPlug (Mailbox)       │
                  │  High-level commands:             │
                  │  "swap to nouveau"                │
                  │  "bootstrap FECS+GPCCS"           │
                  │  "dispatch shader X"              │
                  └──────────┬──────────────────────┘
                             │ JSON-RPC / SCM_RIGHTS
                  ┌──────────▼──────────────────────┐
                  │       Ember (Ring Manager)        │
                  │                                   │
                  │  ┌─────────┐ ┌─────────┐         │
                  │  │ SEC2    │ │ FECS    │         │
                  │  │ CMDQ    │ │ MTHD    │  ...    │
                  │  │ Ring    │ │ Ring    │         │
                  │  └────┬────┘ └────┬────┘         │
                  │       │           │               │
                  │  ┌────┴────┐ ┌────┴────┐         │
                  │  │ GPFIFO  │ │ Trace   │         │
                  │  │ Ring    │ │ Ring    │         │
                  │  └────┬────┘ └────┬────┘         │
                  └───────┼───────────┼──────────────┘
                          │           │
                  ┌───────▼───────────▼──────────────┐
                  │    coral-driver (BAR0 transport)   │
                  └──────────────────────────────────┘
```

Each **Ring** is a typed, observable command channel:

```rust
pub trait EmberRing {
    type Command;
    type Response;

    fn submit(&mut self, cmd: Self::Command) -> RingTicket;
    fn poll(&mut self, ticket: RingTicket) -> RingStatus<Self::Response>;
    fn drain_log(&mut self) -> Vec<RingEvent>;
}
```

### Ring Types

**SEC2 CMDQ Ring** — the immediate blocker for Layer 10:
- Writes properly formatted command entries to the SEC2 DMEM ring buffer
- Tracks CMDQ head/tail pointers (found at known DMEM offsets after ACR init)
- Handles MSGQ responses (ACR writes completion to a separate DMEM queue)
- Wire format: `{ cmd_type: u32, flags: u32, falcon_id: u32, ... }`
- This is what nouveau's `nvkm_falcon_cmdq_send` / `nvkm_falcon_msgq_recv` implement

**FECS MTHD Ring** — already partially implemented in `fecs_method.rs`:
- Submit: write `MTHD_DATA` then `MTHD_CMD`
- Poll: spin on `MTHD_STATUS2` for completion
- Commands: `DISCOVER_IMAGE_SIZE`, `SET_WATCHDOG`, `BIND_POINTER`, `WFI_GOLDEN_SAVE`

**GPFIFO Ring** — the endpoint for compute dispatch:
- Submit: write GP entries to the GPFIFO ring buffer
- Signal: write `GPPUT` doorbell to kick PBDMA
- Response: fence semaphore release

**Trace Ring** — read-only, captures hardware events:
- mmiotrace events from kernel
- BAR0 register snapshots at swap boundaries
- Falcon PC traces, mailbox transitions
- Timestamped for post-hoc analysis

### Multi-Track Switching

Ember can maintain **multiple rings simultaneously** and switch tracks during swaps:

```
Swap to nouveau:
  1. Activate SEC2_CMDQ ring (bootstrap FECS+GPCCS)
  2. Activate FECS_MTHD ring (golden context, discover sizes)
  3. Activate GPFIFO ring (compute dispatch)
  4. Trace ring captures all 3

Swap to nvidia_oracle:
  1. Load custom kernel module
  2. Different SEC2 command protocol (version-indexed)
  3. Same FECS/GPFIFO rings (firmware is compatible)

Swap to amdgpu (failure test):
  1. Only trace ring active
  2. Captures probe failure path
  3. No falcon rings (wrong vendor)
```

### Observability

Each ring maintains a timestamped event log:

```rust
pub struct RingEvent {
    pub timestamp: Instant,
    pub ring_id: &'static str,
    pub direction: Direction, // Submit | Response | Timeout | Error
    pub payload: Vec<u8>,
    pub latency_us: Option<u64>,
}
```

This gives us:
- **Separate timing analysis** per ring (SEC2 latency vs FECS latency vs GPFIFO latency)
- **Message replay** for debugging (feed captured SEC2 commands to a mock)
- **Cross-ring correlation** (FECS stuck → check if SEC2 CMDQ completion arrived)
- **Smaller test surface** — test SEC2 ring format without touching FECS

### Implementation Path

1. **Immediate (Layer 10 unblock):** Implement SEC2 CMDQ ring in coral-driver
   - Reverse-engineer DMEM queue layout from nouveau `nvkm_falcon_cmdq`
   - Send properly formatted `BOOTSTRAP_FALCON(GPCCS)` via the ring
   - Verify GPCCS starts (PC advances from 0x0000)

2. **Near-term:** Extract ring abstraction into coral-ember
   - `EmberRing` trait in coral-ember
   - SEC2, FECS, GPFIFO implementations in coral-driver
   - Ring event logging to trace directory

3. **Evolution:** Multi-ring orchestration in Ember daemon
   - GlowPlug swap triggers ring activation sequence
   - Ember manages ring lifecycle across driver swaps
   - Ring events feed into hw-learn distiller

### Relationship to SEC2 CMDQ (Layer 10 Technical Detail)

The SEC2 ACR firmware on GV100 (Volta) initializes a **CMDQ** (command queue) and
**MSGQ** (message queue) in its DMEM during boot. The host communicates by:

1. Writing a command structure at the CMDQ tail pointer in DMEM
2. Advancing the tail pointer (also in DMEM)
3. Optionally poking an IRQ to wake SEC2
4. SEC2 reads the command, processes it, writes response to MSGQ
5. Host reads MSGQ head for completion

Our current `strategy_mailbox.rs` writes directly to `MAILBOX0`/`MAILBOX1`, which
the firmware may check as a legacy fallback but does NOT use as its primary command
interface. This is why FECS partially bootstraps (MAILBOX might trigger a
one-time bootstrap on first write) but GPCCS fails (subsequent MAILBOX writes
are not processed as ring commands).

The DMEM layout from our diagnostic dumps shows non-zero ranges at:
- `0x020-0x04c`: BL descriptor / ACR metadata
- `0x210-0x264`: ACR descriptor (WPR addresses, falcon counts)
- `0xB00-0xB10`: Possible queue headers or crypto state
- `0xF20-0xF30`: Possible MSGQ region

These offsets need correlation with nouveau's `nvkm_falcon_cmdq` / `nvkm_falcon_msgq`
structures to identify head/tail pointers and command entry format.

## Relationship to Other Specs

- **[DRIVER_AS_SOFTWARE.md](DRIVER_AS_SOFTWARE.md):** Established the swap-capture-return
  cycle and recipe distillation. This document extends it with open targets and the
  reagent safety model.

- **[GPU_CRACKING_GAP_TRACKER.md](archive/GPU_CRACKING_GAP_TRACKER.md):** Open target acceptance
  is prerequisite for Gap 6 (nvidia_oracle) and enables novel driver experimentation
  beyond the current allowlist.

- **CORALREEF_TRACE_INTEGRATION_HANDOFF.md:** Specifies the mmiotrace module and
  VendorLifecycle trace hooks. This document adds the default-on policy and the
  Generic/Custom handling for unknown drivers.
