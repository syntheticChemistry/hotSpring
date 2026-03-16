# Handoff: hotSpring Pin — Primal Evolution Sprint

**Date:** March 16, 2026
**From:** hotSpring (experiments 001-069)
**To:** coralReef, toadStool, barraCuda
**License:** AGPL-3.0-only
**Covers:** hotSpring v0.6.31, Exp 060-069, coral-glowplug v0.1.0

---

## Executive Summary

hotSpring is pinning to let the primals absorb, harden, and evolve the sovereign
GPU infrastructure for a few days. When hotSpring returns, it will pull evolved
primals, revalidate 39/39 suites, and resume hardware experiments (GPCCS discovery,
sovereign HBM2 training, AMD MI50 bring-up).

**What hotSpring proved:**
- 848 tests, 39/39 validation suites, 69 experiments — science is stable
- coral-glowplug: boot-persistent PCIe lifecycle broker (systemd, VFIO-first, graceful shutdown)
- FECS firmware direct execution from host-loaded IMEM (LS security bypass on clean falcon)
- D3hot→D0 VRAM recovery, 24/26 hardware tests
- DRM render node fencing lesson (kernel oops root cause + fix)
- Full reproducibility checklist for adding new GPUs

**What hotSpring needs back:**
- Hardened coral-glowplug with JSON-RPC 2.0, trait-based personalities, SCM_RIGHTS
- toadStool GlowPlug socket client
- AMD Vega metal implementation (MI50 readiness)
- GP_PUT DMA read fix (PFIFO last mile)

---

## Part 1: What Each Primal Should Do

### coralReef — Primary Consumer (Est. 3-4 Weeks of Backlog)

coral-glowplug lives in `coralReef/crates/coral-glowplug/`. It was prototyped
rapidly during hotSpring's experiment loop. It works but needs hardening.

**Priority 1 — Socket Protocol (Days)**

Current: ad-hoc JSON lines over Unix socket.
Target: JSON-RPC 2.0 (ecosystem standard), matching toadStool and barraCuda IPC.

Methods to implement:
- `glowplug.device.list` → `[{bdf, name, personality, vram_alive, power, chip}]`
- `glowplug.device.health` → `{vram, power, domains, pci_link_width}`
- `glowplug.device.swap` → `{bdf, target_personality}` → `{ok, snapshot_id}`
- `glowplug.device.resurrect` → `{bdf}` → `{ok, vram_alive}`
- `glowplug.device.status` → full daemon status
- `glowplug.shutdown` → graceful stop

**Priority 2 — SCM_RIGHTS fd Passing (Days)**

toadStool needs VFIO container file descriptors for sovereign dispatch.
After `device.swap(bdf, "vfio")`, the socket should pass the VFIO group fd
via `SCM_RIGHTS` ancillary message. Standard pattern: `sendmsg` with `cmsg`.

**Priority 3 — Personality Trait System (1 Week)**

Current: `enum Personality { Vfio, Nouveau, Amdgpu, NvidiaProprietary, Unbound }`.
Target: trait-based so new vendors don't require enum modification.

```rust
pub trait GpuPersonality: Send + Sync {
    fn name(&self) -> &str;
    fn bind(&self, bdf: &str) -> Result<()>;
    fn unbind(&self, bdf: &str) -> Result<()>;
    fn capabilities(&self) -> DeviceCapabilities;
    fn health_probe(&self, bar0: &MappedBar) -> DeviceHealth;
}
```

Register `VfioPersonality`, `NouveauPersonality`, `AmdgpuPersonality` etc.
via a `PersonalityRegistry`. New vendors add a struct, not modify an enum.

**Priority 4 — AMD Vega Metal (`amd_metal.rs`) (2-3 Weeks)**

Current: 6 TODO stubs. Target: full register map for MI50/GFX906.

Sources:
- `drivers/gpu/drm/amd/amdgpu/` in Linux kernel (fully open)
- `gfx_v9_0.c` for engine topology
- `soc15.c` for register base addresses
- `umc_v6_1.c` for HBM2 controller registers

What to implement:
- `power_domains()` → SMC, GRBM, SRBM registers
- `memory_controllers()` → UMC (HBM2 controller), GC L2 cache
- `compute_engines()` → GFX, SDMA, VCN
- `mmio_register_domains()` → BAR0 register map
- `bar0_domain_map()` → named regions for health probing
- `power_on_sequence()` → register writes for D0 initialization

**Priority 5 — GP_PUT DMA Read (Days)**

Exp 058 handoff documents the fix: USERD_TARGET in the runlist entry must
point to system memory, not VRAM. The register values are known. This is
the last step before PFIFO channel dispatch works.

**Priority 6 — Privilege Model (Days)**

Replace `sudo tee` fallback in `sysfs_write()` with proper capabilities:
- Run daemon with `CAP_SYS_ADMIN` (for VFIO)
- Use polkit for user-initiated swaps via socket
- Remove all `Command::new("sudo")` calls

**Priority 7 — DRM Consumer Fence (Hours)**

In `resurrect_hbm2()`, before binding nouveau:
1. Check `lsof /dev/dri/renderD*` for open consumers
2. If any found on the target BDF, abort resurrection with clear error
3. Only proceed when confirmed safe

---

### toadStool — Integration Consumer (Est. 1-2 Weeks)

**1. GlowPlug Socket Client Crate**

New crate: `toadstool-glowplug` or module in `toadstool-runtime-gpu`.

```rust
pub struct GlowPlugClient {
    socket: UnixStream,
}

impl GlowPlugClient {
    pub fn connect(path: &str) -> Result<Self>;
    pub fn list_devices(&self) -> Result<Vec<DeviceInfo>>;
    pub fn health(&self, bdf: &str) -> Result<DeviceHealth>;
    pub fn swap(&self, bdf: &str, target: &str) -> Result<SwapResult>;
    pub fn resurrect(&self, bdf: &str) -> Result<ResurrectResult>;
}
```

Discovery: check `XDG_RUNTIME_DIR/coralreef/glowplug.sock` then `/run/coralreef/glowplug.sock`.

**2. VFIO Device in sysmon**

`toadstool-sysmon` should detect vfio-pci bound devices:
- Scan `/sys/bus/pci/drivers/vfio-pci/` for bound BDFs
- Read IOMMU group from `/sys/kernel/iommu_groups/`
- Report as `GpuDevice` with `driver: "vfio-pci"` and `sovereign: true`

**3. hw-learn: GlowPlug Health Feed**

Feed `DeviceHealth` from GlowPlug into the learning pipeline:
- VRAM alive/dead transitions → training data for HBM2 lifecycle model
- Power state changes → D0/D3hot pattern recognition
- Domain fault history → predict hardware failures

---

### barraCuda — No Action Needed

barraCuda's IPC-first design means zero changes. When toadStool wires
GlowPlug discovery, barraCuda automatically sees VFIO devices through
the existing `sovereign-dispatch` feature gate. The compile→dispatch
pipeline flows: `barraCuda → coralReef (compile) → toadStool (dispatch) → GlowPlug (VFIO)`.

**One future opportunity:** when PFIFO dispatch works, barraCuda could
add a `GlowPlugBackend` variant to `GpuBackend` that dispatches directly
through GlowPlug's socket (skipping wgpu/Vulkan entirely). But this
depends on coralReef completing GP_PUT + FECS/GPCCS.

---

## Part 2: What hotSpring Will Do When It Returns

### Phase 1: Revalidation (1-2 hours)

1. Pull latest coralReef, toadStool, barraCuda
2. Update Cargo.toml pins
3. `cargo test` → confirm 848 tests pass
4. `cargo run --release --bin validate_all` → confirm 39/39 suites

### Phase 2: GPCCS Discovery Experiment (hours to days)

Scan GV100 BAR0 for GPCCS falcon:
- Read at 0x1000 intervals through 0x400000-0x500000
- Look for falcon BOOT0 signature pattern
- Cross-reference with nouveau source (`gv100_gr_init`)
- If found: load GPCCS firmware, start GPCCS, re-run FECS

### Phase 3: Sovereign Dispatch Attempt

If GPCCS + FECS are both running:
- Complete GP_PUT DMA read (if coralReef hasn't already)
- Submit a nop shader through PFIFO
- If it runs: first sovereign compute dispatch on GV100
- Write up as Exp 070

### Phase 4: AMD MI50 Bring-Up (if card arrives)

- Add MI50 BDF to glowplug.toml
- Verify auto-discovery finds it
- Test health monitoring with AMD Vega metal (if coralReef implemented it)
- Run first coralReef → GFX906 compilation → amdgpu dispatch

---

## Part 3: What Not to Change

### Stability Invariants

- **Do NOT change `glowplug.toml` boot_personality** — both Titans must stay
  on `vfio` at boot. The DRM render node kernel oops will recur if nouveau
  boots on a non-display GPU.
- **Do NOT remove `disable_idle_d3`** — GV100 PM reset blocks indefinitely
  without this.
- **Do NOT remove `reset_method` disable** from shutdown handler — same PM
  reset issue.

### Architecture Invariants

- GlowPlug is part of coralReef (not toadStool) — it manages hardware at
  the PCIe level, below toadStool's abstraction layer
- Personalities are device-level, not system-level — one device = one personality
- Socket API is control-only until SCM_RIGHTS is implemented — no VFIO fds
  flow through the socket yet
- Display GPU (RTX 5060 on nvidia) is NEVER managed by GlowPlug

---

## Part 4: Open Research Questions (For Next hotSpring Session)

| Question | Approach | Expected Difficulty |
|----------|----------|-------------------|
| Where is GPCCS on GV100? | BAR0 scan, nouveau source analysis | Medium (hours to days) |
| Why does FECS halt at PC=0x2835? | Load + start GPCCS first, then re-run FECS | Depends on GPCCS |
| Can we write instance block via PRAMIN? | Write to BAR0 0x700000 window | Medium |
| Is sovereign HBM2 training possible? | Mmiotrace BIOS POST, diff with D3hot wake | Hard (weeks) |
| Does D3hot→D0 work on MI50? | PCIe PM spec says yes; test when card arrives | Should be easy |

---

## License

AGPL-3.0-only. Part of the ecoPrimals sovereign compute fossil record.
