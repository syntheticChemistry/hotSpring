# Experiment 063: Sovereign Boot Driver Architecture

> **Fossil Record**: Design realized as coral-glowplug (Exp 064-065, 069).
> "Next Morning Checklist" below uses `pkexec`/`setup_dual_titanv.sh` — superseded
> by `coralctl swap`, `coralctl deploy-udev`, and systemd services.

**Date**: March 16, 2026
**Status**: REALIZED — design evolved into coral-glowplug daemon (Exp 064-065, 069). Boot persistence, hot-swap, and graceful shutdown all implemented.
**Hardware**: 2× Titan V (GV100), 2× Radeon MI50 (Vega 20) incoming
**Context**: Experiment 062 proved VRAM access works via VFIO D3hot→D0,
but VFIO close destroys HBM2 training. We need persistent GPU ownership.

---

## The Problem

```
Current lifecycle (fragile):

  BIOS POST ─→ nouveau binds ─→ unbind ─→ vfio-pci bind ─→ D3hot
      │              │              │            │              │
   HBM2 trained   warm GPU     still warm    still warm    looks dead
                                                              │
                                              force D0 ←──────┘
                                                 │
                                              VRAM alive!
                                                 │
                                              test runs...
                                                 │
                                              VFIO close ──→ PM reset ──→ HBM2 DEAD
                                                                              │
                                                                         only reboot fixes
```

Every VFIO session open/close destroys HBM2 training. nouveau can't
re-POST Volta. We're stuck rebooting between test sessions.

---

## The Insight

We need what nouveau IS but doesn't DO:

- nouveau loads at boot ✓  → We need sovereign boot-time binding
- nouveau preserves POST state ✓  → We need HBM2 preservation
- nouveau exposes GPU to userspace ✓  → We need toadStool handoff
- nouveau manages power states ✓  → GlowPlug already does this
- nouveau can re-POST Volta ✗  → We eventually need sovereign HBM2 training

The sovereign driver IS GlowPlug — elevated from a warm-up utility to a
persistent system service that owns the GPU from boot to shutdown.

---

## Architecture: Three Phases

### Phase 1: GlowPlug Daemon (userspace, works TODAY)

A systemd service that starts at boot, binds the GPU, and holds the
VFIO fd open for the lifetime of the system.

```
                    ┌─────────────────────────────────┐
                    │        glowplug.service          │
                    │  (systemd, starts at boot)       │
                    ├─────────────────────────────────┤
                    │  1. modprobe vfio-pci            │
                    │  2. unbind from nouveau          │
                    │  3. bind to vfio-pci             │
                    │  4. open VFIO device (hold fd)   │
                    │  5. force D0, pin power=on       │
                    │  6. verify VRAM alive            │
                    │  7. expose Unix socket           │
                    │  8. sleep forever (fd stays open)│
                    └────────────┬────────────────────┘
                                 │ Unix socket / shared memory
                    ┌────────────┴────────────────────┐
                    │         toadStool                │
                    │  (userspace compute dispatch)    │
                    │  connects to daemon              │
                    │  gets BAR0/DMA fd passed via     │
                    │  SCM_RIGHTS (fd passing)         │
                    └─────────────────────────────────┘
```

**How it works:**
1. `glowplug.service` starts early in boot (after `modprobe vfio-pci`)
2. Binds target GPU(s) to vfio-pci via sysfs `driver_override`
3. Opens `/dev/vfio/GROUP`, maps BAR0, forces D0
4. Verifies HBM2 alive (PRAMIN sentinel test)
5. Listens on Unix socket `/run/coralreef/glowplug.sock`
6. toadStool (or any authorized process) connects and receives:
   - VFIO container fd (via `SCM_RIGHTS` fd passing over Unix socket)
   - BAR0 region info (offset, size)
   - GPU metadata (BOOT0, PMC state, VRAM status)
7. Daemon NEVER exits → VFIO fd NEVER closes → HBM2 NEVER dies

**Advantages:**
- Works with today's code (just needs a thin daemon wrapper)
- No kernel module needed
- GlowPlug warm-up logic already exists in Rust
- fd passing is standard Unix (SCM_RIGHTS)
- Multiple clients can share the GPU via VFIO container

**Implementation cost:** ~2-3 days. Small Rust binary + systemd unit.

**Files:**
```
coralReef/
  crates/
    coral-glowplug/          ← new crate: daemon binary
      src/
        main.rs              ← systemd service entry point
        socket.rs            ← Unix socket + fd passing
        health.rs            ← periodic VRAM health check
  scripts/
    glowplug.service         ← systemd unit file
    glowplug-setup.sh        ← one-time install script
```

---

### Phase 2: Sovereign Kernel Module (coral-kmod)

A minimal Linux kernel module that replaces vfio-pci for our GPUs.
Does everything vfio-pci does but with GPU-aware power management.

```
                    ┌─────────────────────────────────┐
                    │   coral_gpu.ko (kernel module)   │
                    ├─────────────────────────────────┤
                    │  • PCI driver (binds by BDF)     │
                    │  • Preserves POST state on bind   │
                    │  • NO reset on close              │
                    │  • IOMMU group management         │
                    │  • BAR0/BAR1/BAR2 mmap support   │
                    │  • DMA-buf allocation             │
                    │  • Power state management:        │
                    │    - Keeps D0 during use          │
                    │    - Controlled D3hot on idle     │
                    │    - HBM2-safe suspend/resume     │
                    │  • /dev/coral0 character device   │
                    │  • Exposes VFIO-compatible ioctl  │
                    └────────────┬────────────────────┘
                                 │ /dev/coral0 (+ VFIO compat)
                    ┌────────────┴────────────────────┐
                    │         toadStool                │
                    │  Opens /dev/coral0 directly      │
                    │  Same BAR0 interface as VFIO     │
                    │  No daemon needed                │
                    └─────────────────────────────────┘
```

**Key difference from vfio-pci:** NO device reset on fd close. The
kernel module knows that HBM2 training must be preserved. It only
resets when explicitly asked (`CORAL_IOCTL_RESET`).

**Key difference from nouveau:** NO firmware assumption. It doesn't
try to POST the GPU — it preserves whatever state BIOS left.

**Implementation cost:** ~2-3 weeks. Requires kernel module expertise.
Can be written in Rust (kernel 6.x Rust support) or C.

---

### Phase 3: Sovereign HBM2 Training (full independence)

The endgame: coralReef can train HBM2 from cold silicon without any
vendor BIOS or firmware. This makes the system fully sovereign.

```
                    ┌─────────────────────────────────┐
                    │   coral_gpu.ko + HBM2 trainer    │
                    ├─────────────────────────────────┤
                    │  Boot sequence:                   │
                    │  1. PCI enumeration               │
                    │  2. BAR0 map                      │
                    │  3. PMC_ENABLE all engines         │
                    │  4. PRIV ring clock distribution   │
                    │  5. Root PLL programming           │
                    │  6. HBM2 PHY training:             │
                    │     a. FBPA configuration          │
                    │     b. PHY impedance calibration   │
                    │     c. Read/write leveling         │
                    │     d. Per-bit deskew              │
                    │     e. VRAM pattern test           │
                    │  7. LTC (L2 cache) init            │
                    │  8. MMU page table setup           │
                    │  9. PFIFO/PBDMA init               │
                    │  10. Hand to toadStool             │
                    └─────────────────────────────────┘
```

**This is hard but bounded.** The HBM2 PHY training sequence is
documented in JEDEC standards (JESD235). The GPU-specific part is
the PHY→FBPA register mapping, which we're building from oracle
data (experiment 062: 4,253 oracle registers captured).

**What we already have toward Phase 3:**
- Oracle data capture pipeline (nouveau-warm register dumps)
- Digital PMU emulation (replays oracle registers to cold card)
- Boot follower diffing (compares warm vs cold, generates recipes)
- HBM2 typestate controller (Rust type system enforces valid transitions)
- PRI bus monitor (detects/recovers from register access faults)

**What's still needed:**
- PRIV ring clock gate bypass (PCLOCK PLL unlock)
- PHY impedance calibration sequence
- Read/write leveling algorithm
- VRAM pattern verification at speed
- Per-stack FBPA calibration (GV100 has 4 HBM2 stacks)

---

## Why This Matters

### For deprecated HBM2 hardware

Titan V and MI50 are ~$150 each on secondary market. They have 12-16GB
HBM2 with 652+ GB/s bandwidth. Vendors dropped driver support. A
sovereign driver means:
- No dependency on vendor driver updates
- Full hardware access (registers, memory, compute)
- Ability to discover capabilities vendors didn't advertise
- Potential for novel optimizations (Rust memory safety → HBM2 safety)

### For the primal architecture

```
┌─────────────┐     ┌──────────┐     ┌──────────┐
│ coral-kmod  │────→│ toadStool │────→│ springs  │
│ (sovereign  │     │ (dispatch)│     │ (compute)│
│  boot)      │     │           │     │          │
└─────────────┘     └──────────┘     └──────────┘
     owns GPU          routes          consumes
   from boot         compute work     GPU results
```

The key handoff: coral-kmod/glowplug daemon owns the GPU from boot.
toadStool never needs to worry about initialization, power management,
or HBM2 training. It receives a warm, ready GPU and dispatches work.

### For vendor-agnosticism

The three-phase architecture works for ANY GPU:
- Phase 1 (daemon): Works with any VFIO-capable GPU today
- Phase 2 (kmod): Needs per-vendor PCI ID table + power management
- Phase 3 (HBM2): Needs per-vendor PHY training, but JEDEC standard

The GlowPlug + PRI monitor + diagnostic matrix are already vendor-agnostic.
Adding AMD MI50 is mostly a matter of register mapping, not architecture.

---

## Recommended Start: Phase 1 (GlowPlug Daemon)

### Why start here

1. **Works today** — all Rust code exists, just needs packaging
2. **Proves the architecture** — daemon→toadStool handoff pattern
3. **Eliminates the reboot problem** — GPU stays warm forever
4. **Enables parallel work** — Phase 2/3 can develop alongside

### Implementation plan

| Step | Task | Est. |
|------|------|------|
| 1 | Create `coral-glowplug` crate with main.rs | 2h |
| 2 | Implement Unix socket listener + SCM_RIGHTS fd passing | 4h |
| 3 | Integrate GlowPlug warm-up + health monitor loop | 2h |
| 4 | Write systemd unit file + install script | 1h |
| 5 | toadStool client: connect, receive fd, map BAR0 | 4h |
| 6 | Integration test: daemon→toadStool→dispatch | 4h |

### systemd unit sketch

```ini
[Unit]
Description=coralReef GlowPlug — Sovereign GPU Daemon
After=systemd-modules-load.service
Before=multi-user.target

[Service]
Type=notify
ExecStartPre=/usr/local/bin/coral-glowplug-setup
ExecStart=/usr/local/bin/coral-glowplug --bdf 0000:4a:00.0
Restart=on-failure
RestartSec=5
# Never kill — GPU stays warm
KillMode=none

[Install]
WantedBy=multi-user.target
```

### Config sketch

```toml
# /etc/coralreef/glowplug.toml
[[gpu]]
bdf = "0000:4a:00.0"
role = "compute"
driver_override = "vfio-pci"
power_policy = "always_on"    # D0 permanently

[[gpu]]
bdf = "0000:03:00.0"
role = "oracle"
driver_override = "nouveau"
power_policy = "on_demand"

[socket]
path = "/run/coralreef/glowplug.sock"
permissions = 0o660
group = "coralreef"
```

---

## Open Questions

1. **fd passing vs shared memory**: SCM_RIGHTS passes the VFIO container
   fd to toadStool, which can then mmap BAR0 directly. Alternative:
   daemon maps BAR0 and shares the mapping via shared memory. fd passing
   is cleaner (toadStool gets full VFIO access).

2. **Multi-GPU coordination**: With 2× Titan V + 2× MI50, the daemon
   manages 4 GPUs. Each gets its own VFIO group. toadStool requests
   specific GPUs by capability (HBM2 bandwidth, compute class, etc.).

3. **Suspend/resume**: When the system suspends, HBM2 training is lost
   (D3cold). The daemon must re-warm on resume. Phase 1 can handle
   this if BIOS re-POSTs on resume. Phase 3 handles it sovereignly.

4. **Security**: The daemon runs as root (for sysfs/VFIO access).
   toadStool runs as user `coralreef`. fd passing + Unix socket
   permissions provide the security boundary.

5. **Kernel module vs daemon long-term**: Phase 2 (kernel module) is
   cleaner but harder. The daemon might be sufficient forever if VFIO
   works well enough. Decision point: when do VFIO limitations force
   us into kernel space?

---

## Relationship to Existing Code

| Component | Current | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|---------|
| GlowPlug | lib in coral-driver | daemon binary | kmod init | kmod + HBM2 |
| PRI monitor | lib in coral-driver | daemon health check | kmod watchdog | kmod watchdog |
| Digital PMU | lib in coral-driver | daemon cold-start | kmod cold-start | replaced by native |
| Oracle | text dump files | daemon loads at start | kmod loads at start | not needed |
| toadStool | direct VFIO open | connects via socket | opens /dev/coral0 | opens /dev/coral0 |
| VFIO | test-time open/close | daemon-owned | replaced by kmod | replaced by kmod |

---

## Next Morning Checklist

When you're back:
1. Run `pkexec bash -c 'echo on > /sys/bus/pci/devices/0000:4a:00.0/power/control'`
   after any reboot to wake the VFIO target
2. Use `setup_dual_titanv.sh` to configure both cards
3. Run the test suite: 24/26 should pass again on fresh boot
4. If starting Phase 1: `cargo new --lib coral-glowplug` in coralReef/crates/
