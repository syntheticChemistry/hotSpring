# Firmware Learning Matrix

**Updated:** 2026-04-12
**Purpose:** Authoritative reference for GPU firmware interfaces per generation. Not a lock to pick — a system to learn.

## Core Principle

Every NVIDIA GPU since Maxwell has a firmware management interface. The host CPU does not directly control the GPU's compute engines — it communicates with falcon microcontrollers (FECS, GPCCS, PMU, SEC2) that manage execution context, scheduling, and security. Understanding these interfaces is the path to sovereign compute.

## Generation Matrix

### Firmware Subsystems

| Subsystem | Role | How Host Interacts |
|-----------|------|-------------------|
| **FWSEC** | GPU BIOS — runs at POST, sets up WPR2, locks memory protections | Not directly — runs before any OS driver. Hardware boundary. |
| **ACR** | Authenticated Code Loader — loads signed firmware into WPR2 | SEC2 falcon runs ACR code; host provides firmware blobs via DMA |
| **SEC2** | Security Engine 2 — falcon that executes ACR bootloader | Host uploads code via PIO (IMEMC/IMEMD), starts via CPUCTL |
| **FECS** | Front-End Command Scheduler — manages GR compute contexts | Host sends methods via MTHD_DATA/MTHD_CMD (BAR0 0x409500/504) |
| **GPCCS** | GPC Command Scheduler — per-GPC execution control | Follows FECS; booted by ACR after FECS. Same falcon register layout |
| **PMU** | Power Management Unit — clock gating, thermal, power states | Mailbox protocol; required for nouveau DRM channel alloc on Volta |
| **GSP** | GPU System Processor (Turing+) — entire driver runs on GPU | RM ioctls over PCIe; host driver is thin RPC client |

### Security Model by Generation

| Generation | Arch | FECS Loading | Security Model | Reagent | WPR | Our Status |
|------------|------|-------------|---------------|---------|-----|------------|
| **Kepler** | GK110/GK210 | PIO direct from host | None | None needed | No WPR | IMEM upload validated, clocks blocking |
| **Maxwell** | GM200+ | ACR v1 via SEC2 | Signed firmware | nouveau | WPR1 | Not tested (no hardware) |
| **Pascal** | GP100+ | ACR v2 via SEC2 | Signed + WPR2 | nouveau | WPR2 | Not tested (no hardware) |
| **Volta** | GV100 | ACR v2 + FWSEC | WPR2 fuse-locked | nouveau (warm) | WPR2 locked | **DRM dispatch PROVEN (Exp 164, 5/5)**. SovereignInit stages 0-5 via ember. Falcon boot blocked (FBP=0). |
| **Turing** | TU10x | GSP (optional) | GSP or ACR | nvidia proprietary | WPR2/GSP | Not tested |
| **Ampere** | GA10x | GSP mandatory | GSP-locked | nvidia proprietary | GSP manages | VM captures analyzed (RTX 3090) |
| **Ada** | AD10x | GSP mandatory | GSP-locked | nvidia proprietary | GSP manages | RTX 5060 display + UVM code-complete |
| **Blackwell** | GB20x | GSP mandatory | GSP-locked | nvidia proprietary | GSP manages | RTX 5060 on-site |

### FECS Method Interface

The FECS method interface is the primary host->firmware communication channel for GR compute. Methods are sent via BAR0 register writes:

```
Write MTHD_DATA (base + 0x500) = parameter
Write MTHD_CMD  (base + 0x504) = method_id
Poll  MTHD_STATUS (base + 0x800) until completion
```

| Method ID | Name | Purpose | Kepler | Volta | Status |
|-----------|------|---------|--------|-------|--------|
| 0x10 | CTX_IMAGE_SIZE | Query context image size | Yes | Yes | Implemented |
| 0x16 | ZCULL_INFO | Query zcull geometry | Yes | Yes | Implemented |
| 0x25 | PM_MODE | Performance monitor config | Unknown | Yes | Implemented |
| 0x21 | WATCHDOG | Set timeout | Unknown | Yes | Implemented |
| 0x03 | BIND_POINTER | Bind context pointer | Yes | Yes | Implemented |
| 0x09 | WFI_GOLDEN_SAVE | Wait-for-idle + save golden context | Yes | Yes | Implemented |
| 0x01 | STOP_CTXSW | Stop context switching | Unknown | Yes | Not implemented |
| 0x02 | START_CTXSW | Start context switching | Unknown | Yes | Not implemented |

### Falcon Mailbox Protocol

Host<->falcon communication via MAILBOX0/MAILBOX1 registers:

| Step | Host Action | Falcon Response |
|------|------------|----------------|
| 1 | Write request to MAILBOX0 | — |
| 2 | Write STARTCPU or method | — |
| 3 | Poll MAILBOX0/SCRATCH0 | Firmware writes result |
| 4 | Read result | — |

MAILBOX0 is at `base + 0x040` on all generations (falcon standard register).
On Kepler, SCRATCH0 at `FECS_BASE + 0x500` overlaps MTHD_DATA — some code uses
SCRATCH0 for boot handshake while MTHD_DATA/CMD is the method interface.

### GPFIFO Channel Requirements by Generation

| Generation | Channel Class | RAMFC Layout | Doorbell | USERD |
|------------|--------------|-------------|----------|-------|
| Kepler | KEPLER_CHANNEL_GPFIFO_B (0xA16F) | 512B, GP_BASE at +0x00 | N/A (GP_PUT write) | USERD segment |
| Volta | VOLTA_CHANNEL_GPFIFO_A (0xC36F) | 512B, GP_BASE at +0x00 | VOLTA_USERMODE_A (0xC361) | Mapped doorbell page |
| Ampere+ | AMPERE_CHANNEL_GPFIFO_A (0xC56F) | 512B | Doorbell via UVM | UVM-managed |

### Boot Path Decision Tree

SCTL bits `[13:12]` encode the falcon security mode: 0=NS, 1=LS, 2=HS, 3=HS+ (locked).

```
GPU arrives (read FECS SCTL at BAR0 + 0x409240)
  │
  ├─ SCTL[13:12] = 0b00 (NS) ─────→ PIO direct: upload FECS via IMEMC/IMEMD
  │   e.g. Kepler GK210                Requires: clock/PLL init from DEVINIT
  │
  ├─ SCTL[13:12] = 0b10 (HS) ─────→ FECS in HS mode after ACR boot
  │   e.g. Volta post-nouveau          Method interface available at this point
  │                                     This is the target state after warm handoff
  │
  ├─ SCTL[13:12] = 0b01 (LS) ─────→ ACR chain needed: nouveau warm handoff
  │   e.g. Maxwell/Pascal               Requires: nouveau reagent, ember/glowplug
  │
  ├─ SCTL[13:12] = 0b11 (HS+) ────→ Fully locked — warm handoff + livepatch
  │   e.g. Volta cold (FWSEC locked)   Requires: livepatch, ember/glowplug
  │
  ├─ GSP present (Turing+) ───────→ nvidia proprietary reagent
  │                                   RM initializes everything via GSP
  │
  └─ Unknown ──────────────────────→ Run `coralctl onboard`, add to matrix
```

## Register Offset Cross-Reference

All offsets verified against `coral-driver/src/vfio/channel/registers/falcon.rs`:

| Register | Offset (from base) | Source |
|----------|-------------------|--------|
| CPUCTL | +0x100 | `falcon::CPUCTL` |
| SCTL | +0x240 | `falcon::SCTL` |
| PC | +0x030 | `falcon::PC` |
| EXCI | +0x148 | `falcon::EXCI` |
| BOOTVEC | +0x104 | `falcon::BOOTVEC` |
| HWCFG | +0x108 | `falcon::HWCFG` |
| MAILBOX0 | +0x040 | `falcon::MAILBOX0` |
| MAILBOX1 | +0x044 | `falcon::MAILBOX1` |
| IMEMC | +0x180 | `falcon::IMEMC` |
| IMEMD | +0x184 | `falcon::IMEMD` |
| DMEMC | +0x1C0 | `falcon::DMEMC` |
| DMEMD | +0x1C4 | `falcon::DMEMD` |
| MTHD_DATA | +0x500 | `falcon::MTHD_DATA` |
| MTHD_CMD | +0x504 | `falcon::MTHD_CMD` |
| MTHD_STATUS | +0x800 | `falcon::MTHD_STATUS` |
| FBIF_TRANSCFG | +0x624 | `falcon::FBIF_TRANSCFG` |
| EXCEPTION_REG | +0xC24 | `falcon::EXCEPTION_REG` |

Falcon bases: FECS=0x409000, GPCCS=0x41A000, PMU=0x10A000, SEC2=0x087000.

## Firmware Probe Integration

coralReef's `firmware_probe.rs` captures a `FirmwareSnapshot` at any point:

```rust
use coral_driver::vfio::channel::diagnostic::firmware_probe;

let snap = firmware_probe::capture_firmware_snapshot(&bar0, "post-warm-fecs");
firmware_probe::log_firmware_summary(&snap);

// Save to JSON for analysis
let json = serde_json::to_string_pretty(&snap).unwrap();
std::fs::write("snapshot.json", json).unwrap();

// Diff two snapshots
let diffs = firmware_probe::diff_snapshots(&before, &after);
for (path, old, new) in &diffs {
    println!("{path}: {old} -> {new}");
}
```

### What Snapshots Teach Us

| Snapshot Pair | What We Learn |
|--------------|---------------|
| cold -> post-nouveau | What nouveau initializes (ACR chain, FECS boot sequence) |
| post-nouveau -> post-vfio (no livepatch) | What nouveau teardown destroys |
| post-nouveau -> post-vfio (with livepatch) | What the livepatch preserves |
| post-vfio -> post-dispatch | What dispatch changes (channel state, FECS methods) |
| nvidia cold -> nvidia warm | What RM initializes (proprietary FECS boot) |

Each new GPU goes through the same snapshot sequence, building the matrix automatically.

## Hardware Currently Profiled

| GPU | Generation | BDF | Firmware Probe | Boot Path | Dispatch |
|-----|-----------|-----|---------------|-----------|----------|
| Titan V #1 | Volta (GV100) | 0000:03:00.0 | Full (Exp 058-125) | Warm handoff (livepatch) | FRONTIER |
| Tesla K80 die1 | Kepler (GK210) | 0000:4c:00.0 | Partial (Exp 123) | PIO direct (needs DEVINIT) | Blocked (clocks) |
| Tesla K80 die2 | Kepler (GK210) | 0000:4d:00.0 | Identity only | PIO direct | Not started |
| RTX 5060 | Blackwell (GB206) | 0000:21:00.0 | Identity only | nvidia+UVM | Code-complete |
| RTX 3090 (decom.) | Ampere (GA102) | — | VM captures | nvidia+UVM | Code-complete |
| RX 6950 XT (decom.) | RDNA2 | — | Full | AMD DRM | PROVEN (6/6) |

## Hardware Onboarding Protocol

New GPUs are onboarded via `coralctl onboard <BDF>`, which produces a structured JSON report:

```bash
coralctl onboard 0000:03:00.0 --output titan_v_onboard.json
```

The report includes:
- **Identity**: BOOT0, architecture, chip ID, PMC_ENABLE, VRAM status
- **Firmware Census**: FECS/GPCCS/PMU state (CPUCTL, SCTL, halted/hreset, reachable)
- **Boot Path Recommendation**: based on architecture + SCTL security level
- **Dispatch Readiness**: PFIFO alive, FECS running, GR enabled, concrete blockers

This replaces ad-hoc manual probing and ensures every GPU goes through the same diagnostic sequence.

## Gaps to Fill

1. **FECS method enumeration** — we implement 6 methods; the firmware supports more. Systematic probing (try method IDs 0x00-0xFF, record which succeed) would map the full interface.
2. **GPCCS methods** — GPCCS has the same method interface but we don't use it directly. Document which methods GPCCS accepts.
3. **PMU mailbox protocol** — PMU communication is required for nouveau DRM channels. Tracing nouveau's PMU init would document this protocol.
4. **GSP RM protocol** — on Turing+, the entire driver<->GPU interface is RM ioctls to GSP. nvidia-open partially documents this.
5. **WPR2 boundary registers** — which registers define WPR2 boundaries and are they readable? Documenting this constrains what's possible for cold boot.
6. **Clock domain map per generation** — critical for K80 cold boot. Which PLL registers must be configured before falcon CPUs can run?
