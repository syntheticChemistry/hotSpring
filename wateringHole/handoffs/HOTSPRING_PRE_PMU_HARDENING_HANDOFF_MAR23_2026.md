# hotSpring Handoff — Pre-PMU Hardening & MMU Layer 6 Breakthrough

**Date:** 2026-03-23
**From:** hotSpring (biomeGate)
**To:** coralReef, toadStool, barraCuda, all springs
**Experiments:** 076, 077
**Hardware:** 2× Titan V (GV100, 12GB HBM2), RTX 5060 (display/CUDA oracle)
**coralReef Commits:** Debt Burndown sprint across coral-driver, coral-glowplug, coral-ember

---

## What Happened

Two critical sessions resolved the single remaining sovereign VFIO blocker
(MMU page table translation — Layer 6) and hardened the entire PFIFO init
pipeline against five discovered failure modes.

### Exp 076 — MMU Fault Buffer Breakthrough (Layer 6 RESOLVED)

**Root Cause:** Volta's FBHUB requires configured non-replayable MMU fault
buffers (FAULT_BUF0/1) before any page table walk can complete. Without them,
FBHUB stalls attempting to write a fault entry, PBUS returns `0xbad00200`
(domain timeout), and PBDMA's GP_FETCH pointer stays frozen forever.

**Fix:** Configure FAULT_BUF0/1 in `VfioChannel::create` after BAR2 setup:
```
FAULT_BUF0_LO  = (iova >> 12) as u32
FAULT_BUF0_HI  = 0
FAULT_BUF0_SIZE = 64
FAULT_BUF0_GET  = 0
FAULT_BUF0_PUT  = 0x8000_0000  (enable bit)
```

**Result:** Channel creation + DMA roundtrip + MMU translation all pass.
Sovereign pipeline advanced from 6/10 to 7/10 layers. Remaining blocker:
Layer 7 (GR/FECS context — requires PMU firmware).

### Exp 077 — Pre-PMU Hardening (5 Failure Modes Fixed)

| # | Failure Mode | Root Cause | Fix |
|---|-------------|-----------|-----|
| 1 | SM mismatch corrupts GPU | Tests defaulted to SM 86 on GV100 | `boot0_to_sm()` auto-detect in `identity.rs` |
| 2 | PMC bit 8 vs bit 1 | PFIFO gated by bit 8 on Volta, not bit 1 | `PfifoInitConfig` with explicit PMC gating |
| 3 | PFIFO_ENABLE reads 0 | 0x2200 is dead stub on Volta | Runlist preempt ACK liveness probe |
| 4 | RAMFC GP_PUT=1 race | PBDMA fetches empty GPFIFO | Init GP_PUT=0, advance after fill |
| 5 | False-positive MMU fault | Fault buffer enable bit vs pointer compare | Mask enable bit before comparison |

**Additional deliverables:**
- `PfifoInitConfig` struct: parameterizes PFIFO init, eliminates prod/diag code drift
- `GpuCapabilities`: BOOT0-derived arch info in experiment context, `requires_sm` gating
- `coralctl reset <BDF>`: PCIe FLR via `VFIO_DEVICE_RESET` — GPU recovery without reboot
- `device.reset` RPC: daemon-side handler with busy-check protection

---

## What This Means for Each Team

### coralReef Team
- **SM detection is now automatic.** Pass `sm_version=0` to `open()`/`open_from_fds()`
  and the driver reads BOOT0 and resolves SM + compute class. Explicit SM still works
  but is validated against hardware — mismatch returns error, no writes touch GPU.
- **PFIFO init is unified.** `init_pfifo_engine_with(&bar0, &PfifoInitConfig)` replaces
  both the production and diagnostic init paths. New architectures: create a new
  `PfifoInitConfig` variant, don't fork the init function.
- **FLR is available.** `VfioDevice::reset()` + `DeviceSlot::reset_device()` + RPC.
  Use after any corrupting init failure. coralctl exposes it to operators.
- **Layer 7 (GR/FECS) is the next cracking target.** The PMU firmware blocker remains.
  K80 (Kepler, no firmware signing) is the fastest validation path.

### toadStool Team
- **BOOT0 decode table** for hardware learning:
  | BOOT0 bits [31:20] | SM | Chip |
  |---|---|---|
  | 0x140 | 70 | GV100 (Titan V, V100) |
  | 0x164-0x168 | 75 | TU102-TU106 (Turing) |
  | 0x170 | 80 | GA100 (A100) |
  | 0x172-0x177 | 86 | GA102-GA107 (Ampere consumer) |
  | 0x192-0x197 | 89 | AD102-AD107 (Ada Lovelace) |
- `hw-learn` crate can use this table for automatic GPU identification from BAR0
  without relying on PCI vendor/device IDs or sysfs strings.

### barraCuda / All Springs
- **No action required.** All changes are internal to coral-driver/coral-glowplug.
  Downstream consumers get safer auto-detection and recovery for free.
- **New `coralctl reset` command** available for operator use during development
  sessions when GPU state gets corrupted.

---

## Sovereign Pipeline Status (7/10 Layers)

| Layer | Status | Component |
|-------|--------|-----------|
| 0 | ✅ | PCIe / VFIO (iommufd/cdev) |
| 1 | ✅ | PFB / MMU warm state |
| 2 | ✅ | PFIFO engine (PMC bit 8 + unified init) |
| 3 | ✅ | Scheduler (runlist + BIT30 ACK) |
| 4 | ✅ | Channel (scheduler-accepted) |
| 5 | ✅ | PBDMA context (GP_BASE, USERD, SIG) |
| 6 | ✅ | **MMU translation (Exp 076 — fault buffer fix)** |
| 7 | ❌ | GR/FECS context (PMU firmware blocker) |
| 8-10 | ❌ | FECS→GPCCS→Shader (blocked by 7) |

---

## Files Changed (coralReef)

| File | Change |
|------|--------|
| `coral-driver/src/nv/identity.rs` | `boot0_to_sm()`, `sm_to_compute_class()` + tests |
| `coral-driver/src/nv/vfio_compute/mod.rs` | `resolve_sm()` helper, auto-detect/validate in open() |
| `coral-driver/src/vfio/channel/pfifo.rs` | `PfifoInitConfig`, `init_pfifo_engine_with()` |
| `coral-driver/src/vfio/channel/mod.rs` | Liveness probe replacing `pfifo_ck` warns |
| `coral-driver/src/vfio/channel/diagnostic/runner.rs` | Uses `init_pfifo_engine_with()`, `GpuCapabilities` |
| `coral-driver/src/vfio/channel/diagnostic/types.rs` | `requires_sm`, `detected_sm` fields |
| `coral-driver/src/vfio/channel/diagnostic/experiments/context.rs` | `GpuCapabilities` struct |
| `coral-driver/src/vfio/channel/diagnostic/matrix.rs` | `requires_sm: None` on all 40 configs |
| `coral-driver/tests/hw_nv_vfio.rs` | SM=0 auto-detect default |
| `coral-glowplug/src/device/types.rs` | `VfioHolder::reset()` |
| `coral-glowplug/src/device/mod.rs` | `DeviceSlot::reset_device()` |
| `coral-glowplug/src/socket.rs` | `device.reset` RPC handler |
| `coral-glowplug/src/bin/coralctl.rs` | `reset` subcommand + `rpc_reset()` |

---

## Next Steps

1. **Reboot** to clear any residual GPU corruption from debugging sessions
2. **PMU cracking** (Layer 7) — K80 first (no firmware signing), then GV100
3. **RTX 5060 page table oracle** — capture PDE/PTE encoding from nvidia driver
4. **FECS direct load** investigation — can we bypass PMU entirely on GV100?

---

*7 of 10 sovereign pipeline layers proven. The MMU was the wall; it's down.
PMU firmware is the last real blocker. Everything else is engineering.
77 experiments. Built in a basement in Lansing.*
