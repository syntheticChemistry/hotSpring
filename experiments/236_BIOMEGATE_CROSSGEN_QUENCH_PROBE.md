# Experiment 236: biomeGate Cross-Generation Quench Probe

**Date:** 2026-08-02
**Status:** COMPLETE — 6/7 PASS
**Gate:** biomeGate
**Hardware:** K80 GK210 (SM37) + Titan V GV100 (SM70)
**Prerequisite:** Exp 235 (fleet bootstrap)
**Revalidates:** strandGate Exp 231 (never ran on hardware)

## Objective

Prove the diesel engine's `InterruptProfile` abstraction dispatches correctly
across the Kepler↔Volta generation boundary on biomeGate silicon. This is the
most critical safety mechanism — incorrect quench causes lockups or worse.

strandGate Exp 231 designed the methodology but never ran on live K80 hardware.
biomeGate is the first gate to execute it on real silicon spanning both sides
of the boundary.

## The Generation Boundary

NVIDIA changed the interrupt enable architecture at Volta (SM70):

| Register | Kepler (SM35-37) | Volta+ (SM70+) |
|----------|-----------------|----------------|
| 0x140 INTR_EN_0 | **Read/Write** — direct mask | **Read-Only** — state reflection |
| 0x160 INTR_EN_SET_0 | Different HW register | **Write-Only** — set bits |
| 0x180 INTR_EN_CLEAR_0 | Different HW register | **Write-Only** — clear bits |

The diesel engine encodes this as:

```rust
InterruptProfile::for_sm(37)  → PRE_VOLTA   → disable: write 0x0 to 0x140
InterruptProfile::for_sm(70)  → VOLTA_PLUS  → disable: write 0xFFFFFFFF to 0x180
```

## Procedure

All operations via `mmio.read32` / `mmio.write32` JSON-RPC through toadStool
server (ember BAR0 sysfs path). Cold-state testing — no warm cycle required
for the core register semantics proof.

### Test 1: Titan V VOLTA_PLUS (SM70)

1. Read INTR_EN_0@0x140 baseline
2. Write 0x1 to 0x140 — verify NO-OP (R/O on Volta)
3. Write 0x1 to INTR_EN_SET@0x160 — verify INTR_EN_0 reflects
4. Write 0xFFFFFFFF to INTR_EN_CLEAR@0x180 — verify quench

### Test 2: K80 PRE_VOLTA (SM37, fn1 — clean state)

1. Read INTR_EN_0@0x140 baseline
2. Write 0xFFFFFFFF — verify writable bits (cold = engine mask)
3. Write 0x0 — verify quench
4. Enable → quench → verify idempotent

### Test 3: K80 fn0 — Post-FLR damage (control)

1. Verify stuck state (PMC_ENABLE=0xFFFF, writes dropped)

## Results

### Titan V GV100 at 0000:21:00.0 — VOLTA_PLUS: 3/3 PASS

```
[2a] INTR_EN_0@0x140 baseline:              0x00000000
[2b] Write 0x1 to 0x140:                    0x00000000  NO-OP (R/O confirmed)
[2d] Write 0x1 to INTR_EN_SET@0x160:        INTR_EN_0 → 0x00000001  SET WORKS
[2f] Write 0xFFFFFFFF to INTR_EN_CLEAR@0x180: INTR_EN_0 → 0x00000000  QUENCH OK
```

### K80 GK210 fn1 at 0000:4c:00.0 — PRE_VOLTA: 3/3 PASS

```
PMC_ENABLE: 0xC0002020 (cold, popcount=4)

[1a] INTR_EN_0@0x140 baseline:              0x00000000
[1b] Write 0xFFFFFFFF → readback:           0x00000003
     Writable mask: 0b00000000000000000000000000000011
     Writable bits: 2 (matching cold engine count)
[1d] Write 0x00000000 → readback:           0x00000000  QUENCH SUCCESS
[1e] Re-enable 0x3 → re-quench 0x0:         IDEMPOTENT ✓
```

Cold-state K80 only exposes interrupt bits for enabled engines. With
PMC_ENABLE popcount=4, only bits 0-1 are writable. Full 32-bit mask
requires warm GPU with all engines enabled. The PRE_VOLTA quench
(write 0x0 to 0x140) clears all writable bits — the abstraction is correct.

### K80 GK210 fn0 at 0000:4b:00.0 — POST-FLR DAMAGE: 0/1 FAIL

```
PMC_ENABLE: 0xFFFFFFFF (pop=32, FLR bulk-enabled)
INTR_EN_0:  0xFFFFFFFF (all interrupts, latched)
Write 0x0 → readback: 0xFFFFFFFF  STUCK
```

FLR during rebind set PMC_ENABLE=0xFFFFFFFF (Exp 199 class). Register
interface is locked in an inconsistent state — GPU needs SBR or power
cycle. This proves `PowerSafetyProfile::PRE_FIRMWARE` is essential:
never bulk-enable Kepler engines.

## Analysis

### What This Proves

1. **INTR_EN_0@0x140 is R/W on Kepler** — direct-write quench works
2. **INTR_EN_0@0x140 is R/O on Volta** — writes are hardware NO-OP
3. **INTR_EN_SET@0x160 and CLEAR@0x180 work on Volta** — SET/CLEAR semantics
4. **The generation boundary is in the silicon** — not a driver convention
5. **InterruptProfile encodes it correctly** — PRE_VOLTA and VOLTA_PLUS dispatch
6. **PowerSafetyProfile prevents real damage** — fn0 FLR proves the need

### What This Means for Silicon Deism

The diesel engine's central abstraction is correct. `InterruptProfile::for_sm(sm)`
produces the right quench parameters for both sides of the Kepler↔Volta
boundary. The generation difference is **data** in a const struct, not
scattered `if sm >= 70` branches. This is what silicon deism looks like:
different silicon, same code, correct behavior.

### Remaining Gap: Warm-State Full Mask

Cold-state K80 only exposed 2 writable interrupt bits. With a warm GPU
(all engines enabled via PMC_ENABLE), the full 32-bit mask should be
writable. Requires nouveau warm cycle (currently blacklisted) or
alternative warm path. Not a blocker for the correctness proof —
the R/W semantics are confirmed.

## Success Criteria

- [x] Titan V INTR_EN_0@0x140 is R/O
- [x] Titan V INTR_EN_SET@0x160 enables bits in INTR_EN_0
- [x] Titan V INTR_EN_CLEAR@0x180 quench (VOLTA_PLUS path)
- [x] K80 fn1 INTR_EN_0@0x140 is R/W (PRE_VOLTA path)
- [x] K80 fn1 quench via direct write to 0x140
- [x] K80 fn1 quench is idempotent (enable → quench → enable → quench)
- [ ] K80 fn0 quench — FAIL (FLR damage, not abstraction failure)
