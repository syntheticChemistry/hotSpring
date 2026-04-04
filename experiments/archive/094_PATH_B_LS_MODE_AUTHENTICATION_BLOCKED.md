# Exp 094: Path B Dead — LS Mode Authentication Blocks Direct PIO on GV100

**Date:** 2026-03-25
**Status:** CONFIRMED — Path B (direct PIO + BOOTVEC) is hardware-blocked
**Depends:** Exp 093 (W1 header fix, BOOTVEC wiring), Exp 091 (BOOTVEC discovery)
**Goal:** Determine whether the fixed PIO boot path (correct IMEM layout + BOOTVEC) can start FECS/GPCCS on GV100

## Summary

Exp 093 fixed three bugs in the direct PIO boot path: stripped BL headers, corrected IMEM layout (inst at 0, BL at bl_imem_off), and wired BOOTVEC from firmware metadata. Hardware validation showed the fixes are mechanically correct — IMEM readback confirms firmware is present at the right addresses. But FECS and GPCCS still fault immediately at PC=0x0000 with exception 0x02070000.

The root cause is **LS mode authentication**. GV100 FECS/GPCCS are fuse-enforced LS mode (SCTL=0x3000). In this mode, the falcon's secure boot hardware validates that code was loaded through the ACR DMA path — not host PIO. PIO-uploaded code is present and readable in IMEM but is rejected at execution time.

This definitively closes Path B for GV100 FECS/GPCCS. The only route to sovereign compute on Volta is Path A: SEC2 ACR boot via DMA.

## Evidence

### Exception Analysis

All three hardware runs (Exp 093) produced the same result:

| Falcon | cpuctl | exci | PC | Interpretation |
|--------|--------|------|----|----------------|
| GPCCS | 0x10 | 0x02070000 | 0x0000 | Auth failure at entry point |
| FECS | 0x10 | 0x02070000 | 0x0ae1 | Auth failure → ROM exception handler |

Exception code `0x02` = instruction authentication failure. The falcon's secure boot hardware checks a DMA-load authentication tag before allowing execution. PIO writes bypass this tag mechanism.

### IMEM Verification

IMEM readback confirms the firmware is present and correctly laid out:

```
GPCCS IMEM[0x0000] = 0x001400d0  (inst code — correct)
GPCCS IMEM[0x3400] = 0x000400d0  (BL code — correct offset)
```

The fix is mechanically sound. The failure is a hardware security enforcement, not a software bug.

### LS Mode is Fuse-Enforced

SCTL=0x3000 on both Titan V cards. This is set by fuses during manufacturing — not by software, not by the ACR, and not clearable by any reset mechanism available to the host. The SCTL myth (busted in Exp 091) confirmed PIO *writes* work regardless of SCTL, but LS mode still enforces *execution* authentication.

### Boot Solver Confirmation

The full 12-strategy boot solver (Exp 093 Run 3) tested every combination:
- Direct PIO with/without secure flag: auth failure
- Direct PIO with PMC GR reset: CPUCTL accepts STARTCPU but falcon halts immediately
- SEC2 strategies (1, 3, 11): SEC2 starts successfully (mb0 changes), but cannot DMA-load FECS/GPCCS due to FBIF issues

SEC2 is the only falcon that can start from PIO on GV100 because it enters Heavy Secure mode via its own BL — FECS/GPCCS require SEC2's ACR to authenticate their code via DMA.

## Strategic Conclusion

```
Path B: Host PIO → FECS/GPCCS IMEM → BOOTVEC → STARTCPU
  Result: DEAD on GV100 (LS auth blocks execution)
  Would work on: Pre-Maxwell GPUs without secure boot

Path A: Host PIO → SEC2 IMEM → SEC2 BL enters HS → ACR DMA → FECS/GPCCS
  Result: ONLY VIABLE PATH on GV100
  Blocker: SEC2 DMA configuration (FBIF mode, page tables, FBHUB state)
```

## Impact on Architecture

1. **All `fecs_boot.rs` direct PIO paths** are diagnostic-only on GV100. They verify firmware presence but cannot achieve execution.
2. **`strategy_mailbox.rs`** BOOTSTRAP_FALCON command (Strategy 5) is also PIO-based and blocked for the same reason.
3. **The ACR boot chain** (SEC2 → ACR → DMA load → FECS/GPCCS) is the only production path. All engineering effort must focus on SEC2 DMA (Gap 14).
4. **`FalconCapabilityProbe`** correctly identifies PIO capability — the probe works, the falcon just won't execute PIO-loaded code in LS mode.

## Next Step

Focus on Path A: SEC2 DMA. The immediate blocker is FBHUB degradation after VFIO takeover — VRAM DMA reads are corrupted. System memory DMA bypasses FBHUB. See Exp 095.
