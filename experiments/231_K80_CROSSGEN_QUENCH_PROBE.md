# Experiment 231: K80 Cross-Generation Quench Probe

**Date:** 2026-05-28
**Status:** READY — Awaiting K80 hardware availability
**Hardware:** Tesla K80 (GK210, SM35) — BDF TBD
**Prerequisite:** Exp 230 (Diesel Abstraction Revalidation on Titan V)

## Objective

Test the generation-dispatched interrupt quench on pre-Volta hardware. On Kepler
(GK210, SM35), `InterruptProfile::PRE_VOLTA` should write 0x00000000 directly to
INTR_EN_0@0x140 (which is writable on Kepler), instead of the Volta+ CLEAR
register at 0x180.

This validates the `InterruptProfile` abstraction works correctly across the
generation boundary — the most critical safety mechanism in the diesel engine.

## Key Difference: Kepler vs Volta Interrupt Registers

| Register | Kepler (pre-Volta) | Volta+ |
|----------|-------------------|--------|
| 0x140 INTR_EN_0 | Read/Write | Read-Only |
| 0x160 INTR_EN_SET_0 | N/A | Write-Only (set bits) |
| 0x180 INTR_EN_CLEAR_0 | N/A | Write-Only (clear bits) |

On Kepler: writing 0 to 0x140 disables interrupts.
On Volta: writing to 0x140 is a NO-OP — must use CLEAR@0x180.

## Profile Dispatch Path

```
InterruptProfile::for_sm(35)
  → InterruptProfile::PRE_VOLTA
    → disable_offset() = 0x140
    → disable_value()  = 0x00000000
```

vs Titan V:

```
InterruptProfile::for_sm(70)
  → InterruptProfile::VOLTA_PLUS
    → disable_offset() = 0x180
    → disable_value()  = 0xFFFFFFFF
```

## Run Plan

1. Install K80 or connect via remote host
2. Identify BDF (`lspci -d 10de:`)
3. Build: `cargo build --release --bin rm_trigger`
4. Execute Kepler warm handoff:
   ```bash
   socat - UNIX-CONNECT:/run/toadstool-ember.sock <<'EOF'
   {"jsonrpc":"2.0","method":"sovereign.warm_handoff","params":{"bdf":"<K80_BDF>","strategy":"nouveau_k80"},"id":1}
   EOF
   ```
5. Verify journal for PRE_VOLTA quench dispatch:
   - `disable_offset = 0x140` (not 0x180)
   - Interrupt quench writes 0x00000000 (not 0xFFFFFFFF)

## Success Criteria

- [ ] Kepler warm handoff completes without lockup
- [ ] Journal confirms PRE_VOLTA dispatch (0x140 direct write)
- [ ] GPC probe uses `gpc_count=2` (GK210 has 2 GPCs)
- [ ] Firmware artifacts named `*_gk210.bin` (not `*_gv100.bin`)
