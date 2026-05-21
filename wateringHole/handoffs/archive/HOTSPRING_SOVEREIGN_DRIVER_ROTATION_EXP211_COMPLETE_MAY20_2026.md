# Sovereign Driver Rotation + Exp 211 Complete — hotSpring Handoff

**Date:** May 20, 2026
**From:** hotSpring
**To:** primalSpring (audit), toadStool (upstream code landed)
**Status:** Exp 211 complete, driver rotation codified, docs reconciled

## Summary

Exp 211 (PMU Mailbox Tier 2 Investigation) is **fully complete**. The Volta PMU
software path is closed — HS lock is total. The binary-patch warm handoff
technique has been **codified into the diesel engine** as `sovereign.warm_handoff`,
a single JSON-RPC call that manages per-GPU driver rotation without the operator
ever touching the kernel.

## Exp 211 Final Status

| Phase | Status | Finding |
|-------|--------|---------|
| A: PMU Liveness | ✅ | PMU alive but HS-locked (SCTL=0x3000) |
| B: Ungating Attempts | ✅ | 6 strategies attempted, all failed |
| C: DMEM Access Probe | ✅ | `0xDEAD5EC2` sentinel — HS lock total |
| Warm Handoff Execution | ✅ | PMC preserved (23 engines), TPC/CE gated (no signed PMU firmware) |
| Driver Rotation Codification | ✅ | `sovereign.warm_handoff` RPC landed in toadStool |

## What Was Codified in toadStool (S267)

- **`cylinder::vfio::kmod`** — insmod/rmmod/modinfo lifecycle
- **`cylinder::vfio::module_patch`** — binary NOP patcher (ret after ftrace, proven on 6.17.9)
- **`cylinder::vfio::sovereign_handoff`** — 8-step pipeline orchestrator
- **`sovereign.warm_handoff` RPC** — strategies: `nouveau_titanv`, `nouveau_k80`
- **`glowplug::ModuleSource`** — `System` vs `Patched` per seeder driver

## Documentation Reconciled

| File | Update |
|------|--------|
| `EXPERIMENT_INDEX.md` | Exp 211 → Complete, 17 RPC methods, K80 Priority 1 |
| `README.md` | Status date → May 20, 17 RPCs, active range 191-211 |
| `experiments/README.md` | Active table extended through Exp 208-211 |
| `CHANGELOG.md` | New entry: Sovereign Driver Rotation Codified |

## Revised Sovereign Compute Priorities

| Priority | Path | Why |
|----------|------|-----|
| **1** | K80 Cross-Gen (incoming HW) | Unsigned falcons → nouveau fully inits → stock warm handoff → Tier 2 |
| **2** | PMU Firmware Extraction | Extract signed blobs from nvidia-470 → load via toadstool (vendor atheistic) |
| **3** | VBIOS Interpreter Completion | True silicon deism — GPU initializes itself |
| closed | PMU Queue Protocol | DMEM inaccessible (HS lock) |
| deprioritized | nvidia-470 Warm Handoff | DRM contamination — requires agentReagents VM path |

## Stale Handoffs Archived

The May 18 comprehensive evolution handoff (`HOTSPRING_COMPREHENSIVE_EVOLUTION_HANDOFF_MAY18_2026.md`)
references "210 experiments" — now superseded by 211. Moved to archive along with
the two other May 18 handoffs that are now downstream of more current state.

## For primalSpring Audit

- [ ] Verify `sovereign.warm_handoff` aligns with universal driver reagent architecture
- [ ] Review `ModuleSource::Patched` — does the patch-at-runtime approach need
      a safety gate or config allowlist?
- [ ] K80 readiness: does the `nouveau_k80` strategy need PLX keepalive
      integration (SwapGuard burst)?
- [ ] Review toadStool CHANGELOG gap: S264-S266 hardware validation work (May 15)
      has handoffs but no CHANGELOG entries yet
