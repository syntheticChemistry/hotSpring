# hotSpring v0.6.31 — Gap Closure Rewire

**Date**: March 12, 2026
**Version**: v0.6.31
**Pins**: barraCuda `82ff983`, coralReef Iter 37 (`fe9fae4`), toadStool S147 (`ac3ea6d6`)

---

## Summary

Absorbed latest upstream from all three sovereign compute trio primals.
Gap 1 (dispatch_binary) and Gap 4 (RegisterAccess bridge) are now CLOSED.
Gap 3 (FECS channel) is UNBLOCKED by our DRM fix from the previous session.

### Key Changes
- barraCuda pin updated: `d761c5d` → `82ff983`
- MD precision routing now checks `sovereign_resolves_poisoning()` — enables DF64 transcendentals through sovereign bypass even when naga SPIR-V poisoning would normally block them
- SM86/Ampere prioritized in sovereign dispatch probe order
- 848 tests, 0 failures, 0 clippy warnings

### What's Left
- QMD/pushbuf tuning for compute kernel execution (returns 0, expects 42)
- Gap 2 (NVIDIA proprietary UVM) — partially closed, needs remaining stubs
- Gap 5 (knowledge base → compute init) — sovereign GSP module exists, needs wiring
- Gap 6 (error recovery) — last mile
