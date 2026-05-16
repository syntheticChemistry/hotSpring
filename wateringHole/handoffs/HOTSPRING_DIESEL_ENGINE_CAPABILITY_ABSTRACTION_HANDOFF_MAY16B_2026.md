# Diesel Engine Capability Abstraction — Handoff

**Date:** May 16, 2026  
**Status:** VALIDATED — deployed, 6,989 tests pass  
**Scope:** 6 GPU-specific subsystems generalized into diesel engine capabilities

## Summary

Evolved GPU-specific subsystems proven on individual hardware into
generation-agnostic diesel engine capabilities. The sovereign init pipeline
now uses traits, strategy patterns, and generalized dispatch instead of
scattered if/else chains. All three GPUs (Titan V, Tesla K80, RTX 5060)
can use the same pipeline with generation-specific behavior injected
through `GenerationProfile`.

## Abstractions Implemented

### 1. PCIe Bridge Health (from PLX Keepalive)
- `PlxKeepalive` → `PcieBridgeKeepalive` (type alias preserved)
- `PlxGuardian` → `BridgeGuardian` (type alias preserved)
- `detect_pcie_bridges()` returns all upstream bridges, PLX first
- `PLX_VENDOR_ID` consolidated as shared constant

### 2. GspBridge Capabilities
- 4 capability queries: `supports_acr()`, `supports_pgob()`, `supports_pmu()`, `supports_gr_init()`
- `pmu_boot()` default method added
- `StubGspBridge` inherits all defaults (false)

### 3. Memory Training Dispatch
- `MemoryTrainingStrategy` enum: GDDR5 → Devinit, HBM2 → Controller, others → Unsupported
- `dispatch_memory_training()` centralizes warm-detection + execution
- `sovereign_init` caller collapsed from 100+ lines to single match

### 4. Falcon Boot Wiring
- `PmuBootstrap::for_chip(ChipFamily)` — parametric beyond Kepler
- `GspBridge::pmu_boot()` — trait method for PMU bootstrap

### 5. Engine Ungating
- `engine_ungate(bar0, seq, engine_name, status_reg)` — replaces `kepler_pgraph_ungate`
- `SovereignInitOptions::engine_init_sequences` — Vec of per-engine sequences
- Legacy `kepler_gr_init` preserved as fallback

### 6. DriverLab Executor
- `DriverLabExecutor::execute()` with callback-driven architecture
- Lifecycle: power cycle → swap → settle → capture → persist → pairwise diff
- `LabExecutionResult`, `TrialExecutionResult`, `DiffSummary` result types

## Files Modified

| File | Change |
|------|--------|
| `ember/src/plx_keepalive.rs` | Renamed struct/fns, added `detect_pcie_bridges()`, `PLX_VENDOR_ID` |
| `ember/src/lib.rs` | Updated re-exports |
| `glowplug/src/plx.rs` | Renamed `PlxGuardian` → `BridgeGuardian` |
| `glowplug/src/lib.rs` | Updated re-exports |
| `glowplug/src/warm_init.rs` | Added `DriverLabExecutor` + result types + tests |
| `cylinder/src/nv/gsp_bridge.rs` | Added capability queries + `pmu_boot()` |
| `cylinder/src/nv/pmu_init.rs` | Added `PmuBootstrap::for_chip()` |
| `cylinder/src/vfio/sovereign_init.rs` | `engine_ungate`, memory training dispatch |
| `cylinder/src/vfio/sovereign_stages.rs` | `MemoryTrainingStrategy`, `dispatch_memory_training()` |
| `cylinder/src/vfio/sovereign_types.rs` | Added `engine_init_sequences` field |
| `server/src/background/pcie_keepalive.rs` | Import `PLX_VENDOR_ID` from ember |

## Deployment Validation

- Built release binary, deployed to `toadstool-ember.service`
- PCIe bridge keepalive confirmed: 3 PLX bridges discovered, 2 K80 GPUs protected
- All 3 GPUs alive: K80 (D0), Titan V (D0), RTX 5060 (nvidia driver)
- `d3cold_allowed=0`, `power/control=on` across all bridges and GPUs

## Upstream Patterns

- **Backward compatibility via type aliases**: `PlxKeepalive`, `PlxGuardian`, `PlxDeviceStatus`
- **Strategy pattern for dispatch**: `MemoryTrainingStrategy::for_memory_type()`
- **Default trait methods for capabilities**: `GspBridge::supports_*()` → false
- **Callback-driven executor**: `DriverLabExecutor` composable with any swap mechanism
- **Legacy fallback fields**: `kepler_gr_init` → `engine_init_sequences`
