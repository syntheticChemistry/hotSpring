# hotSpring Compute Trio Pipeline Wiring — May 12, 2026

**From:** hotSpring (biomeGate sovereign GPU validation niche)
**For:** primalSpring, toadStool, coralReef, barraCuda teams

---

## Summary

hotSpring has completed the compute trio pipeline wiring after toadStool's
Phase C absorption of glowplug/cylinder/ember from coralReef. The key change:
`GlowplugClient` lifecycle methods now prefer NUCLEUS `call_by_capability`
routing (compute domain → toadStool) with glowplug socket fallback.

Two new validation scenarios exercise the full pipeline end-to-end:
- `compute-trio-pipeline`: barraCuda→coralReef→toadStool generic dispatch
- `hotqcd-dispatch`: 6 core lattice QCD shaders through the trio pipeline

---

## Hardware State (biomeGate, May 12 2026)

| GPU | BDF | Driver | Warm Boot | Dispatch | Blocker |
|-----|-----|--------|-----------|----------|---------|
| RTX 5060 (SM120) | `21:00.0` | nvidia 580 | Fully warm | **PROVEN** (12/12 wgpu) | None |
| Titan V (SM70) | `02:00.0` | vfio-pci | warm-catch FECS running | **BLOCKED** | wgpu can't enumerate VFIO device |
| K80 (SM37) | `4b:00.0` / `4c:00.0` | vfio-pci | warm-catch GDDR5+GPCs | **BLOCKED** | wgpu can't enumerate VFIO device |

### Warm Boot Status Detail

- **RTX 5060**: nvidia shared stack. All 12 sovereign roundtrip checks pass.
  154.2 steps/s on Yukawa benchmark. No gaps.
- **Titan V**: Binary-patched nouveau warm-catch (GAP-HS-073 RESOLVED).
  FECS_MC=0x0c060006 (running), PMC pop=23, 1 GPC. HBM2 warm state
  survived BIOS POST → vfio-pci bind cycle.
- **K80**: Binary-patched nouveau warm-catch (GAP-HS-076 RESOLVED).
  GDDR5 trained (12 GiB), 5 GPCs/30 TPCs active, FECS_MC=0x00060005 (running).
  Both dies initialized. PLX 8747 keepalive active.

---

## What Changed

### GlowplugClient NUCLEUS Evolution (GAP-HS-094)

Added `call_with_nucleus_fallback(domain, method, params)` helper to
`GlowplugClient`. 7 lifecycle methods evolved:

| Method | RPC | Domain | Pattern |
|--------|-----|--------|---------|
| `list_devices()` | `device.list` | compute | NUCLEUS-first → glowplug fallback |
| `dispatch_with_options()` | `device.dispatch` | compute | NUCLEUS-first → glowplug fallback |
| `device_swap()` | `device.swap` | compute | NUCLEUS-first → glowplug fallback |
| `device_health()` | `device.health` | compute | NUCLEUS-first → glowplug fallback |
| `device_resurrect()` | `device.resurrect` | compute | NUCLEUS-first → glowplug fallback |
| `sovereign_boot()` | `sovereign.boot` | compute | NUCLEUS-first → glowplug fallback |

**Intentionally NOT evolved** (device-specific low-level ops):
- `register_dump`, `register_snapshot`, `read_bar0_range` — BAR0 MMIO
- `oracle_capture` — MMU page table capture
- `capture_training` — training recipe capture (orchestration)
- `experiment_start/end`, `device_reset` — device lifecycle bookkeeping

These remain on direct glowplug socket by design — they operate on specific
hardware via specific file descriptors and cannot be load-balanced.

### New ROUTED_CAPABILITIES

```
ember.device.health → toadstool (compute)
ember.device.recover → toadstool (compute)
device.list → toadstool (compute)
sovereign.boot → toadstool (compute)
```

### New Validation Scenarios

1. **`compute-trio-pipeline`** (GpuCompute track, Live tier):
   - Phase 1: Precision advisory via `precision.route` (math domain)
   - Phase 2: Shader compilation via `shader.compile.wgsl` (shader domain → coralReef)
   - Phase 3: toadStool workload preflight
   - Phase 4: toadStool dispatch probe (dry-run, compute domain)
   - Phase 5: Per-GPU hardware readiness check (RTX 5060, Titan V, K80)

2. **`hotqcd-dispatch`** (GpuCompute track, Live tier):
   - 6 core QCD shaders: `wilson_plaquette_f64`, `su3_gauge_force_f64`,
     `su3_link_update_f64`, `dirac_staggered_f64`, `cg_kernels_f64`,
     `hmc_leapfrog_f64`
   - 3 silicon routing shaders: `reduce_shared`, `force_alu`, `stencil_storage`
   - Precision advisory confirms f64/mixed/df64 for lattice QCD
   - QCD-specific toadStool dispatch probe

---

## Gaps for Upstream

### coralReef (critical for Titan V + K80 dispatch)

| ID | Gap | Impact | Resolution Path |
|----|-----|--------|-----------------|
| CR-1 | SM70 wgpu backend rebuild | Blocks Titan V sovereign dispatch validation | coralReef ships SM70 VFIO-aware adapter path so wgpu can enumerate warm Volta |
| CR-2 | SM37 wgpu backend rebuild | Blocks K80 sovereign dispatch validation | coralReef ships SM37 VFIO-aware adapter path for Kepler |
| CR-3 | FECS sentinel structured errors | hotSpring captures, hands back | Continued FECS/GPCCS hardening — `falcon_boot()` timeouts should produce `FecsState` JSON |

### toadStool (Phase D + pipeline consolidation)

| ID | Gap | Impact | Resolution Path |
|----|-----|--------|-----------------|
| TS-1 | `try_local_dispatch()` production default | Phase D not yet production default | Flip `local-dispatch` once parity verified |
| TS-2 | `device.list` handler | hotSpring routes via compute domain | toadStool serves `device.list` once glowplug lifecycle absorbed |
| TS-3 | `sovereign.boot` handler | hotSpring routes via compute domain | toadStool serves `sovereign.boot` once lifecycle absorbed |
| TS-4 | Fleet topology via compute domain | `fleet_client.rs` still reads glowplug fleet JSON | Phase C should expose fleet topology via `compute` domain NUCLEUS |

### barraCuda (precision + TensorSession)

| ID | Gap | Impact | Resolution Path |
|----|-----|--------|-----------------|
| BC-1 | `precision.route` JSON-RPC method | May be library-only | Expose as JSON-RPC in `REGISTERED_METHODS` for Tier 2 |
| BC-2 | `TensorSession` upstream adoption | `FusedPipeline` is hotSpring-side wiring | Ship stable `TensorSession` API for lattice workloads |

### Nouveau / Kernel (long-term)

| ID | Gap | Impact | Resolution Path |
|----|-----|--------|-----------------|
| NV-1 | GK210 chipset ID patch | Requires binary-patched nouveau.ko | Upstream `case 0x0f2: device->chip = &nvf1_chipset;` to kernel nouveau |

---

## Metrics

| Metric | Value |
|--------|-------|
| Lib tests (default) | 590 |
| Lib tests (barracuda-local) | 1,042 |
| Validation scenarios | 13 default / 16 with barracuda-local |
| GlowplugClient methods evolved | 7 of 18 (low-level intentionally excluded) |
| New ROUTED_CAPABILITIES | 4 |
| New scenarios | 2 |
| WGSL shaders validated | 51 lattice + 7 silicon routing |

---

## Adoption Guidance for Sibling Springs

Any spring with GPU dispatch should adopt this pattern:

```rust
fn call_with_nucleus_fallback(
    &self,
    domain: &str,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value, Error> {
    let ctx = NucleusContext::detect();
    if let Ok(resp) = ctx.call_by_capability(domain, method, params.clone()) {
        return Ok(resp);
    }
    // Direct socket fallback
    self.call(method, params)
}
```

This is the same 3-tier pattern from GAP-HS-092 (IPC Transport Evolution)
applied to the lifecycle orchestration layer.
