# hotSpring Compute Trio Rewire — May 13, 2026

Pulled toadStool S255-S257, barraCuda Sprint 64-67, coralReef Sprint 5-6.
Reviewed all changes and rewired hotSpring to align.

---

## Rewiring summary

### toadStool (S255-S257)

**Consumed:**
- FECS warm-state detection (`probe_warm_fecs()`) — BAR0 probe for warm-preserved
  FECS state from nouveau/nvidia-470 handoff. CPUCTL HALTED bit reconciled (bit 5,
  not bit 4).
- Semantic aliases: `ember.swap` → `device_swap`, `sovereign.boot` → `device_swap`.
  hotSpring's `GlowplugClient::sovereign_boot()` now routes correctly.
- 77 direct JSON-RPC methods (up from 74 in S252). New: `auth.peer_info`,
  `provenance.get`, plus semantic surface.
- `NvVfioComputeDevice` now returns a live `ComputeDevice` when warm FECS detected.
  PBDMA dispatch stub is the next layer.

**hotSpring rewired:**
- `niche.rs` ROUTED_CAPABILITIES: +4 entries (`ember.swap`, `auth.peer_info`,
  `provenance.get`, and `sovereign.boot` now explicit aliases).
- `config/capability_registry.toml`: +5 entries matching niche.rs additions.

**Next for toadStool:**
- Wire PBDMA dispatch through `NvVfioComputeDevice`'s `ComputeDevice` trait impl:
  `alloc()` → VFIO DMA, `upload()` → DMA copy, `dispatch()` → GPFIFO pushbuf,
  `readback()` → DMA copy. Channel operations already exist in `cylinder/src/vfio/channel/`.
- hotSpring will run `exp184_k80_gr_sovereign` and `validate_vfio_sovereign` once
  PBDMA is wired.

### barraCuda (Sprint 64-67)

**Consumed:**
- `precision.route` response now includes `dispatch_path` field (`"wgpu"` |
  `"sovereign"` | `"unavailable"`). `PrecisionAdvisory` struct updated.
- `TensorSession::sub()` and `TensorSession::negate()` shipped (Sprint 66).
  IPC batch path (`tensor.batch.submit`) handles `sub`/`negate`.
- `stats.entropy` confirmed available (alias of `stats.shannon`). GAP-HS-041 resolved.
- GEMM routing: `kernel_router` routes `DenseMatmul` with tensor-core-eligible
  precision to `KernelTarget::Sovereign` with `HardwareHint::TensorCore`.
- OOM detection: `WgpuDevice::is_oom()`, `BarracudaError::is_oom()`, `is_retriable()`.

**hotSpring rewired:**
- `PrecisionAdvisory` struct: added `dispatch_path: Option<String>`.
- `s_compute_trio.rs` and `s_hotqcd_dispatch.rs`: added `dispatch_path_reported` checks.
- GAP-HS-041 marked RESOLVED. GAP-HS-027 updated with Sprint 66 info.

**Next for barraCuda:**
- hotSpring needs to wire `TensorSession` into `gpu_hmc/mod.rs` (GAP-HS-027).
  Upstream primitives are complete.
- GEMM tensor-core execution depends on coralReef HMMA codegen.

### coralReef (Sprint 5-6)

**Consumed:**
- `naga::Module` direct ingest (`compile_module`/`compile_module_full`).
- PTX SM120: switch lowering, math builtins, atomics, barriers, subgroups.
- Compile timeouts: `CORALREEF_COMPILE_TIMEOUT_SECS` (120s default).
- `shader.compile.capabilities` response enriched (math_ops, sm_target, atomics,
  subgroup_ops).
- `capability.list` gains `protocol` + `transport` (Wire Standard L3).

**hotSpring rewired:**
- **CRITICAL FIX:** Compile request params changed from `"source"` to `"wgsl_source"`
  in 4 call sites (`s_compute_trio.rs`, `s_hotqcd_dispatch.rs` ×2, `compute_dispatch.rs`).
  coralReef's `CompileWgslRequest` expects `wgsl_source` with no alias for `source`.
  Previous calls would have silently failed on deserialization.
- Removed stale `"format": "wgsl"`, `"source_type": "wgsl"`, `"target": "spirv"` params.
- Capability registry: added `shader.compile.wgsl.multi`, `shader.compile.status`,
  `shader.compile.capabilities`.

**Next for coralReef:**
- HMMA/WGMMA codegen: IR/encoder machinery exists but WGSL→HMMA lowering not wired.
  barraCuda's GEMM routing is ready to consume once this ships.
- SM120 texture instruction set completion.
- `CompileResponse` field naming: `binary_b64`/`shader_info` are the canonical names.
  hotSpring parsers that consume compile results should migrate.

---

## Env var deprecation

- `CORALREEF_SOCKET` in `precision_brain.rs`: now emits deprecation warning →
  `TOADSTOOL_SOCKET`.
- `CORALREEF_MANIFEST` fallback removed.
- `CORALREEF_RUN_DIR` already deprecated in `fleet_client.rs` (prior sprint).

## Quality

- 591/591 lib tests pass
- Zero clippy warnings
- `config/capability_registry.toml` ↔ `niche.rs` sync test passes

---

**Filed by:** hotSpring trio rewire | **Date:** May 13, 2026
