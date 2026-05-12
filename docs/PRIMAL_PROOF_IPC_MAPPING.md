# Primal Proof IPC Mapping — hotSpring

**Version**: v0.6.32 → post-interstadial Tier 4  
**Last updated**: May 12, 2026  
**Purpose**: Maps every `barracuda::` library call used in hotSpring to its JSON-RPC equivalent for IPC-first operation.

## Overview

hotSpring validates physics computations through two paths:

1. **Library path** (`barracuda::`): Direct Rust crate import — used for Tier 1 (structural) validation
2. **IPC path** (JSON-RPC 2.0): Primal composition via `CompositionContext` — used for Tier 2 (live) validation

This document maps every `barracuda::` call to the JSON-RPC method that would produce the same result when routed through the NUCLEUS composition.

## Math / Statistics

| Library Call | JSON-RPC Method | Capability | Provider |
|-------------|----------------|------------|----------|
| `barracuda::stats::mean(&data)` | `stats.mean` | `tensor` | barraCuda |
| `barracuda::stats::variance(&data)` | `stats.variance` | `tensor` | barraCuda |
| `barracuda::stats::std_dev(&data)` | `stats.std_dev` | `tensor` | barraCuda |
| `barracuda::stats::median(&data)` | `stats.median` | `tensor` | barraCuda |
| `barracuda::stats::jackknife_error(&data)` | `stats.jackknife_error` | `tensor` | barraCuda |
| `barracuda::stats::bootstrap(&data, n)` | `stats.bootstrap` | `tensor` | barraCuda |
| `barracuda::stats::autocorrelation(&data)` | `stats.autocorrelation` | `tensor` | barraCuda |

## Linear Algebra

| Library Call | JSON-RPC Method | Capability | Provider |
|-------------|----------------|------------|----------|
| `barracuda::linalg::matmul(a, b)` | `tensor.matmul` | `tensor` | barraCuda |
| `barracuda::linalg::eigenvalues(m)` | `linalg.eigenvalues` | `tensor` | barraCuda |
| `barracuda::linalg::svd(m)` | `linalg.svd` | `tensor` | barraCuda |
| `barracuda::linalg::solve(a, b)` | `linalg.solve` | `tensor` | barraCuda |
| `barracuda::linalg::det(m)` | `linalg.determinant` | `tensor` | barraCuda |
| `barracuda::linalg::inverse(m)` | `linalg.inverse` | `tensor` | barraCuda |

## Special Functions

| Library Call | JSON-RPC Method | Capability | Provider |
|-------------|----------------|------------|----------|
| `barracuda::special::erf(x)` | `special.erf` | `math` | barraCuda |
| `barracuda::special::erfc(x)` | `special.erfc` | `math` | barraCuda |
| `barracuda::special::gamma(x)` | `special.gamma` | `math` | barraCuda |
| `barracuda::special::bessel_j(n, x)` | `special.bessel_j` | `math` | barraCuda |
| `barracuda::special::spherical_bessel_j(l, x)` | `special.spherical_bessel_j` | `math` | barraCuda |

## GPU Compute / Shader Dispatch

| Library Call | JSON-RPC Method | Capability | Provider |
|-------------|----------------|------------|----------|
| `GpuF64::dispatch(&shader, data)` | `device.dispatch` | `compute` | toadStool |
| `GpuF64::create_pipeline(wgsl)` | `device.compile` | `compute` | toadStool |
| `ToadStoolDispatchClient::submit()` | `compute.dispatch.submit` | `compute` | toadStool (absorbs ember Phase A+B) |
| `ToadStoolDispatchClient::capabilities()` | `compute.dispatch.capabilities` | `compute` | toadStool (absorbs ember Phase A+B) |
| `glowplug_client::dispatch()` | `device.dispatch` | `shader` | coralReef (glowplug — soft-deprecated, absorbed by toadStool) |
| `glowplug_client::list_devices()` | `device.list` | `shader` | coralReef (glowplug — soft-deprecated, absorbed by toadStool) |
| `coral_gpu::SovereignPipeline` | `sovereign.boot` | `shader` | coralReef (sovereign compiler) |

## Spectral Theory

| Library Call | JSON-RPC Method | Capability | Provider |
|-------------|----------------|------------|----------|
| `barracuda::spectral::anderson_2d(lx, ly, w, seed)` | `spectral.anderson_2d` | `math` | barraCuda |
| `barracuda::spectral::lanczos(matrix, n, seed)` | `spectral.lanczos` | `math` | barraCuda |
| `barracuda::spectral::spmv(matrix, vec)` | `spectral.spmv` | `math` | barraCuda |

## Security / Cryptography

| Library Call | JSON-RPC Method | Capability | Provider |
|-------------|----------------|------------|----------|
| `blake3::hash(data)` | `crypto.hash` | `security` | bearDog |
| `receipt_signing::sign_receipt()` | `crypto.sign_ed25519` | `security` | bearDog |
| N/A (token verification) | `auth.check` | `security` | bearDog |

## Provenance / Storage

| Library Call | JSON-RPC Method | Capability | Provider |
|-------------|----------------|------------|----------|
| `dag_provenance::DagSession` | `dag.commit` | `dag` | rhizoCrypt |
| N/A (ledger) | `ledger.append` | `ledger` | loamSpine |
| N/A (attribution) | `attribution.record` | `attribution` | sweetGrass |
| N/A (storage) | `storage.put` / `storage.get` | `storage` | NestGate |

## Discovery / Health

| Library Call | JSON-RPC Method | Capability | Provider |
|-------------|----------------|------------|----------|
| `NucleusContext::detect()` | `health.liveness` | all | per-primal |
| `NucleusContext::by_domain(cap)` | `capability.list` | all | per-primal |
| `NucleusContext::call_by_capability()` | `capability.call` | routed | songBird |

## Inference (via Squirrel / neuralSpring)

| Library Call | JSON-RPC Method | Capability | Provider |
|-------------|----------------|------------|----------|
| `squirrel_client::inference_complete()` | `inference.complete` | `inference` | squirrel |
| `squirrel_client::inference_embed()` | `inference.embed` | `inference` | squirrel |
| `squirrel_client::inference_models()` | `inference.models` | `inference` | squirrel |

## Ember (GPU Hardware)

| Library Call | JSON-RPC Method | Capability | Provider |
|-------------|----------------|------------|----------|
| `EmberClient::mmio_read()` | `mmio.read` | `hardware` | coralReef (ember) |
| `EmberClient::mmio_write()` | `mmio.write` | `hardware` | coralReef (ember) |
| `EmberClient::falcon_upload()` | `falcon.upload` | `hardware` | coralReef (ember) |
| `EmberClient::falcon_start()` | `falcon.start` | `hardware` | coralReef (ember) |
| `EmberClient::sec2_prepare()` | `sec2.prepare` | `hardware` | coralReef (ember) |

## Migration Path

### Current (v0.6.32)

hotSpring uses a **dual-lane** model:
- **Lane 1**: Library calls via `barracuda` path dependency (fast, zero IPC overhead)
- **Lane 2**: IPC calls via `NucleusContext` / `CompositionContext` (composition-validated)

### Achieved (Tier 4 — May 10, 2026)

- `barracuda` is `optional = true` with `barracuda-local = ["dep:barracuda"]`
- `primal-proof` feature flag added — library compiles without barraCuda
- 25+ GPU/compute modules gated behind `#[cfg(feature = "barracuda-local")]`
- Local `Complex64` fallback enables lattice QCD core without barraCuda
- `hermite`, `factorial`, `bisect`, `lu_solve` — local fallbacks provided
- `composition.status`, `method.register` — biomeOS v3.51 IPC wired
- Per-trio provenance modules: `ipc::provenance::{rhizocrypt, loamspine, sweetgrass}`
- guideStone L6 certification: NUCLEUS deployment validation
- Cross-sync: zero drift against primalSpring canonical 403 methods

### Remaining IPC-first targets

- Default to IPC-first via `CompositionContext` (currently opt-in)
- Library calls become opt-in fallback (reverse default)
- All validation binaries confirm parity between lanes

### CompositionContext Migration

```rust
// Old: Direct library call
let mean = barracuda::stats::mean(&data);

// New: IPC via CompositionContext
let ctx = CompositionContext::from_live_discovery_with_fallback();
let result = ctx.call("tensor", "stats.mean", json!({"data": data}))?;
let mean: f64 = result["result"].as_f64().unwrap();
```

## License

AGPL-3.0-or-later
