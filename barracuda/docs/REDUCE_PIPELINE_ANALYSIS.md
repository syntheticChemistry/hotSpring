# CG Reduce vs Upstream ReduceScalarPipeline

**Date:** 2026-03-01  
**Scope:** `barracuda/src/lattice/gpu_hmc/` resident CG solver

## Summary

The local GPU-resident CG solver uses a **multi-pass tree reduction** (`sum_reduce_f64.wgsl`) for dot-product accumulation. The upstream barracuda crate provides `ReduceScalarPipeline` (in `phase1/toadStool/crates/barracuda/src/pipeline/reduce.rs`). **Direct replacement is not possible** due to API and capability mismatches.

## Local Implementation

| Location | Responsibility |
|----------|----------------|
| `resident_cg_pipelines.rs` | Compiles `reduce_pipeline` from `WGSL_SUM_REDUCE_F64` (via `cg::WGSL_SUM_REDUCE_F64` → `lattice/shaders/sum_reduce_f64.wgsl`) |
| `resident_cg_buffers.rs` | `build_reduce_chain()` builds a multi-pass chain; `encode_reduce_chain()` encodes passes into a command encoder |

### Bind layout (matches upstream)

- **binding 0:** input (storage, read)
- **binding 1:** output (storage, read_write)
- **binding 2:** params (uniform) — `{ size: u32, pad: u32×3 }`

### Multi-pass behavior

- Each pass reduces 256 elements per workgroup → `ceil(N/256)` partial sums
- Passes alternate between `scratch_a` and `scratch_b` until 1 scalar remains
- Final scalar written to caller-provided target: `rz_buf`, `pap_buf`, `rz_new_buf`
- **GPU-resident:** no readback; result consumed by subsequent kernels (compute_alpha, compute_beta)

### Usage sites

- `encode_cg_batch()` — reduce_to_pap, reduce_to_rz_new
- `resident_cg.rs`, `resident_cg_async.rs`, `resident_cg_brain.rs` — encode_reduce_chain for rz, pap, rz_new

## Upstream ReduceScalarPipeline

| Aspect | ReduceScalarPipeline |
|--------|----------------------|
| **Passes** | Exactly 2: input → partials → 1 scalar |
| **Max N** | `ceil(N/256) ≤ 256` → N ≤ 65,536 |
| **API** | `sum_f64(&input) -> Result<f64>` — always submit + readback |
| **Device** | `Arc<WgpuDevice>` |
| **Shader** | `shaders/reduce/sum_reduce_f64.wgsl`, entry `sum_reduce_f64` |

### Why 2 passes fail for large lattices

For `n_pairs = vol × 3`:

- 32⁴ lattice → n_pairs = 3,145,728 → ceil(n/256) = 12,288 partials
- Pass 2 would need 12,288 workgroups → 12,288 outputs (not 1)
- Requires **3+ passes** for N > 65,536

## Incompatibilities

1. **GPU-resident vs readback:** CG reduce writes to `rz_buf`/`pap_buf`/`rz_new_buf` for consumption by compute_alpha/compute_beta. `ReduceScalarPipeline::sum_f64()` always does `submit_and_poll` + `map_async` — no encode-only API.

2. **Multi-pass vs 2-pass:** Local `build_reduce_chain` supports arbitrary N. Upstream hardcodes 2 passes.

3. **Encoder integration:** Local `encode_reduce_chain(enc, pipeline, chain)` encodes into the caller's encoder. Upstream creates its own encoder and submits.

## What Upstream Would Need

To support the resident CG use case, `ReduceScalarPipeline` (or a sibling type) would need:

1. **`encode_reduce_to_buffer(enc, input, target)`** — encode reduction passes into `enc`, write final scalar to `target`, no submit, no readback.

2. **Multi-pass support** — when `ceil(N/256) > 256`, chain additional passes (alternating scratch buffers) until 1 scalar.

3. **Optional target buffer** — allow caller to provide the final 8-byte output buffer instead of internal `scalar_output`.

## Recommendation

**Do not replace** the local reduce with `ReduceScalarPipeline` in its current form. The local implementation is correct and necessary for:

- GPU-resident CG (no readback)
- Production lattice sizes (N > 65,536)
- Batched encoder integration

Consider proposing the above extensions to the upstream barracuda crate for future unification.
