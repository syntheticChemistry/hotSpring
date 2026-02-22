# hotSpring v0.6.3 — Evolution Handoff

**Date:** February 22, 2026
**From:** hotSpring (computational physics biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only

---

## Summary

v0.6.3 completes inline WGSL elimination, pushes deformed HFB test coverage
to near-100%, evolves metalForge forge to v0.2.0 with enriched hardware
probing and physics workload profiles, and synchronizes all documentation
across the project. hotSpring is now following the absorption pattern more
cleanly: all shaders in dedicated `.wgsl` files, all physics modules well
tested, and forge explicitly designed for toadstool consumption.

---

## 1. Inline WGSL Extraction Complete

**Zero inline WGSL in production library code.**

All remaining inline `const SHADER_*: &str = r"..."` definitions extracted
to dedicated `.wgsl` files loaded via `include_str!()`:

| Source File | Shader Constant | Extracted To |
|-------------|----------------|-------------|
| `md/shaders.rs` | `SHADER_VV_HALF_KICK` | `md/shaders/vv_half_kick_f64.wgsl` |
| `md/shaders.rs` | `SHADER_BERENDSEN` | `md/shaders/berendsen_f64.wgsl` |
| `md/shaders.rs` | `SHADER_KINETIC_ENERGY` | `md/shaders/kinetic_energy_f64.wgsl` |
| `lattice/complex_f64.rs` | `WGSL_COMPLEX64` | `lattice/shaders/complex_f64.wgsl` |
| `lattice/su3.rs` | `WGSL_SU3` | `lattice/shaders/su3_f64.wgsl` |

This means every shader is:
- Independently lintable and diffable
- Loadable by toadstool's shader pipeline without Rust parsing
- Visible to tools (IDE WGSL support, shader linters)

**WGSL inventory**: 41 `.wgsl` files, ~3,600 lines across 5 directories.

---

## 2. Deformed HFB Coverage Push

13 new unit and integration tests pushed `hfb_deformed/` coverage:

| Module | Before | After |
|--------|--------|-------|
| `hfb_deformed/mod.rs` | 29.3% region | **94.9%** |
| `hfb_deformed/basis.rs` | 65.8% region | **98.7%** |

Key tests added:
- `diagonalize_blocks` with V=0 (HO eigenvalues), constant V (shift), sharp Fermi
- `potential_matrix_element` diagonal constancy and Hermitian symmetry
- `solve()` SCF loop: smoke test, deterministic, physical plausibility
- Hermite and Laguerre oscillator numerical norm integrals

---

## 3. metalForge Forge v0.2.0

Forge crate evolved from v0.1.0 (16 tests) to v0.2.0 (19 tests):

### New capabilities
- **GPU VRAM probing**: `adapter.limits().max_buffer_size` reported as `memory_bytes`
- **NPU PCIe sysfs scan**: Reads `/sys/bus/pci/devices/*/vendor` for BrainChip
  vendor ID `0x1e7c`, reports vendor:device string and 8 MB SRAM
- **Multi-device NPU scan**: Probes `/dev/akida0` through `/dev/akida3`
- **Physics workload profiles**: `dispatch::profiles` module with 8 predefined
  workloads encoding capability requirements for hotSpring physics domains:
  - `md_force()` — GPU f64 + scalar reduce
  - `hfb_eigensolve()` — GPU f64 + eigensolve
  - `lattice_cg()` — GPU f64 + CG
  - `esn_npu_inference()` — quantized int8, prefer NPU
  - `esn_gpu_inference()` — f32 shader, prefer GPU
  - `cpu_validation()` — f64, prefer CPU
  - `spectral_spmv()` — GPU f64 + sparse SpMV
  - `hetero_npu_phase_classifier()` — quantized int4, prefer NPU

### Absorption-ready design
- Zero clippy warnings (pedantic + nursery)
- `#![deny(clippy::expect_used, clippy::unwrap_used)]`
- Bridge module connects directly to barracuda `WgpuDevice`
- Absorption mapping table in `ABSORPTION_MANIFEST.md`

---

## 4. Documentation Synchronization

All project documents now reflect v0.6.3 state:

| Document | Key Updates |
|----------|-------------|
| `README.md` | v0.6.3, 648 tests, bench/ module, zero inline WGSL |
| `whitePaper/README.md` | 648 tests, 19 forge tests, 74.9%/83.8% coverage |
| `specs/README.md` | v0.6.3, 648 tests |
| `specs/PAPER_REVIEW_QUEUE.md` | 22 papers consistent, Feb 22 date |
| `metalForge/README.md` | forge v0.2.0, 19 tests, enriched NPU probe |
| `EVOLUTION_READINESS.md` | v0.6.3 completed section, updated shader inventory |
| `ABSORPTION_MANIFEST.md` | v0.6.3, forge v0.2.0, updated absorption targets |

---

## 5. Current Metrics

| Metric | Value |
|--------|-------|
| barracuda crate version | v0.6.3 |
| forge crate version | v0.2.0 |
| Unit tests | 648 (648 pass + 6 GPU/heavy-ignored) |
| Integration tests | 24 (3 suites) |
| Forge tests | 19 |
| Validation suites | 33/33 |
| NPU HW checks | 34/35 |
| Test coverage | 74.9% region / 83.8% function |
| Clippy warnings | 0 (barracuda + forge) |
| Doc warnings | 0 |
| Inline WGSL | 0 (production library) |
| TODO/FIXME markers | 0 |
| `.wgsl` shader files | 41 (~3,600 lines) |

---

## 6. Absorption Roadmap

### Ready now (Tier 1 — high GPU acceleration value)

| Module | Shaders | Tests | Notes |
|--------|---------|-------|-------|
| CSR SpMV | `spmv_csr_f64.wgsl` | 8/8 | Machine-epsilon parity |
| Lanczos | Uses SpMV | 6/6 | CPU control + GPU SpMV inner loop |
| Staggered Dirac | `dirac_staggered_f64.wgsl` | 8/8 | Max error 4.44e-16 |
| CG Solver | 3 WGSL shaders | 9/9 | Iteration counts match CPU exactly |
| ESN Reservoir | 2 WGSL shaders | 16+ | GPU + NPU validated |

### Next evolution targets

| Target | Status | What's needed |
|--------|--------|--------------|
| `hfb_deformed_gpu` H-build | 5 shaders exist, unwired | Wire deformed_*.wgsl into GPU SCF loop |
| Sturm tridiagonal eigensolve | CPU, 23/23 checks | General-purpose tridiag solver for barracuda |
| `StatefulPipeline` for MD | Deferred | Replace manual encoder batching |

---

## 7. Pattern Compliance

hotSpring follows the biome model:

- **Write → Absorb → Lean**: All absorbed modules now use upstream imports
- **Shaders in `.wgsl` files**: Zero inline WGSL (was 8+ in v0.5.x)
- **No cross-spring imports**: hotSpring doesn't import neuralSpring or vice versa
- **Handoffs in `wateringHole/`**: All absorption documented
- **Forge as absorption seam**: Bridge module explicitly connects to barracuda API

---

*hotSpring continues to write physics extensions, validate them locally, and
prepare them for toadstool/barracuda absorption. The cleaner the code, the
faster the fungus grows.*
