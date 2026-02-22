# hotSpring Specifications

**Last Updated**: February 22, 2026
**Status**: Phase A-I complete — 637 tests (637 passing + 6 GPU/heavy-ignored; spectral tests upstream), 33/33 validation suites, crate v0.6.4
**Domain**: Computational plasma physics, nuclear structure, transport, lattice QCD, spectral theory, surrogate learning

---

## Quick Status

| Metric | Value |
|--------|-------|
| Phase A (Python) | 86/86 PASS — Sarkas MD, TTM, surrogate, nuclear EOS |
| Phase B (BarraCUDA GPU) | Validated — 478x speedup, 44.8x energy reduction |
| Phase C (GPU MD) | 9/9 PP Yukawa — N=2,000, 80k steps, 0.000% drift |
| Phase D (f64 + N-scaling) | N=10,000 in 5.3 min, native WGSL builtins |
| Phase E (Paper parity) | 9/9 PP Yukawa — N=10,000, 80k steps, $0.044 |
| Phase F (Full nuclear EOS) | 2,042 nuclei on consumer GPU |
| Phase G (Transport + Lattice) | 13/13 transport, 12/12 pure gauge SU(3), 17/17 Abelian Higgs |
| Phase H (Spectral Theory) | 41/41 (Anderson 1D/2D/3D, Lanczos, Hofstadter butterfly) |
| Phase I (Heterogeneous) | 68/68 (NPU quantization, beyond-SDK, pipeline, lattice NPU, hetero monitor) |
| ToadStool Rewire v3 | CellListGpu fixed, lattice GPU shaders, **FFT f64** — Tier 3 unblocked |
| ToadStool Rewire v4 | Spectral module fully leaning on upstream (41 KB deleted), `CsrMatrix` alias, `BatchIprGpu` available |
| Faculty | Murillo (CMSE, MSU — MSDS professor) |
| Faculty extension | Bazavov (CMSE + Physics, MSU — master's professor) |
| Faculty extension | Kachkovskiy (Math, MSU — spectral theory) |

---

## Specifications

### Validation & Reproduction

| Spec | Status | Description |
|------|--------|-------------|
| [PAPER_REVIEW_QUEUE.md](PAPER_REVIEW_QUEUE.md) | Active | Papers to review/reproduce, prioritized by tier |
| [BARRACUDA_REQUIREMENTS.md](BARRACUDA_REQUIREMENTS.md) | Active | GPU kernel requirements and gap analysis |

### Existing Documentation (in parent directories)

| Document | Location | Description |
|----------|----------|-------------|
| CONTROL_EXPERIMENT_STATUS.md | `../` | Detailed experiment logs and check counts |
| PHYSICS.md | `../` | Physics background and equations |
| NUCLEAR_EOS_STRATEGY.md | `../` | L1-L3 nuclear EOS approach |
| whitePaper/STUDY.md | `../whitePaper/` | Full study narrative |
| whitePaper/METHODOLOGY.md | `../whitePaper/` | Two-phase validation protocol |
| whitePaper/BARRACUDA_SCIENCE_VALIDATION.md | `../whitePaper/` | Phase B GPU results |

---

## Scope

### hotSpring IS:
- **Plasma physics validation** — Sarkas Yukawa MD, OCP thermodynamics
- **Nuclear structure computation** — SEMF → HFB → deformed HFB on consumer GPU
- **Transport coefficients** — Green-Kubo D*/η*/λ*, Stanton-Murillo fits
- **Lattice gauge theory** — SU(3) pure gauge, Wilson action, HMC, staggered Dirac
- **Surrogate learning** — Diaw et al. (2024) neural surrogates for physics
- **BarraCUDA science driver** — the primary workload pushing GPU f64 capabilities

### hotSpring IS NOT:
- Machine learning research (neuralSpring)
- Sensor noise characterization (groundSpring)
- Biological computation (wetSpring)
- Weather/irrigation modeling (airSpring)

### hotSpring EXTENDS TO (via Bazavov):
- Lattice QCD equation of state
- Spectral reconstruction inverse problems
- Hadronic vacuum polarization (muon g-2)

### hotSpring EXTENDS TO (via Kachkovskiy):
- Anderson localization (1D/2D/3D)
- Almost-Mathieu operator / Hofstadter butterfly
- Spectral theory of quasiperiodic operators

---

## Reading Order

**New to hotSpring** (20 min):
1. This README (5 min)
2. `../whitePaper/README.md` — overview and key results (10 min)
3. PAPER_REVIEW_QUEUE.md — what's next (5 min)

**Deep dive** (2 hours):
`../whitePaper/STUDY.md` → `../PHYSICS.md` → `../NUCLEAR_EOS_STRATEGY.md` → BARRACUDA_REQUIREMENTS.md

---

## License

**AGPL-3.0** — GNU Affero General Public License v3.0

All hotSpring code, data, and documentation are aggressively open science. See `../LICENSE` for full text. Any derivative work, including network-accessible services using hotSpring code, must publish source under the same license.
