# hotSpring Specifications

**Last Updated**: February 19, 2026
**Status**: Phase A-G complete — 283 unit tests (278 pass + 5 GPU-ignored), 16/16 validation suites
**Domain**: Computational plasma physics, nuclear structure, transport, lattice QCD, surrogate learning

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
| Phase G (Transport + Lattice) | 13/13 transport, 12/12 pure gauge SU(3) |
| Faculty | Murillo (CMSE, MSU — MSDS professor) |
| Faculty extension | Bazavov (CMSE + Physics, MSU — master's professor) |

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
