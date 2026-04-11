# hotSpring Specifications

**Last Updated**: April 11, 2026
**Status**: Phase A-J complete + NUCLEUS Composition Validation — 4,065+ tests (964 lib), 140 binaries, crate v0.6.32, barraCuda v0.3.11, toadStool S168+, coralReef Phase 10 Iter 78+. **Three-tier validation: Python → Rust → NUCLEUS IPC.** SovereignInit pipeline: 8-stage pure Rust nouveau replacement. NVIDIA GPFIFO pipeline OPERATIONAL on RTX 3090. AMD scratch/local memory OPERATIONAL on RX 6950 XT. 165+ experiments, 128 WGSL shaders. Ember Survivability Hardening COMPLETE. Squirrel wired in proto-nucleate. Science composition probes operational.
**Domain**: Computational plasma physics, nuclear structure, transport, lattice QCD, spectral theory, surrogate learning

---

## Quick Status

| Metric | Value |
|--------|-------|
| Phase A (Python) | 86/86 PASS — Sarkas MD, TTM, surrogate, nuclear EOS |
| Phase B (BarraCuda GPU) | Validated — 478x speedup, 44.8x energy reduction |
| Phase C (GPU MD) | 9/9 PP Yukawa — N=2,000, 80k steps, 0.000% drift |
| Phase D (f64 + N-scaling) | N=10,000 in 5.3 min, native WGSL builtins |
| Phase E (Paper parity) | 9/9 PP Yukawa — N=10,000, 80k steps, $0.044 |
| Phase F (Full nuclear EOS) | 2,042 nuclei on consumer GPU |
| Phase G (Transport + Lattice) | 13/13 transport, 12/12 pure gauge SU(3), 7/7 dynamical QCD, 17/17 Abelian Higgs |
| Phase H (Spectral Theory) | 41/41 (Anderson 1D/2D/3D, Lanczos, Hofstadter butterfly) |
| Phase I (Heterogeneous) | 68/68 (NPU quantization, beyond-SDK, pipeline, lattice NPU, hetero monitor) |
| ToadStool Rewire v3 | CellListGpu fixed, lattice GPU shaders, **FFT f64** — Tier 3 unblocked |
| ToadStool Rewire v4 | Spectral module fully leaning on upstream (41 KB deleted), `CsrMatrix` alias, `BatchIprGpu` available |
| ToadStool S42+ Catch-Up | 612 shaders absorbed. Dirac+CG GPU absorbed. HFB+ESN absorbed. Remaining: pseudofermion HMC |
| Phase J (Debt Reduction) | v0.6.14: 0 clippy (lib+bins), cross-primal discovery, β_c provenance, WGSL dedup |
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
| [ANDERSON_4D_WEGNER_PROXY.md](ANDERSON_4D_WEGNER_PROXY.md) | Draft | 4D Anderson & Wegner block proxy for CG prediction |
| [BIOMEGATE_BRAIN_ARCHITECTURE.md](BIOMEGATE_BRAIN_ARCHITECTURE.md) | Active | 4-substrate brain architecture: NPU steering, Nautilus Shell, concept edges |
| [PRECISION_STABILITY_SPECIFICATION.md](PRECISION_STABILITY_SPECIFICATION.md) | Active | Numerical stability across f32/DF64/f64, GPU precision routing, cross-spring impact |
| [MULTI_BACKEND_DISPATCH.md](MULTI_BACKEND_DISPATCH.md) | Active | Three-tier dispatch: wgpu/Vulkan (production), coralReef sovereign (long-term), Kokkos/LAMMPS (reference target). NVK discovery, gap analysis |
| [SOVEREIGN_VALIDATION_MATRIX.md](SOVEREIGN_VALIDATION_MATRIX.md) | Active | Pipeline layer x dispatch path x hardware substrate validation matrix. Consolidates gap tracker, experiments 058-165, DRM-from-both-sides strategy, and SovereignInit pipeline status. |
| [COMPUTATIONAL_OMICS.md](COMPUTATIONAL_OMICS.md) | Active | Environmental genomics for sovereign hardware — register trace alignment, firmware motif search, subsystem phylogeny. Long-term compute trio evolution |

### Sovereign GPU & Hardware

| Spec | Status | Description |
|------|--------|-------------|
| [CORALREEF_DISPATCH_FRONTIER_HANDOFF.md](archive/CORALREEF_DISPATCH_FRONTIER_HANDOFF.md) | Historical (archived) | Dispatch frontier handoff from coralReef — sovereign pipeline integration points |
| [DRIVER_AS_SOFTWARE.md](DRIVER_AS_SOFTWARE.md) | Active | GPU driver as evolved software — ember/glowplug lifecycle, livepatch, warm handoff |
| [FIRMWARE_LEARNING_MATRIX.md](FIRMWARE_LEARNING_MATRIX.md) | Active | Firmware learning matrix — ACR, SEC2, FECS, GPCCS, PMU subsystem knowledge map |
| [GPU_CRACKING_GAP_TRACKER.md](archive/GPU_CRACKING_GAP_TRACKER.md) | Superseded (archived) | Original gap tracker — consolidated into SOVEREIGN_VALIDATION_MATRIX.md |
| [NATIVE_COMPUTE_ROADMAP.md](NATIVE_COMPUTE_ROADMAP.md) | Active | Native compute roadmap — coralReef sovereign dispatch evolution targets |
| [SILICON_INVENTORY.md](SILICON_INVENTORY.md) | Active | Fleet hardware inventory — GPU/NPU/CPU specs, PCIe topology, IOMMU groups |
| [SILICON_TIER_ROUTING.md](SILICON_TIER_ROUTING.md) | Active | 7-tier silicon routing spec — TMU, ROP, subgroup, shader core characterization |
| [CROSS_SPRING_EVOLUTION.md](CROSS_SPRING_EVOLUTION.md) | Active | Cross-spring shader ecosystem evolution — 164+ shaders, Write→Absorb→Lean cycle |
| [UNIVERSAL_DRIVER_REAGENT_ARCHITECTURE.md](UNIVERSAL_DRIVER_REAGENT_ARCHITECTURE.md) | Active | agentReagents + benchScale architecture for driver capture and analysis VMs |

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
- **Lattice gauge theory** — SU(3) pure gauge, Wilson action, HMC, staggered Dirac, dynamical fermion pseudofermion HMC
- **Surrogate learning** — Diaw et al. (2024) neural surrogates for physics
- **BarraCuda science driver** — the primary workload pushing GPU f64 capabilities

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
