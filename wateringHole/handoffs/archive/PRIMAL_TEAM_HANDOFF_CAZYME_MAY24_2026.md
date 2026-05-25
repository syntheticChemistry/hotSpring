# Primal Team Handoff — CAZyme FEL (May 24, 2026)

**From:** hotSpring  
**To:** primalSpring (audit), barraCuda, coralReef, toadStool, petalTongue, sweetGrass, rhizoCrypt, loamSpine, lithoSpore  
**Context:** Experiment 220 — CAZyme conformational free energy landscapes  
**Action:** Upstream absorption of patterns, gaps, and learnings from first biomolecular MD domain

---

## Per-Primal Status and Evolution Requests

### sweetGrass — Live IPC Braid ✅ Working

Live IPC braid produced (v0.7.27). The `live_braid.json` + `provo_export.jsonld` pattern is reusable for any spring producing ferment transcripts. No evolution needed — sweetGrass already handles this workflow correctly.

**Pattern for other springs:** Any computation-heavy spring producing quantitative results can emit a FermentBraid transcript and have sweetGrass weave it into the attribution layer. The CAZyme FEL flow proves this works end-to-end.

---

### rhizoCrypt — Pseudo DAG (ready for promotion)

Pseudo DAG (`dag.json`) with 11 events and Merkle root. Schema matches anticipated wire format. Ready for promotion when `dag.session.create` / `event.append` IPC is available.

**What's needed:** The `dag.session.create` and `event.append` IPC methods. Once these exist, the 11-event DAG structure from CAZyme FEL can migrate from pseudo JSON to live IPC with minimal adaptation.

**Bottleneck note:** rhizoCrypt + loamSpine are the two primals blocking full provenance-live promotion for any spring. sweetGrass already works.

---

### loamSpine — Pseudo Spine (ready for promotion)

Pseudo spine (`spine.json`) with 3 ledger entries. Ready for `ledger.append` + DID anchoring.

**What's needed:** `ledger.append` IPC method + DID anchoring support. The ledger structure (3 entries: session open, computation complete, handoff sealed) is generic enough to template for other springs.

---

### barraCuda — GAP-HS-111 (4 new shaders + supporting infrastructure)

**New WGSL shaders required:**
1. `harmonic_bond.wgsl` — V(r) = ½k(r - r₀)²
2. `harmonic_angle.wgsl` — V(θ) = ½k(θ - θ₀)²
3. `dihedral_torsion.wgsl` — V(φ) = Σ kₙ(1 + cos(nφ - δₙ))
4. `improper_dihedral.wgsl` — V(ψ) = ½k(ψ - ψ₀)²

**Supporting infrastructure:**
- CHARMM36 topology reader (PSF/PRMTOP → GPU buffers)
- Cremer-Pople collective variable WGSL kernel (ring puckering angles from 6 atoms)
- Metadynamics bias engine (Gaussian deposition + sum_hills reconstruction)

**What already works:** The existing nonbonded engine (LJ, Coulomb, PPPM), Verlet/cell list neighbor finding, velocity Verlet integrator, and thermostat are validated and sufficient. The bonded shaders follow the same pattern (f64 + DF64 paths, workgroup dispatch).

**Reference implementations for parity testing:**
- Python: `notebooks/cazyme_fel/puckering_fel.py`
- Rust: `staging/cazyme-fel/src/lib.rs`

---

### coralReef — No New Requirement

Full f64/DF64 compilation for all GPU targets already works. The bonded force shaders use the same WGSL→native ISA compilation path. No evolution needed.

---

### toadStool — No New Requirement

GPU dispatch for bonded shaders uses the same `TensorSession` pattern (sovereign VFIO dispatch via PBDMA). The existing alloc/dispatch/free cycle is adequate for both prototyping and validation.

For sustained production MD (long trajectories), persistent VRAM handles and fused buffer submission would improve throughput — but this is an optimization, not a requirement.

---

### petalTongue — GAP-HS-112 (FEL Visualization)

**New capability needed:**
- 2D heatmap / 3D surface rendering over Cremer-Pople (θ, φ) space
- CV trajectory overlay on FEL surface
- Interactive ring puckering visualization (6-atom ring conformations)
- Convergence diagnostics (hill height vs time, FES evolution)

**DataBinding adapter spec needed from ludoSpring.** The FEL data format is a regular grid (`grid_min`, `grid_max`, `nbins`) with energy values — straightforward to bind.

---

### lithoSpore — First Non-LTEE Module Ready for Promotion

The CAZyme FEL work is the first module outside LTEE ready for lithoSpore chassis integration:

- **`staging/cazyme-fel/`** — Rust crate implementing sum_hills (1D + 2D FEL reconstruction with periodic CV handling, linear interpolation for grid alignment)
- **`notebooks/cazyme_fel/puckering_fel.py`** — Python reference implementation
- **Parity:** MATCH at <1 kJ/mol vs GROMACS+PLUMED industry control
- **FermentBraid aligned:** `provenance/braids/hotspring_cazyme_fel.json` (BLAKE3 hashes, lithoSpore wire format)
- **Promotion plan:** `docs/LITHOSPORE_PROMOTION.md` (7-step path)

**Integration point:** `lithoSpore::modules::cazyme_fel` — the crate API already exposes `parse_hills`, `reconstruct_fes`, `find_basins`, `find_barriers`, `check_parity`.

---

## Key Learning for Ecosystem

**The provenance trio (DAG + spine + braid) pattern works for any computation-heavy spring producing quantitative results.** The pseudo → live promotion path is smooth when sweetGrass is running. rhizoCrypt and loamSpine IPC availability are the bottleneck for full live provenance across all springs.

**The Python → Rust → WGSL validation ladder works for biomolecular science.** The same tolerance-driven, exit-code-gated methodology that proved Rust matches Python for plasma physics and nuclear EOS also applies to molecular dynamics. The key challenges are:
- Periodic boundary handling for collective variables (dihedrals wrap at ±π)
- Grid alignment between implementations (interpolation needed)
- Well-tempered metadynamics reweighting (heights decay — no explicit γ/(γ-1) correction)

**NUCLEUS composition pattern:** When GAP-HS-111 is resolved, a Tier 3 (IPC-composed) validation can run: hotSpring orchestrates barraCuda (bonded shaders) + toadStool (dispatch) to reproduce the same FEL that standalone Rust/Python produce. This extends the existing composition validation arc (quenched QCD, LTEE) to biomolecular science.

---

## Action Items by Team

| Team | Action | Priority |
|------|--------|----------|
| barraCuda | Implement 4 bonded force WGSL shaders (GAP-HS-111) | Medium |
| barraCuda | CHARMM36 topology reader | Medium |
| barraCuda | Cremer-Pople CV kernel + metadynamics bias | Medium |
| rhizoCrypt | Expose `dag.session.create` / `event.append` IPC | Medium |
| loamSpine | Expose `ledger.append` + DID anchoring IPC | Medium |
| petalTongue | FEL 2D/3D visualization (GAP-HS-112) | Low |
| ludoSpring | DataBinding adapter spec for FEL grid data | Low |
| lithoSpore | Absorb `staging/cazyme-fel/` as `modules::cazyme_fel` | Low |
| primalSpring | Audit this handoff; validate gap severity | — |
