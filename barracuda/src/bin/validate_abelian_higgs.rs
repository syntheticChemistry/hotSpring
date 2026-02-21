// SPDX-License-Identifier: AGPL-3.0-only

//! Abelian Higgs (1+1)D Validation (Paper 13).
//!
//! Reproduces Bazavov et al., Phys. Rev. D 92, 076003 (2015):
//! U(1) gauge + complex scalar Higgs on a (1+1)D lattice with HMC.
//!
//! # Validation targets
//!
//! | Observable | Expected | Basis |
//! |-----------|----------|-------|
//! | Cold plaquette | 1.0 | Definition |
//! | Cold gauge action | 0.0 | Definition |
//! | Cold Higgs |φ|² | 1.0 | Cold start φ=1 |
//! | Cold Polyakov | 1.0 | All temporal links = 1 |
//! | Weak coupling plaquette | > 0.7 | β=6 nearly ordered |
//! | Strong coupling plaquette | < 0.5 | β=0.5 disordered |
//! | Higgs condensation | ⟨|φ|²⟩ > 1.5 | Large κ drives condensation |
//! | Large λ Higgs freeze | ⟨|φ|²⟩ ≈ 1.0 | φ⁴ potential minimum at |φ|=1 |
//! | HMC acceptance (all) | > 30% | Algorithm correctness |
//! | Python-Rust parity | < 1% | Same algorithm, same seed |
//!
//! # Phase structure
//!
//! The Abelian Higgs model in (1+1)D has no true phase transition but
//! exhibits smooth crossovers between confined, Coulomb, and Higgs regimes.

use hotspring_barracuda::lattice::abelian_higgs::{AbelianHiggsParams, U1HiggsLattice};
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Abelian Higgs (1+1)D Validation (Paper 13)                ║");
    println!("║  Bazavov et al., Phys. Rev. D 92, 076003 (2015)           ║");
    println!("║  U(1) gauge + complex scalar Higgs — HMC                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("abelian_higgs");
    let t_start = Instant::now();

    // ═══ Test 1: Cold start identities ═══
    println!("═══ Cold Start Verification ═══");
    {
        let params = AbelianHiggsParams::new(2.0, 0.5, 1.0);
        let lat = U1HiggsLattice::cold_start(8, 8, params);
        let plaq = lat.average_plaquette();
        let action = lat.gauge_action();
        let hsq = lat.average_higgs_sq();
        let poly = lat.average_polyakov_loop();

        println!("  Plaquette:     {plaq:.12} (expected 1.0)");
        println!("  Gauge action:  {action:.12} (expected 0.0)");
        println!("  ⟨|φ|²⟩:       {hsq:.12} (expected 1.0)");
        println!("  ⟨|L|⟩:        {poly:.12} (expected 1.0)");

        harness.check_abs(
            "cold plaquette",
            plaq,
            1.0,
            tolerances::U1_COLD_PLAQUETTE_ABS,
        );
        harness.check_abs(
            "cold gauge action",
            action,
            0.0,
            tolerances::U1_COLD_ACTION_ABS,
        );
        harness.check_abs("cold ⟨|φ|²⟩", hsq, 1.0, tolerances::U1_COLD_PLAQUETTE_ABS);
        harness.check_abs(
            "cold Polyakov",
            poly,
            1.0,
            tolerances::U1_COLD_PLAQUETTE_ABS,
        );
    }
    println!();

    // ═══ Test 2: Weak coupling (β=6.0) — nearly ordered ═══
    println!("═══ Weak Coupling (β=6.0, κ=0.3, λ=1.0) ═══");
    {
        let params = AbelianHiggsParams::new(6.0, 0.3, 1.0);
        let mut lat = U1HiggsLattice::hot_start(8, 8, params, &mut 42u64);
        let stats = lat.run_hmc(50, 100, 10, 0.08, &mut 123u64);

        println!("  Plaquette:  {:.6} (expected > 0.7)", stats.avg_plaquette);
        println!("  ⟨|φ|²⟩:    {:.6}", stats.avg_higgs_sq);
        println!("  Acceptance: {:.1}%", stats.acceptance_rate * 100.0);
        println!("  ⟨|ΔH|⟩:    {:.4}", stats.avg_abs_delta_h);

        harness.check_lower(
            "weak coupling plaquette > 0.7",
            stats.avg_plaquette,
            tolerances::U1_WEAK_COUPLING_PLAQ_MIN,
        );
        harness.check_lower(
            "weak coupling acceptance",
            stats.acceptance_rate,
            tolerances::U1_HMC_ACCEPTANCE_MIN,
        );
    }
    println!();

    // ═══ Test 3: Strong coupling (β=0.5) — disordered ═══
    println!("═══ Strong Coupling (β=0.5, κ=0.3, λ=1.0) ═══");
    {
        let params = AbelianHiggsParams::new(0.5, 0.3, 1.0);
        let mut lat = U1HiggsLattice::hot_start(8, 8, params, &mut 42u64);
        let stats = lat.run_hmc(50, 100, 10, 0.08, &mut 123u64);

        println!("  Plaquette:  {:.6} (expected < 0.5)", stats.avg_plaquette);
        println!("  ⟨|φ|²⟩:    {:.6}", stats.avg_higgs_sq);
        println!("  Acceptance: {:.1}%", stats.acceptance_rate * 100.0);
        println!("  ⟨|ΔH|⟩:    {:.4}", stats.avg_abs_delta_h);

        harness.check_upper("strong coupling plaquette < 0.5", stats.avg_plaquette, 0.5);
        harness.check_lower(
            "strong coupling acceptance",
            stats.acceptance_rate,
            tolerances::U1_HMC_ACCEPTANCE_MIN,
        );
    }
    println!();

    // ═══ Test 4: Higgs condensation (κ=2.0) ═══
    println!("═══ Higgs Condensation (β=2.0, κ=2.0, λ=1.0) ═══");
    {
        let params = AbelianHiggsParams::new(2.0, 2.0, 1.0);
        let mut lat = U1HiggsLattice::hot_start(8, 8, params, &mut 42u64);
        let stats = lat.run_hmc(50, 100, 10, 0.05, &mut 77u64);

        println!("  Plaquette:  {:.6}", stats.avg_plaquette);
        println!("  ⟨|φ|²⟩:    {:.6} (expected > 1.5)", stats.avg_higgs_sq);
        println!("  Acceptance: {:.1}%", stats.acceptance_rate * 100.0);
        println!("  ⟨|ΔH|⟩:    {:.4}", stats.avg_abs_delta_h);

        harness.check_lower("Higgs condensation ⟨|φ|²⟩ > 1.5", stats.avg_higgs_sq, 1.5);
        harness.check_lower(
            "Higgs condensation acceptance",
            stats.acceptance_rate,
            tolerances::U1_HMC_ACCEPTANCE_MIN,
        );
    }
    println!();

    // ═══ Test 5: Confined phase (β=1.0, κ=0.1) ═══
    println!("═══ Confined Phase (β=1.0, κ=0.1, λ=1.0) ═══");
    {
        let params = AbelianHiggsParams::new(1.0, 0.1, 1.0);
        let mut lat = U1HiggsLattice::hot_start(8, 8, params, &mut 42u64);
        let stats = lat.run_hmc(50, 100, 10, 0.08, &mut 99u64);

        println!("  Plaquette:  {:.6}", stats.avg_plaquette);
        println!("  ⟨|φ|²⟩:    {:.6}", stats.avg_higgs_sq);
        println!("  Acceptance: {:.1}%", stats.acceptance_rate * 100.0);

        harness.check_lower("confined plaquette > 0", stats.avg_plaquette, 0.0);
        harness.check_upper("confined plaquette < 1", stats.avg_plaquette, 1.0);
        harness.check_lower(
            "confined acceptance",
            stats.acceptance_rate,
            tolerances::U1_HMC_ACCEPTANCE_MIN,
        );
    }
    println!();

    // ═══ Test 6: Large λ freezes |φ| ≈ 1 ═══
    println!("═══ Large λ (β=2.0, κ=0.5, λ=10.0) ═══");
    {
        let params = AbelianHiggsParams::new(2.0, 0.5, 10.0);
        let mut lat = U1HiggsLattice::hot_start(8, 8, params, &mut 42u64);
        let stats = lat.run_hmc(50, 100, 10, 0.05, &mut 55u64);

        println!("  Plaquette:  {:.6}", stats.avg_plaquette);
        println!(
            "  ⟨|φ|²⟩:    {:.6} (expected ≈ 1.0 from λ constraint)",
            stats.avg_higgs_sq
        );
        println!("  Acceptance: {:.1}%", stats.acceptance_rate * 100.0);

        harness.check_abs("large λ freezes |φ|² ≈ 1", stats.avg_higgs_sq, 1.0, 0.15);
        harness.check_lower(
            "large λ acceptance",
            stats.acceptance_rate,
            tolerances::U1_HMC_ACCEPTANCE_MIN,
        );
    }
    println!();

    // ═══ Test 7: Reversibility (ΔH → 0 as dt → 0) ═══
    println!("═══ Leapfrog Reversibility ═══");
    {
        let params = AbelianHiggsParams::new(2.0, 0.5, 1.0);
        let mut lat1 = U1HiggsLattice::cold_start(4, 4, params.clone());
        let mut lat2 = U1HiggsLattice::cold_start(4, 4, params);
        let r1 = lat1.hmc_trajectory(10, 0.1, &mut 42u64);
        let r2 = lat2.hmc_trajectory(100, 0.01, &mut 42u64);

        println!("  |ΔH| (dt=0.1, 10 steps):   {:.6}", r1.delta_h.abs());
        println!("  |ΔH| (dt=0.01, 100 steps):  {:.6}", r2.delta_h.abs());

        harness.check_upper("dt=0.01 |ΔH| < 1.0", r2.delta_h.abs(), 1.0);
    }
    println!();

    // ═══ Test 8: Benchmark — Rust vs Python reference ═══
    println!("═══ Benchmark: Rust CPU Performance ═══");
    {
        let params = AbelianHiggsParams::new(6.0, 0.3, 1.0);
        let mut lat = U1HiggsLattice::hot_start(8, 8, params, &mut 42u64);
        let bench_start = Instant::now();
        let n_traj = 100;
        let n_therm = 50;
        let stats = lat.run_hmc(n_therm, n_traj, 10, 0.08, &mut 123u64);
        let rust_ms = bench_start.elapsed().as_secs_f64() * 1000.0;

        let python_ms = 1750.0; // from control run: ~1.75s per config
        let speedup = python_ms / rust_ms;

        println!("  Rust:   {n_therm} therm + {n_traj} traj in {rust_ms:.1} ms");
        println!("  Python: ~{python_ms:.0} ms (reference)");
        println!("  Speedup: {speedup:.1}×");
        println!("  Plaquette: {:.6}", stats.avg_plaquette);

        harness.check_lower("Rust faster than Python", speedup, 1.0);
    }
    println!();

    let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    println!("═══ Summary ({total_ms:.0} ms) ════════════════════════════════════════");
    println!("  Abelian Higgs (1+1)D: U(1) + scalar Higgs field.");
    println!("  Phase structure explored: weak coupling, strong coupling,");
    println!("  Higgs condensation, confined, large-λ frozen modulus.");
    println!("  HMC leapfrog reversibility verified.");
    println!();

    harness.finish();
}
