// SPDX-License-Identifier: AGPL-3.0-only

//! Production Lattice QCD Validation (Papers 9-12).
//!
//! Runs a quenched SU(3) β-scan across the deconfinement transition
//! on 4^4 and 8^4 lattices, validating observables against literature
//! and the HotQCD reference table.
//!
//! This binary establishes the **Rust CPU baseline** for lattice QCD
//! production. The evolution path:
//!   Python control → **this binary (Rust CPU)** → GPU shaders → sovereign pipeline
//!
//! # Validation targets
//!
//! | Observable | Expected | Tolerance | Basis |
//! |-----------|----------|-----------|-------|
//! | Plaquette monotonicity | `<P>` increases with β | exact | Strong-coupling expansion |
//! | Plaquette at β=6.0 | ~0.594 | 10% | Bali et al. (1993) |
//! | Polyakov confined | |L| < 0.4 at β=4.0-5.0 | upper bound | Confinement |
//! | Polyakov transition | |L|(β>6) > |L|(β<5) | ordering | Deconfinement |
//! | Acceptance rate | > 30% at all β | lower bound | Algorithm sanity |
//! | 8^4 scaling | matches 4^4 at β=6.0 within 5% | relative | Finite-size |
//! | Determinism | rerun-identical | exact | LCG seed control |
//!
//! # Provenance
//!
//! - Bali et al., PLB 309, 378 (1993) — quenched plaquette reference
//! - Creutz, PRD 21, 2308 (1980) — SU(3) phase transition
//! - Python control: `control/lattice_qcd/scripts/quenched_beta_scan.py`

use hotspring_barracuda::lattice::hmc::{self, HmcConfig};
use hotspring_barracuda::lattice::multi_gpu::{self, TemperatureScanConfig};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Production Lattice QCD Validation (Papers 9-12)           ║");
    println!("║  Quenched SU(3) β-scan: 4^4 → 8^4 scaling                 ║");
    println!("║  Deconfinement transition at β_c ≈ 5.69                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("production_qcd");

    // ═══ Phase 1: 4^4 β-scan ═══
    println!("═══ Phase 1: 4^4 β-scan (9 β values, 50 traj + 20 therm) ═══");
    println!();

    let scan_config = TemperatureScanConfig {
        dims: [4, 4, 4, 4],
        beta_values: vec![4.0, 4.5, 5.0, 5.5, 5.7, 5.8, 6.0, 6.2, 6.5],
        n_trajectories: 50,
        n_thermalization: 20,
        hmc_dt: 0.05,
        hmc_n_md_steps: 15,
        n_workers: 4,
        seed_base: 42,
    };

    let result_4x4 = multi_gpu::run_temperature_scan(&scan_config);
    println!();
    multi_gpu::print_scan_summary(&result_4x4);
    println!();

    // ─── Check 1: Plaquette monotonicity ───
    let plaq_monotonic = result_4x4
        .points
        .windows(2)
        .all(|w| w[1].mean_plaquette > w[0].mean_plaquette);
    harness.check_bool("4^4 plaquette monotonic with β", plaq_monotonic);

    // ─── Check 2: Acceptance > 30% at all β ───
    let all_accept = result_4x4
        .points
        .iter()
        .all(|p| p.acceptance_rate > tolerances::BETA_SCAN_ACCEPTANCE_MIN);
    harness.check_bool("4^4 HMC acceptance > 30% at all β", all_accept);

    // ─── Check 3: Plaquette at β=6.0 matches reference ───
    if let Some(p6) = result_4x4
        .points
        .iter()
        .find(|p| (p.beta - 6.0).abs() < 0.01)
    {
        harness.check_rel(
            "4^4 plaquette β=6.0 vs Bali (0.594)",
            p6.mean_plaquette,
            tolerances::BETA6_PLAQUETTE_REF,
            tolerances::BETA6_PLAQUETTE_TOLERANCE,
        );
        println!(
            "  β=6.0: <plaq>={:.6}, ref=0.594, err={:.2}%",
            p6.mean_plaquette,
            ((p6.mean_plaquette - 0.594) / 0.594).abs() * 100.0
        );
    }

    // ─── Check 4: Polyakov loop confined at low β ───
    let confined_poly: Vec<f64> = result_4x4
        .points
        .iter()
        .filter(|p| p.beta <= 5.0)
        .map(|p| p.polyakov_loop)
        .collect();
    if !confined_poly.is_empty() {
        let avg_confined = confined_poly.iter().sum::<f64>() / confined_poly.len() as f64;
        harness.check_upper(
            "4^4 confined <|L|> (β≤5.0)",
            avg_confined,
            tolerances::BETA_SCAN_CONFINED_POLYAKOV_MAX,
        );
        println!("  Confined <|L|> (β≤5.0): {avg_confined:.6}");
    }

    // ─── Check 5: Polyakov transition: deconfined > confined ───
    let deconfined_poly: Vec<f64> = result_4x4
        .points
        .iter()
        .filter(|p| p.beta >= 6.0)
        .map(|p| p.polyakov_loop)
        .collect();
    if !confined_poly.is_empty() && !deconfined_poly.is_empty() {
        let avg_confined = confined_poly.iter().sum::<f64>() / confined_poly.len() as f64;
        let avg_deconfined = deconfined_poly.iter().sum::<f64>() / deconfined_poly.len() as f64;
        harness.check_bool(
            "4^4 Polyakov transition (deconfined > confined)",
            avg_deconfined > avg_confined,
        );
        println!(
            "  Polyakov transition: confined={avg_confined:.4} → deconfined={avg_deconfined:.4}"
        );
    }
    println!();

    // ═══ Phase 2: 8^4 scaling validation ═══
    println!("═══ Phase 2: 8^4 scaling (3 β values) ═══");
    println!();

    let scan_8x8 = TemperatureScanConfig {
        dims: [8, 8, 8, 8],
        beta_values: vec![5.5, 6.0, 6.5],
        n_trajectories: 30,
        n_thermalization: 15,
        hmc_dt: 0.03,
        hmc_n_md_steps: 20,
        n_workers: 3,
        seed_base: 42,
    };

    let result_8x8 = multi_gpu::run_temperature_scan(&scan_8x8);
    println!();
    multi_gpu::print_scan_summary(&result_8x8);
    println!();

    // ─── Check 6: 8^4 plaquette at β=6.0 matches 4^4 within 5% ───
    let plaq_4_at_6 = result_4x4
        .points
        .iter()
        .find(|p| (p.beta - 6.0).abs() < 0.01)
        .map(|p| p.mean_plaquette);
    let plaq_8_at_6 = result_8x8
        .points
        .iter()
        .find(|p| (p.beta - 6.0).abs() < 0.01)
        .map(|p| p.mean_plaquette);
    if let (Some(p4), Some(p8)) = (plaq_4_at_6, plaq_8_at_6) {
        harness.check_rel(
            "8^4 vs 4^4 plaquette at β=6.0",
            p8,
            p4,
            tolerances::BETA_SCAN_SCALING_PARITY,
        );
        println!(
            "  Scaling: 4^4 plaq={p4:.6}, 8^4 plaq={p8:.6}, diff={:.2}%",
            ((p8 - p4) / p4).abs() * 100.0
        );
    }

    // ─── Check 7: 8^4 acceptance ───
    let accept_8 = result_8x8
        .points
        .iter()
        .all(|p| p.acceptance_rate > tolerances::BETA_SCAN_ACCEPTANCE_MIN);
    harness.check_bool("8^4 HMC acceptance > 30% at all β", accept_8);

    // ─── Check 8: 8^4 plaquette monotonicity ───
    let mono_8 = result_8x8
        .points
        .windows(2)
        .all(|w| w[1].mean_plaquette > w[0].mean_plaquette);
    harness.check_bool("8^4 plaquette monotonic with β", mono_8);
    println!();

    // ═══ Phase 3: Determinism check ═══
    println!("═══ Phase 3: Determinism (rerun 4^4 at β=5.5) ═══");
    {
        let results: Vec<f64> = (0..2)
            .map(|_| {
                let mut lat = Lattice::hot_start([4, 4, 4, 4], 5.5, 42);
                let mut cfg = HmcConfig {
                    n_md_steps: 15,
                    dt: 0.05,
                    seed: 42,
                };
                let stats = hmc::run_hmc(&mut lat, 10, 5, &mut cfg);
                stats.mean_plaquette
            })
            .collect();

        let diff = (results[0] - results[1]).abs();
        println!("  Run 1: {:.15}", results[0]);
        println!("  Run 2: {:.15}", results[1]);
        println!("  Diff:  {diff:.2e}");

        harness.check_abs(
            "determinism: rerun-identical plaquette",
            diff,
            0.0,
            f64::EPSILON,
        );
    }
    println!();

    // ═══ Phase 4: Thermodynamic observables ═══
    println!("═══ Phase 4: Thermodynamic observables ═══");
    {
        // Action density: S/(6V) = 1 - <P>
        // This connects to the trace anomaly in the continuum limit
        println!("  Action density S/(6V) = 1 - <P>:");
        for p in &result_4x4.points {
            let action_density = 1.0 - p.mean_plaquette;
            println!("    β={:.2}: S/(6V) = {action_density:.6}", p.beta);
        }

        // Plaquette susceptibility (variance × volume)
        // χ_P = V × (⟨P²⟩ - ⟨P⟩²) — peaks at β_c
        println!("\n  Plaquette fluctuation σ(P) — should peak near β_c:");
        let mut max_std = 0.0_f64;
        let mut max_std_beta = 0.0;
        for p in &result_4x4.points {
            println!("    β={:.2}: σ(P) = {:.6}", p.beta, p.std_plaquette);
            if p.std_plaquette > max_std {
                max_std = p.std_plaquette;
                max_std_beta = p.beta;
            }
        }

        // The susceptibility should peak in the transition region
        let peak_in_transition = (5.0..=6.5).contains(&max_std_beta);
        harness.check_bool(
            "plaquette susceptibility peaks in transition region",
            peak_in_transition,
        );
        println!("\n  Peak σ(P) at β={max_std_beta:.2} (transition region: 5.0-6.5)");
    }
    println!();

    println!(
        "═══ Summary: {} β-points, 4^4 + 8^4, {:.1}s total ═══",
        result_4x4.points.len() + result_8x8.points.len(),
        result_4x4.total_wall_time_s + result_8x8.total_wall_time_s
    );
    println!();

    harness.finish();
}
