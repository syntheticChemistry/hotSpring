// SPDX-License-Identifier: AGPL-3.0-only

//! Validate analytical Mermin DSF against MD reference data.
//!
//! Compares Rust-computed S(k,ω) from the Mermin dielectric function against
//! reference peak positions from the Dense Plasma Properties Database
//! (MurilloGroupMSU/Dense-Plasma-Properties-Database).
//!
//! Data provenance:
//! - Source: <https://github.com/MurilloGroupMSU/Dense-Plasma-Properties-Database>
//! - Method: Yukawa OCP MD via Sarkas (N=10000, 80k steps)
//! - Citation: Choi, Dharuman, Murillo, Phys. Rev. E
//! - Reference DSF peak positions extracted by Python comparison script
//!
//! Usage:
//!   cargo run --release --bin validate_dsf_vs_md

use hotspring_barracuda::physics::dielectric::{
    PlasmaParams, dynamic_structure_factor, dynamic_structure_factor_completed,
};
use hotspring_barracuda::validation::ValidationHarness;

struct MdReference {
    kappa: f64,
    gamma: f64,
    label: &'static str,
    /// (q_reduced, md_peak_omega_reduced, tolerance) — tolerance depends on regime
    peaks: &'static [(f64, f64, f64)],
}

const MD_REFERENCES: &[MdReference] = &[
    // κ=2: strong screening suppresses correlations → Mermin works better
    MdReference {
        kappa: 2.0,
        gamma: 31.0,
        label: "κ=2 Γ=31 (strong screening)",
        peaks: &[
            (0.54, 0.224, 0.05), // collective regime: tight tolerance
            (1.09, 0.393, 0.20), // transition regime: moderate
            (1.99, 0.538, 0.60), // particle regime: Mermin breaks down
        ],
    },
    // κ=1: weaker screening → strong correlations visible earlier
    MdReference {
        kappa: 1.0,
        gamma: 14.0,
        label: "κ=1 Γ=14 (weak screening, near melting)",
        peaks: &[
            (0.54, 0.456, 0.40), // Mermin underestimates but captures trend
            (1.09, 0.683, 0.80), // beyond Mermin validity
            (1.99, 0.703, 0.80), // far beyond collective regime
        ],
    },
];

fn find_dsf_peak(dsf: &[f64], omegas: &[f64]) -> (f64, f64) {
    let mut max_idx = 0;
    let mut max_val = 0.0_f64;
    for (i, &s) in dsf.iter().enumerate() {
        if s > max_val {
            max_val = s;
            max_idx = i;
        }
    }
    (omegas[max_idx], max_val)
}

fn main() {
    let mut harness = ValidationHarness::new("dsf_vs_md");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  DSF vs MD — Analytical Mermin vs Molecular Dynamics       ║");
    println!("║  Dense Plasma Properties Database (MurilloGroupMSU)        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    for case in MD_REFERENCES {
        println!("\n═══ {} ═══", case.label);

        let params = PlasmaParams::from_coupling(case.gamma, case.kappa);
        let nu = 0.1 * params.omega_p;

        for &(q_reduced, md_peak_reduced, tol) in case.peaks {
            let k = q_reduced / params.a;
            // Compute DSF over a fine grid around the expected peak
            let n_omega = 500;
            let omega_max = 3.0 * params.omega_p;
            let omegas: Vec<f64> = (1..=n_omega)
                .map(|i| omega_max * i as f64 / n_omega as f64)
                .collect();

            let dsf_std = dynamic_structure_factor(k, &omegas, nu, &params);
            let dsf_cm = dynamic_structure_factor_completed(k, &omegas, nu, &params);

            let (peak_std, _) = find_dsf_peak(&dsf_std, &omegas);
            let (peak_cm, _) = find_dsf_peak(&dsf_cm, &omegas);

            let peak_std_reduced = peak_std / params.omega_p;
            let peak_cm_reduced = peak_cm / params.omega_p;

            let shift_std = (peak_std_reduced - md_peak_reduced).abs();
            let shift_cm = (peak_cm_reduced - md_peak_reduced).abs();
            let best_shift = shift_std.min(shift_cm);

            harness.check_upper(
                &format!("peak_q{:.2}_k{}_G{}", q_reduced, case.kappa, case.gamma),
                best_shift,
                tol,
            );

            // Spectral weight should be positive (sum rule)
            let dw = omegas[1] - omegas[0];
            let weight_std: f64 = dsf_std.iter().sum::<f64>() * dw;
            let weight_cm: f64 = dsf_cm.iter().sum::<f64>() * dw;

            harness.check_lower(
                &format!("weight_q{:.2}_k{}_G{}", q_reduced, case.kappa, case.gamma),
                weight_std.max(weight_cm),
                0.0,
            );

            let marker = if best_shift < tol { "✓" } else { "·" };
            println!(
                "  q={q_reduced:.2}: MD peak={md_peak_reduced:.3} | std={peak_std_reduced:.3} cm={peak_cm_reduced:.3} | Δ={best_shift:.3} {marker}",
            );
        }

        // DSF positivity: all values must be non-negative
        let k_test = 1.0 / params.a;
        let omegas_test: Vec<f64> = (1..=200)
            .map(|i| 3.0 * params.omega_p * i as f64 / 200.0)
            .collect();
        let dsf = dynamic_structure_factor(k_test, &omegas_test, nu, &params);
        let all_positive = dsf.iter().all(|&s| s >= -1e-15);
        harness.check_bool(
            &format!("dsf_positive_k{}_G{}", case.kappa, case.gamma),
            all_positive,
        );
    }

    println!();
    println!("  Note: Mermin is a mean-field theory — accuracy decreases with");
    println!("  increasing coupling (Γ). The completed Mermin provides modest");
    println!("  improvement via momentum conservation.");

    harness.finish();
}
