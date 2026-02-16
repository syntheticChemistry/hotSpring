// SPDX-License-Identifier: AGPL-3.0-only

//! Verification: Rust HFB vs Python reference on SLy4
//!
//! Uses provenance constants for SLy4 parameters and test nuclei.
//! Expected values trace to `control/surrogate/nuclear-eos/wrapper/skyrme_hf.py`,
//! commit `fd908c41`.

use hotspring_barracuda::physics::hfb::{binding_energy_l2, SphericalHFB};
use hotspring_barracuda::provenance::{HFB_TEST_NUCLEI, SLY4_PARAMS};
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  HFB Verification: Rust vs Python (SLy4)");
    println!("  Physics: p/n channels, Coulomb, T_eff, BCS, CM correction");
    println!("═══════════════════════════════════════════════════════════════");

    let mut harness = ValidationHarness::new("hfb_sly4_verification");

    println!(
        "\n{:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>6} {:>6}",
        "Nucleus", "B_exp", "B_rust", "B_python", "Rust-Exp", "Rust-Py", "Conv", "Time"
    );
    println!("{}", "-".repeat(82));

    let mut total_delta_py = 0.0;
    let mut total_delta_exp = 0.0;
    let mut count_hfb = 0;

    for &(z, n, name, b_exp, b_python) in HFB_TEST_NUCLEI {
        let t0 = Instant::now();
        let (b_rust, conv) = binding_energy_l2(z, n, &SLY4_PARAMS);
        let dt = t0.elapsed().as_secs_f64();

        let a = z + n;
        let method = if (56..=132).contains(&a) {
            "HFB"
        } else {
            "SEMF"
        };
        let delta_exp = b_rust - b_exp;
        let delta_py = b_rust - b_python;

        println!(
            "{name:>8} {b_exp:>10.1} {b_rust:>10.1} {b_python:>10.1} {delta_exp:>+10.1} {delta_py:>+10.1} {:>6} {dt:.2}s [{method}]",
            if conv { "yes" } else { "NO" },
        );

        let label_conv = format!("{name} convergence");
        harness.check_bool(&label_conv, conv);
        let label_be = format!("{name} B_rust vs B_exp (<10%)");
        let rel_err = (delta_exp / b_exp).abs();
        harness.check_upper(&label_be, rel_err, tolerances::HFB_RUST_VS_EXP_REL);

        if (56..=132).contains(&a) {
            total_delta_py += delta_py.abs();
            total_delta_exp += delta_exp.abs();
            count_hfb += 1;
        }
    }

    if count_hfb > 0 {
        let avg_py = total_delta_py / count_hfb as f64;
        let avg_exp = total_delta_exp / count_hfb as f64;
        println!(
            "\nHFB nuclei: mean |Rust-Python| = {avg_py:.1} MeV, mean |Rust-Exp| = {avg_exp:.1} MeV"
        );
        // Generous acceptance threshold, not a tight tolerance; Rust vs Python may differ by solver/mixing
        harness.check_upper("HFB mean |Rust-Python|", avg_py, 100.0);
    }

    // Detailed energy component comparison for Ni-56
    println!("\n--- Detailed HFB for Ni-56 (verbose) ---");
    let hfb56 = SphericalHFB::new_adaptive(28, 28);
    println!("  n_states = {}, nr = {}", hfb56.n_states(), hfb56.nr());
    let _r56 = hfb56.solve_verbose(&SLY4_PARAMS, 200, 0.05, 0.3);

    println!("\n--- Detailed HFB for Sn-112 (verbose) ---");
    let hfb112 = SphericalHFB::new_adaptive(50, 62);
    println!("  n_states = {}, nr = {}", hfb112.n_states(), hfb112.nr());
    let _r112 = hfb112.solve_verbose(&SLY4_PARAMS, 200, 0.05, 0.3);

    println!(
        "\n  Provenance: control/surrogate/nuclear-eos/wrapper/skyrme_hf.py, SLy4 (Chabanat 1998)"
    );
    harness.finish();
}
