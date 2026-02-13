//! Verification: Rust HFB vs Python reference on SLy4

use hotspring_barracuda::physics::hfb::{binding_energy_l2, SphericalHFB};
use std::time::Instant;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  HFB Verification: Rust vs Python (SLy4)");
    println!("  Physics: p/n channels, Coulomb, T_eff, BCS, CM correction");
    println!("═══════════════════════════════════════════════════════════════");

    let sly4: [f64; 10] = [
        -2488.91, 486.82, -546.39, 13777.0,
        0.834, -0.344, -1.0, 1.354, 0.1667, 123.0,
    ];

    let test_nuclei: Vec<(usize, usize, &str, f64, f64)> = vec![
        (28, 28, "Ni-56",  483.988, 556.031),
        (40, 50, "Zr-90",  783.893, 845.669),
        (50, 82, "Sn-132", 1102.851, 1063.401),
        (82, 126,"Pb-208", 1636.430, 1577.561),
        (50, 62, "Sn-112", 953.531, 948.591),
        (40, 54, "Zr-94",  812.990, 847.982),
    ];

    println!("\n{:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>6} {:>6}",
        "Nucleus", "B_exp", "B_rust", "B_python", "Rust-Exp", "Rust-Py", "Conv", "Time");
    println!("{}", "-".repeat(82));

    let mut total_delta_py = 0.0;
    let mut total_delta_exp = 0.0;
    let mut count_hfb = 0;

    for &(z, n, name, b_exp, b_python) in &test_nuclei {
        let t0 = Instant::now();
        let (b_rust, conv) = binding_energy_l2(z, n, &sly4);
        let dt = t0.elapsed().as_secs_f64();

        let a = z + n;
        let method = if a >= 56 && a <= 132 { "HFB" } else { "SEMF" };
        let delta_exp = b_rust - b_exp;
        let delta_py = b_rust - b_python;

        println!("{:>8} {:>10.1} {:>10.1} {:>10.1} {:>+10.1} {:>+10.1} {:>6} {:.2}s [{}]",
            name, b_exp, b_rust, b_python, delta_exp, delta_py,
            if conv { "yes" } else { "NO" }, dt, method);

        if a >= 56 && a <= 132 {
            total_delta_py += delta_py.abs();
            total_delta_exp += delta_exp.abs();
            count_hfb += 1;
        }
    }

    if count_hfb > 0 {
        let avg_py = total_delta_py / count_hfb as f64;
        let avg_exp = total_delta_exp / count_hfb as f64;
        println!("\nHFB nuclei: mean |Rust-Python| = {:.1} MeV, mean |Rust-Exp| = {:.1} MeV", avg_py, avg_exp);
    }

    // Detailed energy component comparison for Ni-56
    println!("\n--- Detailed HFB for Ni-56 (verbose) ---");
    let hfb56 = SphericalHFB::new_adaptive(28, 28);
    println!("  n_states = {}, nr = {}", hfb56.n_states(), hfb56.nr());
    let _r56 = hfb56.solve_verbose(&sly4, 200, 0.05, 0.3);

    println!("\n--- Detailed HFB for Sn-112 (verbose) ---");
    let hfb112 = SphericalHFB::new_adaptive(50, 62);
    println!("  n_states = {}, nr = {}", hfb112.n_states(), hfb112.nr());
    let _r112 = hfb112.solve_verbose(&sly4, 200, 0.05, 0.3);
}
