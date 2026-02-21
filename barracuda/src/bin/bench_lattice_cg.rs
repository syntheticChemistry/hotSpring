// SPDX-License-Identifier: AGPL-3.0-only

//! Lattice QCD CG Benchmark — Rust CPU vs Python baseline
//!
//! Runs the same CG solver (D†D x = b) as the Python control with
//! identical parameters, LCG seeds, and tolerance, then reports:
//!   - Iteration count (should match Python exactly)
//!   - Residual (should match Python exactly)
//!   - Wall time (should be significantly faster than Python)
//!
//! This proves: "barracuda CPU is pure math and faster than interpreted language."

use hotspring_barracuda::lattice::cg::cg_solve;
use hotspring_barracuda::lattice::dirac::{apply_dirac, apply_dirac_sq, FermionField};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Lattice QCD CG Benchmark — Rust CPU                      ║");
    println!("║  Same algorithm, same seeds as Python control              ║");
    println!("║  Proving: pure Rust math is faster than interpreted        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Cold lattice CG (4⁴, mass=1.0) ─────────────────────────────
    println!("═══ Cold lattice CG (4⁴, mass=1.0, tol=1e-8) ════════════════");
    {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let vol = lat.volume();
        let b = FermionField::random(vol, 42);
        let mut x = FermionField::zeros(vol);

        let t0 = Instant::now();
        let result = cg_solve(&lat, &mut x, &b, 1.0, 1e-8, 500);
        let dt = t0.elapsed();

        println!(
            "  iters={}, residual={:.2e}, time={:.3}ms",
            result.iterations,
            result.final_residual,
            dt.as_secs_f64() * 1000.0
        );

        let ax = apply_dirac_sq(&lat, &x, 1.0);
        let max_diff: f64 = ax
            .data
            .iter()
            .zip(b.data.iter())
            .flat_map(|(a, b)| {
                (0..3).map(move |c| ((a[c].re - b[c].re).abs()).max((a[c].im - b[c].im).abs()))
            })
            .fold(0.0, f64::max);
        println!("  max |D†D x - b|: {max_diff:.2e}");
    }

    // ── Hot lattice CG (4⁴, mass=0.5) ──────────────────────────────
    println!();
    println!("═══ Hot lattice CG (4⁴, mass=0.5, tol=1e-6) ═════════════════");
    {
        let lat = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let vol = lat.volume();
        let b = FermionField::random(vol, 99);
        let mut x = FermionField::zeros(vol);

        let t0 = Instant::now();
        let result = cg_solve(&lat, &mut x, &b, 0.5, 1e-6, 2000);
        let dt = t0.elapsed();

        println!(
            "  iters={}, residual={:.2e}, time={:.3}ms",
            result.iterations,
            result.final_residual,
            dt.as_secs_f64() * 1000.0
        );
    }

    // ── Dirac apply benchmark (4⁴, 100 iterations) ─────────────────
    println!();
    println!("═══ Dirac apply benchmark (4⁴, 100 applies) ══════════════════");
    {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let vol = lat.volume();
        let psi = FermionField::random(vol, 77);

        let n_reps = 100;
        let t0 = Instant::now();
        for _ in 0..n_reps {
            let _ = apply_dirac(&lat, &psi, 0.5);
        }
        let dt = t0.elapsed();
        let per_apply_ms = dt.as_secs_f64() * 1000.0 / n_reps as f64;
        println!(
            "  {n_reps} Dirac applies: {:.1}ms total, {per_apply_ms:.4}ms/apply",
            dt.as_secs_f64() * 1000.0
        );
    }

    // ── Larger lattice benchmark (6⁴ and 8⁴) ───────────────────────
    println!();
    println!("═══ Scaling: CG on larger lattices ═══════════════════════════");
    for dims in [[6, 6, 6, 6], [8, 8, 8, 4]] {
        let lat = Lattice::hot_start(dims, 6.0, 42);
        let vol = lat.volume();
        let b = FermionField::random(vol, 99);
        let mut x = FermionField::zeros(vol);

        let t0 = Instant::now();
        let result = cg_solve(&lat, &mut x, &b, 0.5, 1e-6, 5000);
        let dt = t0.elapsed();

        let label = format!("{}×{}×{}×{}", dims[0], dims[1], dims[2], dims[3]);
        println!(
            "  {label} (V={}): iters={}, res={:.2e}, time={:.1}ms",
            vol,
            result.iterations,
            result.final_residual,
            dt.as_secs_f64() * 1000.0
        );
    }

    println!();
    println!("═══ Summary ══════════════════════════════════════════════════");
    println!("  Compare iterations and residuals with Python control:");
    println!("  control/lattice_qcd/scripts/lattice_cg_control.py");
    println!("  Iterations and residuals should match EXACTLY.");
    println!("  Rust wall time should be significantly faster.");
    println!();
}
