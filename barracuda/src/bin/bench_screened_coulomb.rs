// SPDX-License-Identifier: AGPL-3.0-only

//! Screened Coulomb performance benchmark: pure Rust vs Python/scipy.
//!
//! Measures wall-clock time for the same computations performed by the
//! Python control (`scipy.linalg.eigh_tridiagonal`), demonstrating that
//! pure Rust math (Sturm bisection) matches or exceeds compiled LAPACK.
//!
//! | Computation        | Python/scipy     | Rust/barracuda     |
//! |--------------------|------------------|--------------------|
//! | Eigensolve N=2000  | ~50 ms (LAPACK)  | measured here      |
//! | κ_c(1s) bisection  | ~4100 ms         | measured here      |
//! | Full validation    | ~13000 ms        | measured here      |

use hotspring_barracuda::physics::screened_coulomb::{
    self, critical_screening, eigenvalues, DEFAULT_N_GRID, DEFAULT_R_MAX,
};

use std::time::Instant;

fn bench<F: Fn() -> R, R>(label: &str, n_iter: usize, f: F) -> f64 {
    let mut times = Vec::with_capacity(n_iter);
    for _ in 0..n_iter {
        let t = Instant::now();
        std::hint::black_box(f());
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let std = if times.len() > 1 {
        let var = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (times.len() - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };
    println!("  {label:35}  {mean:8.2} ± {std:5.2} ms");
    mean
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Screened Coulomb Benchmark: Pure Rust (Sturm bisection)");
    println!("  vs Python/scipy reference timing");
    println!("═══════════════════════════════════════════════════════════════════");

    // Python reference timings (from bench_eigenvalues.py)
    let py_single_ms = 50.0;
    let py_critical_1s_ms = 4116.0;
    let py_full_ms = 12965.0;

    println!("\n── Single eigensolve (N={DEFAULT_N_GRID}, r_max={DEFAULT_R_MAX}) ──");
    let rust_single = bench("H l=0 κ=0", 20, || {
        eigenvalues(1.0, 0.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX)
    });
    bench("H l=0 κ=0.5", 20, || {
        eigenvalues(1.0, 0.5, 0, DEFAULT_N_GRID, DEFAULT_R_MAX)
    });
    bench("H l=0 κ=1.0", 20, || {
        eigenvalues(1.0, 1.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX)
    });
    bench("He+ l=0 κ=0", 20, || {
        eigenvalues(2.0, 0.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX)
    });

    println!("\n── Critical screening (bisection) ──");
    let rust_critical = bench("κ_c(1s)", 5, || {
        critical_screening(1.0, 1, 0, DEFAULT_N_GRID, DEFAULT_R_MAX)
    });
    bench("κ_c(2s)", 5, || {
        critical_screening(1.0, 2, 0, DEFAULT_N_GRID, DEFAULT_R_MAX)
    });
    bench("κ_c(2p)", 5, || {
        critical_screening(1.0, 2, 1, DEFAULT_N_GRID, DEFAULT_R_MAX)
    });

    println!("\n── Grid-size scaling (H l=0 κ=0) ──");
    for &n in &[500, 1000, 2000, 5000, 10000, 20000] {
        bench(&format!("N={n}"), 10, || {
            eigenvalues(1.0, 0.0, 0, n, DEFAULT_R_MAX)
        });
    }

    println!("\n── Full validation equivalent ──");
    let rust_full_t = Instant::now();

    // Match the Python benchmark: all eigensolve + critical screening.
    // black_box prevents dead-code elimination.
    let _ = std::hint::black_box(eigenvalues(1.0, 0.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX));
    let _ = std::hint::black_box(eigenvalues(1.0, 0.0, 1, DEFAULT_N_GRID, DEFAULT_R_MAX));
    let _ = std::hint::black_box(eigenvalues(1.0, 0.1, 0, DEFAULT_N_GRID, DEFAULT_R_MAX));
    let _ = std::hint::black_box(eigenvalues(1.0, 0.5, 0, DEFAULT_N_GRID, DEFAULT_R_MAX));
    let _ = std::hint::black_box(eigenvalues(1.0, 1.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX));
    let _ = std::hint::black_box(eigenvalues(2.0, 0.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX));
    let _ = std::hint::black_box(eigenvalues(1.0, 0.1, 1, DEFAULT_N_GRID, DEFAULT_R_MAX));
    let _ = std::hint::black_box(screened_coulomb::bound_state_count(
        1.0,
        5.0,
        0,
        DEFAULT_N_GRID,
        DEFAULT_R_MAX,
    ));
    let _ = std::hint::black_box(critical_screening(1.0, 1, 0, DEFAULT_N_GRID, DEFAULT_R_MAX));
    let _ = std::hint::black_box(critical_screening(1.0, 2, 0, DEFAULT_N_GRID, DEFAULT_R_MAX));
    let _ = std::hint::black_box(critical_screening(1.0, 2, 1, DEFAULT_N_GRID, DEFAULT_R_MAX));

    let rust_full = rust_full_t.elapsed().as_secs_f64() * 1000.0;
    println!("  Full validation equivalent:      {rust_full:8.1} ms");

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  SPEEDUP SUMMARY (Rust vs Python/scipy)");
    println!("───────────────────────────────────────────────────────────────────");
    println!(
        "  Single eigensolve:  Rust {rust_single:.2} ms vs Python {py_single_ms:.0} ms  → {:.0}× faster",
        py_single_ms / rust_single
    );
    println!(
        "  Critical screening: Rust {rust_critical:.1} ms vs Python {py_critical_1s_ms:.0} ms  → {:.0}× faster",
        py_critical_1s_ms / rust_critical
    );
    println!(
        "  Full validation:    Rust {rust_full:.1} ms vs Python {py_full_ms:.0} ms  → {:.0}× faster",
        py_full_ms / rust_full
    );
    println!("═══════════════════════════════════════════════════════════════════");
}
