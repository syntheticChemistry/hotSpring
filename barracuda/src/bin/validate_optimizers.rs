// SPDX-License-Identifier: AGPL-3.0-only

//! Validate `barracuda::optimize` functions (BFGS, Nelder-Mead, RK45 ODE)
//!
//! Tests classic optimization problems + ODE integration
//! Reference: scipy.optimize, scipy.integrate

use std::f64::consts::PI;

use hotspring_barracuda::provenance;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  BarraCUDA Optimizer & Numerical Methods Validation");
    println!("  Reference: scipy.optimize / scipy.integrate");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut harness = ValidationHarness::new("optimizers_numerical");

    // ─── BFGS: Rosenbrock ─────────────────────────────────────────
    println!("── BFGS: Rosenbrock f(x) = (1-x₀)² + 100(x₁-x₀²)² ──");
    {
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let grad = |x: &[f64]| {
            vec![
                -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
                200.0 * (x[1] - x[0] * x[0]),
            ]
        };

        let config = barracuda::optimize::bfgs::BfgsConfig {
            max_iter: 5000,
            gtol: 1e-8,
            ..Default::default()
        };

        match barracuda::optimize::bfgs::bfgs(&f, &grad, &[-1.0, 1.0], &config) {
            Ok(result) => {
                let ok = result.converged
                    && (result.x[0] - 1.0).abs() < tolerances::BFGS_TOLERANCE
                    && (result.x[1] - 1.0).abs() < tolerances::BFGS_TOLERANCE;
                if ok {
                    println!(
                        "  ✅ Converged to ({:.6}, {:.6}) in {} iters, {} fevals",
                        result.x[0], result.x[1], result.n_iter, result.n_feval
                    );
                } else {
                    println!(
                        "  ❌ Got ({:.6}, {:.6}), converged={}",
                        result.x[0], result.x[1], result.converged
                    );
                }
                harness.check_bool("BFGS Rosenbrock → (1,1)", ok);
            }
            Err(e) => {
                println!("  ❌ BFGS failed: {e}");
                harness.check_bool("BFGS Rosenbrock", false);
            }
        }
    }

    // ─── BFGS with numerical gradient ─────────────────────────────
    println!("\n── BFGS (numerical gradient): Sphere f(x) = Σxᵢ² ──");
    {
        let n = 5;
        let f = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let x0: Vec<f64> = (0..n).map(|i| f64::from(i + 1)).collect();

        let config = barracuda::optimize::bfgs::BfgsConfig::default();
        match barracuda::optimize::bfgs::bfgs_numerical(&f, &x0, &config) {
            Ok(result) => {
                let max_err = result.x.iter().map(|xi| xi.abs()).fold(0.0_f64, f64::max);
                let ok = result.converged && max_err < tolerances::BFGS_TOLERANCE;
                if ok {
                    println!(
                        "  ✅ 5D sphere: converged, max |xᵢ| = {:.2e}, {} fevals",
                        max_err, result.n_feval
                    );
                } else {
                    println!("  ❌ 5D sphere: max |xᵢ| = {max_err:.2e}");
                }
                harness.check_bool("BFGS numerical 5D sphere", ok);
            }
            Err(e) => {
                println!("  ❌ BFGS numerical failed: {e}");
                harness.check_bool("BFGS numerical 5D sphere", false);
            }
        }
    }

    // ─── Nelder-Mead: Rosenbrock ──────────────────────────────────
    println!("\n── Nelder-Mead: Rosenbrock ──");
    {
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let x0 = vec![0.0, 0.0];

        match barracuda::optimize::nelder_mead(f, &x0, &bounds, 5000, 1e-10) {
            Ok((x, fval, n_eval)) => {
                let ok = (x[0] - 1.0).abs() < 0.1 && (x[1] - 1.0).abs() < 0.1;
                if ok {
                    println!(
                        "  ✅ NM Rosenbrock: ({:.4}, {:.4}), f={:.6}, {} evals",
                        x[0], x[1], fval, n_eval
                    );
                } else {
                    println!(
                        "  ❌ NM Rosenbrock: ({:.4}, {:.4}), f={:.6}",
                        x[0], x[1], fval
                    );
                }
                harness.check_bool("NM Rosenbrock", ok);
            }
            Err(e) => {
                println!("  ❌ NM failed: {e}");
                harness.check_bool("NM Rosenbrock", false);
            }
        }
    }

    // ─── Multi-start Nelder-Mead ──────────────────────────────────
    println!("\n── Multi-start Nelder-Mead: Ackley function ──");
    {
        // Ackley: global min at (0, 0) = 0
        let ackley = |x: &[f64]| {
            let n = x.len() as f64;
            let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum::<f64>() / n;
            let sum_cos: f64 = x.iter().map(|xi| (2.0 * PI * xi).cos()).sum::<f64>() / n;
            -20.0 * (-0.2 * sum_sq.sqrt()).exp() - sum_cos.exp() + 20.0 + std::f64::consts::E
        };

        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        match barracuda::optimize::multi_start_nelder_mead(ackley, &bounds, 8, 500, 1e-8, 42) {
            Ok((best, _cache, _all)) => {
                let ok = best.f_best < 1.0;
                if ok {
                    println!(
                        "  ✅ Ackley: best f = {:.6} at ({:.4}, {:.4})",
                        best.f_best, best.x_best[0], best.x_best[1]
                    );
                } else {
                    println!("  ❌ Ackley: best f = {:.6}", best.f_best);
                }
                harness.check_bool("Multi-start NM Ackley", ok);
            }
            Err(e) => {
                println!("  ❌ Multi-start NM failed: {e}");
                harness.check_bool("Multi-start NM Ackley", false);
            }
        }
    }

    // ─── Bisection root-finding ───────────────────────────────────
    println!("\n── Bisection: x² - 2 = 0 → x = √2 ──");
    {
        let f = |x: f64| x * x - 2.0;
        match barracuda::optimize::bisect(f, 1.0, 2.0, 1e-12, 100) {
            Ok(root) => {
                let ok = (root - std::f64::consts::SQRT_2).abs() < tolerances::EXACT_F64;
                if ok {
                    println!("  ✅ √2 = {root:.14}");
                } else {
                    println!(
                        "  ❌ Got {:.14}, expected {:.14}",
                        root,
                        std::f64::consts::SQRT_2
                    );
                }
                harness.check_bool("Bisection sqrt(2)", ok);
            }
            Err(e) => {
                println!("  ❌ Bisect failed: {e}");
                harness.check_bool("Bisection sqrt(2)", false);
            }
        }
    }

    // ─── RK45 ODE: Exponential decay ──────────────────────────────
    println!("\n── RK45 ODE: dy/dt = -y → y(t) = e^(-t) ──");
    {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let config = barracuda::numerical::rk45::Rk45Config::new(1e-8, 1e-10);

        match barracuda::numerical::rk45::rk45_solve(&f, 0.0, 2.0, &[1.0], &config) {
            Ok(result) => {
                let expected = (-2.0_f64).exp();
                let err = (result.y_final[0] - expected).abs();
                let ok = err < tolerances::GPU_VS_CPU_F64;
                if ok {
                    println!(
                        "  ✅ y(2) = {:.10} (expected {:.10}), {} steps",
                        result.y_final[0], expected, result.n_steps
                    );
                } else {
                    println!(
                        "  ❌ y(2) = {:.10} (expected {:.10}), err={:.2e}",
                        result.y_final[0], expected, err
                    );
                }
                harness.check_bool("RK45 exp decay", ok);
            }
            Err(e) => {
                println!("  ❌ RK45 exp decay failed: {e}");
                harness.check_bool("RK45 exp decay", false);
            }
        }
    }

    // ─── RK45 ODE: Harmonic oscillator ────────────────────────────
    println!("\n── RK45 ODE: x'' + x = 0 → x(t) = cos(t) ──");
    {
        // y = [x, v], dy/dt = [v, -x]
        let f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let config = barracuda::numerical::rk45::Rk45Config::new(1e-8, 1e-10);

        match barracuda::numerical::rk45::rk45_solve(&f, 0.0, 2.0 * PI, &[1.0, 0.0], &config) {
            Ok(result) => {
                // After one full period: x(2π) = cos(2π) = 1, v(2π) = -sin(2π) = 0
                let x_err = (result.y_final[0] - 1.0).abs();
                let v_err = result.y_final[1].abs();
                let ok = x_err < tolerances::RK45_TOLERANCE && v_err < tolerances::RK45_TOLERANCE;
                if ok {
                    println!(
                        "  ✅ x(2π)={:.8}, v(2π)={:.8} ({} steps, {} rejected)",
                        result.y_final[0], result.y_final[1], result.n_steps, result.n_rejected
                    );
                } else {
                    println!(
                        "  ❌ x(2π)={:.8} (expect 1), v(2π)={:.8} (expect 0)",
                        result.y_final[0], result.y_final[1]
                    );
                }
                harness.check_bool("RK45 harmonic oscillator", ok);
            }
            Err(e) => {
                println!("  ❌ RK45 harmonic oscillator failed: {e}");
                harness.check_bool("RK45 harmonic oscillator", false);
            }
        }
    }

    // ─── RK45 ODE: Lotka-Volterra (predator-prey) ─────────────────
    println!("\n── RK45 ODE: Lotka-Volterra predator-prey ──");
    {
        let alpha = 1.5;
        let beta = 1.0;
        let delta = 1.0;
        let gamma_lv = 3.0;

        let f = move |_t: f64, y: &[f64]| {
            vec![
                alpha * y[0] - beta * y[0] * y[1],
                delta * y[0] * y[1] - gamma_lv * y[1],
            ]
        };
        let config =
            barracuda::numerical::rk45::Rk45Config::new(1e-6, 1e-8).with_step_bounds(1e-8, 1.0);

        match barracuda::numerical::rk45::rk45_solve(&f, 0.0, 10.0, &[10.0, 5.0], &config) {
            Ok(result) => {
                let ok = result.y_final[0] > 0.0
                    && result.y_final[1] > 0.0
                    && result.y_final[0] < 1e6
                    && result.y_final[1] < 1e6;
                if ok {
                    println!(
                        "  ✅ Prey={:.4}, Pred={:.4} at t=10 ({} steps)",
                        result.y_final[0], result.y_final[1], result.n_steps
                    );
                } else {
                    println!(
                        "  ❌ Unstable: Prey={:.4}, Pred={:.4}",
                        result.y_final[0], result.y_final[1]
                    );
                }
                harness.check_bool("RK45 Lotka-Volterra", ok);
            }
            Err(e) => {
                println!("  ❌ RK45 Lotka-Volterra failed: {e}");
                harness.check_bool("RK45 Lotka-Volterra", false);
            }
        }
    }
    println!();

    // ─── Crank-Nicolson PDE: Heat diffusion ───────────────────────
    println!("── Crank-Nicolson: 1D Heat Diffusion ──");
    {
        // u(x,0) = sin(πx), α=1, L=1
        // Analytical: u(x,t) = exp(-π²t) sin(πx)
        let nx = 101;
        let dx = 1.0 / (nx - 1) as f64;
        let dt = 0.0001;
        let alpha = 1.0;

        let config = barracuda::pde::CrankNicolsonConfig::new(alpha, dx, dt, nx)
            .with_boundary_conditions(0.0, 0.0);

        let initial: Vec<f64> = (0..nx)
            .map(|i| (PI * i as f64 / (nx - 1) as f64).sin())
            .collect();

        match barracuda::pde::HeatEquation1D::new(config, &initial) {
            Ok(mut solver) => {
                let n_steps = 1000; // t = 0.1s
                let t_final = n_steps as f64 * dt;
                let _ = solver.advance(n_steps);

                let sol = solver.solution();
                let mid_idx = nx / 2;
                let expected = (-PI * PI * t_final).exp(); // sin(π/2) = 1

                let err = (sol[mid_idx] - expected).abs();
                let ok = err < tolerances::SOBOL_TOLERANCE;
                if ok {
                    println!(
                        "  ✅ u(0.5, {:.1}) = {:.6} (analytical {:.6}), err={:.2e}",
                        t_final, sol[mid_idx], expected, err
                    );
                } else {
                    println!(
                        "  ❌ u(0.5, {:.1}) = {:.6} (analytical {:.6}), err={:.2e}",
                        t_final, sol[mid_idx], expected, err
                    );
                }
                harness.check_bool("Crank-Nicolson midpoint", ok);

                // Test stability: should not oscillate
                // The true solution is smooth, so minor boundary effects are ok
                // but check the central region
                let central = &sol[10..nx - 10];
                let mut central_oscillations = 0;
                for w in central.windows(3) {
                    if (w[1] > w[0] + tolerances::EXACT_F64 && w[1] > w[2] + tolerances::EXACT_F64)
                        || (w[1] + tolerances::EXACT_F64 < w[0]
                            && w[1] + tolerances::EXACT_F64 < w[2])
                    {
                        central_oscillations += 1;
                    }
                }
                let stable_ok = central_oscillations <= 1;
                if central_oscillations == 0 {
                    println!("  ✅ No spurious oscillations (stable)");
                } else if central_oscillations <= 1 {
                    println!("  ✅ {central_oscillations} minor oscillation (acceptable at peak)");
                } else {
                    println!("  ❌ {central_oscillations} central oscillations detected");
                }
                harness.check_bool("Crank-Nicolson stability", stable_ok);
            }
            Err(e) => {
                println!("  ❌ CN solver failed: {e}");
                harness.check_bool("Crank-Nicolson midpoint", false);
                harness.check_bool("Crank-Nicolson stability", false);
            }
        }
    }
    println!();

    // ─── Statistics: Normal CDF/PPF ───────────────────────────────
    println!("── Statistics: Normal Distribution ──");
    {
        // Normal CDF
        let cdf_tests = vec![
            (0.0, 0.5, "Φ(0) = 0.5"),
            (1.96, 0.9750021049, "Φ(1.96) ≈ 0.975"),
            (-1.96, 0.0249978951, "Φ(-1.96) ≈ 0.025"),
            (3.0, 0.9986501020, "Φ(3) ≈ 0.9987"),
        ];
        for (x, expected, desc) in &cdf_tests {
            let got = barracuda::stats::norm_cdf(*x);
            let err = (got - expected).abs();
            let ok = err < 1e-4;
            if ok {
                println!("  ✅ {desc} | {got:.8} (expected {expected:.8})");
            } else {
                println!("  ❌ {desc} | {got:.8} (expected {expected:.8}), err={err:.2e}");
            }
            harness.check_bool(desc, ok);
        }

        // Normal PPF (inverse CDF)
        let ppf_tests = vec![
            (0.5, 0.0, "Φ⁻¹(0.5) = 0"),
            (0.975, 1.959964, "Φ⁻¹(0.975) ≈ 1.96"),
            (0.025, -1.959964, "Φ⁻¹(0.025) ≈ -1.96"),
            (0.999, 3.090232, "Φ⁻¹(0.999) ≈ 3.09"),
        ];
        for (p, expected, desc) in &ppf_tests {
            let got = barracuda::stats::norm_ppf(*p);
            let err = (got - expected).abs();
            let ok = err < 1e-3;
            if ok {
                println!("  ✅ {desc} | {got:.6} (expected {expected:.6})");
            } else {
                println!("  ❌ {desc} | {got:.6} (expected {expected:.6}), err={err:.2e}");
            }
            harness.check_bool(desc, ok);
        }
    }
    println!();

    // ─── Statistics: Correlation ──────────────────────────────────
    println!("── Statistics: Pearson Correlation ──");
    {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        match barracuda::stats::pearson_correlation(&x, &y) {
            Ok(r) => {
                let ok = (r - 1.0).abs() < tolerances::EXACT_F64;
                if ok {
                    println!("  ✅ Perfect positive: r = {r:.10}");
                } else {
                    println!("  ❌ Perfect positive: r = {r:.10} (expected 1.0)");
                }
                harness.check_bool("Pearson perfect positive", ok);
            }
            Err(e) => {
                println!("  ❌ pearson_correlation failed: {e}");
                harness.check_bool("Pearson perfect positive", false);
            }
        }

        // Zero correlation (orthogonal)
        let x2 = vec![1.0, 0.0, -1.0, 0.0];
        let y2 = vec![0.0, 1.0, 0.0, -1.0];
        match barracuda::stats::pearson_correlation(&x2, &y2) {
            Ok(r2) => {
                let ok = r2.abs() < tolerances::EXACT_F64;
                if ok {
                    println!("  ✅ Orthogonal: r = {r2:.10}");
                } else {
                    println!("  ❌ Orthogonal: r = {r2:.10} (expected 0.0)");
                }
                harness.check_bool("Pearson orthogonal", ok);
            }
            Err(e) => {
                println!("  ❌ pearson_correlation failed: {e}");
                harness.check_bool("Pearson orthogonal", false);
            }
        }
    }
    println!();

    // ─── Sobol sequences ─────────────────────────────────────────
    println!("── Sampling: Sobol Quasi-Random ──");
    {
        match barracuda::sample::sobol::sobol_sequence(100, 3) {
            Ok(points) => {
                // All points should be in [0, 1]
                let all_in_unit = points
                    .iter()
                    .all(|p| p.iter().all(|&v| (0.0..=1.0).contains(&v)));
                let ok = all_in_unit && points.len() == 100 && points[0].len() == 3;
                if ok {
                    println!(
                        "  ✅ 100 points in 3D, all in [0,1]³, first: ({:.4}, {:.4}, {:.4})",
                        points[0][0], points[0][1], points[0][2]
                    );
                } else {
                    println!(
                        "  ❌ Points outside [0,1] or wrong dimensions: {} × {}",
                        points.len(),
                        if points.is_empty() {
                            0
                        } else {
                            points[0].len()
                        }
                    );
                }
                harness.check_bool("Sobol in-unit", ok);

                // Check uniformity: mean should ≈ 0.5 for each dimension
                let mut means_ok = true;
                for d in 0..3 {
                    let mean: f64 = points.iter().map(|p| p[d]).sum::<f64>() / points.len() as f64;
                    if (mean - 0.5).abs() > 0.1 {
                        means_ok = false;
                    }
                }
                if means_ok {
                    println!("  ✅ Sobol mean ≈ 0.5 per dimension (good uniformity)");
                } else {
                    println!("  ❌ Sobol mean deviates from 0.5");
                }
                harness.check_bool("Sobol uniformity", means_ok);
            }
            Err(e) => {
                println!("  ❌ Sobol generation failed: {e}");
                harness.check_bool("Sobol in-unit", false);
                harness.check_bool("Sobol uniformity", false);
            }
        }
    }
    println!();

    // ─── Summary (with exit code) ──────────────────────────────────
    println!("\n  Reference: {}", provenance::OPTIMIZER_REFS);
    harness.finish();
}
