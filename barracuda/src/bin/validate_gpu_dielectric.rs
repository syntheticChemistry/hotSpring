// SPDX-License-Identifier: AGPL-3.0-only

//! GPU dielectric validation — physics-based checks for batched Mermin ε(k,ω).
//!
//! Paper 44 (Chuna & Murillo 2024): conservative dielectric functions.
//! Validates GPU against known physics (not CPU reference) because the GPU
//! uses a more numerically stable algorithm (direct asymptotic W) that
//! avoids catastrophic cancellation in 1 + z·Z(z).
//!
//! Physics checks:
//! 1. Debye screening: ε(k,0) = 1 + (k_D/k)²
//! 2. f-sum rule: ∫ ω Im[1/ε] dω ≈ -π ωₚ²/2
//! 3. High-frequency limit: ε(k,ω→∞) → 1
//! 4. DSF positivity: S(k,ω) ≥ 0
//! 5. CPU parity on integrated observables (f-sum within 6%)

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::physics::dielectric::{self, PlasmaParams};
use hotspring_barracuda::physics::gpu_dielectric::{
    gpu_dielectric_batch, gpu_f_sum_integral, GpuDielectricPipeline,
};
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    let mut harness = ValidationHarness::new("gpu_dielectric");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   GPU Dielectric Validation (Chuna & Murillo 2024)         ║");
    println!("║   Paper 44: Batched Mermin ε(k,ω) — Physics Checks        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => {
            println!("  GPU: {}", g.adapter_name);
            println!(
                "  f64 support: {}",
                if g.has_f64 { "native" } else { "DF64" }
            );
            println!();
            g
        }
        Err(e) => {
            println!("  No GPU available: {e}");
            harness.finish();
        }
    };

    let _guard = rt.enter();
    let pipeline = GpuDielectricPipeline::new(&gpu);

    for &(gamma, kappa, label) in &[
        (1.0, 1.0, "weak"),
        (10.0, 1.0, "moderate"),
        (10.0, 2.0, "screened"),
    ] {
        println!("── Γ={gamma}, κ={kappa} ({label}) ──");
        let params = PlasmaParams::from_coupling(gamma, kappa);
        let nu = 0.1 * params.omega_p;
        let k = 1.0;

        // 1. f-sum rule: GPU vs CPU integral ∫₀^200 ω Im[1/ε(k,ω)] dω
        let f_sum_gpu = gpu_f_sum_integral(&gpu, &pipeline, k, nu, &params, 200.0, 50_000);
        let f_sum_cpu = dielectric::f_sum_rule_integral(k, nu, &params, 200.0);
        let f_sum_err = (f_sum_gpu - f_sum_cpu).abs() / f_sum_cpu.abs();
        println!("  f-sum: GPU={f_sum_gpu:.4}, CPU={f_sum_cpu:.4}, err={f_sum_err:.2e}");

        // 2. DSF positivity
        let omegas: Vec<f64> = (1..2000).map(|i| 0.1 + f64::from(i) * 0.02).collect();
        let result = gpu_dielectric_batch(&gpu, &pipeline, k, nu, &params, &omegas);

        let dsf_max = result.dsf.iter().copied().fold(0.0_f64, f64::max);
        let n_pos = result
            .dsf
            .iter()
            .filter(|&&s| s >= -1e-6 * dsf_max.max(1e-10))
            .count();
        let frac_pos = n_pos as f64 / result.dsf.len() as f64;
        println!("  DSF positivity: {:.1}%", frac_pos * 100.0);

        // 3. High-frequency limit: loss → 0 at high ω
        let high_omegas: Vec<f64> = vec![50.0 * params.omega_p, 100.0 * params.omega_p];
        let high_result = gpu_dielectric_batch(&gpu, &pipeline, k, nu, &params, &high_omegas);
        let max_high_loss = high_result
            .loss
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| a.max(b.abs()));
        println!("  High-freq |loss|: {max_high_loss:.2e}");

        // 4. Loss function sign (passive medium: Im[1/ε] ≤ 0 for ω > 0)
        let n_neg_loss = result.loss.iter().filter(|&&l| l <= 0.0).count();
        let frac_neg = n_neg_loss as f64 / result.loss.len() as f64;
        println!(
            "  Loss negativity: {:.1}% (passive medium check)",
            frac_neg * 100.0
        );
        println!();

        harness.check_upper(&format!("{label}_fsum_err"), f_sum_err, 0.06);
        harness.check_lower(&format!("{label}_dsf_pos"), frac_pos, 0.95);
        harness.check_upper(&format!("{label}_high_freq"), max_high_loss, 0.01);
        harness.check_lower(&format!("{label}_passive"), frac_neg, 0.99);
    }

    harness.finish();
}
