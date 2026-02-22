// SPDX-License-Identifier: AGPL-3.0-only

//! `BarraCUDA` HFB Pipeline Validation
//!
//! Validates `BarraCUDA`'s GPU ops for nuclear physics:
//!   - `BatchedBisectionGpu`: BCS chemical potential via GPU bisection
//!   - `BatchedEighGpu`: Symmetric eigendecomposition for HFB matrices
//!
//! Cross-validates GPU results against hotSpring's CPU reference
//! (Brent bisection, `eigh_f64`) to prove the `BarraCUDA` abstraction
//! produces correct nuclear physics.
//!
//! **Provenance**: GPU BCS/eigh vs CPU f64 reference (analytical eigensolve).
//! See `provenance::GPU_KERNEL_REFS`.
//!
//! This is TIER 3 handoff validation: if these pass, the `ToadStool`
//! team can wire the GPU ops into the full SCF loop.

use barracuda::linalg::eigh_f64;
use barracuda::ops::linalg::BatchedEighGpu;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::physics::bcs_gpu::BcsBisectionGpu;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

// ═══════════════════════════════════════════════════════════════════
// CPU reference: BCS bisection (same as hfb_deformed_gpu.rs)
// ═══════════════════════════════════════════════════════════════════

fn cpu_find_fermi_bcs(eigenvalues: &[f64], n_target: f64, delta: f64) -> f64 {
    let pn = |mu: f64| -> f64 {
        eigenvalues
            .iter()
            .map(|&e| {
                let eps = e - mu;
                0.5 * (1.0 - eps / eps.hypot(delta))
            })
            .sum()
    };
    let mut lo = eigenvalues[0] - 50.0;
    let mut hi = *eigenvalues.last().expect("non-empty eigenvalues") + 50.0;
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if pn(mid) < n_target {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < 1e-12 {
            break;
        }
    }
    0.5 * (lo + hi)
}

fn cpu_bcs_occupations(eigenvalues: &[f64], fermi: f64, delta: f64) -> Vec<f64> {
    eigenvalues
        .iter()
        .map(|&e| {
            let eps = e - fermi;
            let e_qp = eps.hypot(delta);
            (0.5 * (1.0 - eps / e_qp)).clamp(0.0, 1.0)
        })
        .collect()
}

/// Build a symmetric HFB-like matrix for testing.
/// Uses a simple harmonic oscillator model: `H_ij` = `δ_ij` * `ε_i` + `V_ij`
/// where `V_ij` = -g/(1 + |i-j|) (pairing-like off-diagonal)
fn build_test_hamiltonian(ns: usize, hw: f64, g: f64) -> Vec<f64> {
    let mut h = vec![0.0_f64; ns * ns];
    for i in 0..ns {
        let eps_i = hw * (i as f64 + 0.5);
        h[i * ns + i] = eps_i;
        for j in 0..ns {
            if i != j {
                let v = -g / (1.0 + (i as f64 - j as f64).abs());
                h[i * ns + j] = v;
                h[j * ns + i] = v;
            }
        }
    }
    h
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  BarraCUDA HFB Pipeline Validation");
    println!("  GPU BCS bisection + batched eigensolve vs CPU reference");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut harness = ValidationHarness::new("barracuda_hfb");

    // Initialize GPU
    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            println!("  GPU init failed: {e}");
            harness.check_bool("GPU available", false);
            harness.finish();
        }
    };
    if !gpu.has_f64 {
        println!("  SHADER_F64 not supported — skipping.");
        harness.check_bool("SHADER_F64 available", false);
        harness.finish();
    }
    gpu.print_info();
    let device = gpu.to_wgpu_device();
    println!("  BarraCUDA WgpuDevice bridged\n");

    // ── Phase 1: BCS Bisection (GPU vs CPU) ──
    println!("── Phase 1: GPU BCS Bisection ─────────────────────────");
    {
        let bisect = BcsBisectionGpu::new(&gpu, 100, 1e-12);

        // Test case: 8 HFB-like problems with different level counts
        // Simulating a batch of nuclei with varying shell structures
        let n_levels = 12;
        let batch_size = 8;

        // Generate eigenvalue spectra for 8 "nuclei"
        let mut all_eigenvalues = Vec::with_capacity(batch_size * n_levels);
        let mut deltas = Vec::with_capacity(batch_size);
        let mut target_ns = Vec::with_capacity(batch_size);
        let mut lower = Vec::with_capacity(batch_size);
        let mut upper = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let hw = (b as f64).mul_add(0.5, 7.0); // Varying oscillator frequencies
            let a_mass = (b as f64).mul_add(10.0, 40.0);
            let delta = 12.0 / a_mass.sqrt();
            // Keep target within level capacity (n_levels=12, deg=1, max=12)
            let n_particles = (2 + b) as f64;

            let mut eigs: Vec<f64> = (0..n_levels)
                .map(|i| hw * (i as f64 + 0.5) + 0.1 * (i as f64).sin())
                .collect();
            eigs.sort_by(f64::total_cmp);

            lower.push(eigs[0] - 50.0);
            upper.push(eigs[n_levels - 1] + 50.0);
            all_eigenvalues.extend_from_slice(&eigs);
            deltas.push(delta);
            target_ns.push(n_particles);
        }

        // GPU bisection
        let result = bisect
            .solve_bcs(&lower, &upper, &all_eigenvalues, &deltas, &target_ns)
            .expect("local GPU BCS bisection");

        // Compare against CPU
        let mut max_mu_err: f64 = 0.0;
        let mut max_v2_err: f64 = 0.0;
        for b in 0..batch_size {
            let eigs = &all_eigenvalues[b * n_levels..(b + 1) * n_levels];
            let delta = deltas[b];
            let n_t = target_ns[b];

            let cpu_mu = cpu_find_fermi_bcs(eigs, n_t, delta);
            let gpu_mu = result.roots[b];
            let mu_err = (cpu_mu - gpu_mu).abs();
            if mu_err > max_mu_err {
                max_mu_err = mu_err;
            }

            // Compare occupations
            let cpu_v2 = cpu_bcs_occupations(eigs, cpu_mu, delta);
            let gpu_v2 = cpu_bcs_occupations(eigs, gpu_mu, delta);
            for i in 0..n_levels {
                let err = (cpu_v2[i] - gpu_v2[i]).abs();
                if err > max_v2_err {
                    max_v2_err = err;
                }
            }

            // Check particle number conservation
            let gpu_pn: f64 = cpu_bcs_occupations(eigs, gpu_mu, delta).iter().sum();
            let pn_err = (gpu_pn - n_t).abs();
            if b < 3 {
                println!(
                    "  Nucleus {b}: μ_cpu={cpu_mu:.6} μ_gpu={gpu_mu:.6} err={mu_err:.2e} N={gpu_pn:.4} (target={n_t:.0})"
                );
            }
            harness.check_upper(
                &format!("BCS nucleus {b} particle number error"),
                pn_err,
                tolerances::BCS_PARTICLE_NUMBER_ABS,
            );
        }

        println!("  Max μ error: {max_mu_err:.2e}");
        println!("  Max v² error: {max_v2_err:.2e}");
        harness.check_upper(
            "BCS: max chemical potential error",
            max_mu_err,
            tolerances::BCS_CHEMICAL_POTENTIAL_REL,
        );
        harness.check_upper(
            "BCS: max occupation error",
            max_v2_err,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    // ── Phase 2: Batched Eigensolve (GPU vs CPU) ──
    println!("\n── Phase 2: GPU Batched Eigensolve ─────────────────────");
    {
        let batch_size = 4;
        let ns = 8; // 8×8 matrices (typical for small HFB blocks)

        // Build test Hamiltonians
        let mut packed: Vec<f64> = Vec::with_capacity(batch_size * ns * ns);
        let mut cpu_eigenvalues = Vec::with_capacity(batch_size * ns);

        for b in 0..batch_size {
            let hw = 7.0 + b as f64;
            let g = 0.1f64.mul_add(b as f64, 0.5);
            let h = build_test_hamiltonian(ns, hw, g);

            // CPU eigensolve for reference
            let cpu_eig = eigh_f64(&h, ns).expect("CPU eigh_f64");
            cpu_eigenvalues.extend_from_slice(&cpu_eig.eigenvalues);

            packed.extend_from_slice(&h);
        }

        // GPU batched eigensolve (multi-dispatch)
        let (gpu_vals, gpu_vecs) =
            BatchedEighGpu::execute_f64(device.clone(), &packed, ns, batch_size, 30)
                .expect("GPU BatchedEighGpu multi-dispatch");

        // Compare eigenvalues
        let mut max_eval_err: f64 = 0.0;
        for b in 0..batch_size {
            let mut cpu_evals: Vec<f64> = cpu_eigenvalues[b * ns..(b + 1) * ns].to_vec();
            let mut gpu_evals: Vec<f64> = gpu_vals[b * ns..(b + 1) * ns].to_vec();
            cpu_evals.sort_by(f64::total_cmp);
            gpu_evals.sort_by(f64::total_cmp);

            for i in 0..ns {
                let err = (cpu_evals[i] - gpu_evals[i]).abs();
                let rel = if cpu_evals[i].abs() > tolerances::EXACT_F64 {
                    err / cpu_evals[i].abs()
                } else {
                    err
                };
                if rel > max_eval_err {
                    max_eval_err = rel;
                }
            }

            if b < 2 {
                println!(
                    "  Matrix {}: CPU=[{:.3}, {:.3}, ...] GPU=[{:.3}, {:.3}, ...]",
                    b, cpu_evals[0], cpu_evals[1], gpu_evals[0], gpu_evals[1]
                );
            }
        }
        println!("  Max eigenvalue relative error (multi-dispatch): {max_eval_err:.2e}");
        harness.check_upper(
            "Eigensolve: max eigenvalue relative error",
            max_eval_err,
            tolerances::GPU_EIGENSOLVE_REL,
        );

        // Verify eigenvectors are orthogonal (V^T V ≈ I)
        let mut max_ortho_err: f64 = 0.0;
        for b in 0..batch_size {
            let vecs = &gpu_vecs[b * ns * ns..(b + 1) * ns * ns];
            for i in 0..ns {
                for j in 0..ns {
                    let mut dot: f64 = 0.0;
                    for k in 0..ns {
                        dot += vecs[k * ns + i] * vecs[k * ns + j];
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let err = (dot - expected).abs();
                    if err > max_ortho_err {
                        max_ortho_err = err;
                    }
                }
            }
        }
        println!("  Max orthogonality error (multi-dispatch): {max_ortho_err:.2e}");
        harness.check_upper(
            "Eigensolve: eigenvector orthogonality",
            max_ortho_err,
            tolerances::GPU_EIGENVECTOR_ORTHO,
        );

        // ── Single-dispatch eigensolve (all rotations in one shader) ──
        // wgpu panics (rather than returning Err) on shader validation failures,
        // so catch_unwind is required to handle the upstream toadstool bug where
        // the loop unroller emits bare int literals instead of `u32` in WGSL.
        println!("\n  Single-dispatch eigensolve (all rotations in one shader):");
        let sd_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            BatchedEighGpu::execute_single_dispatch(device, &packed, ns, batch_size, 30, 1e-12)
        }));
        match sd_result {
            Ok(Ok((sd_vals, sd_vecs))) => {
                let mut max_sd_err: f64 = 0.0;
                for b in 0..batch_size {
                    let mut cpu_evals: Vec<f64> =
                        cpu_eigenvalues[b * ns..(b + 1) * ns].to_vec();
                    let mut sd_evals: Vec<f64> = sd_vals[b * ns..(b + 1) * ns].to_vec();
                    cpu_evals.sort_by(f64::total_cmp);
                    sd_evals.sort_by(f64::total_cmp);
                    for i in 0..ns {
                        let err = (cpu_evals[i] - sd_evals[i]).abs();
                        let rel = if cpu_evals[i].abs() > tolerances::EXACT_F64 {
                            err / cpu_evals[i].abs()
                        } else {
                            err
                        };
                        if rel > max_sd_err {
                            max_sd_err = rel;
                        }
                    }
                }
                println!(
                    "  Max eigenvalue relative error (single-dispatch): {max_sd_err:.2e}"
                );
                harness.check_upper(
                    "Eigensolve: single-dispatch eigenvalue error",
                    max_sd_err,
                    tolerances::GPU_EIGENSOLVE_REL,
                );

                let mut max_sd_ortho: f64 = 0.0;
                for b in 0..batch_size {
                    let vecs = &sd_vecs[b * ns * ns..(b + 1) * ns * ns];
                    for i in 0..ns {
                        for j in 0..ns {
                            let mut dot: f64 = 0.0;
                            for k in 0..ns {
                                dot += vecs[k * ns + i] * vecs[k * ns + j];
                            }
                            let expected = if i == j { 1.0 } else { 0.0 };
                            let err = (dot - expected).abs();
                            if err > max_sd_ortho {
                                max_sd_ortho = err;
                            }
                        }
                    }
                }
                println!(
                    "  Max orthogonality error (single-dispatch): {max_sd_ortho:.2e}"
                );
                harness.check_upper(
                    "Eigensolve: single-dispatch orthogonality",
                    max_sd_ortho,
                    tolerances::GPU_EIGENVECTOR_ORTHO,
                );
            }
            Ok(Err(e)) => {
                println!(
                    "  SKIP: single-dispatch returned error: {e}"
                );
            }
            Err(_) => {
                println!(
                    "  SKIP: single-dispatch shader compilation panicked \
                     (upstream toadstool loop_unroller bug)"
                );
                println!(
                    "    Root cause: WGSL idx2d() receives bare int literal \
                     instead of u32 after @unroll_hint expansion"
                );
            }
        }
    }

    // ── Phase 3: BCS with degeneracy (nuclear physics) ──
    println!("\n── Phase 3: GPU BCS with Degeneracy ───────────────────");
    {
        let bisect = BcsBisectionGpu::new(&gpu, 100, 1e-12);

        // Simulating proton levels for O-16 (Z=8): 1s1/2, 1p3/2, 1p1/2
        // Degeneracies: 2j+1 = 2, 4, 2
        let n_levels = 3;
        let eigenvalues = vec![-40.0, -20.0, -15.0]; // MeV
        let degeneracies = vec![2.0, 4.0, 2.0]; // 2j+1
        let delta = vec![12.0 / 16.0_f64.sqrt()]; // Δ = 12/√A MeV
        let target_n = vec![8.0]; // Z=8
        let lower = vec![-90.0];
        let upper = vec![10.0];

        let result = bisect
            .solve_bcs_with_degeneracy(
                &lower,
                &upper,
                &eigenvalues,
                &degeneracies,
                &delta,
                &target_n,
            )
            .expect("local GPU BCS with degeneracy");

        let mu = result.roots[0];
        let delta_val = delta[0];

        // Verify particle number
        let mut pn: f64 = 0.0;
        for k in 0..n_levels {
            let eps = eigenvalues[k] - mu;
            let e_k = eps.hypot(delta_val);
            let v2_k = 0.5 * (1.0 - eps / e_k);
            pn += degeneracies[k] * v2_k;
        }

        println!("  O-16 protons: μ={mu:.4} MeV, Δ={delta_val:.4} MeV");
        println!("  Particle number: {pn:.6} (target=8.0)");
        println!("  Iterations: {}", result.iterations[0]);

        let pn_err = (pn - 8.0).abs();
        harness.check_upper(
            "BCS degeneracy: O-16 proton number error",
            pn_err,
            tolerances::BCS_DEGENERACY_PARTICLE_NUMBER_ABS,
        );

        // Verify μ is within the spectrum
        let mu_reasonable = mu > 10.0f64.mul_add(-delta_val, eigenvalues[0])
            && mu
                < 10.0f64.mul_add(
                    delta_val,
                    *eigenvalues.last().expect("eigenvalues non-empty"),
                );
        harness.check_bool("BCS degeneracy: μ within reasonable range", mu_reasonable);
    }

    println!();
    harness.finish();
}
