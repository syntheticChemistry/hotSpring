// SPDX-License-Identifier: AGPL-3.0-only

//! Titan V (or CPU) validation oracle for critical configurations.
//!
//! Extracted from production_mixed_pipeline to reduce binary size.
//! Runs quenched HMC on a secondary GPU (Titan V) or CPU f64 to validate
//! plaquette and Polyakov loop against primary GPU results.

use crate::gpu::GpuF64;
use crate::lattice::gpu_hmc::{
    gpu_hmc_trajectory_streaming, gpu_polyakov_loop, GpuHmcState, GpuHmcStreamingPipelines,
};
use crate::lattice::hmc::{self, HmcConfig, IntegratorType};
use crate::lattice::wilson::Lattice;
use crate::production::BetaResult;

/// Run Titan V (or CPU) validation oracle on critical configurations.
///
/// For each transition-region β point, runs quenched HMC on the validation
/// substrate and compares plaquette and Polyakov loop to the primary GPU result.
pub fn run_titan_validation(
    gpu_titan: Option<&GpuF64>,
    results: &[BetaResult],
    dims: [usize; 4],
    n_md: usize,
    dt: f64,
) {
    let transition_results: Vec<&BetaResult> =
        results.iter().filter(|r| r.phase == "transition").collect();

    if transition_results.is_empty() {
        println!("  No transition-region points to validate");
        return;
    }

    let titan_dims = if dims[0] > 16 { [16, 16, 16, 16] } else { dims };

    for r in &transition_results {
        if let Some(titan) = gpu_titan {
            let titan_pipelines = GpuHmcStreamingPipelines::new(titan);
            let mut lat = Lattice::hot_start(titan_dims, r.beta, 77777);

            let mut cfg = HmcConfig {
                n_md_steps: n_md.min(30),
                dt: dt.max(0.01),
                seed: 77777,
                integrator: IntegratorType::Omelyan,
            };
            for _ in 0..20 {
                hmc::hmc_trajectory(&mut lat, &mut cfg);
            }

            let state = GpuHmcState::from_lattice(titan, &lat, r.beta);
            let mut seed = 88888u64;
            for t in 0..20 {
                gpu_hmc_trajectory_streaming(
                    titan,
                    &titan_pipelines,
                    &state,
                    n_md.min(30),
                    dt.max(0.01),
                    t as u32,
                    &mut seed,
                );
            }

            let mut plaq_sum = 0.0;
            let n_verify = 50;
            for t in 0..n_verify {
                let tr = gpu_hmc_trajectory_streaming(
                    titan,
                    &titan_pipelines,
                    &state,
                    n_md.min(30),
                    dt.max(0.01),
                    (20 + t) as u32,
                    &mut seed,
                );
                plaq_sum += tr.plaquette;
            }
            let titan_plaq = plaq_sum / n_verify as f64;
            let (titan_poly, _) = gpu_polyakov_loop(titan, &titan_pipelines.hmc, &state);

            let plaq_diff = (titan_plaq - r.mean_plaq).abs();
            let agree = plaq_diff < 0.05;

            println!(
                "  β={:.4}: 3090 ⟨P⟩={:.6} vs Titan V ⟨P⟩={:.6} ({}⁴, native f64) Δ={:.6} {}",
                r.beta,
                r.mean_plaq,
                titan_plaq,
                titan_dims[0],
                plaq_diff,
                if agree { "✓ AGREE" } else { "△ CHECK" },
            );
            println!(
                "           3090 |L|={:.4} vs Titan V |L|={:.4}",
                r.polyakov, titan_poly,
            );
        } else {
            let mut lat = Lattice::hot_start(dims, r.beta, 77777);
            let mut cfg = HmcConfig {
                n_md_steps: n_md,
                dt,
                seed: 77777,
                integrator: IntegratorType::Omelyan,
            };
            for _ in 0..20 {
                hmc::hmc_trajectory(&mut lat, &mut cfg);
            }
            let stats = hmc::run_hmc(&mut lat, 50, 0, &mut cfg);
            let poly = lat.average_polyakov_loop();
            let plaq_diff = (stats.mean_plaquette - r.mean_plaq).abs();
            let agree = plaq_diff < 0.05;

            println!(
                "  β={:.4}: GPU ⟨P⟩={:.6} vs CPU f64 ⟨P⟩={:.6} Δ={:.6} {}",
                r.beta,
                r.mean_plaq,
                stats.mean_plaquette,
                plaq_diff,
                if agree { "✓ AGREE" } else { "△ CHECK" },
            );
            println!(
                "           GPU |L|={:.4} vs CPU |L|={:.4}",
                r.polyakov, poly,
            );
        }
    }
}
