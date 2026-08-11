// SPDX-License-Identifier: AGPL-3.0-or-later

//! arXiv production campaign — node-atomic composition (v0.7).
//!
//! Thin orchestrator: load grid → compose primal calls → run → emit provenance.
//! All GPU physics via barraCuda's GpuHmcTrajectory (no local shaders).

use barracuda::ops::lattice::gpu_hmc_types::GpuHmcConfig;
use hotspring_barracuda::node_atomic::NodeAtomicQcd;
use hotspring_barracuda::spring::campaign::{CampaignGrid, CampaignResult};
use hotspring_barracuda::spring::provenance::CampaignProvenance;
use hotspring_barracuda::spring::validation;

use std::path::PathBuf;

fn main() {

    let output_dir = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hotspring/node_atomic_v2");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output dir");

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Node-Atomic arXiv Campaign — barraCuda composition             ║");
    println!("║  Protocol: cold start → warmup → production → provenance        ║");
    println!("║  HMC: barraCuda GpuHmcTrajectory (Omelyan 2MN)                  ║");
    println!("║  Grid: 3 volumes × 3 β × 5 seeds = 45 configs                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let grid = CampaignGrid::arxiv_standard(output_dir.clone());
    let remaining = grid.remaining();

    println!(
        "  Grid: {} total, {} already complete, {} remaining",
        grid.configs.len(),
        grid.configs.len() - remaining.len(),
        remaining.len()
    );
    println!("  Output: {:?}", output_dir);
    println!();

    for config in remaining {
        let label = format!(
            "{}⁴ β={:.2} seed={}",
            config.dims[0], config.beta, config.seed
        );
        println!("━━━ {label} ━━━");

        let hmc_config = GpuHmcConfig {
            nt: config.dims[0],
            nx: config.dims[1],
            ny: config.dims[2],
            nz: config.dims[3],
            beta: config.beta,
            mass: 1.0,
            n_md_steps: config.n_md_steps,
            dt: config.dt,
            cg_tol: 1e-10,
            cg_max_iter: 1000,
            n_flavors_over_4: 0,
        };

        let qcd = match NodeAtomicQcd::new(hmc_config.clone(), config.seed as u64) {
            Ok(q) => q,
            Err(e) => {
                eprintln!("  ERROR creating QCD state: {e}");
                continue;
            }
        };

        qcd.upload_topology();
        qcd.seed_rng(config.seed);

        println!("  GPU: {}", qcd.device.adapter_info().name);
        println!("  Volume: {} sites, {} links", qcd.volume(), qcd.volume() * 4);

        let runner = hotspring_barracuda::node_atomic::TrajectoryRunner {
            warmup_count: config.n_warmup,
            production_count: config.n_production,
            ..Default::default()
        };

        let warmup_result = runner.run_warmup(&qcd, 100, |step, plaq, acc| {
            println!("    warmup {step}/{}: ⟨P⟩ = {plaq:.8}, accept = {:.0}%", config.n_warmup, acc * 100.0);
        });

        match warmup_result {
            Ok(result) => {
                println!(
                    "  Warmup done: P={:.6}, accept={:.0}%",
                    result.final_plaquette,
                    result.accepted as f64 / result.trajectories as f64 * 100.0
                );
            }
            Err(e) => {
                eprintln!("  ERROR in warmup: {e}");
                continue;
            }
        }

        let mut measurements = Vec::with_capacity(config.n_production);
        let prod_result = runner.run_production(&qcd, &mut measurements);

        match prod_result {
            Ok(result) => {
                let mean_plaq: f64 = measurements.iter().sum::<f64>() / measurements.len() as f64;
                let variance: f64 = measurements.iter().map(|p| (p - mean_plaq).powi(2)).sum::<f64>()
                    / (measurements.len() - 1) as f64;
                let std_plaq = variance.sqrt();
                let acc_rate = result.accepted as f64 / result.trajectories as f64;

                println!("  Production: ⟨P⟩ = {mean_plaq:.6} ± {std_plaq:.6}, accept = {:.0}%", acc_rate * 100.0);

                let v = validation::validate_plaquette(config.beta, mean_plaq);
                println!("  {v}");

                let mut provenance = CampaignProvenance::new(config.dims, config.beta, config.seed);
                provenance.n_warmup = config.n_warmup;
                provenance.n_production = config.n_production;
                for (i, &plaq) in measurements.iter().enumerate() {
                    provenance.add_plaquette(plaq, i);
                }

                let campaign_result = CampaignResult {
                    config: config.clone(),
                    mean_plaquette: mean_plaq,
                    plaquette_std: std_plaq,
                    acceptance_rate: acc_rate,
                    measurements,
                };

                let filename = format!(
                    "su3_{}x{}x{}x{}_b{:.2}_s{}.json",
                    config.dims[0], config.dims[1], config.dims[2], config.dims[3],
                    config.beta, config.seed
                );
                let path = output_dir.join(&filename);
                if let Ok(json) = serde_json::to_string_pretty(&campaign_result) {
                    std::fs::write(&path, json).ok();
                    println!("  Saved: {filename}");
                }
            }
            Err(e) => {
                eprintln!("  ERROR in production: {e}");
            }
        }
        println!();
    }

    println!("Campaign complete.");
}
