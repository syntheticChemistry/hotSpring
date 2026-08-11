// SPDX-License-Identifier: AGPL-3.0-or-later

//! Focused 32⁴ β=5.90 validation — confirms the WG128 dispatch fix.
//! Expected: plaquette should descend from 1.0 (cold start) toward ~0.578.

use barracuda::ops::lattice::gpu_hmc_types::GpuHmcConfig;
use hotspring_barracuda::node_atomic::NodeAtomicQcd;

fn main() {
    let config = GpuHmcConfig {
        nt: 32,
        nx: 32,
        ny: 32,
        nz: 32,
        beta: 5.90,
        mass: 1.0,
        n_md_steps: 20,
        dt: 0.01,
        cg_tol: 1e-10,
        cg_max_iter: 1000,
        n_flavors_over_4: 0,
    };

    println!("32⁴ β=5.90 Node-Atomic Validation");
    println!("Expected: ⟨P⟩ → 0.578 (from cold start)");
    println!("If dispatch fix works: should see P drop below 0.6");
    println!("If still broken: P stays near 0.786-0.789");
    println!();

    let qcd = NodeAtomicQcd::new(config.clone(), 42).expect("Failed to create QCD state");
    qcd.upload_topology();
    qcd.seed_rng(42);

    let volume = qcd.volume();
    let n_links = volume * 4;
    println!("  Volume: {volume} sites, {n_links} links");
    println!("  Workgroups (WG128): {} per-link dispatches", n_links / 128);
    println!("  GPU: {}", qcd.device.adapter_info().name);
    println!();

    let n_warmup = 1000;
    let mut accepted = 0usize;

    for i in 0..n_warmup {
        let result = qcd.run_trajectory().expect("Trajectory failed");
        if result.accepted {
            accepted += 1;
        }

        if (i + 1) % 50 == 0 {
            let plaq = 1.0 - result.gauge_action / (6.0 * volume as f64 * config.beta);
            let acc_rate = accepted as f64 / (i + 1) as f64;
            println!(
                "  warmup {:4}/{}: ⟨P⟩ = {:.8}, accept = {:.0}%, ΔH = {:.4}",
                i + 1,
                n_warmup,
                plaq,
                acc_rate * 100.0,
                result.delta_h
            );
        }
    }

    println!();
    println!("Running 200 production measurements...");
    let mut measurements = Vec::with_capacity(200);
    for _ in 0..200 {
        let result = qcd.run_trajectory().expect("Trajectory failed");
        let plaq = 1.0 - result.gauge_action / (6.0 * volume as f64 * config.beta);
        measurements.push(plaq);
    }

    let mean: f64 = measurements.iter().sum::<f64>() / measurements.len() as f64;
    let variance: f64 = measurements
        .iter()
        .map(|p| (p - mean).powi(2))
        .sum::<f64>()
        / (measurements.len() - 1) as f64;
    let std = variance.sqrt();

    println!();
    println!("═══ RESULT ═══");
    println!("  ⟨P⟩ = {mean:.8} ± {std:.8}");
    println!("  Expected: 0.5780 ± 0.0005");

    if (mean - 0.578).abs() < 0.005 {
        println!("  ✓ PASS — plaquette consistent with literature");
    } else if mean > 0.75 {
        println!("  ✗ FAIL — still frozen (dispatch bug persists)");
    } else {
        println!("  ? INCONCLUSIVE — plaquette outside expected range");
    }
}
