// SPDX-License-Identifier: AGPL-3.0-or-later

//! 32⁴ Node-Atomic validation — confirms WG128 dispatch + multi-pass reduce at scale.
//!
//! Key test parameters:
//!   - 4,194,304 links → 32,768 WG128 dispatches (under 65535 limit)
//!   - Reduce: n_partial=16,384 → needs 3-pass iterative reduction
//!   - Expected equilibrium: ⟨P⟩ ≈ 0.578 at β=5.90

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
        n_md_steps: 40,
        dt: 0.0025,
        cg_tol: 1e-10,
        cg_max_iter: 1000,
        n_flavors_over_4: 0,
    };

    println!("═══ 32⁴ β=5.90 Node-Atomic Validation ═══");
    println!("  Omelyan 2MN: n_md={}, dt={}, τ={:.3}", config.n_md_steps, config.dt,
        config.n_md_steps as f64 * config.dt);
    println!("  Tests: WG128 dispatch (32768 WGs), 3-pass reduce (n_partial=16384)");
    println!();

    let t0 = std::time::Instant::now();
    let qcd = NodeAtomicQcd::new(config.clone(), 42).expect("Failed to create QCD state");
    println!("  Device: {} ({:.2}s)", qcd.device.adapter_info().name, t0.elapsed().as_secs_f64());

    qcd.upload_topology();
    qcd.cold_start().expect("cold_start failed");
    qcd.seed_rng(42);

    let volume = qcd.volume();
    let n_links = volume * 4;
    println!("  Volume: {volume} sites, {n_links} links, {} WG128 dispatches",
        n_links / 128);
    println!("  Initialized ({:.2}s)", t0.elapsed().as_secs_f64());
    println!();

    // Quick diagnostic: 20 trajectories to verify dynamics work at 32⁴
    let n_warmup = 20;
    let mut accepted = 0usize;

    for i in 0..n_warmup {
        let result = qcd.run_trajectory().expect("Trajectory failed");
        if result.accepted {
            accepted += 1;
        }

        if (i + 1) % 5 == 0 {
            let plaq = 1.0 - result.gauge_action / (6.0 * volume as f64 * config.beta);
            let acc_rate = accepted as f64 / (i + 1) as f64;
            println!(
                "  warmup {:2}/{}: P={:.8}, acc={:.0}%, ΔH={:+.4} ({:.1}s)",
                i + 1, n_warmup, plaq, acc_rate * 100.0, result.delta_h,
                t0.elapsed().as_secs_f64()
            );
        }
    }

    let final_result = qcd.run_trajectory().expect("Final trajectory");
    let final_plaq = 1.0 - final_result.gauge_action / (6.0 * volume as f64 * config.beta);
    let acc_rate = accepted as f64 / n_warmup as f64;

    let passed = final_plaq < 1.0 && acc_rate > 0.1 && final_result.delta_h.abs() < 10.0;
    hotspring_barracuda::gossip::validation_result(passed, if passed { 1 } else { 0 }, 1);

    println!("\n═══ RESULT ═══");
    println!("  Final P = {:.8} (from cold, {} trajectories)", final_plaq, n_warmup + 1);
    println!("  Acceptance: {:.0}%", acc_rate * 100.0);
    println!("  ΔH (last): {:+.4}", final_result.delta_h);
    println!("  Time: {:.1}s ({:.2}s/traj)", t0.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64() / (n_warmup + 1) as f64);

    if passed {
        println!("  ✓ PASS — 32⁴ dispatch + 3-pass reduce operational");
    } else if final_plaq > 0.99 {
        println!("  ✗ FAIL — lattice frozen at identity (cold_start or dynamics broken)");
    } else {
        println!("  ? INCONCLUSIVE — check ΔH and acceptance");
    }
}
