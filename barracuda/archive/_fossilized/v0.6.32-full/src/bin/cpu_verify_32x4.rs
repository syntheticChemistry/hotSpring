// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quick CPU verification: does 32^4 at β=5.9 converge to ~0.578?
//! Uses native f64 on CPU with the same HMC parameters as the GPU campaign.

use hotspring_barracuda::lattice::hmc::{HmcConfig, HmcResult, IntegratorType, hmc_trajectory};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  CPU Native f64 Verification — 32⁴ β=5.9                        ║");
    println!("║  Protocol: hot start, Omelyan, dt=0.01, n_md=20                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let dims = [32, 32, 32, 32];
    let beta = 5.9;
    let seed = 42u64;
    let n_therm = 200;

    println!("  Starting hot (disordered) lattice — should converge to ~0.578");
    println!("  Volume: {}⁴ = {} sites", dims[0], dims[0]*dims[1]*dims[2]*dims[3]);
    println!();

    let mut lat = Lattice::hot_start(dims, beta, seed);
    let mut cfg = HmcConfig {
        n_md_steps: 20,
        dt: 0.01,
        seed,
        integrator: IntegratorType::Omelyan,
    };

    let t0 = Instant::now();
    let mut accepted = 0u32;

    for i in 0..n_therm {
        let result = hmc_trajectory(&mut lat, &mut cfg);
        if result.accepted {
            accepted += 1;
        }

        if (i + 1) % 10 == 0 {
            let plaq = lat.average_plaquette();
            let rate = accepted as f64 / (i + 1) as f64 * 100.0;
            let elapsed = t0.elapsed().as_secs_f64();
            let per_traj = elapsed / (i + 1) as f64;
            println!("    traj {:>3}/{}: P = {:.8}, accept = {:.0}%, {:.1}s/traj",
                     i + 1, n_therm, plaq, rate, per_traj);
        }
    }

    let final_plaq = lat.average_plaquette();
    let elapsed = t0.elapsed().as_secs_f64();
    println!();
    println!("  Final plaquette: {:.8}", final_plaq);
    println!("  Acceptance rate: {:.1}%", accepted as f64 / n_therm as f64 * 100.0);
    println!("  Wall time: {:.1}s ({:.1}s/traj)", elapsed, elapsed / n_therm as f64);
    println!();

    if (final_plaq - 0.578).abs() < 0.01 {
        println!("  ✓ CONVERGED to expected ~0.578 — precision is the issue, not protocol");
    } else if (final_plaq - 0.786).abs() < 0.01 {
        println!("  ✗ CONVERGED to 0.786 — protocol issue (needs more warmup or smaller dt)");
    } else {
        println!("  ? Still thermalizing — P={:.4} (need more trajectories)", final_plaq);
    }
}
