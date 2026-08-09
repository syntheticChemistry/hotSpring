// SPDX-License-Identifier: AGPL-3.0-or-later
//! Large Lattice HMC — Push to 20⁴ and 24⁴ on both cards.
//!
//! Based on tiling measurements: both cards scale monotonically to L=24.
//! The real question: can we run ACTUAL SU(3) HMC at these sizes?
//!
//! Working set for SU(3) gauge field:
//! - 4 links per site × 18 f64 per link = 72 f64/site = 576 bytes/site
//! - 16⁴ (65536 sites): 36 MB (fits IC)
//! - 20⁴ (160000 sites): 88 MB (fits IC!)
//! - 24⁴ (331776 sites): 182 MB (EXCEEDS IC → VRAM cliff expected)
//!
//! Plus momenta (same size) and neighbor tables.
//! Total working set ≈ 2.5× field size.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Large Lattice HMC — Pushing to 20⁴ and 24⁴                    ║");
    println!("║  Testing IC absorption limits for production QCD                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => { println!("  GPU not available: {e}"); return; }
    };
    println!("  GPU: {}", gpu.adapter_name);
    println!();

    let pipelines = GpuHmcStreamingPipelines::new(&gpu);

    let configs: &[(&str, [usize; 4])] = &[
        ("16⁴", [16, 16, 16, 16]),
        ("20⁴", [20, 20, 20, 20]),
        ("24⁴", [24, 24, 24, 24]),
        ("32⁴", [32, 32, 32, 32]),
    ];

    let beta = 6.0;
    let n_md = 10;
    let dt = 0.05;

    println!("  {:>8}  {:>10}  {:>8}  {:>12}  {:>8}  {:>8}", "Lattice", "Volume", "WS MB", "ms/traj", "Acc", "Status");
    println!("  ──────────────────────────────────────────────────────────────────");

    for &(label, dims) in configs {
        let vol: usize = dims.iter().product();
        let n_links = vol * 4;
        let ws_bytes = n_links * 18 * 8 * 2; // links + momenta
        let ws_mb = ws_bytes as f64 / (1024.0 * 1024.0);

        // Check if buffer size is reasonable (2 GB limit for AMD RADV)
        let link_buf_bytes = (n_links * 18 * 8) as u64;
        if link_buf_bytes > 2_000_000_000 {
            println!("  {:>8}  {:>10}  {:>6.0}  {:>12}  {:>8}  {:>8}",
                label, vol, ws_mb, "—", "—", "SKIP(buf)");
            continue;
        }

        // Hot start (random SU(3) links) — skip CPU thermalization for large lattices
        // since we're measuring GPU timing, not physics quality
        let mut lat = Lattice::hot_start(dims, beta, 42);
        let mut cfg = HmcConfig { n_md_steps: n_md, dt, seed: 42, integrator: IntegratorType::Omelyan };

        // Only CPU-thermalize small lattices (large ones take too long on CPU)
        let cpu_therm = if vol <= 65536 { 3 } else { 0 };
        for _ in 0..cpu_therm {
            hmc::hmc_trajectory(&mut lat, &mut cfg);
        }

        // Upload to GPU
        let state = GpuHmcState::from_lattice(&gpu, &lat, beta);

        // Warmup (1 GPU trajectory)
        let mut seed = 2000u64;
        match gpu_hmc_trajectory_streaming(&gpu, &pipelines, &state, n_md, dt, 0, &mut seed) {
            Ok(_) => {},
            Err(e) => {
                println!("  {:>8}  {:>10}  {:>6.0}  {:>12}  {:>8}  ERR: {}",
                    label, vol, ws_mb, "—", "—", e);
                continue;
            }
        };

        // Benchmark (3 trajectories for statistics)
        let n_traj = 3;
        let start = Instant::now();
        let mut accepted = 0;
        for i in 0..n_traj {
            match gpu_hmc_trajectory_streaming(&gpu, &pipelines, &state, n_md, dt, (i + 1) as u32, &mut seed) {
                Ok(r) => { if r.accepted { accepted += 1; } },
                Err(e) => {
                    println!("  {:>8}  {:>10}  {:>6.0}  traj {} failed: {}",
                        label, vol, ws_mb, i, e);
                    break;
                }
            }
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let ms_per_traj = elapsed_ms / n_traj as f64;

        let status = if ws_mb < 6.0 { "L2-fit" }
            else if ws_mb < 128.0 { "IC-fit" }
            else { "VRAM" };

        println!("  {:>8}  {:>10}  {:>6.0}  {:>10.1}  {:>6}/{:<2}  {:>8}",
            label, vol, ws_mb, ms_per_traj, accepted, n_traj, status);
    }

    println!();
    println!("  ── Scaling Analysis ──");
    println!("    Working set = 2 × vol × 4 dirs × 18 f64 × 8 bytes");
    println!("    IC boundary at 128 MB → vol ≈ 232K sites (≈ 22⁴)");
    println!("    Above 22⁴: expect AMD performance cliff (VRAM-bound)");
    println!("    Below 22⁴: IC absorbs everything (20× vs NVIDIA)");
    println!();
    println!("  ── Physics Significance ──");
    println!("    16⁴: Rung 1 (quenched SU(3), strong coupling)");
    println!("    20⁴: Rung 1.5 (larger volume, reduced finite-size effects)");
    println!("    24⁴: Rung 2 target (approaching continuum limit)");
    println!("    32⁴: Rung 3 target (production MILC-comparable)");
}
