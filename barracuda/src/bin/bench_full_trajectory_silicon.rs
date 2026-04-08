// SPDX-License-Identifier: AGPL-3.0-or-later

//! Full-trajectory silicon comparison: actual RHMC/HMC wall-clock on both GPUs.
//!
//! Runs real streaming quenched HMC trajectories at each lattice size on every
//! discrete GPU, reporting wall-clock per trajectory, VRAM allocation, and the
//! largest lattice that fits.
//!
//! Usage:
//!   cargo run --release --bin bench_full_trajectory_silicon

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::wilson::Lattice;

use std::time::Instant;

struct CardResult {
    adapter_name: String,
    max_buffer_bytes: u64,
    lattice_results: Vec<LatticeResult>,
}

struct LatticeResult {
    dims: [usize; 4],
    volume: usize,
    vram_estimate_mb: f64,
    mean_traj_ms: f64,
    min_traj_ms: f64,
    max_traj_ms: f64,
    acceptance_rate: f64,
    mean_plaquette: f64,
    n_traj: usize,
}

fn estimate_vram_bytes(dims: [usize; 4]) -> u64 {
    let vol: usize = dims.iter().product();
    let n_links = vol * 4;
    let link_bytes = (n_links * 18 * 8) as u64;
    // link_buf + link_backup + mom_buf + force_buf + ke_out + plaq_out + poly_out + nbr
    // + reduce scratch + observable buffers
    6 * link_bytes
        + (vol as u64 * 8)       // plaq_out
        + (n_links as u64 * 8)   // ke_out
        + (vol as u64 * 8 * 4)   // nbr (u32 × 8 per site)
        + (vol as u64 * 2 * 8)   // poly_out
        + 4 * 1024 * 1024 // reduce scratch + staging + misc
}

fn bench_one_card(gpu: &GpuF64, max_buffer: u64) -> CardResult {
    let pipelines = GpuHmcStreamingPipelines::new_with_tmu(gpu);

    let lattice_sizes: &[[usize; 4]] = &[
        [4, 4, 4, 4],
        [8, 8, 8, 8],
        [12, 12, 12, 12],
        [16, 16, 16, 16],
        [20, 20, 20, 20],
        [24, 24, 24, 24],
    ];

    let beta = 6.0;
    let n_md = 10;
    let dt = 0.05;

    let mut lattice_results = Vec::new();

    for dims in lattice_sizes {
        let vol: usize = dims.iter().product();
        let vram_est = estimate_vram_bytes(*dims);
        let vram_est_mb = vram_est as f64 / (1024.0 * 1024.0);

        let single_link_buf = (vol * 4 * 18 * 8) as u64;
        if single_link_buf > max_buffer {
            println!(
                "    {}^4 (V={vol}): SKIP — link buffer {:.0} MB > max_buffer_size {:.0} MB",
                dims[0],
                single_link_buf as f64 / (1024.0 * 1024.0),
                max_buffer as f64 / (1024.0 * 1024.0),
            );
            continue;
        }

        let n_traj = if vol <= 4096 {
            20
        } else if vol <= 65536 {
            10
        } else {
            5
        };

        print!("    {}^4 (V={vol}, {n_traj} traj) ... ", dims[0]);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        let mut lat = Lattice::hot_start(*dims, beta, 42);

        let state = GpuHmcState::from_lattice(gpu, &lat, beta);
        let mut seed = 12345u64;

        // Warmup: 3 trajectories
        for i in 0..3 {
            gpu_hmc_trajectory_streaming(gpu, &pipelines, &state, n_md, dt, i, &mut seed)
                .expect("streaming HMC trajectory");
        }

        let mut times = Vec::with_capacity(n_traj);
        let mut accepted = 0usize;
        let mut plaq_sum = 0.0_f64;

        for i in 0..n_traj {
            let start = Instant::now();
            let r = gpu_hmc_trajectory_streaming(
                gpu,
                &pipelines,
                &state,
                n_md,
                dt,
                (3 + i) as u32,
                &mut seed,
            )
            .expect("streaming HMC trajectory");
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            times.push(elapsed_ms);
            if r.accepted {
                accepted += 1;
            }
            plaq_sum += r.plaquette;
        }

        // Download final state for plaquette cross-check
        hotspring_barracuda::lattice::gpu_hmc::gpu_links_to_lattice(gpu, &state, &mut lat);

        let mean_ms = times.iter().sum::<f64>() / n_traj as f64;
        let min_ms = times.iter().copied().fold(f64::INFINITY, f64::min);
        let max_ms = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let acc_rate = accepted as f64 / n_traj as f64;
        let mean_plaq = plaq_sum / n_traj as f64;

        println!(
            "{mean_ms:.1} ms/traj (min={min_ms:.1} max={max_ms:.1}), \
             acc={:.0}%, plaq={mean_plaq:.6}, VRAM~{vram_est_mb:.0} MB",
            acc_rate * 100.0,
        );

        lattice_results.push(LatticeResult {
            dims: *dims,
            volume: vol,
            vram_estimate_mb: vram_est_mb,
            mean_traj_ms: mean_ms,
            min_traj_ms: min_ms,
            max_traj_ms: max_ms,
            acceptance_rate: acc_rate,
            mean_plaquette: mean_plaq,
            n_traj,
        });
    }

    CardResult {
        adapter_name: gpu.adapter_name.clone(),
        max_buffer_bytes: max_buffer,
        lattice_results,
    }
}

fn print_comparison(cards: &[CardResult]) {
    println!(
        "\n╔══════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("║  Full-Trajectory Silicon Comparison                                             ║");
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════╣"
    );

    if cards.len() >= 2 {
        let a = &cards[0];
        let b = &cards[1];
        println!(
            "║  {:>30}  vs  {:<30}    ║",
            &a.adapter_name[..a.adapter_name.len().min(30)],
            &b.adapter_name[..b.adapter_name.len().min(30)],
        );
        println!(
            "╠════════╤════════╤══════════════╤══════════════╤═══════════╤═══════════════════╣"
        );
        println!(
            "║ L      │ Volume │ {:<12} │ {:<12} │ Ratio     │ Largest?          ║",
            "Card A ms", "Card B ms"
        );
        println!(
            "╠════════╪════════╪══════════════╪══════════════╪═══════════╪═══════════════════╣"
        );

        let all_sizes: std::collections::BTreeSet<usize> = a
            .lattice_results
            .iter()
            .chain(b.lattice_results.iter())
            .map(|r| r.volume)
            .collect();

        for vol in &all_sizes {
            let ra = a.lattice_results.iter().find(|r| r.volume == *vol);
            let rb = b.lattice_results.iter().find(|r| r.volume == *vol);
            let l = match ra.or(rb) {
                Some(r) => r.dims[0],
                None => continue,
            };
            let (a_ms, b_ms) = (ra.map(|r| r.mean_traj_ms), rb.map(|r| r.mean_traj_ms));
            let ratio_str = match (a_ms, b_ms) {
                (Some(a), Some(b)) => {
                    let r = a / b;
                    if r < 1.0 {
                        format!("A {:.1}x", 1.0 / r)
                    } else {
                        format!("B {r:.1}x")
                    }
                }
                _ => "—".to_string(),
            };
            let a_str = a_ms.map_or_else(|| "SKIP".to_string(), |v| format!("{v:.1}"));
            let b_str = b_ms.map_or_else(|| "SKIP".to_string(), |v| format!("{v:.1}"));
            let largest = match (ra, rb) {
                (Some(_), None) => "A only",
                (None, Some(_)) => "B only",
                _ => "",
            };
            println!(
                "║ {l:>4}^4 │ {vol:>6} │ {a_str:>12} │ {b_str:>12} │ {ratio_str:>9} │ {largest:<17} ║"
            );
        }
        println!(
            "╚════════╧════════╧══════════════╧══════════════╧═══════════╧═══════════════════╝"
        );
    } else {
        for card in cards {
            println!(
                "║  {} (max_buf: {:.0} MB)",
                card.adapter_name,
                card.max_buffer_bytes as f64 / (1024.0 * 1024.0)
            );
            println!(
                "╠════════╤════════╤══════════════╤═══════════════╤═══════════════════════════╣"
            );
            println!(
                "║ L      │ Volume │  ms/traj     │  Accept%      │  Plaquette                ║"
            );
            println!(
                "╠════════╪════════╪══════════════╪═══════════════╪═══════════════════════════╣"
            );
            for r in &card.lattice_results {
                println!(
                    "║ {:>4}^4 │ {:>6} │ {:>10.1}   │ {:>10.0}%   │ {:>10.6}                  ║",
                    r.dims[0],
                    r.volume,
                    r.mean_traj_ms,
                    r.acceptance_rate * 100.0,
                    r.mean_plaquette
                );
            }
            println!(
                "╚════════╧════════╧══════════════╧═══════════════╧═══════════════════════════╝"
            );
        }
    }
}

fn save_results(cards: &[CardResult]) {
    let root = std::env::var("HOTSPRING_ROOT").unwrap_or_else(|_| ".".to_string());
    let dir = std::path::PathBuf::from(&root).join("results");
    std::fs::create_dir_all(&dir).ok();

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut entries = Vec::new();
    for card in cards {
        for r in &card.lattice_results {
            entries.push(serde_json::json!({
                "adapter": card.adapter_name,
                "max_buffer_bytes": card.max_buffer_bytes,
                "dims": r.dims,
                "volume": r.volume,
                "vram_estimate_mb": r.vram_estimate_mb,
                "mean_traj_ms": r.mean_traj_ms,
                "min_traj_ms": r.min_traj_ms,
                "max_traj_ms": r.max_traj_ms,
                "acceptance_rate": r.acceptance_rate,
                "mean_plaquette": r.mean_plaquette,
                "n_traj": r.n_traj,
            }));
        }
    }

    let json = serde_json::json!({
        "benchmark": "bench_full_trajectory_silicon",
        "timestamp": timestamp,
        "beta": 6.0,
        "n_md": 10,
        "dt": 0.05,
        "results": entries,
    });

    let path = dir.join(format!("bench_full_trajectory_silicon_{timestamp}.json"));
    match std::fs::write(
        &path,
        serde_json::to_string_pretty(&json).unwrap_or_default(),
    ) {
        Ok(()) => println!("\n  Results saved → {}", path.display()),
        Err(e) => eprintln!("\n  Save failed: {e}"),
    }
}

/// Compute buffer overhead for quenched HMC (links, momenta, force, observables).
fn quenched_buffer_bytes(vol: usize) -> u64 {
    let n_links = vol * 4;
    let link_bytes = (n_links * 18 * 8) as u64;
    // link + backup + momenta + force
    4 * link_bytes
        + (n_links as u64 * 8)   // ke_out
        + (vol as u64 * 8)       // plaq_out
        + (vol as u64 * 16)      // poly_out (2 f64)
        + (vol as u64 * 8 * 4)   // nbr table (8 u32 per site)
        + 4 * 1024 * 1024 // reduce scratch + staging
}

/// Compute buffer overhead for dynamical RHMC: CG + pseudofermion + multi-shift.
fn dynamical_buffer_bytes(vol: usize, n_sectors: usize, n_poles: usize) -> u64 {
    let fermion_vec = (vol * 6 * 8) as u64; // one staggered fermion field (3 colors, complex)
    let n_links = vol * 4;

    // CG solver shared buffers: x, r, p, ap, temp, y, dot
    let cg_shared = 6 * fermion_vec + (vol as u64 * 8);
    // Fermion force output buffer
    let ferm_force = (n_links * 18 * 8) as u64;
    // Per-sector: phi + x_bufs[n_poles]
    let per_sector = fermion_vec + (n_poles as u64) * fermion_vec;
    // Multi-shift CG: residual vectors per shift (shared workspace)
    let multi_shift = (n_poles as u64) * fermion_vec;
    // Scalar reduce chains + staging (small)
    let cg_scratch = 2 * 1024 * 1024u64;

    cg_shared + ferm_force + (n_sectors as u64) * per_sector + multi_shift + cg_scratch
}

/// Largest single buffer in the allocation set: the link buffer.
fn largest_single_buffer(vol: usize) -> u64 {
    (vol * 4 * 18 * 8) as u64
}

/// Estimate total VRAM from adapter name (wgpu doesn't expose this directly).
fn estimate_total_vram_bytes(adapter_name: &str) -> u64 {
    let name = adapter_name.to_lowercase();
    if name.contains("3090") || name.contains("4090") {
        24 * 1024 * 1024 * 1024
    } else if name.contains("4080") {
        16 * 1024 * 1024 * 1024
    } else if name.contains("4070") {
        12 * 1024 * 1024 * 1024
    } else if name.contains("6950") || name.contains("6900") {
        16 * 1024 * 1024 * 1024
    } else if name.contains("7900 xtx") {
        24 * 1024 * 1024 * 1024
    } else if name.contains("7900 xt") {
        20 * 1024 * 1024 * 1024
    } else if name.contains("titan v") {
        12 * 1024 * 1024 * 1024
    } else {
        8 * 1024 * 1024 * 1024
    }
}

/// Compute the largest isotropic lattice L^4 that fits in VRAM.
///
/// Two constraints:
/// 1. Largest single buffer <= `max_buf_size` (wgpu per-allocation limit)
/// 2. Total allocation <= `total_vram` (physical VRAM)
fn max_lattice_for_vram(
    max_buf_size: u64,
    total_vram: u64,
    n_sectors: usize,
    n_poles: usize,
) -> (usize, u64) {
    let mut best_l = 4usize;
    let mut best_bytes = 0u64;
    for l in (4..=128).step_by(2) {
        let vol = l * l * l * l;
        if largest_single_buffer(vol) > max_buf_size {
            break;
        }
        let total = quenched_buffer_bytes(vol) + dynamical_buffer_bytes(vol, n_sectors, n_poles);
        if total > total_vram {
            break;
        }
        best_l = l;
        best_bytes = total;
    }
    (best_l, best_bytes)
}

fn print_capacity_analysis(cards: &[CardResult]) {
    println!(
        "\n╔══════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("║  Max-Lattice Capacity Analysis                                                  ║");
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!("║  Buffer accounting: quenched + dynamical RHMC (Nf=2+1, 15 poles/sector)        ║");
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════╣"
    );

    for card in cards {
        let max_buf = card.max_buffer_bytes;
        let total_vram = estimate_total_vram_bytes(&card.adapter_name);
        let n_sectors = 2;
        let n_poles = 15;

        let (max_l_quenched, qb) = {
            let mut bl = 4;
            let mut bb = 0;
            for l in (4..=128).step_by(2) {
                let vol = l * l * l * l;
                if largest_single_buffer(vol) > max_buf {
                    break;
                }
                let total = quenched_buffer_bytes(vol);
                if total > total_vram {
                    break;
                }
                bl = l;
                bb = total;
            }
            (bl, bb)
        };

        let (max_l_dyn, db) = max_lattice_for_vram(max_buf, total_vram, n_sectors, n_poles);

        let qvol = max_l_quenched.pow(4);
        let dvol = max_l_dyn.pow(4);

        println!(
            "║                                                                                  ║"
        );
        println!("║  {:50}                        ║", card.adapter_name);
        println!(
            "║  max_buffer_size: {:.0} MB, estimated VRAM: {:.0} MB                        ║",
            max_buf as f64 / (1024.0 * 1024.0),
            total_vram as f64 / (1024.0 * 1024.0)
        );
        println!(
            "║                                                                                  ║"
        );
        println!(
            "║  Quenched HMC:    L={:>3}^4 (V={:>9})  {:.1} MB used                      ║",
            max_l_quenched,
            qvol,
            qb as f64 / (1024.0 * 1024.0)
        );
        println!(
            "║  Dynamical RHMC:  L={:>3}^4 (V={:>9})  {:.1} MB used (2 sectors × 15 poles)║",
            max_l_dyn,
            dvol,
            db as f64 / (1024.0 * 1024.0)
        );

        // Per-lattice breakdown for reference sizes
        for l in [8, 16, 24, 32, 48] {
            let vol = l * l * l * l;
            let q = quenched_buffer_bytes(vol);
            let d = dynamical_buffer_bytes(vol, n_sectors, n_poles);
            let total = q + d;
            let single_ok = largest_single_buffer(vol) <= max_buf;
            let total_ok = total <= total_vram;
            let fits = if single_ok && total_ok { "✓" } else { "✗" };
            println!(
                "║  {:>2}^4: quench {:.0} + dyn {:.0} = {:.0} MB (link {:.0} MB) {fits}          ║",
                l,
                q as f64 / (1024.0 * 1024.0),
                d as f64 / (1024.0 * 1024.0),
                total as f64 / (1024.0 * 1024.0),
                largest_single_buffer(vol) as f64 / (1024.0 * 1024.0),
            );
        }
    }
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════╝"
    );
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Full-Trajectory Silicon Benchmark                          ║");
    println!("║  Streaming quenched HMC on every GPU, 4^4 → 24^4          ║");
    println!("║  β=6.0, n_md=10, dt=0.05, Omelyan                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> =
        rt.block_on(instance.enumerate_adapters(wgpu::Backends::all()));

    let mut cards = Vec::new();

    for adapter in adapters {
        let info = adapter.get_info();
        if info.device_type != wgpu::DeviceType::DiscreteGpu {
            continue;
        }

        let max_buffer = adapter.limits().max_buffer_size;
        println!(
            "━━━ {} (max_buffer: {:.0} MB) ━━━\n",
            info.name,
            max_buffer as f64 / (1024.0 * 1024.0),
        );

        let gpu = match rt.block_on(GpuF64::from_adapter(adapter)) {
            Ok(g) => g,
            Err(e) => {
                println!("  Skip: {e}\n");
                continue;
            }
        };

        let result = bench_one_card(&gpu, max_buffer);
        cards.push(result);
        println!();
    }

    print_comparison(&cards);
    print_capacity_analysis(&cards);
    save_results(&cards);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!(
        "  Full-Trajectory Silicon Benchmark Complete — {} GPUs",
        cards.len()
    );
    println!("═══════════════════════════════════════════════════════════════");
}
