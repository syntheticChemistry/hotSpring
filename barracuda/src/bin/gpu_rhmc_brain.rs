// SPDX-License-Identifier: AGPL-3.0-only

//! Brain-steered dual-GPU RHMC — hardware NPU cortex drives physics.
//!
//! Runs RHMC trajectories on both GPUs simultaneously. After each trajectory
//! pair, observables from *both* GPUs stream into a unified physics-focused
//! input sequence for the NPU (Akida hardware-first, software fallback).
//! The NPU learns the physics parameters (dt, n_md, CG tol) rather than
//! hardware-specific quirks, and gets 2× learning rate from dual-GPU input.
//! Cross-GPU agreement on acceptance/ΔH signals robust parameter regions.
//!
//! Usage:
//!   cargo run --release --bin gpu_rhmc_brain -- \
//!     --lattice 12 --beta 6.0 --nf 2 --mass 0.05 --trajs 100 \
//!     --checkpoint 25 --report-interval 20

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::brain_rhmc::{
    BrainRhmcRunner, NpuCortex, load_brain_state,
};
use hotspring_barracuda::lattice::gpu_hmc::dynamical::{GpuDynHmcPipelines, GpuDynHmcState};
#[allow(deprecated)]
use hotspring_barracuda::lattice::gpu_hmc::gpu_rhmc::{
    GpuRhmcPipelines, GpuRhmcState, gpu_rhmc_trajectory,
};
use hotspring_barracuda::lattice::gpu_hmc::rhmc_calibrator::RhmcCalibrator;
use hotspring_barracuda::lattice::gpu_hmc::unidirectional_cortex::UnidirectionalRhmc;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::md::reservoir::Activation;
use hotspring_barracuda::md::reservoir::heads;
use hotspring_barracuda::md::reservoir::npu::ExportedWeights;

use hotspring_forge::pipeline::topologies;

use std::time::Instant;

// Legacy RHMC trajectory calls used for dt-discovery probes only;
// the main brain trajectory uses the unidirectional path.
#[allow(deprecated)]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lattice_size = parse_arg(&args, "--lattice", 8);
    let beta: f64 = parse_arg_f64(&args, "--beta", 6.0);
    let nf: usize = parse_nf(&args);
    let mass: f64 = parse_arg_f64(&args, "--mass", 0.05);
    let strange_mass: f64 = parse_arg_f64(&args, "--strange-mass", 0.5);
    let n_trajs: usize = parse_arg(&args, "--trajs", 40);
    let report_interval: usize = parse_arg(&args, "--report-interval", 10);
    let checkpoint_interval: usize = parse_arg(&args, "--checkpoint", 50);
    let seed: u64 = parse_arg(&args, "--seed", 42) as u64;

    let dims = [lattice_size; 4];
    let vol: usize = dims.iter().product();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Brain-Steered Dual-GPU RHMC — NPU Cortex Drives Parameters");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  Lattice: {}^4 ({} sites)  beta={beta:.4}  Nf={}",
        lattice_size,
        vol,
        nf_label(nf, mass, strange_mass)
    );
    println!("  Trajectories: {n_trajs}  |  Report interval: {report_interval}");
    println!();

    // ═══ Discover GPUs ═══════════════════════════════════════════════
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    let available = GpuF64::enumerate_adapters();
    let f64_adapters: Vec<_> = available.iter().filter(|a| a.has_f64).collect();

    println!("  Available adapters ({} with f64):", f64_adapters.len());
    for a in &f64_adapters {
        println!("    [{}] {} ({}B VRAM)", a.index, a.name, a.memory_bytes);
    }
    println!();

    let (gpu_a, gpu_b) = if f64_adapters.len() >= 2 {
        let a = rt
            .block_on(GpuF64::from_adapter_name(&f64_adapters[0].name))
            .expect("GPU A");
        let b = rt
            .block_on(GpuF64::from_adapter_name(&f64_adapters[1].name))
            .expect("GPU B");
        println!("  GPU A: {}", a.adapter_name);
        println!("  GPU B: {}", b.adapter_name);
        (a, b)
    } else if !f64_adapters.is_empty() {
        let a = rt.block_on(GpuF64::new()).expect("GPU A");
        let b = rt.block_on(GpuF64::new()).expect("GPU B");
        println!("  GPU A: {} (single-GPU mode)", a.adapter_name);
        println!("  GPU B: {} (same device)", b.adapter_name);
        (a, b)
    } else {
        eprintln!("ERROR: No f64-capable GPU found");
        std::process::exit(1);
    };

    // ═══ metalForge pipeline topology ═══════════════════════════════
    let topology = topologies::qcd_gpu_npu_oracle();
    println!("\n  Pipeline topology: {}", topology.name);
    for stage in topology.ordered_stages() {
        let bound = stage.substrate_name.as_deref().unwrap_or("unbound");
        println!("    [{:?}] {} ({bound})", stage.role, stage.name);
    }

    // ═══ metalForge substrate census ═══════════════════════════════
    let substrates = hotspring_forge::inventory::discover();
    println!("\n  Substrate census ({} devices):", substrates.len());
    for (i, s) in substrates.iter().enumerate() {
        println!(
            "    {i}: {} [{:?}] — {:?}",
            s.identity.name,
            s.kind,
            s.capabilities.len()
        );
    }

    let t_total = Instant::now();

    // ═══ Pipeline compilation on both GPUs ══════════════════════════
    println!("\n--- Pipeline compilation ---");
    let t0 = Instant::now();

    let dyn_pl_a = GpuDynHmcPipelines::new(&gpu_a);
    let rhmc_pl_a = GpuRhmcPipelines::new(&gpu_a);
    let dyn_pl_b = GpuDynHmcPipelines::new(&gpu_b);
    let rhmc_pl_b = GpuRhmcPipelines::new(&gpu_b);

    println!(
        "  Compiled on both GPUs: {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    // ═══ Quenched pre-thermalization (GPU A only) ═══════════════════
    println!("\n--- Quenched pre-therm (GPU A) ---");
    let lat = Lattice::hot_start(dims, beta, seed);
    let hmc_pipelines = GpuHmcStreamingPipelines::new(&gpu_a);
    let hmc_state = GpuHmcState::from_lattice(&gpu_a, &lat, beta);

    let t1 = Instant::now();
    let mut plaq_window: Vec<f64> = Vec::new();
    let mut pretherm_count = 0;
    let mut rng_seed = seed;

    loop {
        let r = gpu_hmc_trajectory_streaming(
            &gpu_a,
            &hmc_pipelines,
            &hmc_state,
            10,
            0.1,
            pretherm_count as u32,
            &mut rng_seed,
        );
        pretherm_count += 1;
        plaq_window.push(r.plaquette);
        if plaq_window.len() > 10 {
            plaq_window.remove(0);
        }
        if pretherm_count % 20 == 0 {
            let mean = plaq_window.iter().sum::<f64>() / plaq_window.len() as f64;
            let var = plaq_window.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
                / plaq_window.len() as f64;
            let rv = var.sqrt() / mean.max(1e-10);
            println!("  pretherm {pretherm_count}: <P>={mean:.6} sigma/<P>={rv:.2e}");
            if plaq_window.len() >= 10 && rv < 1e-3 {
                println!("  -> Stable.");
                break;
            }
        }
        if pretherm_count >= 200 {
            println!("  -> Max reached.");
            break;
        }
    }
    println!(
        "  {} traj in {:.1}s",
        pretherm_count,
        t1.elapsed().as_secs_f64()
    );

    let mut lat_cpu = Lattice::hot_start(dims, beta, seed + 1);
    hotspring_barracuda::lattice::gpu_hmc::gpu_links_to_lattice(&gpu_a, &hmc_state, &mut lat_cpu);

    // ═══ RHMC calibration ═══════════════════════════════════════════
    println!("\n--- RHMC calibration ---");
    let mut calibrator = if nf == 3 {
        RhmcCalibrator::new_nf2p1(mass, strange_mass, beta, dims)
    } else {
        RhmcCalibrator::new(nf, mass, beta, dims)
    };

    let base_config = calibrator.produce_config();

    // ═══ Initialize RHMC state on both GPUs ═════════════════════════
    let dyn_state_a = GpuDynHmcState::from_lattice(
        &gpu_a,
        &lat_cpu,
        beta,
        mass,
        base_config.cg_tol,
        base_config.cg_max_iter,
    );
    let rhmc_state_a = GpuRhmcState::new(&gpu_a, &base_config, dyn_state_a);

    let dyn_state_b = GpuDynHmcState::from_lattice(
        &gpu_b,
        &lat_cpu,
        beta,
        mass,
        base_config.cg_tol,
        base_config.cg_max_iter,
    );
    let rhmc_state_b = GpuRhmcState::new(&gpu_b, &base_config, dyn_state_b);

    let spectral = calibrator.calibrate_spectral(&gpu_a, &dyn_pl_a, &rhmc_state_a.gauge);
    println!(
        "  Spectral: lambda_min>={:.4e} lambda_max~={:.2}",
        spectral.lambda_min, spectral.lambda_max
    );

    // ═══ dt discovery (GPU A, sync path) ════════════════════════════
    println!("\n--- dt discovery ---");
    let mut probe_dt = 0.02_f64;
    for round in 0..20 {
        let mut config = calibrator.produce_config();
        config.dt = probe_dt;
        config.n_md_steps = 1;
        let r = gpu_rhmc_trajectory(
            &gpu_a,
            &dyn_pl_a,
            &rhmc_pl_a,
            &rhmc_state_a,
            &config,
            &mut rng_seed,
        );
        let dh = r.delta_h.abs();
        println!(
            "  probe {:>2}: dt={:.2e} |dH|={:.2e} {} cg={}",
            round + 1,
            probe_dt,
            dh,
            if r.accepted { "Y" } else { "N" },
            r.total_cg_iterations
        );
        if dh < 1.5 {
            println!("  -> dt={probe_dt:.2e} viable");
            break;
        }
        if probe_dt < 1e-6 {
            break;
        }
        probe_dt *= 0.5;
    }

    let mut config = calibrator.produce_config();
    config.dt = probe_dt;
    config.n_md_steps = 1;

    // ═══ Build UnidirectionalRhmc on each GPU ═══════════════════════
    let mut uni_a = UnidirectionalRhmc::new(gpu_a, dyn_pl_a, rhmc_pl_a, rhmc_state_a);
    let mut uni_b = UnidirectionalRhmc::new(gpu_b, dyn_pl_b, rhmc_pl_b, rhmc_state_b);
    println!("  Unidirectional pipeline initialized on both GPUs.");

    // ═══ ESN weights (random init for cold start) ═══════════════════
    let esn_weights = build_default_esn_weights();

    // Check for persisted brain state
    if let Some(state) = load_brain_state(uni_a.adapter_name()) {
        println!(
            "  Loaded brain state for {}: dt={:.2e}, {} prior trajectories",
            state.gpu_name, state.optimal_dt, state.trajectories_observed
        );
        config.dt = state.optimal_dt;
    }

    // ═══ Brain channels + cortex thread ═════════════════════════════
    let (mut runner, obs_rx, sug_tx) =
        BrainRhmcRunner::new(config, rng_seed, rng_seed.wrapping_add(0x0005_DEEC_E66D));

    let cortex_weights = esn_weights;
    let cortex_handle = std::thread::spawn(move || {
        let mut cortex = NpuCortex::new(
            &cortex_weights,
            obs_rx,
            sug_tx,
            report_interval,
            checkpoint_interval,
        );
        cortex.run();
        cortex.stats().clone()
    });

    // ═══ Brain-steered production run ═══════════════════════════════
    println!("\n--- Brain-steered production ({n_trajs} trajectory pairs) ---");
    println!(
        "  {:>4} {:>12} {:>12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "traj", "GPU_A", "GPU_B", "P_A", "P_B", "dH_A", "dH_B", "acc_A", "acc_B"
    );

    let mut all_plaq_a: Vec<f64> = Vec::new();
    let mut all_plaq_b: Vec<f64> = Vec::new();
    let mut accept_a = 0usize;
    let mut accept_b = 0usize;

    for i in 0..n_trajs {
        let result = match runner.run_iteration(&mut uni_a, &mut uni_b) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("ERROR: {e}");
                std::process::exit(1);
            }
        };

        all_plaq_a.push(result.gpu_a.plaquette);
        all_plaq_b.push(result.gpu_b.plaquette);
        if result.gpu_a.accepted {
            accept_a += 1;
        }
        if result.gpu_b.accepted {
            accept_b += 1;
        }

        if (i + 1) % 5 == 0 || i == 0 || i + 1 == n_trajs {
            println!(
                "  {:>4} {:>12.3}s {:>12.3}s {:>8.6} {:>8.6} {:>8.3} {:>8.3} {:>8} {:>8}",
                i + 1,
                result.gpu_a.elapsed_secs,
                result.gpu_b.elapsed_secs,
                result.gpu_a.plaquette,
                result.gpu_b.plaquette,
                result.gpu_a.delta_h,
                result.gpu_b.delta_h,
                if result.gpu_a.accepted { "Y" } else { "N" },
                if result.gpu_b.accepted { "Y" } else { "N" },
            );
        }
    }

    // ═══ Shut down cortex ═══════════════════════════════════════════
    drop(runner);

    let cortex_summary = cortex_handle.join().ok();

    // ═══ Persist brain state ═══════════════════════════════════════
    // (simplified — we don't have access to runner internals after drop)

    // ═══ Summary ════════════════════════════════════════════════════
    let total_secs = t_total.elapsed().as_secs_f64();
    let n_a = all_plaq_a.len().max(1) as f64;
    let n_b = all_plaq_b.len().max(1) as f64;
    let mean_plaq_a = all_plaq_a.iter().sum::<f64>() / n_a;
    let mean_plaq_b = all_plaq_b.iter().sum::<f64>() / n_b;

    println!("\n══════════════════════════════════════════════════════════");
    println!("  Brain-Steered Dual-GPU RHMC Summary");
    println!("══════════════════════════════════════════════════════════");
    println!(
        "  GPU A ({}): <P>={mean_plaq_a:.6}  acc={:.0}%",
        uni_a.adapter_name(),
        accept_a as f64 / n_a * 100.0
    );
    println!(
        "  GPU B ({}): <P>={mean_plaq_b:.6}  acc={:.0}%",
        uni_b.adapter_name(),
        accept_b as f64 / n_b * 100.0
    );
    println!("  Trajectories: {n_trajs} pairs");
    println!(
        "  Wall time: {total_secs:.1}s ({:.2}h)",
        total_secs / 3600.0
    );

    if let Some(ref stats) = cortex_summary {
        let range = format!("all {} observations", stats.total_observations);
        println!("\n{}", stats.display_table(&range));
    }

    println!("\n  Pipeline topology: {}", topology.name);
    println!("  Substrate count: {}", substrates.len());
    println!("  Pipeline: brain-steered unidirectional (GPU-resident CG + NPU cortex)");
}

fn build_default_esn_weights() -> ExportedWeights {
    let input_size = 6;
    let reservoir_size = 64;
    let output_size = heads::NUM_HEADS;

    let mut seed = 0xCAFE_BABE_u64;
    let mut rng = || -> f32 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f32 / u64::MAX as f32) * 2.0 - 1.0
    };

    let w_in: Vec<f32> = (0..reservoir_size * input_size)
        .map(|_| rng() * 0.1)
        .collect();
    let w_res: Vec<f32> = (0..reservoir_size * reservoir_size)
        .map(|_| {
            let v = rng();
            if v.abs() > 0.8 { v * 0.5 } else { 0.0 }
        })
        .collect();
    let w_out: Vec<f32> = (0..output_size * reservoir_size)
        .map(|_| rng() * 0.01)
        .collect();

    ExportedWeights {
        w_in,
        w_res,
        w_out,
        input_size,
        reservoir_size,
        output_size,
        leak_rate: 0.3,
        activation: Activation::default(),
    }
}

fn nf_label(nf: usize, mass: f64, strange_mass: f64) -> String {
    match nf {
        2 => format!("2 (m={mass})"),
        3 => format!("2+1 (m_l={mass}, m_s={strange_mass})"),
        _ => format!("{nf}"),
    }
}

fn parse_nf(args: &[String]) -> usize {
    for (i, a) in args.iter().enumerate() {
        if a == "--nf"
            && let Some(v) = args.get(i + 1)
        {
            return match v.as_str() {
                "2+1" | "3" => 3,
                "2" => 2,
                o => o.parse().unwrap_or(2),
            };
        }
    }
    2
}

fn parse_arg(args: &[String], name: &str, default: usize) -> usize {
    for (i, a) in args.iter().enumerate() {
        if a.starts_with(&format!("{name}=")) {
            return a.split('=').nth(1).unwrap().parse().unwrap_or(default);
        }
        if a == name
            && let Some(v) = args.get(i + 1)
        {
            return v.parse().unwrap_or(default);
        }
    }
    default
}

fn parse_arg_f64(args: &[String], name: &str, default: f64) -> f64 {
    for (i, a) in args.iter().enumerate() {
        if a.starts_with(&format!("{name}=")) {
            return a.split('=').nth(1).unwrap().parse().unwrap_or(default);
        }
        if a == name
            && let Some(v) = args.get(i + 1)
        {
            return v.parse().unwrap_or(default);
        }
    }
    default
}
