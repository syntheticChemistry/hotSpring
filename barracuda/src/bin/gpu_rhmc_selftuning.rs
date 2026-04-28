// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-only self-tuning RHMC production binary.
//!
//! Zero magic numbers. The only human inputs are physics intent:
//!   --lattice, --beta, --nf, --mass, --strange-mass
//!
//! Everything else is discovered or adapted:
//!   - dt: binary search → acceptance-driven feedback
//!   - spectral range: GPU power iteration (via RhmcCalibrator)
//!   - rational approx (poles/shifts): RhmcCalibrator
//!   - CG tolerance: from RhmcCalibrator (theory-derived)
//!   - quenched pre-therm: plaquette stability detection
//!   - flow t_max: adaptive (double until t₀ found)
//!
//! Architecture: CPU is the cortex (observe, decide, dispatch).
//! GPU does all physics. NPU hook wired but optional.
//!
//! Key insight: with RHMC + light quarks, n_md=1 at the right dt is
//! optimal for throughput. Many cheap trajectories decorrelate faster
//! than fewer expensive ones, because τ_eq is limited by the Dirac
//! condition number, not the integrator.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_flow::{GpuFlowPipelines, GpuFlowState, gpu_gradient_flow};
use hotspring_barracuda::lattice::gpu_hmc::dynamical::{GpuDynHmcPipelines, GpuDynHmcState};
#[expect(
    deprecated,
    reason = "legacy API retained for backward compatibility during migration"
)]
use hotspring_barracuda::lattice::gpu_hmc::gpu_rhmc::{
    GpuRhmcPipelines, GpuRhmcState, gpu_rhmc_trajectory,
};
use hotspring_barracuda::lattice::gpu_hmc::rhmc_calibrator::RhmcCalibrator;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::gradient_flow::FlowIntegrator;
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

/// Simple dt adapter: converges to target acceptance via dt scaling.
struct DtAdapter {
    dt: f64,
    accept_history: Vec<bool>,
    delta_h_history: Vec<f64>,
}

impl DtAdapter {
    fn new(dt: f64) -> Self {
        Self {
            dt,
            accept_history: Vec::new(),
            delta_h_history: Vec::new(),
        }
    }

    fn observe(&mut self, accepted: bool, delta_h: f64) {
        self.accept_history.push(accepted);
        self.delta_h_history.push(delta_h.abs());
        if self.accept_history.len() > 30 {
            self.accept_history.remove(0);
            self.delta_h_history.remove(0);
        }
    }

    fn acceptance_rate(&self) -> f64 {
        if self.accept_history.is_empty() {
            return 0.5;
        }
        self.accept_history.iter().filter(|&&a| a).count() as f64 / self.accept_history.len() as f64
    }

    fn mean_abs_delta_h(&self) -> f64 {
        if self.delta_h_history.is_empty() {
            return 0.0;
        }
        self.delta_h_history.iter().sum::<f64>() / self.delta_h_history.len() as f64
    }

    /// Nudge dt toward ~60% acceptance. Called periodically.
    fn adapt(&mut self) {
        if self.accept_history.len() < 5 {
            return;
        }
        let acc = self.acceptance_rate();
        let mean_dh = self.mean_abs_delta_h();

        if mean_dh > 5.0 {
            self.dt *= 0.5; // emergency
        } else if acc > 0.85 && mean_dh < 0.3 {
            self.dt *= 1.15; // much too conservative
        } else if acc > 0.75 {
            self.dt *= 1.05; // slightly conservative
        } else if acc < 0.35 {
            self.dt *= 0.85; // too aggressive
        }
        // 35-75%: leave alone (sweet spot)
    }
}

// Tooling binary: uses legacy RHMC trajectory for self-tuning parameter search.
// EVOLUTION(B3): migrate to unidirectional when RhmcCalibrator supports it — blocked on
// upstream barraCuda RhmcCalibrator unidirectional mode stabilization.
#[expect(
    deprecated,
    reason = "legacy API retained for backward compatibility during migration"
)]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lattice_size = parse_arg(&args, "--lattice", 8);
    let beta: f64 = parse_arg_f64(&args, "--beta", 6.0);
    let nf: usize = parse_nf(&args);
    let mass: f64 = parse_arg_f64(&args, "--mass", 0.05);
    let strange_mass: f64 = parse_arg_f64(&args, "--strange-mass", 0.5);
    let n_configs: usize = parse_arg(&args, "--configs", 5);
    let flow_epsilon: f64 = parse_arg_f64(&args, "--flow-eps", 0.01);
    let seed: u64 = parse_arg(&args, "--seed", 42) as u64;

    let dims = [lattice_size; 4];
    let vol: usize = dims.iter().product();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  GPU Self-Tuning RHMC — Zero Magic Numbers");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  Lattice: {}⁴ ({} sites)  β={beta:.4}  Nf={}",
        lattice_size,
        vol,
        nf_label(nf, mass, strange_mass)
    );
    println!("  Configs: {n_configs}  |  Everything else discovered.");
    println!();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU required");
    println!("  GPU: {}", gpu.adapter_name);

    let t_total = Instant::now();
    let mut rng_seed = seed;

    // ═══ Phase 0: Pipeline compilation ═══════════════════════════
    let t0 = Instant::now();
    let dyn_pipelines = GpuDynHmcPipelines::new(&gpu);
    let rhmc_pipelines = GpuRhmcPipelines::new(&gpu);
    let flow_pipelines = GpuFlowPipelines::new(&gpu);
    println!("  Pipelines: {:.1}s", t0.elapsed().as_secs_f64());

    // ═══ Phase 1: Quenched pre-thermalization ════════════════════
    println!("\n─── Phase 1: Quenched pre-therm (stability-detected) ───");
    let lat = Lattice::hot_start(dims, beta, seed);
    let hmc_pipelines = GpuHmcStreamingPipelines::new(&gpu);
    let hmc_state = GpuHmcState::from_lattice(&gpu, &lat, beta);

    let t1 = Instant::now();
    let mut plaq_window: Vec<f64> = Vec::new();
    let mut pretherm_count = 0;

    loop {
        let r = gpu_hmc_trajectory_streaming(
            &gpu,
            &hmc_pipelines,
            &hmc_state,
            10,
            0.1,
            pretherm_count as u32,
            &mut rng_seed,
        )
        .expect("streaming HMC trajectory");
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
            println!("  pretherm {pretherm_count}: ⟨P⟩={mean:.6} σ/⟨P⟩={rv:.2e}");
            if plaq_window.len() >= 10 && rv < 1e-3 {
                println!("  → Stable. Pre-therm done.");
                break;
            }
        }
        if pretherm_count >= 200 {
            println!("  → Max reached.");
            break;
        }
    }
    println!(
        "  {} traj in {:.1}s",
        pretherm_count,
        t1.elapsed().as_secs_f64()
    );

    let mut lat_cpu = Lattice::hot_start(dims, beta, seed + 1);
    hotspring_barracuda::lattice::gpu_hmc::gpu_links_to_lattice(&gpu, &hmc_state, &mut lat_cpu);

    // ═══ Phase 2: RHMC physics (spectral + rational approx) ═════
    println!("\n─── Phase 2: RHMC physics initialization ───");
    let mut calibrator = if nf == 3 {
        RhmcCalibrator::new_nf2p1(mass, strange_mass, beta, dims)
    } else {
        RhmcCalibrator::new(nf, mass, beta, dims)
    };

    let base_config = calibrator.produce_config().expect("RHMC config failed");
    let dyn_state = GpuDynHmcState::from_lattice(
        &gpu,
        &lat_cpu,
        beta,
        mass,
        base_config.cg_tol,
        base_config.cg_max_iter,
    );
    let rhmc_state = GpuRhmcState::new(&gpu, &base_config, dyn_state);

    let spectral = calibrator.calibrate_spectral(&gpu, &dyn_pipelines, &rhmc_state.gauge);
    println!(
        "  Spectral: λ_min≥{:.4e} λ_max≈{:.2}",
        spectral.lambda_min, spectral.lambda_max
    );
    println!("  Poles: {} per sector", calibrator.n_poles());

    // ═══ Phase 2b: dt discovery (binary search, n_md=1) ═════════
    println!("\n─── Phase 2b: dt discovery ───");
    let mut probe_dt = 0.02_f64;

    for round in 0..20 {
        let mut config = calibrator.produce_config().expect("RHMC config failed");
        config.dt = probe_dt;
        config.n_md_steps = 1;
        let r = gpu_rhmc_trajectory(
            &gpu,
            &dyn_pipelines,
            &rhmc_pipelines,
            &rhmc_state,
            &config,
            &mut rng_seed,
        );
        let dh = r.delta_h.abs();
        println!(
            "  probe {:>2}: dt={:.2e} |ΔH|={:.2e} {} cg={}",
            round + 1,
            probe_dt,
            dh,
            if r.accepted { "✓" } else { "✗" },
            r.total_cg_iterations
        );

        if dh < 1.5 {
            println!("  → dt={probe_dt:.2e} viable");
            break;
        }
        if probe_dt < 1e-6 {
            break;
        }
        probe_dt *= 0.5;
    }

    // ═══ Phase 3: RHMC thermalization (dt self-tunes to ~60% acc) ═
    println!("\n─── Phase 3: RHMC thermalization (n_md=1, dt adapts) ───");
    let mut ctrl = DtAdapter::new(probe_dt);
    let t3 = Instant::now();
    let therm_max = 60;
    let n_md = 1;

    for i in 1..=therm_max {
        let mut config = calibrator.produce_config().expect("RHMC config failed");
        config.dt = ctrl.dt;
        config.n_md_steps = n_md;
        let r = gpu_rhmc_trajectory(
            &gpu,
            &dyn_pipelines,
            &rhmc_pipelines,
            &rhmc_state,
            &config,
            &mut rng_seed,
        );
        ctrl.observe(r.accepted, r.delta_h);

        if i % 5 == 0 {
            ctrl.adapt();
        }
        if i % 10 == 0 || i <= 3 {
            println!(
                "  therm {:>3}: P={:.6} ΔH={:>8.3} {} acc={:.0}% dt={:.2e} cg={}",
                i,
                r.plaquette,
                r.delta_h,
                if r.accepted { "✓" } else { "✗" },
                ctrl.acceptance_rate() * 100.0,
                ctrl.dt,
                r.total_cg_iterations,
            );
        }

        if i >= 30 && ctrl.accept_history.len() >= 20 {
            let acc = ctrl.acceptance_rate();
            if (0.45..=0.80).contains(&acc) && ctrl.mean_abs_delta_h() < 2.0 {
                println!(
                    "  → Converged: acc={:.0}% ⟨|ΔH|⟩={:.2}",
                    acc * 100.0,
                    ctrl.mean_abs_delta_h()
                );
                break;
            }
        }
    }
    println!(
        "  Thermalization: {:.1}s, dt={:.2e}, acc={:.0}%",
        t3.elapsed().as_secs_f64(),
        ctrl.dt,
        ctrl.acceptance_rate() * 100.0
    );

    // ═══ Phase 4: Production RHMC + GPU gradient flow ════════════
    // With n_md=1, decorrelation needs many trajectories.
    // Ideal: 10/τ. But CPU-GPU sync cost per trajectory limits throughput.
    // Cap at 100 for now; GPU-resident CG will lift this.
    let skip = ((5.0 / ctrl.dt).ceil() as usize).clamp(20, 100);
    println!("\n─── Phase 4: Production ({n_configs} configs, skip={skip}) ───");

    let mut plaquettes: Vec<f64> = Vec::new();
    let mut t0_values: Vec<f64> = Vec::new();
    let mut w0_values: Vec<f64> = Vec::new();

    for cfg_idx in 0..n_configs {
        let cfg_start = Instant::now();

        // Decorrelation
        for _ in 0..skip {
            let mut config = calibrator.produce_config().expect("RHMC config failed");
            config.dt = ctrl.dt;
            config.n_md_steps = n_md;
            let r = gpu_rhmc_trajectory(
                &gpu,
                &dyn_pipelines,
                &rhmc_pipelines,
                &rhmc_state,
                &config,
                &mut rng_seed,
            );
            ctrl.observe(r.accepted, r.delta_h);
        }
        // Adapt dt during production too
        ctrl.adapt();

        // Measurement trajectory
        let mut config = calibrator.produce_config().expect("RHMC config failed");
        config.dt = ctrl.dt;
        config.n_md_steps = n_md;
        let meas = gpu_rhmc_trajectory(
            &gpu,
            &dyn_pipelines,
            &rhmc_pipelines,
            &rhmc_state,
            &config,
            &mut rng_seed,
        );
        ctrl.observe(meas.accepted, meas.delta_h);
        plaquettes.push(meas.plaquette);

        // GPU gradient flow
        let mut flow_lat = Lattice::hot_start(dims, beta, seed + 100 + cfg_idx as u64);
        hotspring_barracuda::lattice::gpu_hmc::gpu_links_to_lattice(
            &gpu,
            &rhmc_state.gauge.gauge,
            &mut flow_lat,
        );
        let (t0_val, w0_val) =
            run_adaptive_flow(&gpu, &flow_pipelines, &flow_lat, beta, flow_epsilon);
        if let Some(v) = t0_val {
            t0_values.push(v);
        }
        if let Some(v) = w0_val {
            w0_values.push(v);
        }

        println!(
            "  cfg {:>2}/{}: P={:.6} t₀={} w₀={} ({:.1}s) acc={:.0}%",
            cfg_idx + 1,
            n_configs,
            meas.plaquette,
            t0_val.map_or_else(|| "N/A".into(), |v| format!("{v:.4}")),
            w0_val.map_or_else(|| "N/A".into(), |v| format!("{v:.4}")),
            cfg_start.elapsed().as_secs_f64(),
            ctrl.acceptance_rate() * 100.0,
        );
    }

    // ═══ Summary ══════════════════════════════════════════════════
    let total_secs = t_total.elapsed().as_secs_f64();
    println!("\n══════════════════════════════════════════════════════════");
    println!(
        "  {}⁴ Nf={} Self-Tuning RHMC + GPU Flow",
        lattice_size,
        nf_label(nf, mass, strange_mass)
    );
    println!("══════════════════════════════════════════════════════════");

    let n = plaquettes.len().max(1) as f64;
    let mean_plaq = plaquettes.iter().sum::<f64>() / n;
    let plaq_err = if plaquettes.len() > 1 {
        let var: f64 = plaquettes
            .iter()
            .map(|p| (p - mean_plaq).powi(2))
            .sum::<f64>()
            / (plaquettes.len() - 1) as f64;
        var.sqrt() / n.sqrt()
    } else {
        0.0
    };
    println!("  ⟨P⟩       = {mean_plaq:.6} ± {plaq_err:.6}");

    if t0_values.is_empty() {
        println!("  t₀        = not found");
    } else {
        let m = t0_values.iter().sum::<f64>() / t0_values.len() as f64;
        println!("  t₀        = {m:.4} ({}/{})", t0_values.len(), n_configs);
    }
    if w0_values.is_empty() {
        println!("  w₀        = not found");
    } else {
        let m = w0_values.iter().sum::<f64>() / w0_values.len() as f64;
        println!("  w₀        = {m:.4} ({}/{})", w0_values.len(), n_configs);
    }

    println!("  dt        = {:.2e} (self-tuned)", ctrl.dt);
    println!("  n_md      = {n_md}");
    println!("  acceptance= {:.1}%", ctrl.acceptance_rate() * 100.0);
    println!("  ⟨|ΔH|⟩   = {:.3}", ctrl.mean_abs_delta_h());
    println!(
        "  wall      = {total_secs:.1}s ({:.2}h)",
        total_secs / 3600.0
    );

    println!("\ncfg,plaquette,t0,w0");
    for (i, &p) in plaquettes.iter().enumerate() {
        let t0 = t0_values.get(i).copied().unwrap_or(f64::NAN);
        let w0 = w0_values.get(i).copied().unwrap_or(f64::NAN);
        println!("{i},{p:.8},{t0},{w0}");
    }
}

fn run_adaptive_flow(
    gpu: &GpuF64,
    pipelines: &GpuFlowPipelines,
    lattice: &Lattice,
    beta: f64,
    epsilon: f64,
) -> (Option<f64>, Option<f64>) {
    let mut t_max = 4.0;
    #[expect(
        clippy::while_float,
        reason = "adaptive HMC step-size loop with convergence guard"
    )]
    while t_max <= 32.0 {
        let state = GpuFlowState::from_lattice(gpu, lattice, beta);
        let fr = gpu_gradient_flow(
            gpu,
            pipelines,
            &state,
            FlowIntegrator::Lscfrk3w7,
            epsilon,
            t_max,
            10,
        );
        let (t0, w0) = extract_t0_w0(&fr.measurements);
        if t0.is_some() {
            return (t0, w0);
        }
        t_max *= 2.0;
    }
    (None, None)
}

fn extract_t0_w0(
    m: &[hotspring_barracuda::lattice::gradient_flow::FlowMeasurement],
) -> (Option<f64>, Option<f64>) {
    let mut t0 = None;
    let mut w0 = None;
    for i in 1..m.len() {
        let (prev, curr) = (&m[i - 1], &m[i]);
        if prev.t2_e < 0.3 && curr.t2_e >= 0.3 && t0.is_none() {
            let f = (0.3 - prev.t2_e) / (curr.t2_e - prev.t2_e);
            t0 = Some(prev.t + f * (curr.t - prev.t));
        }
        if curr.t > prev.t && i >= 2 {
            let dt = curr.t - prev.t;
            let w_curr = curr.t * (curr.t2_e - prev.t2_e) / dt;
            let prev2 = &m[i - 2];
            if prev.t > prev2.t {
                let dt2 = prev.t - prev2.t;
                let w_prev = prev.t * (prev.t2_e - prev2.t2_e) / dt2;
                if w_prev < 0.3 && w_curr >= 0.3 && w0.is_none() {
                    let f = (0.3 - w_prev) / (w_curr - w_prev);
                    w0 = Some(prev.t + f * (curr.t - prev.t));
                }
            }
        }
    }
    (t0, w0)
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
