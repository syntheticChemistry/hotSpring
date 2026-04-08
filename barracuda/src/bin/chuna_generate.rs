// SPDX-License-Identifier: AGPL-3.0-or-later

//! Chuna Engine: Generate — produce thermalized ILDG gauge configurations.
//!
//! Runs pure-gauge HMC (or dynamical RHMC) to generate ensembles of
//! thermalized gauge configurations, saved in ILDG/LIME format for
//! interop with MILC/Bazavov analysis tools.
//!
//! Output directory structure:
//! ```text
//! output_dir/
//!   ensemble.json          — manifest with action params + config list
//!   conf_000100.lime       — ILDG gauge config at trajectory 100
//!   conf_000110.lime       — ...
//!   measurements/
//!     conf_000100.json     — gauge observables for conf 100
//!     ...
//! ```
//!
//! # Usage
//!
//! ```bash
//! # Quenched 8⁴ ensemble at β=6.0:
//! cargo run --release --bin chuna_generate -- \
//!   --lattice=8 --beta=6.0 --therm=200 --configs=50 --interval=10 --outdir=data/b6.0_L8
//!
//! # Asymmetric 16³×32 (finite-temperature):
//! cargo run --release --bin chuna_generate -- \
//!   --ns=16 --nt=32 --beta=6.0 --therm=500 --configs=100 --interval=10
//!
//! # Fully specified geometry:
//! cargo run --release --bin chuna_generate -- \
//!   --dims=8,8,8,16 --beta=6.0 --therm=200 --configs=50 --interval=10
//! ```

use hotspring_barracuda::dag_provenance::{DagEvent, DagSession};
use hotspring_barracuda::lattice::gradient_flow::{
    FlowIntegrator, find_t0, find_w0, run_flow, topological_charge,
};
use hotspring_barracuda::lattice::hmc::{HmcConfig, IntegratorType, hmc_trajectory};
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuDynHmcPipelines, GpuDynHmcState, GpuHmcState, GpuHmcStreamingPipelines,
    gpu_dynamical_hmc_trajectory, gpu_hmc_trajectory_streaming, gpu_links_to_lattice,
};
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::ildg::{IldgMetadata, ildg_crc, write_gauge_config_file};
use hotspring_barracuda::lattice::measurement::{
    ConfigEntry, ConfigMeasurement, EnsembleManifest, FlowPoint, FlowResults, GaugeObservables,
    ImplementationInfo, RunManifest, TopologyResults, WilsonLoopEntry,
    format_dims, format_dims_id, min_spatial_dim, parse_dims_from_args,
};
use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::receipt_signing;
use hotspring_barracuda::validation::TelemetryWriter;
use hotspring_barracuda::lattice::qcdml::{
    QcdmlConfigInfo, QcdmlEnsembleInfo, generate_config_xml, generate_ensemble_xml,
};
use hotspring_barracuda::lattice::wilson::Lattice;

use std::time::Instant;

struct CliArgs {
    dims: [usize; 4],
    beta: f64,
    n_therm: usize,
    n_configs: usize,
    meas_interval: usize,
    n_md_steps: usize,
    dt: f64,
    seed: u64,
    outdir: String,
    flow: bool,
    flow_tmax: f64,
    flow_eps: f64,
    wilson_rmax: usize,
    telemetry: Option<String>,
    integrator: IntegratorType,
    gpu: bool,
    mass: f64,
    nf: usize,
    cg_tol: f64,
    cg_max_iter: usize,
}

fn parse_args() -> CliArgs {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let dims = parse_dims_from_args(&raw_args).unwrap_or([8, 8, 8, 8]);

    let mut args = CliArgs {
        dims,
        beta: 6.0,
        n_therm: 200,
        n_configs: 50,
        meas_interval: 10,
        n_md_steps: 20,
        dt: 0.05,
        seed: 42,
        outdir: "chuna_output".to_string(),
        flow: true,
        flow_tmax: 4.0,
        flow_eps: 0.01,
        wilson_rmax: 4,
        telemetry: None,
        integrator: IntegratorType::Omelyan,
        gpu: false,
        mass: 0.1,
        nf: 0,
        cg_tol: 1e-8,
        cg_max_iter: 5000,
    };

    for arg in &raw_args {
        if let Some(v) = arg.strip_prefix("--integrator=") {
            args.integrator = match v {
                "leapfrog" | "Leapfrog" => IntegratorType::Leapfrog,
                "omelyan" | "Omelyan" | "2mn" => IntegratorType::Omelyan,
                _ => {
                    eprintln!("Unknown integrator: {v}. Valid: leapfrog, omelyan");
                    std::process::exit(1);
                }
            };
        } else if let Some(v) = arg.strip_prefix("--beta=") {
            args.beta = v.parse().expect("--beta=F");
        } else if let Some(v) = arg.strip_prefix("--therm=") {
            args.n_therm = v.parse().expect("--therm=N");
        } else if let Some(v) = arg.strip_prefix("--configs=") {
            args.n_configs = v.parse().expect("--configs=N");
        } else if let Some(v) = arg.strip_prefix("--interval=") {
            args.meas_interval = v.parse().expect("--interval=N");
        } else if let Some(v) = arg.strip_prefix("--n-md=") {
            args.n_md_steps = v.parse().expect("--n-md=N");
        } else if let Some(v) = arg.strip_prefix("--dt=") {
            args.dt = v.parse().expect("--dt=F");
        } else if let Some(v) = arg.strip_prefix("--seed=") {
            args.seed = v.parse().expect("--seed=N");
        } else if let Some(v) = arg.strip_prefix("--outdir=") {
            args.outdir = v.to_string();
        } else if arg == "--no-flow" {
            args.flow = false;
        } else if let Some(v) = arg.strip_prefix("--flow-tmax=") {
            args.flow_tmax = v.parse().expect("--flow-tmax=F");
        } else if let Some(v) = arg.strip_prefix("--flow-eps=") {
            args.flow_eps = v.parse().expect("--flow-eps=F");
        } else if let Some(v) = arg.strip_prefix("--wilson-rmax=") {
            args.wilson_rmax = v.parse().expect("--wilson-rmax=N");
        } else if let Some(v) = arg.strip_prefix("--telemetry=") {
            args.telemetry = Some(v.to_string());
        } else if arg == "--gpu" {
            args.gpu = true;
        } else if let Some(v) = arg.strip_prefix("--mass=") {
            args.mass = v.parse().expect("--mass=F");
        } else if let Some(v) = arg.strip_prefix("--nf=") {
            args.nf = v.parse().expect("--nf=N");
        } else if let Some(v) = arg.strip_prefix("--cg-tol=") {
            args.cg_tol = v.parse().expect("--cg-tol=F");
        } else if let Some(v) = arg.strip_prefix("--cg-max-iter=") {
            args.cg_max_iter = v.parse().expect("--cg-max-iter=N");
        }
    }
    args
}

enum GpuBackend {
    Quenched {
        pipelines: GpuHmcStreamingPipelines,
        state: GpuHmcState,
    },
    Dynamical {
        pipelines: GpuDynHmcPipelines,
        state: GpuDynHmcState,
        _n_fields: usize,
    },
}

impl GpuBackend {
    fn gauge_state(&self) -> &GpuHmcState {
        match self {
            GpuBackend::Quenched { state, .. } => state,
            GpuBackend::Dynamical { state, .. } => &state.gauge,
        }
    }
}

fn run_gpu_trajectory(
    gpu: &GpuF64,
    backend: &GpuBackend,
    n_md_steps: usize,
    dt: f64,
    seed: &mut u64,
    traj_id: u32,
) -> (bool, f64, f64) {
    match backend {
        GpuBackend::Quenched { pipelines, state } => {
            let r = gpu_hmc_trajectory_streaming(gpu, pipelines, state, n_md_steps, dt, traj_id, seed)
                .expect("streaming HMC trajectory");
            (r.accepted, r.delta_h, r.plaquette)
        }
        GpuBackend::Dynamical { pipelines, state, .. } => {
            let r = gpu_dynamical_hmc_trajectory(gpu, pipelines, state, n_md_steps, dt, seed);
            (r.accepted, r.delta_h, r.plaquette)
        }
    }
}

fn main() {
    let args = parse_args();
    let run_manifest = RunManifest::capture("chuna_generate");
    let mut telemetry = match &args.telemetry {
        Some(p) => TelemetryWriter::new(p),
        None => TelemetryWriter::disabled(),
    };
    let dims = args.dims;
    let vol: usize = dims.iter().product();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Chuna Engine: Generate ILDG Gauge Configurations          ║");
    println!("║  hotSpring-barracuda — pure Rust (GPU optional)            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Lattice:    {} ({} sites)", format_dims(dims), vol);
    println!("  β:          {:.4}", args.beta);
    println!("  Therm:      {} HMC trajectories", args.n_therm);
    println!("  Configs:    {} (every {} trajectories)", args.n_configs, args.meas_interval);
    let integrator_name = match args.integrator {
        IntegratorType::Leapfrog => "Leapfrog",
        IntegratorType::Omelyan => "Omelyan",
    };
    println!("  MD:         {} steps × dt={} ({})", args.n_md_steps, args.dt, integrator_name);
    if args.gpu {
        println!("  GPU:        enabled (dynamical Nf={}, mass={})", args.nf, args.mass);
    } else {
        println!("  GPU:        disabled (CPU path)");
    }
    println!("  Flow:       {}", if args.flow { "enabled" } else { "disabled" });
    println!("  Output:     {}", args.outdir);

    let nucleus = NucleusContext::detect();
    nucleus.print_banner();
    let mut run_manifest = if nucleus.any_alive() {
        run_manifest.with_nucleus(&nucleus)
    } else {
        run_manifest
    };

    let mut dag_session = DagSession::begin(&nucleus, "chuna_generate");

    // Create output directories
    std::fs::create_dir_all(&args.outdir).expect("create output dir");
    let meas_dir = format!("{}/measurements", args.outdir);
    std::fs::create_dir_all(&meas_dir).expect("create measurements dir");

    let ensemble_id = format!("hotspring_b{:.2}_{}", args.beta, format_dims_id(dims));

    let mut manifest = EnsembleManifest::new(&ensemble_id, dims, args.beta);
    manifest.algorithm.integrator = integrator_name.to_string();
    manifest.algorithm.dt = args.dt;
    manifest.algorithm.n_md_steps = args.n_md_steps;
    manifest.algorithm.n_therm = args.n_therm;
    manifest.algorithm.meas_interval = args.meas_interval;
    manifest.algorithm.seed = args.seed;

    let total_start = Instant::now();

    // Phase 1: Thermalize
    println!("\n═══ Phase 1: Thermalization ({} trajectories) ═══", args.n_therm);
    let mut lattice = Lattice::hot_start(dims, args.beta, args.seed);
    let mut hmc_cfg = HmcConfig {
        n_md_steps: args.n_md_steps,
        dt: args.dt,
        seed: args.seed,
        integrator: args.integrator,
    };

    // GPU initialization (when --gpu is set)
    let gpu_ctx: Option<(GpuF64, GpuBackend)> = if args.gpu {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        let gpu = rt.block_on(GpuF64::new()).expect("GPU initialization failed");
        println!("  GPU:   {}", gpu.adapter_name);
        if args.nf > 0 {
            let n_fields = args.nf / 4 + if args.nf % 4 > 0 { 1 } else { 0 };
            let dyn_pip = GpuDynHmcPipelines::new(&gpu);
            let dyn_state = GpuDynHmcState::from_lattice_multi(
                &gpu, &lattice, args.beta, args.mass,
                args.cg_tol, args.cg_max_iter, n_fields,
            );
            Some((gpu, GpuBackend::Dynamical { pipelines: dyn_pip, state: dyn_state, _n_fields: n_fields }))
        } else {
            let q_pip = GpuHmcStreamingPipelines::new(&gpu);
            let q_state = GpuHmcState::from_lattice(&gpu, &lattice, args.beta);
            Some((gpu, GpuBackend::Quenched { pipelines: q_pip, state: q_state }))
        }
    } else {
        None
    };

    let therm_start = Instant::now();
    let mut n_accepted = 0usize;
    for i in 0..args.n_therm {
        let (accepted, delta_h, plaquette) = if let Some((ref gpu, ref backend)) = gpu_ctx {
            run_gpu_trajectory(gpu, backend, args.n_md_steps, args.dt, &mut hmc_cfg.seed, i as u32)
        } else {
            let r = hmc_trajectory(&mut lattice, &mut hmc_cfg);
            (r.accepted, r.delta_h, r.plaquette)
        };
        if accepted {
            n_accepted += 1;
        }
        telemetry.log("therm", "plaquette", plaquette);
        telemetry.log("therm", "delta_h", delta_h);
        telemetry.log("therm", "accepted", if accepted { 1.0 } else { 0.0 });
        if (i + 1) % 50 == 0 || i + 1 == args.n_therm {
            println!(
                "  [{:>5}/{}] ⟨P⟩={:.6}  acc={:.1}%  ΔH={:.4}  ({:.1}s)",
                i + 1,
                args.n_therm,
                plaquette,
                100.0 * n_accepted as f64 / (i + 1) as f64,
                delta_h,
                therm_start.elapsed().as_secs_f64()
            );
        }
    }
    // Sync GPU -> CPU lattice after thermalization
    if let Some((ref gpu, ref backend)) = gpu_ctx {
        gpu_links_to_lattice(gpu, backend.gauge_state(), &mut lattice);
    }
    let therm_secs = therm_start.elapsed().as_secs_f64();
    println!(
        "  Thermalization done: ⟨P⟩={:.6}, acceptance={:.1}% ({:.1}s)",
        lattice.average_plaquette(),
        100.0 * n_accepted as f64 / args.n_therm as f64,
        therm_secs
    );

    if let Some(ref mut dag) = dag_session {
        dag.append(&nucleus, DagEvent {
            phase: "thermalize".to_string(),
            input_hash: None,
            output_hash: None,
            wall_seconds: therm_secs,
            summary: serde_json::json!({
                "trajectories": args.n_therm,
                "acceptance": 100.0 * n_accepted as f64 / args.n_therm as f64,
                "plaquette": lattice.average_plaquette(),
            }),
        });
    }

    // Phase 2: Generate and save configurations
    println!(
        "\n═══ Phase 2: Generate {} configs (interval={}) ═══",
        args.n_configs, args.meas_interval
    );
    let _gen_start = Instant::now();
    let mut traj_counter = args.n_therm;

    for cfg_idx in 0..args.n_configs {
        let cfg_start = Instant::now();

        // Run interval trajectories
        let mut last_accepted = false;
        let mut last_delta_h = 0.0;
        for j in 0..args.meas_interval {
            let traj_idx = (traj_counter + j + 1) as u32;
            let (accepted, delta_h, plaquette) =
                if let Some((ref gpu, ref backend)) = gpu_ctx {
                    run_gpu_trajectory(gpu, backend, args.n_md_steps, args.dt, &mut hmc_cfg.seed, traj_idx)
                } else {
                    let r = hmc_trajectory(&mut lattice, &mut hmc_cfg);
                    (r.accepted, r.delta_h, r.plaquette)
                };
            telemetry.log("generate", "plaquette", plaquette);
            telemetry.log("generate", "delta_h", delta_h);
            telemetry.log("generate", "accepted", if accepted { 1.0 } else { 0.0 });
            last_accepted = accepted;
            last_delta_h = delta_h;
            traj_counter += 1;
        }
        // Sync GPU -> CPU for saving and measurement
        if let Some((ref gpu, ref backend)) = gpu_ctx {
            gpu_links_to_lattice(gpu, backend.gauge_state(), &mut lattice);
        }

        let hmc_accepted = last_accepted;
        let hmc_delta_h = last_delta_h;

        // Save ILDG config
        let mut meta = IldgMetadata::for_lattice(&lattice, traj_counter);
        meta.ensemble_id = ensemble_id.clone();
        let conf_filename = format!("conf_{traj_counter:06}.lime");
        let conf_path = format!("{}/{conf_filename}", args.outdir);
        write_gauge_config_file(&conf_path, &lattice, &meta)
            .expect("write ILDG config");

        // Compute ILDG CRC (POSIX cksum) of the written file
        let file_bytes = std::fs::read(&conf_path).expect("read back config");
        let checksum = ildg_crc(&file_bytes);

        // Measure gauge observables
        let plaq = lattice.average_plaquette();
        let ns = [lattice.dims[0], lattice.dims[1], lattice.dims[2]];
        let spatial_vol = ns[0] * ns[1] * ns[2];
        let mut poly_re_sum = 0.0;
        let mut poly_im_sum = 0.0;
        for ix in 0..ns[0] {
            for iy in 0..ns[1] {
                for iz in 0..ns[2] {
                    let l = lattice.polyakov_loop([ix, iy, iz]);
                    poly_re_sum += l.re;
                    poly_im_sum += l.im;
                }
            }
        }
        let poly_re = poly_re_sum / spatial_vol as f64;
        let poly_im = poly_im_sum / spatial_vol as f64;
        let poly_abs = (poly_re * poly_re + poly_im * poly_im).sqrt();

        let mut measurement = ConfigMeasurement::new(&ensemble_id, traj_counter, &meta.lfn);
        measurement.implementation = Some(ImplementationInfo::auto_detect_cpu_only());
        measurement.gauge = GaugeObservables {
            plaquette: plaq,
            polyakov_abs: poly_abs,
            polyakov_re: poly_re,
            polyakov_im: poly_im,
            action_density: 6.0 * (1.0 - plaq),
        };
        measurement.diagnostics = Some(
            hotspring_barracuda::lattice::measurement::HmcDiagnostics {
                accepted: hmc_accepted,
                delta_h: hmc_delta_h,
                cg_iterations: None,
                trajectory_seconds: cfg_start.elapsed().as_secs_f64(),
            },
        );

        // Gradient flow
        if args.flow {
            let mut flow_lat = lattice.clone();
            let measure_interval = (0.05 / args.flow_eps).max(1.0) as usize;
            let flow_results = run_flow(
                &mut flow_lat,
                FlowIntegrator::Lscfrk3w7,
                args.flow_eps,
                args.flow_tmax,
                measure_interval,
            );
            let t0 = find_t0(&flow_results);
            let w0 = find_w0(&flow_results);
            let q = topological_charge(&flow_lat);

            let flow_curve: Vec<FlowPoint> = flow_results
                .iter()
                .map(|fm| FlowPoint {
                    t: fm.t,
                    energy_density: fm.energy_density,
                    t2_e: fm.t2_e,
                })
                .collect();

            measurement.flow = Some(FlowResults {
                integrator: "Lscfrk3w7".to_string(),
                epsilon: args.flow_eps,
                t_max: args.flow_tmax,
                t0,
                w0,
                flow_curve,
            });

            measurement.topology = Some(TopologyResults {
                charge: q,
                flow_time: args.flow_tmax,
            });
        }

        // Wilson loops
        let r_max = args.wilson_rmax.min(min_spatial_dim(dims) / 2);
        let t_max_wl = r_max.min(dims[3] / 2);
        let mut wl_entries = Vec::new();
        for r in 1..=r_max {
            for t in 1..=t_max_wl {
                let val = lattice.spatial_temporal_wilson_loop(r, t);
                wl_entries.push(WilsonLoopEntry { r, t, value: val });
            }
        }
        if !wl_entries.is_empty() {
            measurement.wilson_loops = Some(wl_entries);
        }

        measurement.wall_seconds = cfg_start.elapsed().as_secs_f64();

        // Write measurement JSON
        let meas_path = format!("{meas_dir}/conf_{traj_counter:06}.json");
        std::fs::write(&meas_path, measurement.to_json()).expect("write measurement");

        // Add to manifest
        manifest.configs.push(ConfigEntry {
            trajectory: traj_counter,
            filename: conf_filename.clone(),
            ildg_lfn: meta.lfn.clone(),
            checksum_crc32: Some(format!("{checksum:08x}")),
            checksum_ildg_crc: Some(checksum),
            plaquette: plaq,
        });

        // Write QCDml config XML
        let ens_info = QcdmlEnsembleInfo::for_manifest(&manifest);
        let cfg_info = QcdmlConfigInfo::from_provenance(&manifest.provenance);
        let config_xml = generate_config_xml(
            manifest.configs.last().unwrap(),
            &meta,
            &ens_info,
            &cfg_info,
        );
        let xml_path = format!("{}/conf_{traj_counter:06}.xml", args.outdir);
        std::fs::write(&xml_path, &config_xml).expect("write config XML");

        let t0_str = measurement
            .flow
            .as_ref()
            .and_then(|f| f.t0)
            .map_or("N/A".to_string(), |v| format!("{v:.4}"));
        let q_str = measurement
            .topology
            .as_ref()
            .map_or("N/A".to_string(), |t| format!("{:.2}", t.charge));

        println!(
            "  [{:>3}/{}] traj={:>5}  ⟨P⟩={:.6}  |L|={:.4}  t₀={}  Q={}  ({:.1}s)",
            cfg_idx + 1,
            args.n_configs,
            traj_counter,
            plaq,
            poly_abs,
            t0_str,
            q_str,
            cfg_start.elapsed().as_secs_f64()
        );
    }

    // DAG event for generation phase
    if let Some(ref mut dag) = dag_session {
        dag.append(&nucleus, DagEvent {
            phase: "generate".to_string(),
            input_hash: None,
            output_hash: None,
            wall_seconds: total_start.elapsed().as_secs_f64(),
            summary: serde_json::json!({
                "n_configs": args.n_configs,
                "interval": args.meas_interval,
            }),
        });
    }

    // Finalize DAG session
    if let Some(dag) = dag_session {
        let prov = dag.dehydrate(&nucleus);
        run_manifest.set_dag_provenance(&prov);
    }

    // Write manifest (with run metadata for reproducibility)
    manifest.run = Some(run_manifest);
    let manifest_path = format!("{}/ensemble.json", args.outdir);
    let manifest_json = manifest.to_json();
    std::fs::write(&manifest_path, &manifest_json).expect("write manifest");

    // Sign the manifest receipt if bearDog is available
    if let Ok(mut receipt_val) = serde_json::from_str::<serde_json::Value>(&manifest_json) {
        receipt_signing::sign_and_embed(
            &nucleus,
            &mut receipt_val,
            std::path::Path::new(&manifest_path),
        );
    }

    // Write QCDml ensemble XML
    let ens_info = QcdmlEnsembleInfo::for_manifest(&manifest);
    let ensemble_xml = generate_ensemble_xml(&manifest, &ens_info);
    let xml_path = format!("{}/ensemble.xml", args.outdir);
    std::fs::write(&xml_path, &ensemble_xml).expect("write ensemble XML");

    let total_secs = total_start.elapsed().as_secs_f64();
    println!("\n═══ Complete ═══");
    println!("  Configs:   {} ILDG files in {}", args.n_configs, args.outdir);
    println!("  Manifest:  {manifest_path}");
    println!("  QCDml:     {xml_path}");
    println!("  Total:     {total_secs:.1}s");
}
