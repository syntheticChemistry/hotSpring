// SPDX-License-Identifier: AGPL-3.0-only

//! Production silicon-instrumented QCD runner.
//!
//! Runs the full QCD flavor ladder (quenched, Nf=1,2,2+1,3,4,2+1+1) at specified
//! lattice sizes with per-trajectory timing, FLOP counting, energy
//! accounting (RAPL + nvidia-smi/amdgpu), and QUDA comparison columns.
//!
//! # Usage
//!
//! ```bash
//! # Quenched 8^4 beta-scan on default GPU:
//! cargo run --release --bin production_silicon_qcd -- \
//!   --mode=quenched --lattice=8 --betas=5.5,5.69,6.0,6.2 --therm=200 --meas=100
//!
//! # Nf=2 RHMC at 16^4:
//! cargo run --release --bin production_silicon_qcd -- \
//!   --mode=nf2 --lattice=16 --betas=5.5,6.0 --mass=0.1 --therm=100 --meas=50
//!
//! # Nf=2+1 at 16^4 with gradient flow:
//! cargo run --release --bin production_silicon_qcd -- \
//!   --mode=nf2+1 --lattice=16 --betas=6.0 --mass=0.05 --strange-mass=0.5 \
//!   --therm=100 --meas=50 --flow --flow-configs=10
//!
//! # 32^4 quenched with specific GPU:
//! HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin production_silicon_qcd -- \
//!   --mode=quenched --lattice=32 --betas=5.5,6.0 --therm=200 --meas=100
//! ```

#[path = "../production_support.rs"]
mod production_support;

use hotspring_barracuda::bench::{EnergyReport, GpuTelemetry, PowerMonitor};
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_flow::{
    FlowReduceBuffers, GpuFlowPipelines, GpuFlowState, gpu_gradient_flow_resident,
};
use hotspring_barracuda::lattice::gpu_hmc::resident_shifted_cg::GpuResidentShiftedCgBuffers;
use hotspring_barracuda::lattice::gpu_hmc::true_multishift_cg::TrueMultiShiftBuffers;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuDynHmcPipelines, GpuDynHmcState, GpuHmcState, GpuHmcStreamingPipelines, GpuRhmcPipelines,
    GpuRhmcState, TrajectoryResult, UniHamiltonianBuffers, UniPipelines,
    gpu_hmc_trajectory_streaming, gpu_rhmc_trajectory_unidirectional,
};
use hotspring_barracuda::lattice::gradient_flow::{FlowIntegrator, find_t0, find_w0};
use hotspring_barracuda::lattice::rhmc::RhmcConfig;
use hotspring_barracuda::lattice::wilson::Lattice;

use production_support::{mean, std_dev};

use std::io::Write;
use std::time::Instant;

// ── Silicon budget (reused from bench_silicon_budget) ──

#[allow(dead_code)] // Fields used for future reporting / CSV parity with bench_silicon_budget
struct SiliconBudget {
    name: String,
    fp32_tflops: f64,
    df64_tflops: f64,
    memory_bw_gbs: f64,
    tmu_count: u32,
    boost_ghz: f64,
    vram_bytes: u64,
    total_tflops: f64,
    quda_style_tflops: f64,
}

fn lookup_budget(name: &str) -> SiliconBudget {
    let n = name.to_lowercase();
    if n.contains("3090") {
        SiliconBudget {
            name: name.to_string(),
            fp32_tflops: 35.6,
            df64_tflops: 3.24,
            memory_bw_gbs: 936.0,
            tmu_count: 328,
            boost_ghz: 1.70,
            vram_bytes: 24 * 1024 * 1024 * 1024,
            total_tflops: 284.0,
            quda_style_tflops: 2.0,
        }
    } else if n.contains("6950") {
        SiliconBudget {
            name: name.to_string(),
            fp32_tflops: 23.65,
            df64_tflops: 5.9,
            memory_bw_gbs: 576.0,
            tmu_count: 320,
            boost_ghz: 2.31,
            vram_bytes: 16 * 1024 * 1024 * 1024,
            total_tflops: 47.0,
            quda_style_tflops: 3.5,
        }
    } else if n.contains("4070") {
        SiliconBudget {
            name: name.to_string(),
            fp32_tflops: 29.15,
            df64_tflops: 2.6,
            memory_bw_gbs: 504.0,
            tmu_count: 184,
            boost_ghz: 2.48,
            vram_bytes: 12 * 1024 * 1024 * 1024,
            total_tflops: 233.0,
            quda_style_tflops: 1.5,
        }
    } else if n.contains("a100") {
        SiliconBudget {
            name: name.to_string(),
            fp32_tflops: 19.5,
            df64_tflops: 9.7,
            memory_bw_gbs: 2039.0,
            tmu_count: 432,
            boost_ghz: 1.41,
            vram_bytes: 80 * 1024 * 1024 * 1024,
            total_tflops: 830.0,
            quda_style_tflops: 5.0,
        }
    } else {
        SiliconBudget {
            name: name.to_string(),
            fp32_tflops: 10.0,
            df64_tflops: 1.0,
            memory_bw_gbs: 300.0,
            tmu_count: 128,
            boost_ghz: 1.5,
            vram_bytes: 8 * 1024 * 1024 * 1024,
            total_tflops: 20.0,
            quda_style_tflops: 0.5,
        }
    }
}

// ── QCD FLOP model ──

/// Estimated FLOPs for a single RHMC trajectory.
///
/// Based on standard lattice QCD operation counting for staggered fermions:
///   Force:  ~16000 FLOP/site per evaluation (4 directions × staple chains)
///   Dirac:  ~1512 FLOP/site per D†D application (8 neighbors × SU(3) matvec)
///   CG:    ~24 FLOP/site per iteration (dot + axpy)
///   PRNG:  ~100 FLOP/site per momentum/pseudofermion draw
///   Plaq:  ~1224 FLOP/site (6 planes × 204 FLOP)
fn estimate_traj_flops(vol: usize, n_md: usize, total_cg_iters: usize, is_quenched: bool) -> f64 {
    let v = vol as f64;
    let force = 16000.0 * v * n_md as f64;
    let plaq = 1224.0 * v * 2.0; // start + end
    let prng = 100.0 * v * 4.0; // momenta for all links

    if is_quenched {
        force + plaq + prng
    } else {
        let dirac_cg = 1512.0 * v * total_cg_iters as f64;
        let cg_overhead = 24.0 * v * total_cg_iters as f64;
        let fermion_prng = 100.0 * v; // pseudofermion draw
        force + dirac_cg + cg_overhead + plaq + prng + fermion_prng
    }
}

/// Memory traffic estimate for a trajectory (bytes).
fn estimate_traj_bytes(vol: usize, n_md: usize, total_cg_iters: usize, is_quenched: bool) -> f64 {
    let n_links = vol * 4;
    let link_bytes = n_links as f64 * 18.0 * 8.0; // SU(3) = 18 f64
    let force_rw = link_bytes * 2.0 * n_md as f64; // read links + write force per step
    let update_rw = link_bytes * 3.0 * n_md as f64; // read links + mom, write links per step

    if is_quenched {
        force_rw + update_rw
    } else {
        let vec_bytes = vol as f64 * 6.0 * 8.0; // staggered fermion = 3 complex
        let cg_rw = (link_bytes + vec_bytes * 4.0) * total_cg_iters as f64;
        force_rw + update_rw + cg_rw
    }
}

// ── AMD GPU power polling ──

struct AmdGpuPower {
    hwmon_path: Option<String>,
}

impl AmdGpuPower {
    fn detect() -> Self {
        for entry in std::fs::read_dir("/sys/class/hwmon")
            .into_iter()
            .flatten()
            .flatten()
        {
            let name_path = entry.path().join("name");
            if let Ok(name) = std::fs::read_to_string(&name_path)
                && name.trim() == "amdgpu"
            {
                return Self {
                    hwmon_path: Some(entry.path().to_string_lossy().to_string()),
                };
            }
        }
        Self { hwmon_path: None }
    }

    fn read_watts(&self) -> Option<f64> {
        let path = self.hwmon_path.as_ref()?;
        let uw_str = std::fs::read_to_string(format!("{path}/power1_average")).ok()?;
        let microwatts: f64 = uw_str.trim().parse().ok()?;
        Some(microwatts / 1_000_000.0)
    }
}

// ── Trajectory result with silicon metrics ──

#[allow(dead_code)] // traj_idx and other fields align with CSV columns / tooling
struct InstrumentedResult {
    traj_idx: usize,
    accepted: bool,
    delta_h: f64,
    plaquette: f64,
    cg_iters: usize,
    wall_ms: f64,
    est_gflops: f64,
    est_gflop_per_s: f64,
    est_gb_s: f64,
}

// ── Run summary per beta ──

#[allow(dead_code)] // Summary struct consumed by downstream tooling / eprintln
struct BetaSummary {
    beta: f64,
    mode: String,
    lattice: usize,
    gpu_name: String,
    n_meas: usize,
    acceptance_pct: f64,
    mean_plaq: f64,
    std_plaq: f64,
    mean_ms_per_traj: f64,
    mean_gflop_per_s: f64,
    total_wall_s: f64,
    total_gflops: f64,
    energy: EnergyReport,
    joules_per_traj: f64,
    quda_tflops: f64,
    eco_tflops: f64,
    silicon_util_pct: f64,
    quda_util_pct: f64,
    flow_results: Option<FlowSummary>,
}

#[allow(dead_code)]
struct FlowSummary {
    n_configs: usize,
    mean_t0: Option<f64>,
    std_t0: Option<f64>,
    mean_w0: Option<f64>,
    std_w0: Option<f64>,
}

// ── CLI ──

#[derive(Clone)]
struct CliArgs {
    mode: String,
    lattice: usize,
    betas: Vec<f64>,
    mass: f64,
    strange_mass: f64,
    charm_mass: f64,
    n_therm: usize,
    n_quenched_pretherm: usize,
    n_meas: usize,
    n_md_steps: usize,
    dt: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    seed: u64,
    output: Option<String>,
    flow: bool,
    flow_configs: usize,
    flow_skip: usize,
    flow_epsilon: f64,
    flow_t_max: f64,
}

fn parse_args() -> CliArgs {
    let mut args = CliArgs {
        mode: "quenched".to_string(),
        lattice: 8,
        betas: vec![5.5, 5.69, 6.0, 6.2],
        mass: 0.1,
        strange_mass: 0.5,
        charm_mass: 1.5,
        n_therm: 200,
        n_quenched_pretherm: 50,
        n_meas: 100,
        n_md_steps: 20,
        dt: 0.01,
        cg_tol: 1e-8,
        cg_max_iter: 5000,
        seed: 42,
        output: None,
        flow: false,
        flow_configs: 10,
        flow_skip: 5,
        flow_epsilon: 0.01,
        flow_t_max: 5.0,
    };

    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--mode=") {
            args.mode = val.to_string();
        } else if let Some(val) = arg.strip_prefix("--lattice=") {
            args.lattice = val.parse().expect("--lattice=N");
        } else if let Some(val) = arg.strip_prefix("--betas=") {
            args.betas = val.split(',').map(|s| s.parse().expect("beta")).collect();
        } else if let Some(val) = arg.strip_prefix("--mass=") {
            args.mass = val.parse().expect("--mass=F");
        } else if let Some(val) = arg.strip_prefix("--strange-mass=") {
            args.strange_mass = val.parse().expect("--strange-mass=F");
        } else if let Some(val) = arg.strip_prefix("--charm-mass=") {
            args.charm_mass = val.parse().expect("--charm-mass=F");
        } else if let Some(val) = arg.strip_prefix("--therm=") {
            args.n_therm = val.parse().expect("--therm=N");
        } else if let Some(val) = arg.strip_prefix("--quenched-pretherm=") {
            args.n_quenched_pretherm = val.parse().expect("--quenched-pretherm=N");
        } else if let Some(val) = arg.strip_prefix("--meas=") {
            args.n_meas = val.parse().expect("--meas=N");
        } else if let Some(val) = arg.strip_prefix("--n-md=") {
            args.n_md_steps = val.parse().expect("--n-md=N");
        } else if let Some(val) = arg.strip_prefix("--dt=") {
            args.dt = val.parse().expect("--dt=F");
        } else if let Some(val) = arg.strip_prefix("--cg-tol=") {
            args.cg_tol = val.parse().expect("--cg-tol=F");
        } else if let Some(val) = arg.strip_prefix("--cg-max-iter=") {
            args.cg_max_iter = val.parse().expect("--cg-max-iter=N");
        } else if let Some(val) = arg.strip_prefix("--seed=") {
            args.seed = val.parse().expect("--seed=N");
        } else if let Some(val) = arg.strip_prefix("--output=") {
            args.output = Some(val.to_string());
        } else if arg == "--flow" {
            args.flow = true;
        } else if let Some(val) = arg.strip_prefix("--flow-configs=") {
            args.flow_configs = val.parse().expect("--flow-configs=N");
            args.flow = true;
        } else if let Some(val) = arg.strip_prefix("--flow-skip=") {
            args.flow_skip = val.parse().expect("--flow-skip=N");
        } else if let Some(val) = arg.strip_prefix("--flow-epsilon=") {
            args.flow_epsilon = val.parse().expect("--flow-epsilon=F");
        } else if let Some(val) = arg.strip_prefix("--flow-tmax=") {
            args.flow_t_max = val.parse().expect("--flow-tmax=F");
        }
    }

    if args.mode == "quenched" {
        args.n_md_steps = args.n_md_steps.max(20);
    }

    args
}

fn mode_label(args: &CliArgs) -> String {
    match args.mode.as_str() {
        "quenched" => "Quenched (pure gauge)".to_string(),
        "nf1" | "1" => format!("Nf=1 (m={})", args.mass),
        "nf2" | "2" => format!("Nf=2 (m={})", args.mass),
        "nf2+1" | "2+1" => format!("Nf=2+1 (m_l={}, m_s={})", args.mass, args.strange_mass),
        "nf3" | "3" => format!("Nf=3 (m={})", args.mass),
        "nf4" | "4" => format!("Nf=4 (m={})", args.mass),
        "nf2+1+1" => format!(
            "Nf=2+1+1 (m_l={}, m_s={}, m_c={})",
            args.mass, args.strange_mass, args.charm_mass
        ),
        other => format!("Unknown mode: {other}"),
    }
}

fn is_dynamical(mode: &str) -> bool {
    matches!(
        mode,
        "nf1" | "1" | "nf2" | "2" | "nf2+1" | "2+1" | "nf3" | "3" | "nf4" | "4" | "nf2+1+1"
    )
}

// ── Main ──

fn main() {
    let args = parse_args();
    let l = args.lattice;
    let dims = [l, l, l, l];
    let vol = l * l * l * l;
    let label = mode_label(&args);

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  Production Silicon-Instrumented QCD                       ║");
    eprintln!("║  ecoPrimals · Exp 105 · Silicon Revalidation Campaign      ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  Mode:        {label}");
    eprintln!("  Lattice:     {l}^4 ({vol} sites)");
    eprintln!("  β points:    {:?}", args.betas);
    eprintln!("  MD:          {} steps × dt={}", args.n_md_steps, args.dt);
    eprintln!(
        "  Therm:       {} (+ {} quenched pre-therm)",
        args.n_therm, args.n_quenched_pretherm
    );
    eprintln!("  Meas:        {} per β", args.n_meas);
    if args.flow {
        eprintln!(
            "  Flow:        {} configs, skip={}, ε={}, t_max={}",
            args.flow_configs, args.flow_skip, args.flow_epsilon, args.flow_t_max
        );
    }
    eprintln!();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("  GPU not available: {e}");
            std::process::exit(1);
        }
    };

    let budget = lookup_budget(&gpu.adapter_name);
    let amd_power = AmdGpuPower::detect();
    let has_amd = amd_power.hwmon_path.is_some();
    let telemetry = GpuTelemetry::start(&gpu.adapter_name);

    eprintln!("  GPU:         {}", gpu.adapter_name);
    eprintln!("  DF64:        {:.2} TFLOPS", budget.df64_tflops);
    eprintln!("  Mem BW:      {:.0} GB/s", budget.memory_bw_gbs);
    eprintln!(
        "  TMU:         {} units × {:.2} GHz = {:.0} GTexels/s",
        budget.tmu_count,
        budget.boost_ghz,
        budget.tmu_count as f64 * budget.boost_ghz
    );
    eprintln!(
        "  VRAM:        {:.1} GB",
        budget.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    eprintln!(
        "  QUDA ref:    {:.1} TFLOPS (FP64 ALU only)",
        budget.quda_style_tflops
    );
    eprintln!(
        "  Total cap:   {:.0} TFLOPS (all silicon)",
        budget.total_tflops
    );
    eprintln!(
        "  Telemetry:   {} ({})",
        telemetry.backend,
        if has_amd { "AMD sysfs" } else { "nvidia-smi" }
    );
    eprintln!();

    let mut out_file = args.output.as_ref().map(|path| {
        let f = std::fs::File::create(path).expect("create output file");
        std::io::BufWriter::new(f)
    });

    let csv_header = "mode,beta,lattice,gpu,traj,accepted,delta_h,plaquette,cg_iters,\
                      wall_ms,est_gflops,est_gflop_per_s,est_gb_s,phase";
    if let Some(f) = &mut out_file {
        writeln!(f, "{csv_header}").ok();
    }
    println!("{csv_header}");

    let run_start = Instant::now();
    let mut all_summaries: Vec<BetaSummary> = Vec::new();
    let mut seed = args.seed;

    for &beta in &args.betas {
        let summary = run_beta_point(
            &gpu,
            &budget,
            &amd_power,
            &telemetry,
            &args,
            beta,
            dims,
            vol,
            &mut seed,
            &mut out_file,
        );
        all_summaries.push(summary);
    }

    let total_time = run_start.elapsed();

    // ── Grand summary ──
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  SILICON UTILIZATION SUMMARY                               ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!(
        "  {:<8} {:<6} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "β", "acc%", "ms/traj", "GFLOP/s", "eco%", "QUDA%", "J/traj", "⟨P⟩", "σ(P)"
    );
    eprintln!("  {}", "─".repeat(88));

    for s in &all_summaries {
        eprintln!(
            "  {:<8.4} {:<6.1} {:<10.1} {:<10.1} {:<10.2} {:<10.2} {:<10.2} {:<10.6} {:<10.2e}",
            s.beta,
            s.acceptance_pct,
            s.mean_ms_per_traj,
            s.mean_gflop_per_s * 1000.0,
            s.silicon_util_pct,
            s.quda_util_pct,
            s.joules_per_traj,
            s.mean_plaq,
            s.std_plaq
        );
    }

    eprintln!();
    eprintln!("  Legend: eco% = ecoPrimals sustained / total silicon capacity");
    eprintln!("          QUDA% = QUDA-style sustained / total silicon capacity");
    eprintln!();

    // QUDA comparison table
    eprintln!("  ┌──────────────────────────────────────────────────────────────┐");
    eprintln!("  │  QUDA vs ecoPrimals Comparison                             │");
    eprintln!("  ├──────────────────────────────────────────────────────────────┤");

    let mean_eco_tflops = if all_summaries.is_empty() {
        0.0
    } else {
        all_summaries.iter().map(|s| s.eco_tflops).sum::<f64>() / all_summaries.len() as f64
    };

    eprintln!("  │  GPU:            {:<43}│", budget.name);
    eprintln!(
        "  │  Total silicon:  {:<6.0} TFLOPS{:>31}│",
        budget.total_tflops, ""
    );
    eprintln!(
        "  │  QUDA sustained: {:<6.2} TFLOPS ({:.2}% util){:>19}│",
        budget.quda_style_tflops,
        budget.quda_style_tflops / budget.total_tflops * 100.0,
        ""
    );
    eprintln!(
        "  │  ecoPrimals:     {:<6.4} TFLOPS ({:.2}% util){:>19}│",
        mean_eco_tflops,
        mean_eco_tflops / budget.total_tflops * 100.0,
        ""
    );
    eprintln!(
        "  │  Speedup:        {:<6.2}x vs QUDA-style{:>23}│",
        if budget.quda_style_tflops > 0.0 {
            mean_eco_tflops / budget.quda_style_tflops
        } else {
            0.0
        },
        ""
    );
    eprintln!("  └──────────────────────────────────────────────────────────────┘");

    // Flow results if any
    let has_flow = all_summaries.iter().any(|s| s.flow_results.is_some());
    if has_flow {
        eprintln!();
        eprintln!("  ┌──────────────────────────────────────────────────────────────┐");
        eprintln!("  │  Gradient Flow Scale Setting                                │");
        eprintln!("  ├──────────────────────────────────────────────────────────────┤");
        for s in &all_summaries {
            if let Some(ref f) = s.flow_results {
                let t0_str = f.mean_t0.map_or_else(
                    || "N/A".to_string(),
                    |v| format!("{:.4} ± {:.4}", v, f.std_t0.unwrap_or(0.0)),
                );
                let w0_str = f.mean_w0.map_or_else(
                    || "N/A".to_string(),
                    |v| format!("{:.4} ± {:.4}", v, f.std_w0.unwrap_or(0.0)),
                );
                eprintln!(
                    "  │  β={:.4}: t₀ = {:<20} w₀ = {:<18}│",
                    s.beta, t0_str, w0_str
                );
            }
        }
        eprintln!("  └──────────────────────────────────────────────────────────────┘");
    }

    // Energy summary
    let total_gpu_j: f64 = all_summaries.iter().map(|s| s.energy.gpu_joules).sum();
    let total_cpu_j: f64 = all_summaries.iter().map(|s| s.energy.cpu_joules).sum();
    let total_traj: usize = all_summaries.iter().map(|s| s.n_meas).sum();

    eprintln!();
    eprintln!(
        "  Energy: GPU {:.1} J + CPU {:.1} J = {:.1} J total ({} measurement trajectories)",
        total_gpu_j,
        total_cpu_j,
        total_gpu_j + total_cpu_j,
        total_traj
    );
    if total_traj > 0 {
        eprintln!(
            "  Average: {:.2} J/trajectory",
            (total_gpu_j + total_cpu_j) / total_traj as f64
        );
    }
    eprintln!(
        "  Total wall: {:.1}s ({:.2}h)",
        total_time.as_secs_f64(),
        total_time.as_secs_f64() / 3600.0
    );
    eprintln!();
}

fn run_beta_point(
    gpu: &GpuF64,
    budget: &SiliconBudget,
    amd_power: &AmdGpuPower,
    telemetry: &GpuTelemetry,
    args: &CliArgs,
    beta: f64,
    dims: [usize; 4],
    vol: usize,
    seed: &mut u64,
    out_file: &mut Option<std::io::BufWriter<std::fs::File>>,
) -> BetaSummary {
    let l = args.lattice;
    let dynamical = is_dynamical(&args.mode);

    let hw = telemetry.snapshot();
    eprintln!(
        "━━━ β = {beta:.4} ({}) ━━━  [{}]",
        mode_label(args),
        hw.status_line()
    );

    // Hot start
    let lattice = Lattice::hot_start(dims, beta, *seed);

    // Quenched pre-thermalization (streaming path — already zero-sync)
    let quenched_state = GpuHmcState::from_lattice(gpu, &lattice, beta);
    if args.n_quenched_pretherm > 0 {
        let quenched_pipelines = GpuHmcStreamingPipelines::new_with_tmu(gpu);
        for i in 0..args.n_quenched_pretherm {
            let r = gpu_hmc_trajectory_streaming(
                gpu,
                &quenched_pipelines,
                &quenched_state,
                20,
                0.1,
                i as u32,
                seed,
            );
            if (i + 1) % 10 == 0 {
                let hw = telemetry.snapshot();
                eprintln!(
                    "  pretherm {}/{}: P={:.6} ΔH={:.4} {}  [{}]",
                    i + 1,
                    args.n_quenched_pretherm,
                    r.plaquette,
                    r.delta_h,
                    if r.accepted { "Y" } else { "N" },
                    hw.status_line()
                );
            }
        }
    }

    if !dynamical {
        return run_quenched_beta(
            gpu,
            budget,
            amd_power,
            telemetry,
            args,
            beta,
            dims,
            vol,
            seed,
            out_file,
            &quenched_state,
        );
    }

    // ── Dynamical: UNIDIRECTIONAL RHMC (GPU-resident CG, minimal readback) ──
    let mut rhmc_config = match args.mode.as_str() {
        "nf1" | "1" => RhmcConfig::nf1(args.mass, beta),
        "nf2+1" | "2+1" => RhmcConfig::nf2p1(args.mass, args.strange_mass, beta),
        "nf3" | "3" => RhmcConfig::nf3(args.mass, beta),
        "nf4" | "4" => RhmcConfig::nf4(args.mass, beta),
        "nf2+1+1" => RhmcConfig::nf2p1p1(args.mass, args.strange_mass, args.charm_mass, beta),
        _ => RhmcConfig::nf2(args.mass, beta),
    };
    rhmc_config.dt = args.dt;
    rhmc_config.n_md_steps = args.n_md_steps;
    rhmc_config.cg_tol = args.cg_tol;
    rhmc_config.cg_max_iter = args.cg_max_iter;

    let dyn_state = GpuDynHmcState::from_lattice(
        gpu,
        &lattice,
        beta,
        rhmc_config.sectors[0].mass,
        args.cg_tol,
        args.cg_max_iter,
    );
    let rhmc_state = GpuRhmcState::new(gpu, &rhmc_config, dyn_state);

    if args.n_quenched_pretherm > 0 {
        let n_bytes = (quenched_state.n_links * 18 * 8) as u64;
        let mut enc = gpu.begin_encoder("copy_therm_links");
        enc.copy_buffer_to_buffer(
            &quenched_state.link_buf,
            0,
            &rhmc_state.gauge.gauge.link_buf,
            0,
            n_bytes,
        );
        gpu.submit_encoder(enc);
    }

    let dyn_pipelines = GpuDynHmcPipelines::new(gpu);
    let rhmc_pipelines = GpuRhmcPipelines::new(gpu);

    // Build unidirectional buffers for GPU-resident CG (~50x fewer sync points)
    let uni_pipelines = UniPipelines::new_saturated(gpu, vol);
    let scg_bufs = GpuResidentShiftedCgBuffers::new(
        gpu,
        &dyn_pipelines,
        &uni_pipelines.shifted_cg,
        &rhmc_state.gauge,
    );
    let ham_bufs = UniHamiltonianBuffers::new(
        gpu,
        &uni_pipelines.shifted_cg.base.reduce_pipeline,
        &rhmc_state.gauge.gauge,
        &rhmc_state.gauge,
    );

    // True multi-shift CG: shared Krylov, N_shifts fewer D†D ops per iteration
    let max_shifts = rhmc_config
        .sectors
        .iter()
        .map(|s| s.action_approx.sigma.len().max(s.force_approx.sigma.len()))
        .max()
        .unwrap_or(0);
    let ms_bufs = if max_shifts > 0 {
        Some(TrueMultiShiftBuffers::new(
            gpu,
            &dyn_pipelines,
            &uni_pipelines.true_ms_cg,
            &rhmc_state.gauge,
            max_shifts,
        ))
    } else {
        None
    };
    eprintln!("  True multi-shift CG: {max_shifts} shifts, shared Krylov basis");

    eprintln!(
        "  Thermalizing: {} RHMC trajectories (unidirectional)...",
        args.n_therm
    );
    let mut therm_accepted = 0;
    for i in 0..args.n_therm {
        let r = run_uni_traj(
            gpu,
            &dyn_pipelines,
            &rhmc_pipelines,
            &uni_pipelines,
            &rhmc_state,
            &scg_bufs,
            ms_bufs.as_ref(),
            &ham_bufs,
            &rhmc_config,
            seed,
        );
        if r.accepted {
            therm_accepted += 1;
        }
        if (i + 1) % 20 == 0 || i == 0 {
            let hw = telemetry.snapshot();
            eprintln!(
                "    therm {}/{}: P={:.6} ΔH={:.4e} CG={} acc={:.0}%  [{}]",
                i + 1,
                args.n_therm,
                r.plaquette,
                r.delta_h,
                r.total_cg_iterations,
                therm_accepted as f64 / (i + 1) as f64 * 100.0,
                hw.status_line()
            );
        }
    }

    // Measurement phase with energy + telemetry
    let monitor = PowerMonitor::start();
    let meas_start = Instant::now();
    let mut amd_power_samples: Vec<f64> = Vec::new();

    let mut results: Vec<InstrumentedResult> = Vec::new();
    let mut meas_accepted = 0;
    let mut plaq_sum = 0.0;

    for i in 0..args.n_meas {
        if let Some(w) = amd_power.read_watts() {
            amd_power_samples.push(w);
        }

        let t0 = Instant::now();
        let r = run_uni_traj(
            gpu,
            &dyn_pipelines,
            &rhmc_pipelines,
            &uni_pipelines,
            &rhmc_state,
            &scg_bufs,
            ms_bufs.as_ref(),
            &ham_bufs,
            &rhmc_config,
            seed,
        );
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let est_flops = estimate_traj_flops(vol, args.n_md_steps, r.total_cg_iterations, false);
        let est_gflops = est_flops / 1e9;
        let est_gflop_per_s = est_gflops / (wall_ms / 1000.0);
        let est_bytes = estimate_traj_bytes(vol, args.n_md_steps, r.total_cg_iterations, false);
        let est_gb_s = (est_bytes / 1e9) / (wall_ms / 1000.0);

        if r.accepted {
            meas_accepted += 1;
        }
        plaq_sum += r.plaquette;

        let ir = InstrumentedResult {
            traj_idx: i,
            accepted: r.accepted,
            delta_h: r.delta_h,
            plaquette: r.plaquette,
            cg_iters: r.total_cg_iterations,
            wall_ms,
            est_gflops,
            est_gflop_per_s,
            est_gb_s,
        };

        // CSV with telemetry columns
        let hw = telemetry.snapshot();
        let line = format!(
            "{},{beta:.4},{l},{},{},{},{:.6e},{:.8},{},{:.1},{:.3},{:.1},{:.1},meas",
            args.mode,
            budget.name,
            i,
            i32::from(ir.accepted),
            ir.delta_h,
            ir.plaquette,
            ir.cg_iters,
            ir.wall_ms,
            ir.est_gflops,
            ir.est_gflop_per_s,
            ir.est_gb_s
        );
        println!("{line}");
        if let Some(f) = out_file.as_mut() {
            writeln!(f, "{line}").ok();
            f.flush().ok();
        }

        results.push(ir);

        if (i + 1) % 10 == 0 || i + 1 == args.n_meas {
            let n = (i + 1) as f64;
            let mean_p = plaq_sum / n;
            let rate = meas_accepted as f64 / n * 100.0;
            eprintln!(
                "    meas {}/{}: ⟨P⟩={:.6} acc={:.0}% {:.0}ms/traj {:.0}GFLOP/s  [{}]",
                i + 1,
                args.n_meas,
                mean_p,
                rate,
                wall_ms,
                est_gflop_per_s,
                hw.status_line()
            );
        }
    }

    let meas_wall_s = meas_start.elapsed().as_secs_f64();
    let mut energy = monitor.stop();

    if energy.gpu_joules < 0.01 && !amd_power_samples.is_empty() {
        let avg_w = amd_power_samples.iter().sum::<f64>() / amd_power_samples.len() as f64;
        energy.gpu_joules = avg_w * meas_wall_s;
        energy.gpu_watts_avg = avg_w;
        energy.gpu_watts_peak = amd_power_samples.iter().copied().fold(0.0f64, f64::max);
        energy.gpu_samples = amd_power_samples.len();
    }

    let flow_results = if args.flow {
        Some(run_gradient_flow_uni(
            (),
            gpu,
            args,
            &rhmc_state,
            &rhmc_config,
            dims,
            seed,
        ))
    } else {
        None
    };

    build_summary(&results, args, beta, budget, energy, flow_results)
}

fn run_quenched_beta(
    gpu: &GpuF64,
    budget: &SiliconBudget,
    amd_power: &AmdGpuPower,
    telemetry: &GpuTelemetry,
    args: &CliArgs,
    beta: f64,
    dims: [usize; 4],
    vol: usize,
    seed: &mut u64,
    out_file: &mut Option<std::io::BufWriter<std::fs::File>>,
    quenched_state: &GpuHmcState,
) -> BetaSummary {
    let l = args.lattice;
    let quenched_pipelines = GpuHmcStreamingPipelines::new_with_tmu(gpu);

    eprintln!(
        "  Thermalizing: {} quenched HMC trajectories (streaming)...",
        args.n_therm
    );
    let mut therm_accepted = 0;
    for i in 0..args.n_therm {
        let r = gpu_hmc_trajectory_streaming(
            gpu,
            &quenched_pipelines,
            quenched_state,
            args.n_md_steps,
            args.dt,
            (args.n_quenched_pretherm + i) as u32,
            seed,
        );
        if r.accepted {
            therm_accepted += 1;
        }
        if (i + 1) % 20 == 0 || i == 0 {
            let hw = telemetry.snapshot();
            eprintln!(
                "    therm {}/{}: P={:.6} ΔH={:.4} acc={:.0}%  [{}]",
                i + 1,
                args.n_therm,
                r.plaquette,
                r.delta_h,
                therm_accepted as f64 / (i + 1) as f64 * 100.0,
                hw.status_line()
            );
        }
    }

    let monitor = PowerMonitor::start();
    let meas_start = Instant::now();
    let mut amd_power_samples: Vec<f64> = Vec::new();

    let mut results: Vec<InstrumentedResult> = Vec::new();
    let mut meas_accepted = 0;
    let mut plaq_sum = 0.0;

    for i in 0..args.n_meas {
        if let Some(w) = amd_power.read_watts() {
            amd_power_samples.push(w);
        }

        let traj_id = (args.n_quenched_pretherm + args.n_therm + i) as u32;
        let t0 = Instant::now();
        let r = gpu_hmc_trajectory_streaming(
            gpu,
            &quenched_pipelines,
            quenched_state,
            args.n_md_steps,
            args.dt,
            traj_id,
            seed,
        );
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let est_flops = estimate_traj_flops(vol, args.n_md_steps, 0, true);
        let est_gflops = est_flops / 1e9;
        let est_gflop_per_s = est_gflops / (wall_ms / 1000.0);
        let est_bytes = estimate_traj_bytes(vol, args.n_md_steps, 0, true);
        let est_gb_s = (est_bytes / 1e9) / (wall_ms / 1000.0);

        if r.accepted {
            meas_accepted += 1;
        }
        plaq_sum += r.plaquette;

        let ir = InstrumentedResult {
            traj_idx: i,
            accepted: r.accepted,
            delta_h: r.delta_h,
            plaquette: r.plaquette,
            cg_iters: 0,
            wall_ms,
            est_gflops,
            est_gflop_per_s,
            est_gb_s,
        };

        let line = format!(
            "{},{beta:.4},{l},{},{},{},{:.6e},{:.8},{},{:.1},{:.3},{:.1},{:.1},meas",
            args.mode,
            budget.name,
            i,
            i32::from(ir.accepted),
            ir.delta_h,
            ir.plaquette,
            ir.cg_iters,
            ir.wall_ms,
            ir.est_gflops,
            ir.est_gflop_per_s,
            ir.est_gb_s
        );
        println!("{line}");
        if let Some(f) = out_file.as_mut() {
            writeln!(f, "{line}").ok();
            f.flush().ok();
        }

        results.push(ir);

        if (i + 1) % 25 == 0 || i + 1 == args.n_meas {
            let n = (i + 1) as f64;
            let mean_p = plaq_sum / n;
            let rate = meas_accepted as f64 / n * 100.0;
            let hw = telemetry.snapshot();
            eprintln!(
                "    meas {}/{}: ⟨P⟩={:.6} acc={:.0}% {:.0}ms/traj {:.0}GFLOP/s  [{}]",
                i + 1,
                args.n_meas,
                mean_p,
                rate,
                wall_ms,
                est_gflop_per_s,
                hw.status_line()
            );
        }
    }

    let meas_wall_s = meas_start.elapsed().as_secs_f64();
    let mut energy = monitor.stop();

    if energy.gpu_joules < 0.01 && !amd_power_samples.is_empty() {
        let avg_w = amd_power_samples.iter().sum::<f64>() / amd_power_samples.len() as f64;
        energy.gpu_joules = avg_w * meas_wall_s;
        energy.gpu_watts_avg = avg_w;
        energy.gpu_watts_peak = amd_power_samples.iter().copied().fold(0.0f64, f64::max);
        energy.gpu_samples = amd_power_samples.len();
    }

    let flow_results = if args.flow {
        Some(run_quenched_gradient_flow(
            gpu,
            args,
            quenched_state,
            dims,
            seed,
        ))
    } else {
        None
    };

    build_summary(&results, args, beta, budget, energy, flow_results)
}

/// Wrapper: run one unidirectional RHMC trajectory with timing.
fn run_uni_traj(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    rhmc_pipelines: &GpuRhmcPipelines,
    uni_pipelines: &UniPipelines,
    rhmc_state: &GpuRhmcState,
    scg_bufs: &GpuResidentShiftedCgBuffers,
    ms_bufs: Option<&TrueMultiShiftBuffers>,
    ham_bufs: &UniHamiltonianBuffers,
    config: &RhmcConfig,
    seed: &mut u64,
) -> TrajectoryResult {
    let t0 = Instant::now();
    let r = gpu_rhmc_trajectory_unidirectional(
        gpu,
        dyn_pipelines,
        rhmc_pipelines,
        uni_pipelines,
        rhmc_state,
        scg_bufs,
        ms_bufs,
        ham_bufs,
        config,
        seed,
    );
    TrajectoryResult {
        accepted: r.accepted,
        delta_h: r.delta_h,
        plaquette: r.plaquette,
        total_cg_iterations: r.total_cg_iterations,
        elapsed_secs: t0.elapsed().as_secs_f64(),
    }
}

/// Gradient flow with unidirectional skip trajectories (no per-iteration sync).
fn run_gradient_flow_uni(
    _uni: (), // placeholder — we use free function directly
    gpu: &GpuF64,
    args: &CliArgs,
    rhmc_state: &GpuRhmcState,
    rhmc_config: &RhmcConfig,
    dims: [usize; 4],
    seed: &mut u64,
) -> FlowSummary {
    eprintln!(
        "  Running gradient flow on {} configs (skip={}, unidirectional)...",
        args.flow_configs, args.flow_skip
    );

    let vol: usize = dims.iter().product();
    let dyn_pipelines = GpuDynHmcPipelines::new(gpu);
    let rhmc_pipelines = GpuRhmcPipelines::new(gpu);
    let uni_pipelines = UniPipelines::new_saturated(gpu, vol);
    let scg_bufs = GpuResidentShiftedCgBuffers::new(
        gpu,
        &dyn_pipelines,
        &uni_pipelines.shifted_cg,
        &rhmc_state.gauge,
    );
    let ham_bufs = UniHamiltonianBuffers::new(
        gpu,
        &uni_pipelines.shifted_cg.base.reduce_pipeline,
        &rhmc_state.gauge.gauge,
        &rhmc_state.gauge,
    );

    // GPU-resident flow: eliminates B4 (gpu_links_to_lattice) transfer
    let flow_pipelines = GpuFlowPipelines::new(gpu);

    let mut all_t0 = Vec::new();
    let mut all_w0 = Vec::new();

    for cfg_idx in 0..args.flow_configs {
        for _ in 0..args.flow_skip {
            gpu_rhmc_trajectory_unidirectional(
                gpu,
                &dyn_pipelines,
                &rhmc_pipelines,
                &uni_pipelines,
                rhmc_state,
                &scg_bufs,
                None,
                &ham_bufs,
                rhmc_config,
                seed,
            );
        }

        // GPU-GPU link copy (no PCI-e round-trip) → GPU gradient flow
        let flow_state = GpuFlowState::from_gpu_gauge(gpu, &rhmc_state.gauge.gauge);
        let flow_reduce = FlowReduceBuffers::new(gpu, &flow_pipelines.reduce_pipeline, &flow_state);

        let flow_result = gpu_gradient_flow_resident(
            gpu,
            &flow_pipelines,
            &flow_state,
            &flow_reduce,
            FlowIntegrator::Lscfrk3w7,
            args.flow_epsilon,
            args.flow_t_max,
            1,
        );

        let t0_val = find_t0(&flow_result.measurements);
        let w0_val = find_w0(&flow_result.measurements);
        if let Some(t) = t0_val {
            all_t0.push(t);
        }
        if let Some(w) = w0_val {
            all_w0.push(w);
        }

        eprintln!(
            "    flow cfg {}/{}: t0={} w0={} ({:.1}s GPU flow)",
            cfg_idx + 1,
            args.flow_configs,
            t0_val.map_or_else(|| "N/A".to_string(), |v| format!("{v:.4}")),
            w0_val.map_or_else(|| "N/A".to_string(), |v| format!("{v:.4}")),
            flow_result.wall_seconds
        );
    }

    FlowSummary {
        n_configs: args.flow_configs,
        mean_t0: if all_t0.is_empty() {
            None
        } else {
            Some(mean(&all_t0))
        },
        std_t0: if all_t0.len() < 2 {
            None
        } else {
            Some(std_dev(&all_t0))
        },
        mean_w0: if all_w0.is_empty() {
            None
        } else {
            Some(mean(&all_w0))
        },
        std_w0: if all_w0.len() < 2 {
            None
        } else {
            Some(std_dev(&all_w0))
        },
    }
}

fn run_quenched_gradient_flow(
    gpu: &GpuF64,
    args: &CliArgs,
    state: &GpuHmcState,
    _dims: [usize; 4],
    seed: &mut u64,
) -> FlowSummary {
    eprintln!(
        "  Running gradient flow on {} quenched configs (skip={})...",
        args.flow_configs, args.flow_skip
    );

    let pipelines = GpuHmcStreamingPipelines::new_with_tmu(gpu);
    let mut all_t0 = Vec::new();
    let mut all_w0 = Vec::new();

    let flow_pipelines = GpuFlowPipelines::new(gpu);

    let beta = args.betas[0]; // current beta from caller context
    let _ = beta; // used implicitly via state.beta
    for cfg_idx in 0..args.flow_configs {
        for s in 0..args.flow_skip {
            let traj_id = (1000 + cfg_idx * args.flow_skip + s) as u32;
            gpu_hmc_trajectory_streaming(
                gpu,
                &pipelines,
                state,
                args.n_md_steps,
                args.dt,
                traj_id,
                seed,
            );
        }

        // GPU-GPU link copy → GPU gradient flow (no B4 transfer)
        let flow_state = GpuFlowState::from_gpu_gauge(gpu, state);
        let flow_reduce = FlowReduceBuffers::new(gpu, &flow_pipelines.reduce_pipeline, &flow_state);

        let flow_result = gpu_gradient_flow_resident(
            gpu,
            &flow_pipelines,
            &flow_state,
            &flow_reduce,
            FlowIntegrator::Lscfrk3w7,
            args.flow_epsilon,
            args.flow_t_max,
            1,
        );

        let t0_val = find_t0(&flow_result.measurements);
        let w0_val = find_w0(&flow_result.measurements);
        if let Some(t) = t0_val {
            all_t0.push(t);
        }
        if let Some(w) = w0_val {
            all_w0.push(w);
        }

        eprintln!(
            "    flow cfg {}/{}: t0={} w0={} ({:.1}s GPU flow)",
            cfg_idx + 1,
            args.flow_configs,
            t0_val.map_or_else(|| "N/A".to_string(), |v| format!("{v:.4}")),
            w0_val.map_or_else(|| "N/A".to_string(), |v| format!("{v:.4}")),
            flow_result.wall_seconds
        );
    }

    FlowSummary {
        n_configs: args.flow_configs,
        mean_t0: if all_t0.is_empty() {
            None
        } else {
            Some(mean(&all_t0))
        },
        std_t0: if all_t0.len() < 2 {
            None
        } else {
            Some(std_dev(&all_t0))
        },
        mean_w0: if all_w0.is_empty() {
            None
        } else {
            Some(mean(&all_w0))
        },
        std_w0: if all_w0.len() < 2 {
            None
        } else {
            Some(std_dev(&all_w0))
        },
    }
}

fn build_summary(
    results: &[InstrumentedResult],
    args: &CliArgs,
    beta: f64,
    budget: &SiliconBudget,
    energy: EnergyReport,
    flow_results: Option<FlowSummary>,
) -> BetaSummary {
    let n = results.len() as f64;
    let plaq_sum: f64 = results.iter().map(|r| r.plaquette).sum();
    let plaq_sq_sum: f64 = results.iter().map(|r| r.plaquette * r.plaquette).sum();
    let mean_plaq = plaq_sum / n;
    let var_plaq = ((plaq_sq_sum / n) - mean_plaq.powi(2)).max(0.0);
    let std_plaq = var_plaq.sqrt();

    let accepted = results.iter().filter(|r| r.accepted).count();
    let acceptance_pct = accepted as f64 / n * 100.0;
    let mean_ms = results.iter().map(|r| r.wall_ms).sum::<f64>() / n;
    let mean_gflop_s = results.iter().map(|r| r.est_gflop_per_s).sum::<f64>() / n;
    let total_gflops: f64 = results.iter().map(|r| r.est_gflops).sum();
    let total_wall_s: f64 = results.iter().map(|r| r.wall_ms).sum::<f64>() / 1000.0;

    let eco_tflops = mean_gflop_s / 1000.0; // GFLOP/s → TFLOP/s
    let silicon_util = eco_tflops / budget.total_tflops * 100.0;
    let quda_util = budget.quda_style_tflops / budget.total_tflops * 100.0;

    let joules_per_traj = if results.is_empty() {
        0.0
    } else {
        (energy.gpu_joules + energy.cpu_joules) / n
    };

    let summary = BetaSummary {
        beta,
        mode: args.mode.clone(),
        lattice: args.lattice,
        gpu_name: budget.name.clone(),
        n_meas: results.len(),
        acceptance_pct,
        mean_plaq,
        std_plaq,
        mean_ms_per_traj: mean_ms,
        mean_gflop_per_s: eco_tflops,
        total_wall_s,
        total_gflops,
        energy,
        joules_per_traj,
        quda_tflops: budget.quda_style_tflops,
        eco_tflops,
        silicon_util_pct: silicon_util,
        quda_util_pct: quda_util,
        flow_results,
    };

    eprintln!();
    eprintln!(
        "  β={beta:.4} summary: ⟨P⟩={mean_plaq:.6}±{std_plaq:.2e} acc={acceptance_pct:.0}% {mean_ms:.0}ms/traj {eco_tflops:.2}TFLOP/s ({silicon_util:.2}% silicon)"
    );
    if summary.energy.gpu_joules > 0.0 {
        eprintln!(
            "  Energy: {:.1}J GPU + {:.1}J CPU = {:.2}J/traj (avg {:.0}W GPU)",
            summary.energy.gpu_joules,
            summary.energy.cpu_joules,
            joules_per_traj,
            summary.energy.gpu_watts_avg
        );
    }
    eprintln!();

    summary
}
