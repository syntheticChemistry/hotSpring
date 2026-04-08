// SPDX-License-Identifier: AGPL-3.0-or-later

//! Silicon budgets, FLOP estimates, AMD power sampling, and CLI for production QCD runs.

use hotspring_barracuda::bench::EnergyReport;

// ── Silicon budget (reused from bench_silicon_budget) ──

#[allow(dead_code)] // Fields used for future reporting / CSV parity with bench_silicon_budget
pub struct SiliconBudget {
    pub name: String,
    pub fp32_tflops: f64,
    pub df64_tflops: f64,
    pub memory_bw_gbs: f64,
    pub tmu_count: u32,
    pub boost_ghz: f64,
    pub vram_bytes: u64,
    pub total_tflops: f64,
    pub quda_style_tflops: f64,
}

pub fn lookup_budget(name: &str) -> SiliconBudget {
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
pub fn estimate_traj_flops(vol: usize, n_md: usize, total_cg_iters: usize, is_quenched: bool) -> f64 {
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
pub fn estimate_traj_bytes(vol: usize, n_md: usize, total_cg_iters: usize, is_quenched: bool) -> f64 {
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

pub struct AmdGpuPower {
    pub hwmon_path: Option<String>,
}

impl AmdGpuPower {
    pub fn detect() -> Self {
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

    pub fn read_watts(&self) -> Option<f64> {
        let path = self.hwmon_path.as_ref()?;
        let uw_str = std::fs::read_to_string(format!("{path}/power1_average")).ok()?;
        let microwatts: f64 = uw_str.trim().parse().ok()?;
        Some(microwatts / 1_000_000.0)
    }
}

// ── Trajectory result with silicon metrics ──

#[allow(dead_code)] // traj_idx and other fields align with CSV columns / tooling
pub struct InstrumentedResult {
    pub traj_idx: usize,
    pub accepted: bool,
    pub delta_h: f64,
    pub plaquette: f64,
    pub cg_iters: usize,
    pub wall_ms: f64,
    pub est_gflops: f64,
    pub est_gflop_per_s: f64,
    pub est_gb_s: f64,
}

// ── Run summary per beta ──

#[allow(dead_code)] // Summary struct consumed by downstream tooling / eprintln
pub struct BetaSummary {
    pub beta: f64,
    pub mode: String,
    pub lattice: usize,
    pub gpu_name: String,
    pub n_meas: usize,
    pub acceptance_pct: f64,
    pub mean_plaq: f64,
    pub std_plaq: f64,
    pub mean_ms_per_traj: f64,
    pub mean_gflop_per_s: f64,
    pub total_wall_s: f64,
    pub total_gflops: f64,
    pub energy: EnergyReport,
    pub joules_per_traj: f64,
    pub quda_tflops: f64,
    pub eco_tflops: f64,
    pub silicon_util_pct: f64,
    pub quda_util_pct: f64,
    pub flow_results: Option<FlowSummary>,
}

#[allow(dead_code)]
pub struct FlowSummary {
    pub n_configs: usize,
    pub mean_t0: Option<f64>,
    pub std_t0: Option<f64>,
    pub mean_w0: Option<f64>,
    pub std_w0: Option<f64>,
}

// ── CLI ──

#[derive(Clone)]
pub struct CliArgs {
    pub mode: String,
    pub lattice: usize,
    pub betas: Vec<f64>,
    pub mass: f64,
    pub strange_mass: f64,
    pub charm_mass: f64,
    pub n_therm: usize,
    pub n_quenched_pretherm: usize,
    pub n_meas: usize,
    pub n_md_steps: usize,
    pub dt: f64,
    pub cg_tol: f64,
    pub cg_max_iter: usize,
    pub seed: u64,
    pub output: Option<String>,
    pub flow: bool,
    pub flow_configs: usize,
    pub flow_skip: usize,
    pub flow_epsilon: f64,
    pub flow_t_max: f64,
}

pub fn parse_args() -> CliArgs {
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

pub fn mode_label(args: &CliArgs) -> String {
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

pub fn is_dynamical(mode: &str) -> bool {
    matches!(
        mode,
        "nf1" | "1" | "nf2" | "2" | "nf2+1" | "2+1" | "nf3" | "3" | "nf4" | "4" | "nf2+1+1"
    )
}

