// SPDX-License-Identifier: AGPL-3.0-only

//! `ComputeBackend` comparison benchmark — runs the same physics on all
//! available backends (CPU, GPU, external) and reports speedup ratios.
//!
//! # Usage
//!
//! ```bash
//! # Symmetric lattice:
//! cargo run --release --bin bench_backends -- --lattice=8 --beta=6.0
//!
//! # Asymmetric lattice:
//! cargo run --release --bin bench_backends -- --dims=16,16,16,4 --beta=5.69
//!
//! # Scale sweep:
//! cargo run --release --bin bench_backends -- --sweep
//! ```

use hotspring_barracuda::bench::compute_backend::{
    BackendKind, BarraCudaCpuBackend, BenchmarkResult, BenchmarkSpec, ComputeBackend,
    PrecisionMode, compare_backends,
};
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

struct BarraCudaGpuBackend {
    gpu: GpuF64,
    pipelines: GpuHmcStreamingPipelines,
}

impl BarraCudaGpuBackend {
    fn new() -> Result<Self, String> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| format!("runtime: {e}"))?;
        let gpu = rt
            .block_on(GpuF64::new())
            .map_err(|e| format!("GPU: {e}"))?;
        let pipelines = GpuHmcStreamingPipelines::new(&gpu);
        Ok(Self { gpu, pipelines })
    }
}

impl ComputeBackend for BarraCudaGpuBackend {
    fn name(&self) -> &'static str {
        "barraCuda-GPU"
    }
    fn kind(&self) -> BackendKind {
        BackendKind::BarraCudaGpu
    }
    fn precision(&self) -> PrecisionMode {
        PrecisionMode::DF64
    }
    fn available(&self) -> bool {
        true
    }

    fn run_quenched_hmc(&self, spec: &BenchmarkSpec) -> Result<BenchmarkResult, String> {
        let mut lat = Lattice::hot_start(spec.dims, spec.beta, spec.seed);
        let mut cfg = HmcConfig {
            n_md_steps: spec.n_md_steps,
            dt: spec.dt,
            seed: spec.seed,
            integrator: IntegratorType::Omelyan,
        };

        let cpu_therm = spec.n_therm.min(10);
        for _ in 0..cpu_therm {
            hmc::hmc_trajectory(&mut lat, &mut cfg);
        }

        let state = GpuHmcState::from_lattice(&self.gpu, &lat, spec.beta);
        let mut seed = spec.seed * 100;

        let remaining_therm = spec.n_therm.saturating_sub(cpu_therm);
        let mut traj_id = 0u32;
        for _ in 0..remaining_therm {
            gpu_hmc_trajectory_streaming(
                &self.gpu,
                &self.pipelines,
                &state,
                spec.n_md_steps,
                spec.dt,
                traj_id,
                &mut seed,
            );
            traj_id = traj_id.wrapping_add(1);
        }

        gpu_hmc_trajectory_streaming(
            &self.gpu,
            &self.pipelines,
            &state,
            spec.n_md_steps,
            spec.dt,
            traj_id,
            &mut seed,
        );
        traj_id = traj_id.wrapping_add(1);

        let start = Instant::now();
        let mut plaq_vals = Vec::with_capacity(spec.n_meas);
        let mut accepted = 0usize;
        for _ in 0..spec.n_meas {
            let r = gpu_hmc_trajectory_streaming(
                &self.gpu,
                &self.pipelines,
                &state,
                spec.n_md_steps,
                spec.dt,
                traj_id,
                &mut seed,
            );
            traj_id = traj_id.wrapping_add(1);
            plaq_vals.push(r.plaquette);
            if r.accepted {
                accepted += 1;
            }
        }
        let wall = start.elapsed();
        let ms_per = wall.as_secs_f64() * 1000.0 / spec.n_meas as f64;

        let mean_plaq = plaq_vals.iter().sum::<f64>() / plaq_vals.len() as f64;
        let var_plaq = plaq_vals
            .iter()
            .map(|p| (p - mean_plaq).powi(2))
            .sum::<f64>()
            / (plaq_vals.len() - 1).max(1) as f64;

        Ok(BenchmarkResult {
            backend_name: format!("{} ({})", self.name(), self.gpu.adapter_name),
            backend_kind: self.kind(),
            spec: spec.clone(),
            mean_plaquette: mean_plaq,
            std_plaquette: var_plaq.sqrt(),
            polyakov_mag: 0.0,
            acceptance_rate: accepted as f64 / spec.n_meas as f64,
            wall_time: wall,
            ms_per_trajectory: ms_per,
            precision_mode: self.precision(),
        })
    }
}

#[allow(deprecated)]
fn main() {
    let mut dims: Option<[usize; 4]> = None;
    let mut lattice: Option<usize> = None;
    let mut beta = 6.0;
    let mut sweep = false;
    let mut output: Option<String> = None;

    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--lattice=") {
            lattice = Some(val.parse().expect("--lattice=N"));
        } else if let Some(val) = arg.strip_prefix("--dims=") {
            let d: Vec<usize> = val
                .split(',')
                .map(|s| s.parse().expect("dims: Nx,Ny,Nz,Nt"))
                .collect();
            assert_eq!(d.len(), 4, "--dims requires 4 values");
            dims = Some([d[0], d[1], d[2], d[3]]);
        } else if let Some(val) = arg.strip_prefix("--beta=") {
            beta = val.parse().expect("--beta=X.X");
        } else if arg == "--sweep" {
            sweep = true;
        } else if let Some(val) = arg.strip_prefix("--output=") {
            output = Some(val.to_string());
        }
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ComputeBackend Comparison — barraCuda CPU vs GPU          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let cpu = BarraCudaCpuBackend;
    let gpu = match BarraCudaGpuBackend::new() {
        Ok(g) => {
            println!("  GPU: {}", g.gpu.adapter_name);
            Some(g)
        }
        Err(e) => {
            println!("  GPU: not available ({e})");
            None
        }
    };
    println!();

    let configs: Vec<BenchmarkSpec> = if sweep {
        vec![
            BenchmarkSpec {
                dims: [4, 4, 4, 4],
                beta,
                n_therm: 50,
                n_meas: 100,
                n_md_steps: 10,
                dt: 0.05,
                seed: 42,
            },
            BenchmarkSpec {
                dims: [8, 8, 8, 8],
                beta,
                n_therm: 50,
                n_meas: 50,
                n_md_steps: 10,
                dt: 0.05,
                seed: 42,
            },
            BenchmarkSpec {
                dims: [8, 8, 8, 4],
                beta: 5.69,
                n_therm: 50,
                n_meas: 50,
                n_md_steps: 10,
                dt: 0.05,
                seed: 42,
            },
            BenchmarkSpec {
                dims: [16, 16, 16, 4],
                beta: 5.69,
                n_therm: 20,
                n_meas: 20,
                n_md_steps: 10,
                dt: 0.03,
                seed: 42,
            },
            BenchmarkSpec {
                dims: [16, 16, 16, 8],
                beta: 6.06,
                n_therm: 20,
                n_meas: 20,
                n_md_steps: 15,
                dt: 0.02,
                seed: 42,
            },
        ]
    } else {
        let d = match (dims, lattice) {
            (Some(d), _) => d,
            (None, Some(l)) => [l, l, l, l],
            (None, None) => [8, 8, 8, 8],
        };
        vec![BenchmarkSpec::quenched_default(d, beta)]
    };

    let mut all_results: Vec<(String, Vec<Result<BenchmarkResult, String>>)> = Vec::new();

    for spec in &configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "  {} β={:.4} (V={})",
            spec.label(),
            spec.beta,
            spec.volume()
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let backends: Vec<&dyn ComputeBackend> = if let Some(ref g) = gpu {
            vec![&cpu, g]
        } else {
            vec![&cpu]
        };

        let results = compare_backends(&backends, spec);
        all_results.push((spec.label(), results));
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  {:>10} {:>8} {:>12} {:>12} {:>10} {:>10}",
        "Lattice", "β", "CPU ms/traj", "GPU ms/traj", "Speedup", "ΔPlaq"
    );
    for (label, results) in &all_results {
        let cpu_r = results.iter().find_map(|r| {
            r.as_ref()
                .ok()
                .filter(|r| r.backend_kind == BackendKind::BarraCudaCpu)
        });
        let gpu_r = results.iter().find_map(|r| {
            r.as_ref()
                .ok()
                .filter(|r| r.backend_kind == BackendKind::BarraCudaGpu)
        });

        let cpu_ms = cpu_r.map_or(f64::NAN, |r| r.ms_per_trajectory);
        let gpu_ms = gpu_r.map_or(f64::NAN, |r| r.ms_per_trajectory);
        let speedup = cpu_ms / gpu_ms;
        let delta_p = match (cpu_r, gpu_r) {
            (Some(c), Some(g)) => (c.mean_plaquette - g.mean_plaquette).abs(),
            _ => f64::NAN,
        };
        let beta_val = cpu_r.or(gpu_r).map_or(f64::NAN, |r| r.spec.beta);

        println!(
            "  {label:>10} {beta_val:>8.4} {cpu_ms:>12.1} {gpu_ms:>12.1} {speedup:>9.1}× {delta_p:>10.6}"
        );
    }
    println!();

    if let Some(path) = output {
        let json_results: Vec<serde_json::Value> = all_results
            .iter()
            .flat_map(|(_, results)| {
                results.iter().filter_map(|r| {
                    r.as_ref().ok().map(|r| {
                        serde_json::json!({
                            "backend": r.backend_name,
                            "kind": format!("{:?}", r.backend_kind),
                            "precision": format!("{:?}", r.precision_mode),
                            "lattice": r.spec.label(),
                            "dims": r.spec.dims,
                            "beta": r.spec.beta,
                            "volume": r.spec.volume(),
                            "mean_plaquette": r.mean_plaquette,
                            "std_plaquette": r.std_plaquette,
                            "acceptance": r.acceptance_rate,
                            "ms_per_trajectory": r.ms_per_trajectory,
                            "wall_s": r.wall_time.as_secs_f64(),
                        })
                    })
                })
            })
            .collect();
        let json = serde_json::to_string_pretty(&json_results).unwrap();
        std::fs::write(&path, json).unwrap_or_else(|e| eprintln!("Failed to write {path}: {e}"));
        println!("  Results saved to: {path}");
    }
}
