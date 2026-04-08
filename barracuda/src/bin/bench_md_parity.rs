// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-backend MD parity benchmark: barraCuda GPU vs Kokkos/LAMMPS.
//!
//! Runs the 9 PP Yukawa DSF cases from the Sarkas study on all available
//! backends and produces a gap analysis table. This is the quantitative
//! measurement of how close our pure Rust GPU pipeline is to Kokkos-CUDA.
//!
//! See `specs/MULTI_BACKEND_DISPATCH.md` for the three-tier dispatch strategy.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --bin bench_md_parity                          # full 9-case sweep
//! cargo run --release --bin bench_md_parity -- --quick               # single case (k2_G158)
//! cargo run --release --bin bench_md_parity -- --output=results.json # save JSON
//! cargo run --release --bin bench_md_parity -- --particles=500       # override N
//! HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin bench_md_parity  # Titan V only
//! ```

use hotspring_barracuda::bench::BackendKind;
use hotspring_barracuda::bench::md_backend::{
    BarraCudaMdBackend, KokkosLammpsBackend, MdBenchmarkBackend, MdBenchmarkResult,
    MdBenchmarkSpec, compare_md_backends, print_gap_analysis,
};

fn main() {
    let mut quick = false;
    let mut output: Option<String> = None;
    let mut n_particles: usize = 2000;

    for arg in std::env::args().skip(1) {
        if arg == "--quick" {
            quick = true;
        } else if let Some(val) = arg.strip_prefix("--output=") {
            output = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--particles=") {
            n_particles = val.parse().expect("--particles=N");
        }
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Multi-Backend MD Parity Benchmark                         ║");
    println!("║  barraCuda wgpu/Vulkan  vs  Kokkos-CUDA (LAMMPS)          ║");
    println!("║  See: specs/MULTI_BACKEND_DISPATCH.md                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Discover backends ──
    let gpu_backend = match BarraCudaMdBackend::new() {
        Ok(b) => {
            println!("  Tier 1 (wgpu/Vulkan): AVAILABLE");
            Some(b)
        }
        Err(e) => {
            println!("  Tier 1 (wgpu/Vulkan): unavailable ({e})");
            None
        }
    };

    let kokkos_backend = KokkosLammpsBackend::new();
    if kokkos_backend.available() {
        println!("  Tier 3 (Kokkos-CUDA): AVAILABLE");
    } else {
        println!("  Tier 3 (Kokkos-CUDA): unavailable (LAMMPS not found in PATH)");
        println!("    Install: cmake -DKokkos_ENABLE_CUDA=ON ... && make lmp");
    }
    println!("  Tier 2 (coralReef):   pending DRM dispatch (compile-only today)");
    println!();

    // ── Build specs ──
    let specs = if quick {
        let config = hotspring_barracuda::md::config::quick_test_case(n_particles);
        vec![MdBenchmarkSpec::from_config(&config)]
    } else {
        MdBenchmarkSpec::kokkos_parity_cases(n_particles)
    };

    println!(
        "  Running {} case{} at N={n_particles}:",
        specs.len(),
        if specs.len() == 1 { "" } else { "s" }
    );
    println!();

    // ── Run benchmarks ──
    let backends: Vec<&dyn MdBenchmarkBackend> = {
        let mut v: Vec<&dyn MdBenchmarkBackend> = Vec::new();
        if let Some(ref g) = gpu_backend {
            v.push(g);
        }
        if kokkos_backend.available() {
            v.push(&kokkos_backend);
        }
        v
    };

    if backends.is_empty() {
        println!("  ERROR: No backends available. Need at least a GPU or LAMMPS.");
        return;
    }

    let mut all_results: Vec<(String, Vec<Result<MdBenchmarkResult, String>>)> = Vec::new();

    for spec in &specs {
        let results = compare_md_backends(&backends, spec);
        all_results.push((spec.label.clone(), results));
        println!();
    }

    // ── Gap analysis ──
    println!("═══════════════════════════════════════════════════════════════");
    println!("  GAP ANALYSIS: barraCuda vs Kokkos-CUDA");
    println!("═══════════════════════════════════════════════════════════════");

    print_gap_analysis(&all_results);

    // ── Summary statistics ──
    let bc_results: Vec<&MdBenchmarkResult> = all_results
        .iter()
        .flat_map(|(_, rs)| {
            rs.iter().filter_map(|r| {
                r.as_ref()
                    .ok()
                    .filter(|r| r.backend_kind == BackendKind::BarraCudaGpu)
            })
        })
        .collect();

    let kk_results: Vec<&MdBenchmarkResult> = all_results
        .iter()
        .flat_map(|(_, rs)| {
            rs.iter().filter_map(|r| {
                r.as_ref()
                    .ok()
                    .filter(|r| r.backend_kind == BackendKind::KokkosCuda)
            })
        })
        .collect();

    println!();
    if !bc_results.is_empty() {
        let avg_sps: f64 =
            bc_results.iter().map(|r| r.steps_per_sec).sum::<f64>() / bc_results.len() as f64;
        let max_drift = bc_results
            .iter()
            .map(|r| r.energy_drift_pct)
            .fold(0.0_f64, f64::max);
        println!(
            "  barraCuda: {}/{} cases, avg {:.1} steps/s, max drift {:.3}%",
            bc_results.len(),
            specs.len(),
            avg_sps,
            max_drift
        );
        if let Some(r) = bc_results.first() {
            println!("    GPU: {}, driver: {}", r.adapter_name, r.driver_info);
        }
    }

    if !kk_results.is_empty() {
        let avg_sps: f64 =
            kk_results.iter().map(|r| r.steps_per_sec).sum::<f64>() / kk_results.len() as f64;
        println!(
            "  Kokkos:    {}/{} cases, avg {:.1} steps/s",
            kk_results.len(),
            specs.len(),
            avg_sps
        );
    }

    if !bc_results.is_empty() && !kk_results.is_empty() {
        let bc_avg =
            bc_results.iter().map(|r| r.steps_per_sec).sum::<f64>() / bc_results.len() as f64;
        let kk_avg =
            kk_results.iter().map(|r| r.steps_per_sec).sum::<f64>() / kk_results.len() as f64;
        println!("  Average gap: {:.1}× (target: 1.0×)", kk_avg / bc_avg);
    } else if kk_results.is_empty() {
        println!();
        println!("  Note: Kokkos/LAMMPS not installed — gap analysis unavailable.");
        println!("  To enable: install LAMMPS with Kokkos-CUDA and ensure `lmp` is in PATH.");
    }

    // ── JSON output ──
    if let Some(path) = output {
        let json_results: Vec<serde_json::Value> = all_results
            .iter()
            .flat_map(|(_, results)| {
                results.iter().filter_map(|r| {
                    r.as_ref().ok().map(|r| {
                        serde_json::json!({
                            "backend": r.backend_name,
                            "kind": format!("{:?}", r.backend_kind),
                            "label": r.label,
                            "n_particles": r.n_particles,
                            "kappa": r.kappa,
                            "gamma": r.gamma,
                            "steps_per_sec": r.steps_per_sec,
                            "energy_drift_pct": r.energy_drift_pct,
                            "wall_s": r.wall_time.as_secs_f64(),
                            "force_method": format!("{}", r.force_method),
                            "adapter": r.adapter_name,
                            "driver": r.driver_info,
                        })
                    })
                })
            })
            .collect();

        match serde_json::to_string_pretty(&json_results) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, json) {
                    eprintln!("  Failed to write {path}: {e}");
                } else {
                    println!();
                    println!("  Results saved to: {path}");
                }
            }
            Err(e) => eprintln!("  JSON serialization failed: {e}"),
        }
    }

    println!();
}
