// SPDX-License-Identifier: AGPL-3.0-only

//! N-scaling complexity benchmark: barraCuda vs Kokkos-CUDA head-to-head.
//!
//! Profiles both stacks across problem sizes (N=500..50k), force algorithms
//! (AllPairs, Verlet), and hardware to produce scaling curves and identify
//! the complexity crossover points where barraCuda catches or loses to Kokkos.
//!
//! Output: table + optional JSON for plotting.
//!
//! Usage:
//!   bench_kokkos_complexity                    # default N sweep
//!   bench_kokkos_complexity --sizes 500,2000,10000,50000
//!   bench_kokkos_complexity --quick            # 500 + 2000 only
//!   bench_kokkos_complexity --output data.json

use hotspring_barracuda::bench::md_backend::{
    BarraCudaMdBackend, ForceMethod, KokkosLammpsBackend, MdBenchmarkBackend, MdBenchmarkResult,
    MdBenchmarkSpec,
};

/// Physics case template (algorithm + plasma parameters).
struct ScalingCase {
    label: &'static str,
    kappa: f64,
    gamma: f64,
    rc: f64,
    expected_method: ForceMethod,
}

const CASES: &[ScalingCase] = &[
    ScalingCase {
        label: "AP_k1_G72",
        kappa: 1.0,
        gamma: 72.0,
        rc: 8.0,
        expected_method: ForceMethod::AllPairs,
    },
    ScalingCase {
        label: "VL_k2_G158",
        kappa: 2.0,
        gamma: 158.0,
        rc: 6.5,
        expected_method: ForceMethod::VerletList,
    },
    ScalingCase {
        label: "VL_k3_G503",
        kappa: 3.0,
        gamma: 503.0,
        rc: 6.0,
        expected_method: ForceMethod::VerletList,
    },
];

fn build_spec(case: &ScalingCase, n: usize, quick: bool) -> MdBenchmarkSpec {
    let (equil, prod) = if quick { (500, 2000) } else { (2000, 10_000) };
    MdBenchmarkSpec {
        label: format!("{}_{}", case.label, n),
        n_particles: n,
        kappa: case.kappa,
        gamma: case.gamma,
        rc: case.rc,
        dt: 0.01,
        equil_steps: equil,
        prod_steps: prod,
        force_method: case.expected_method,
    }
}

#[derive(serde::Serialize)]
struct ScalingResult {
    label: String,
    n_particles: usize,
    method: String,
    kappa: f64,
    gamma: f64,
    barracuda_steps_per_sec: Option<f64>,
    kokkos_steps_per_sec: Option<f64>,
    gap: Option<f64>,
    barracuda_wall_s: Option<f64>,
    kokkos_wall_s: Option<f64>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let output_path = args
        .windows(2)
        .find(|w| w[0] == "--output")
        .map(|w| w[1].clone());

    let sizes: Vec<usize> = args
        .windows(2)
        .find(|w| w[0] == "--sizes")
        .map(|w| {
            w[1].split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect()
        })
        .unwrap_or_else(|| {
            if quick {
                vec![500, 2000]
            } else {
                vec![500, 2000, 10_000, 50_000]
            }
        });

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  N-Scaling Complexity Benchmark                                ║");
    println!("║  barraCuda wgpu/Vulkan  vs  Kokkos-CUDA (LAMMPS)              ║");
    println!("║  Profile: throughput vs N, per-algorithm, per-hardware         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Discover backends ──
    let bc = match BarraCudaMdBackend::new() {
        Ok(b) => {
            println!("  Tier 1 (wgpu/Vulkan): AVAILABLE");
            Some(b)
        }
        Err(e) => {
            println!("  Tier 1 (wgpu/Vulkan): unavailable ({e})");
            None
        }
    };

    let kokkos = KokkosLammpsBackend::new();
    if kokkos.available() {
        println!("  Tier 3 (Kokkos-CUDA): AVAILABLE");
    } else {
        println!("  Tier 3 (Kokkos-CUDA): NOT FOUND (install lmp with Kokkos-CUDA)");
    }

    println!("  Sizes: {:?}", sizes);
    println!(
        "  Cases: {}",
        CASES.iter().map(|c| c.label).collect::<Vec<_>>().join(", ")
    );
    if quick {
        println!("  Mode: quick (reduced steps)");
    }
    println!();

    if bc.is_none() && !kokkos.available() {
        println!("  ERROR: No backends available.");
        return;
    }

    let mut results: Vec<ScalingResult> = Vec::new();

    for case in CASES {
        println!(
            "━━━ {} ({}, κ={}, Γ={}) ━━━",
            case.label, case.expected_method, case.kappa, case.gamma
        );
        println!();
        println!(
            "  {:>8}  {:>12}  {:>12}  {:>8}  {:>8}  {:>8}",
            "N", "barraCuda", "Kokkos", "Gap", "bc(s)", "kk(s)"
        );
        println!(
            "  {:>8}  {:>12}  {:>12}  {:>8}  {:>8}  {:>8}",
            "────", "steps/s", "steps/s", "────", "────", "────"
        );

        for &n in &sizes {
            let spec = build_spec(case, n, quick);

            let bc_result: Option<Result<MdBenchmarkResult, String>> =
                bc.as_ref().map(|b| b.run_yukawa_md(&spec));
            let kk_result: Option<Result<MdBenchmarkResult, String>> = if kokkos.available() {
                Some(kokkos.run_yukawa_md(&spec))
            } else {
                None
            };

            let bc_sps = bc_result
                .as_ref()
                .and_then(|r| r.as_ref().ok())
                .map(|r| r.steps_per_sec);
            let kk_sps = kk_result
                .as_ref()
                .and_then(|r| r.as_ref().ok())
                .map(|r| r.steps_per_sec);
            let gap = match (bc_sps, kk_sps) {
                (Some(b), Some(k)) if b > 0.0 => Some(k / b),
                _ => None,
            };
            let bc_wall = bc_result
                .as_ref()
                .and_then(|r| r.as_ref().ok())
                .map(|r| r.wall_time.as_secs_f64());
            let kk_wall = kk_result
                .as_ref()
                .and_then(|r| r.as_ref().ok())
                .map(|r| r.wall_time.as_secs_f64());

            let bc_str = match &bc_result {
                Some(Ok(r)) => format!("{:.1}", r.steps_per_sec),
                Some(Err(e)) => format!("ERR:{}", &e[..e.len().min(20)]),
                None => "—".to_string(),
            };
            let kk_str = kk_sps.map_or("—".to_string(), |s| format!("{s:.1}"));
            let gap_str = gap.map_or("—".to_string(), |g| format!("{g:.1}×"));
            let bcw_str = bc_wall.map_or("—".to_string(), |w| format!("{w:.1}"));
            let kkw_str = kk_wall.map_or("—".to_string(), |w| format!("{w:.1}"));

            println!(
                "  {:>8}  {:>12}  {:>12}  {:>8}  {:>8}  {:>8}",
                n, bc_str, kk_str, gap_str, bcw_str, kkw_str
            );

            results.push(ScalingResult {
                label: spec.label,
                n_particles: n,
                method: format!("{}", case.expected_method),
                kappa: case.kappa,
                gamma: case.gamma,
                barracuda_steps_per_sec: bc_sps,
                kokkos_steps_per_sec: kk_sps,
                gap,
                barracuda_wall_s: bc_wall,
                kokkos_wall_s: kk_wall,
            });
        }
        println!();
    }

    // ── Complexity analysis ──
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  COMPLEXITY ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    for case in CASES {
        let case_results: Vec<&ScalingResult> = results
            .iter()
            .filter(|r| r.method == format!("{}", case.expected_method) && r.kappa == case.kappa)
            .collect();

        if case_results.len() < 2 {
            continue;
        }

        println!("  {} ({}):", case.label, case.expected_method);

        print_scaling_exponent("barraCuda", &case_results, |r| r.barracuda_steps_per_sec);
        print_scaling_exponent("Kokkos", &case_results, |r| r.kokkos_steps_per_sec);

        // Gap trend
        let gap_points: Vec<(f64, f64)> = case_results
            .iter()
            .filter_map(|r| r.gap.map(|g| (r.n_particles as f64, g)))
            .collect();
        if gap_points.len() >= 2 {
            let (n1, g1) = gap_points[0];
            let (n2, g2) = gap_points[gap_points.len() - 1];
            let trend = if (g2 - g1).abs() < 0.5 {
                "stable"
            } else if g2 > g1 {
                "widening (Kokkos scales better)"
            } else {
                "narrowing (barraCuda scales better)"
            };
            println!("    Gap trend: {g1:.1}× at N={n1:.0} → {g2:.1}× at N={n2:.0} ({trend})");
        }
        println!();
    }

    // ── Summary ──
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  ABSORPTION TARGETS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  Use these results to identify where barraCuda loses ground:");
    println!("  • If gap WIDENS with N → memory access / neighbor-list build overhead");
    println!("  • If gap is STABLE → arithmetic throughput (f64 penalty)");
    println!("  • If gap NARROWS with N → dispatch/launch overhead amortized at scale");
    println!("  • AllPairs α should be ~2 for both; deviation = GPU occupancy issues");
    println!("  • Verlet α should be ~1.3 for both; deviation = neighbor list scaling");
    println!();

    if let Some(path) = output_path {
        let json = serde_json::to_string_pretty(&results)
            .unwrap_or_else(|e| format!("JSON error: {e}"));
        if let Err(e) = std::fs::write(&path, &json) {
            eprintln!("  Failed to write {path}: {e}");
        } else {
            println!("  Results written to: {path}");
        }
    }
}

fn print_scaling_exponent<F>(label: &str, results: &[&ScalingResult], get_sps: F)
where
    F: Fn(&ScalingResult) -> Option<f64>,
{
    let points: Vec<(f64, f64)> = results
        .iter()
        .filter_map(|r| get_sps(r).map(|s| (r.n_particles as f64, s)))
        .collect();

    if points.len() < 2 {
        return;
    }

    let (n1, s1) = points[0];
    let (n2, s2) = points[points.len() - 1];
    if s1 <= 0.0 || s2 <= 0.0 || n1 <= 0.0 || n2 <= 0.0 || (n2 / n1 - 1.0).abs() < 0.01 {
        return;
    }

    // steps/s ∝ 1/N^α  ⟹  α = -log(s2/s1) / log(N2/N1)
    let alpha = -(s2.ln() - s1.ln()) / (n2.ln() - n1.ln());
    let complexity_label = if alpha < 0.8 {
        "sub-linear (GPU saturated at small N)"
    } else if alpha < 1.3 {
        "~O(N) — neighbor-list dominated"
    } else if alpha < 1.8 {
        "~O(N·M) — neighbor-list with growing M"
    } else {
        "~O(N²) — all-pairs dominated"
    };
    println!("    {label:10}: α = {alpha:.2} ({complexity_label})");
    println!("      N={n1:.0}: {s1:.1} steps/s → N={n2:.0}: {s2:.1} steps/s");
}
