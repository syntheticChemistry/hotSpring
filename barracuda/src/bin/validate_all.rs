// SPDX-License-Identifier: AGPL-3.0-or-later

//! Meta-validator: runs all hotSpring validation suites in sequence.
//!
//! Exit code is 0 only if ALL selected validation binaries pass.
//! Follows the hotSpring pattern: explicit pass/fail, exit code 0/1.
//!
//! Three tiers:
//!   `smoke`   — lib tests + structural (no live primals or GPU required)
//!   `nucleus` — requires live NUCLEUS primals for IPC validation
//!   `silicon` — requires GPU hardware
//!
//! Usage:
//!   `validate_all`                 — runs all tiers
//!   `validate_all --tier smoke`    — smoke tier only
//!   `validate_all --tier nucleus`  — smoke + nucleus
//!   `validate_all --tier silicon`  — smoke + nucleus + silicon
//!   `validate_all --skip-gpu`      — legacy alias for `--tier nucleus`

use std::process::{self, Command};
use std::time::Instant;

/// Validation tier for a suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Tier {
    Smoke = 0,
    Nucleus = 1,
    Silicon = 2,
}

/// A validation suite to run.
struct Suite {
    name: &'static str,
    binary: &'static str,
    tier: Tier,
}

// 65 suites total: 38 smoke, 7 nucleus, 20 silicon
const SUITES: &[Suite] = &[
    // ── Smoke tier (no primals, no GPU) ────────────────────────────────
    Suite {
        name: "Special Functions",
        binary: "validate_special_functions",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Linear Algebra",
        binary: "validate_linalg",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Optimizers & Numerics",
        binary: "validate_optimizers",
        tier: Tier::Smoke,
    },
    Suite {
        name: "MD Forces & Integrators",
        binary: "validate_md",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Nuclear EOS (Pure Rust)",
        binary: "validate_nuclear_eos",
        tier: Tier::Smoke,
    },
    Suite {
        name: "HFB Verification (SLy4)",
        binary: "verify_hfb",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Screened Coulomb (Paper 6)",
        binary: "validate_screened_coulomb",
        tier: Tier::Smoke,
    },
    Suite {
        name: "HotQCD EOS Tables (Paper 7)",
        binary: "validate_hotqcd_eos",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Pure Gauge SU(3) (Paper 8)",
        binary: "validate_pure_gauge",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Production QCD β-Scan (Papers 9-12)",
        binary: "validate_production_qcd",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Dynamical Fermion QCD (Paper 10)",
        binary: "validate_dynamical_qcd",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Abelian Higgs (Paper 13)",
        binary: "validate_abelian_higgs",
        tier: Tier::Smoke,
    },
    Suite {
        name: "NPU Quantization Cascade",
        binary: "validate_npu_quantization",
        tier: Tier::Smoke,
    },
    Suite {
        name: "NPU Beyond-SDK Capabilities",
        binary: "validate_npu_beyond_sdk",
        tier: Tier::Smoke,
    },
    Suite {
        name: "NPU Physics Pipeline",
        binary: "validate_npu_pipeline",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Lattice QCD + NPU Phase",
        binary: "validate_lattice_npu",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Heterogeneous Real-Time Monitor",
        binary: "validate_hetero_monitor",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Spectral Theory (Kachkovskiy)",
        binary: "validate_spectral",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Lanczos + 2D Anderson (Kachkovskiy)",
        binary: "validate_lanczos",
        tier: Tier::Smoke,
    },
    Suite {
        name: "3D Anderson (Kachkovskiy)",
        binary: "validate_anderson_3d",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Hofstadter Butterfly (Kachkovskiy)",
        binary: "validate_hofstadter",
        tier: Tier::Smoke,
    },
    Suite {
        name: "BarraCuda Evolution (CPU Foundation)",
        binary: "validate_barracuda_evolution",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Stanton-Murillo Transport (Paper 5)",
        binary: "validate_stanton_murillo",
        tier: Tier::Smoke,
    },
    Suite {
        name: "TTM Laser-Plasma (Paper 2)",
        binary: "validate_ttm",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Gradient Flow (Paper 43)",
        binary: "validate_gradient_flow",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Chuna Papers 43/44/45",
        binary: "validate_chuna",
        tier: Tier::Smoke,
    },
    Suite {
        name: "BGK Dielectric (Paper 44)",
        binary: "validate_dielectric",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Kinetic-Fluid Coupling (Haack)",
        binary: "validate_kinetic_fluid",
        tier: Tier::Smoke,
    },
    Suite {
        name: "DSF vs MD (Chuna/Murillo)",
        binary: "validate_dsf_vs_md",
        tier: Tier::Smoke,
    },
    Suite {
        name: "FPEOS Table (Paper 32)",
        binary: "validate_fpeos",
        tier: Tier::Smoke,
    },
    Suite {
        name: "atoMEC Average-Atom (Paper 33)",
        binary: "validate_atomec",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Freeze-Out Curvature (Paper 12)",
        binary: "validate_freeze_out",
        tier: Tier::Smoke,
    },
    Suite {
        name: "HVP g-2 (Paper 11)",
        binary: "validate_hvp_g2",
        tier: Tier::Smoke,
    },
    Suite {
        name: "Production QCD v2 (Omelyan+Hasenbusch)",
        binary: "validate_production_qcd_v2",
        tier: Tier::Smoke,
    },
    Suite {
        name: "guideStone (Level 5 Certified)",
        binary: "hotspring_guidestone",
        tier: Tier::Smoke,
    },
    // ── Nucleus tier (requires live primals via IPC) ───────────────────
    Suite {
        name: "NUCLEUS Composition",
        binary: "validate_nucleus_composition",
        tier: Tier::Nucleus,
    },
    Suite {
        name: "Primal Proof (Level 5)",
        binary: "validate_primal_proof",
        tier: Tier::Nucleus,
    },
    Suite {
        name: "NUCLEUS Tower Atomic",
        binary: "validate_nucleus_tower",
        tier: Tier::Nucleus,
    },
    Suite {
        name: "NUCLEUS Node Atomic",
        binary: "validate_nucleus_node",
        tier: Tier::Nucleus,
    },
    Suite {
        name: "NUCLEUS Nest Atomic",
        binary: "validate_nucleus_nest",
        tier: Tier::Nucleus,
    },
    Suite {
        name: "Squirrel Inference Round-Trip",
        binary: "validate_squirrel_roundtrip",
        tier: Tier::Nucleus,
    },
    Suite {
        name: "Compute Trio Pipeline",
        binary: "validate_compute_trio_pipeline",
        tier: Tier::Nucleus,
    },
    // ── Silicon tier (requires GPU hardware) ──────────────────────────
    Suite {
        name: "WGSL f64 Builtins",
        binary: "f64_builtin_test",
        tier: Tier::Silicon,
    },
    Suite {
        name: "BarraCuda HFB Pipeline",
        binary: "validate_barracuda_hfb",
        tier: Tier::Silicon,
    },
    Suite {
        name: "BarraCuda MD Pipeline",
        binary: "validate_barracuda_pipeline",
        tier: Tier::Silicon,
    },
    Suite {
        name: "PPPM Coulomb/Ewald",
        binary: "validate_pppm",
        tier: Tier::Silicon,
    },
    Suite {
        name: "CPU/GPU Parity",
        binary: "validate_cpu_gpu_parity",
        tier: Tier::Silicon,
    },
    Suite {
        name: "NAK Eigensolve (correctness)",
        binary: "validate_nak_eigensolve",
        tier: Tier::Silicon,
    },
    Suite {
        name: "GPU-Only Transport (Paper 5)",
        binary: "validate_transport_gpu_only",
        tier: Tier::Silicon,
    },
    Suite {
        name: "GPU SpMV (Spectral Theory P1)",
        binary: "validate_gpu_spmv",
        tier: Tier::Silicon,
    },
    Suite {
        name: "GPU Lanczos Eigensolve",
        binary: "validate_gpu_lanczos",
        tier: Tier::Silicon,
    },
    Suite {
        name: "GPU Staggered Dirac (Papers 9-12)",
        binary: "validate_gpu_dirac",
        tier: Tier::Silicon,
    },
    Suite {
        name: "GPU CG Solver (Papers 9-12)",
        binary: "validate_gpu_cg",
        tier: Tier::Silicon,
    },
    Suite {
        name: "Pure GPU QCD Workload",
        binary: "validate_pure_gpu_qcd",
        tier: Tier::Silicon,
    },
    Suite {
        name: "GPU Streaming HMC",
        binary: "validate_gpu_streaming",
        tier: Tier::Silicon,
    },
    Suite {
        name: "GPU Streaming Dynamical",
        binary: "validate_gpu_streaming_dyn",
        tier: Tier::Silicon,
    },
    Suite {
        name: "Reservoir Transport (ESN)",
        binary: "validate_reservoir_transport",
        tier: Tier::Silicon,
    },
    Suite {
        name: "Transport CPU/GPU Parity",
        binary: "validate_transport",
        tier: Tier::Silicon,
    },
    Suite {
        name: "GPU Gradient Flow (Paper 43 GPU)",
        binary: "validate_gpu_gradient_flow",
        tier: Tier::Silicon,
    },
    Suite {
        name: "GPU Dielectric (Paper 44 GPU)",
        binary: "validate_gpu_dielectric",
        tier: Tier::Silicon,
    },
    Suite {
        name: "GPU β-Scan (Full Temperature Sweep)",
        binary: "validate_gpu_beta_scan",
        tier: Tier::Silicon,
    },
    Suite {
        name: "GPU Dynamical Fermion HMC",
        binary: "validate_gpu_dynamical_hmc",
        tier: Tier::Silicon,
    },
    Suite {
        name: "Sovereign Round-Trip",
        binary: "validate_sovereign_roundtrip",
        tier: Tier::Silicon,
    },
    Suite {
        name: "Precision Matrix (Silicon)",
        binary: "validate_precision_matrix",
        tier: Tier::Silicon,
    },
    Suite {
        name: "BarraCuda CPU/GPU Parity (All Domains)",
        binary: "validate_barracuda_cpu_gpu_parity",
        tier: Tier::Silicon,
    },
];

fn parse_tier(args: &[String]) -> Tier {
    for (i, arg) in args.iter().enumerate() {
        if arg == "--tier" {
            if let Some(val) = args.get(i + 1) {
                return match val.as_str() {
                    "smoke" => Tier::Smoke,
                    "nucleus" => Tier::Nucleus,
                    "silicon" | "gpu" => Tier::Silicon,
                    _ => {
                        eprintln!("Unknown tier: {val} (valid: smoke, nucleus, silicon)");
                        process::exit(2);
                    }
                };
            }
        }
        if arg == "--skip-gpu" {
            return Tier::Nucleus;
        }
    }
    Tier::Silicon
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_tier = parse_tier(&args);

    let tier_label = match max_tier {
        Tier::Smoke => "smoke",
        Tier::Nucleus => "nucleus",
        Tier::Silicon => "silicon (all)",
    };

    println!("═══════════════════════════════════════════════════════════");
    println!("  hotSpring Validation Suite — tier: {tier_label}");
    println!("  {} suites registered", SUITES.len());
    println!("═══════════════════════════════════════════════════════════\n");

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let release_dir = std::path::Path::new(manifest_dir)
        .join("target")
        .join("release");

    let t_total = Instant::now();
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut skipped = 0usize;
    let mut failures: Vec<&str> = Vec::new();

    for suite in SUITES {
        if suite.tier > max_tier {
            println!("  SKIP  {:<40} (tier: {:?})", suite.name, suite.tier);
            skipped += 1;
            continue;
        }

        let t_suite = Instant::now();
        print!("  RUN   {:<40} ", suite.name);

        let pre_built = release_dir.join(suite.binary);
        let result = if pre_built.exists() {
            Command::new(&pre_built).current_dir(manifest_dir).output()
        } else {
            Command::new("cargo")
                .args(["run", "--release", "--bin", suite.binary])
                .current_dir(manifest_dir)
                .output()
        };

        match result {
            Ok(output) => {
                let elapsed = t_suite.elapsed().as_secs_f64();
                if output.status.success() {
                    println!("PASS  ({elapsed:.1}s)");
                    passed += 1;
                } else {
                    println!("FAIL  ({elapsed:.1}s)");
                    failed += 1;
                    failures.push(suite.name);
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for line in stdout
                        .lines()
                        .rev()
                        .take(5)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                    {
                        println!("        {line}");
                    }
                }
            }
            Err(e) => {
                println!("ERROR ({e})");
                failed += 1;
                failures.push(suite.name);
            }
        }
    }

    let total_time = t_total.elapsed().as_secs_f64();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  TOTAL: {passed} passed, {failed} failed, {skipped} skipped ({total_time:.1}s)");

    if !failures.is_empty() {
        println!("  FAILURES: {}", failures.join(", "));
    }

    if failed == 0 {
        println!("  ALL VALIDATION SUITES PASSED");
        println!("═══════════════════════════════════════════════════════════");
        process::exit(0);
    } else {
        println!("  SOME VALIDATION SUITES FAILED");
        println!("═══════════════════════════════════════════════════════════");
        process::exit(1);
    }
}
