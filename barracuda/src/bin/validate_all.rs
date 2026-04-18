// SPDX-License-Identifier: AGPL-3.0-or-later

//! Meta-validator: runs all 63 hotSpring validation suites in sequence.
//!
//! Exit code is 0 only if ALL validation binaries pass.
//! Follows the hotSpring pattern: explicit pass/fail, exit code 0/1.
//!
//! Three-tier validation: Python baselines → Rust validation → NUCLEUS IPC composition.
//! Suites 1–57 prove Rust/GPU parity with Python. Suites 58–62 prove NUCLEUS primal
//! composition produces the same science via IPC (the "primal proof").

use std::process::{self, Command};
use std::time::Instant;

/// A validation suite to run.
struct Suite {
    name: &'static str,
    binary: &'static str,
    requires_gpu: bool,
}

const SUITES: &[Suite] = &[
    Suite {
        name: "Special Functions",
        binary: "validate_special_functions",
        requires_gpu: false,
    },
    Suite {
        name: "Linear Algebra",
        binary: "validate_linalg",
        requires_gpu: false,
    },
    Suite {
        name: "Optimizers & Numerics",
        binary: "validate_optimizers",
        requires_gpu: false,
    },
    Suite {
        name: "MD Forces & Integrators",
        binary: "validate_md",
        requires_gpu: false,
    },
    Suite {
        name: "Nuclear EOS (Pure Rust)",
        binary: "validate_nuclear_eos",
        requires_gpu: false,
    },
    Suite {
        name: "HFB Verification (SLy4)",
        binary: "verify_hfb",
        requires_gpu: false,
    },
    Suite {
        name: "WGSL f64 Builtins",
        binary: "f64_builtin_test",
        requires_gpu: true,
    },
    Suite {
        name: "BarraCuda HFB Pipeline",
        binary: "validate_barracuda_hfb",
        requires_gpu: true,
    },
    Suite {
        name: "BarraCuda MD Pipeline",
        binary: "validate_barracuda_pipeline",
        requires_gpu: true,
    },
    Suite {
        name: "PPPM Coulomb/Ewald",
        binary: "validate_pppm",
        requires_gpu: true,
    },
    Suite {
        name: "CPU/GPU Parity",
        binary: "validate_cpu_gpu_parity",
        requires_gpu: true,
    },
    Suite {
        name: "NAK Eigensolve (correctness)",
        binary: "validate_nak_eigensolve",
        requires_gpu: true,
    },
    Suite {
        name: "GPU-Only Transport (Paper 5)",
        binary: "validate_transport_gpu_only",
        requires_gpu: true,
    },
    Suite {
        name: "Screened Coulomb (Paper 6)",
        binary: "validate_screened_coulomb",
        requires_gpu: false,
    },
    Suite {
        name: "HotQCD EOS Tables (Paper 7)",
        binary: "validate_hotqcd_eos",
        requires_gpu: false,
    },
    Suite {
        name: "Pure Gauge SU(3) (Paper 8)",
        binary: "validate_pure_gauge",
        requires_gpu: false,
    },
    Suite {
        name: "Production QCD β-Scan (Papers 9-12)",
        binary: "validate_production_qcd",
        requires_gpu: false,
    },
    Suite {
        name: "Dynamical Fermion QCD (Paper 10)",
        binary: "validate_dynamical_qcd",
        requires_gpu: false,
    },
    Suite {
        name: "Abelian Higgs (Paper 13)",
        binary: "validate_abelian_higgs",
        requires_gpu: false,
    },
    Suite {
        name: "NPU Quantization Cascade",
        binary: "validate_npu_quantization",
        requires_gpu: false,
    },
    Suite {
        name: "NPU Beyond-SDK Capabilities",
        binary: "validate_npu_beyond_sdk",
        requires_gpu: false,
    },
    Suite {
        name: "NPU Physics Pipeline",
        binary: "validate_npu_pipeline",
        requires_gpu: false,
    },
    Suite {
        name: "Lattice QCD + NPU Phase",
        binary: "validate_lattice_npu",
        requires_gpu: false,
    },
    Suite {
        name: "Heterogeneous Real-Time Monitor",
        binary: "validate_hetero_monitor",
        requires_gpu: false,
    },
    Suite {
        name: "Spectral Theory (Kachkovskiy)",
        binary: "validate_spectral",
        requires_gpu: false,
    },
    Suite {
        name: "Lanczos + 2D Anderson (Kachkovskiy)",
        binary: "validate_lanczos",
        requires_gpu: false,
    },
    Suite {
        name: "3D Anderson (Kachkovskiy)",
        binary: "validate_anderson_3d",
        requires_gpu: false,
    },
    Suite {
        name: "Hofstadter Butterfly (Kachkovskiy)",
        binary: "validate_hofstadter",
        requires_gpu: false,
    },
    Suite {
        name: "BarraCuda Evolution (CPU Foundation)",
        binary: "validate_barracuda_evolution",
        requires_gpu: false,
    },
    Suite {
        name: "GPU SpMV (Spectral Theory P1)",
        binary: "validate_gpu_spmv",
        requires_gpu: true,
    },
    Suite {
        name: "GPU Lanczos Eigensolve",
        binary: "validate_gpu_lanczos",
        requires_gpu: true,
    },
    Suite {
        name: "GPU Staggered Dirac (Papers 9-12)",
        binary: "validate_gpu_dirac",
        requires_gpu: true,
    },
    Suite {
        name: "GPU CG Solver (Papers 9-12)",
        binary: "validate_gpu_cg",
        requires_gpu: true,
    },
    Suite {
        name: "Pure GPU QCD Workload",
        binary: "validate_pure_gpu_qcd",
        requires_gpu: true,
    },
    Suite {
        name: "GPU Streaming HMC",
        binary: "validate_gpu_streaming",
        requires_gpu: true,
    },
    Suite {
        name: "GPU Streaming Dynamical",
        binary: "validate_gpu_streaming_dyn",
        requires_gpu: true,
    },
    Suite {
        name: "Reservoir Transport (ESN)",
        binary: "validate_reservoir_transport",
        requires_gpu: true,
    },
    Suite {
        name: "Stanton-Murillo Transport (Paper 5)",
        binary: "validate_stanton_murillo",
        requires_gpu: false,
    },
    Suite {
        name: "Transport CPU/GPU Parity",
        binary: "validate_transport",
        requires_gpu: true,
    },
    Suite {
        name: "TTM Laser-Plasma (Paper 2)",
        binary: "validate_ttm",
        requires_gpu: false,
    },
    Suite {
        name: "Gradient Flow (Paper 43)",
        binary: "validate_gradient_flow",
        requires_gpu: false,
    },
    Suite {
        name: "GPU Gradient Flow (Paper 43 GPU)",
        binary: "validate_gpu_gradient_flow",
        requires_gpu: true,
    },
    Suite {
        name: "Chuna Papers 43/44/45",
        binary: "validate_chuna",
        requires_gpu: false,
    },
    Suite {
        name: "BGK Dielectric (Paper 44)",
        binary: "validate_dielectric",
        requires_gpu: false,
    },
    Suite {
        name: "GPU Dielectric (Paper 44 GPU)",
        binary: "validate_gpu_dielectric",
        requires_gpu: true,
    },
    Suite {
        name: "Kinetic-Fluid Coupling (Haack)",
        binary: "validate_kinetic_fluid",
        requires_gpu: false,
    },
    Suite {
        name: "DSF vs MD (Chuna/Murillo)",
        binary: "validate_dsf_vs_md",
        requires_gpu: false,
    },
    Suite {
        name: "FPEOS Table (Paper 32)",
        binary: "validate_fpeos",
        requires_gpu: false,
    },
    Suite {
        name: "atoMEC Average-Atom (Paper 33)",
        binary: "validate_atomec",
        requires_gpu: false,
    },
    Suite {
        name: "Freeze-Out Curvature (Paper 12)",
        binary: "validate_freeze_out",
        requires_gpu: false,
    },
    Suite {
        name: "HVP g-2 (Paper 11)",
        binary: "validate_hvp_g2",
        requires_gpu: false,
    },
    Suite {
        name: "Production QCD v2 (Omelyan+Hasenbusch)",
        binary: "validate_production_qcd_v2",
        requires_gpu: false,
    },
    Suite {
        name: "GPU β-Scan (Full Temperature Sweep)",
        binary: "validate_gpu_beta_scan",
        requires_gpu: true,
    },
    Suite {
        name: "GPU Dynamical Fermion HMC",
        binary: "validate_gpu_dynamical_hmc",
        requires_gpu: true,
    },
    Suite {
        name: "Sovereign Round-Trip",
        binary: "validate_sovereign_roundtrip",
        requires_gpu: true,
    },
    Suite {
        name: "Precision Matrix (Silicon)",
        binary: "validate_precision_matrix",
        requires_gpu: true,
    },
    Suite {
        name: "BarraCuda CPU/GPU Parity (All Domains)",
        binary: "validate_barracuda_cpu_gpu_parity",
        requires_gpu: true,
    },
    Suite {
        name: "NUCLEUS Composition",
        binary: "validate_nucleus_composition",
        requires_gpu: false,
    },
    Suite {
        name: "Primal Proof (Level 5)",
        binary: "validate_primal_proof",
        requires_gpu: false,
    },
    Suite {
        name: "NUCLEUS Tower Atomic",
        binary: "validate_nucleus_tower",
        requires_gpu: false,
    },
    Suite {
        name: "NUCLEUS Node Atomic",
        binary: "validate_nucleus_node",
        requires_gpu: false,
    },
    Suite {
        name: "NUCLEUS Nest Atomic",
        binary: "validate_nucleus_nest",
        requires_gpu: false,
    },
    Suite {
        name: "Squirrel Inference Round-Trip",
        binary: "validate_squirrel_roundtrip",
        requires_gpu: false,
    },
];

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  hotSpring Validation Suite — All Domains");
    println!("  Python baseline → Rust → GPU evolution validation");
    println!("═══════════════════════════════════════════════════════════\n");

    let skip_gpu = std::env::args().any(|a| a == "--skip-gpu");
    if skip_gpu {
        println!("  --skip-gpu: skipping GPU validation suites\n");
    }

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let t_total = Instant::now();
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut skipped = 0usize;
    let mut failures: Vec<&str> = Vec::new();

    for suite in SUITES {
        if suite.requires_gpu && skip_gpu {
            println!("  SKIP  {:<30} (GPU required)", suite.name);
            skipped += 1;
            continue;
        }

        let t_suite = Instant::now();
        print!("  RUN   {:<30} ", suite.name);

        let result = Command::new("cargo")
            .args(["run", "--release", "--bin", suite.binary])
            .current_dir(manifest_dir)
            .output();

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
                    // Print last 5 lines of stdout for context
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
