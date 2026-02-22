// SPDX-License-Identifier: AGPL-3.0-only

//! Meta-validator: runs all hotSpring validation suites in sequence.
//!
//! Exit code is 0 only if ALL validation binaries pass.
//! Follows the hotSpring pattern: explicit pass/fail, exit code 0/1.
//!
//! # Validation suites (in order)
//!
//! | Binary | Domain | GPU? |
//! |--------|--------|------|
//! | `validate_special_functions` | Gamma, Bessel, Laguerre | No |
//! | `validate_linalg` | LU, QR, SVD, eigh | No |
//! | `validate_optimizers` | NM, BFGS, RK45, Sobol | No |
//! | `validate_md` | LJ, Coulomb, Morse, VV | No |
//! | `validate_nuclear_eos` | L1 SEMF, L2 HFB, NMP | No |
//! | `verify_hfb` | HFB verbose SLy4 verification | No |
//! | `f64_builtin_test` | WGSL f64 builtins | GPU |
//! | `validate_barracuda_hfb` | BCS bisection, BatchedEigh | GPU |
//! | `validate_barracuda_pipeline` | Yukawa MD GPU ops | GPU |
//! | `validate_pppm` | PPPM Coulomb/Ewald | GPU |
//! | `validate_cpu_gpu_parity` | CPU vs GPU same-physics proof | GPU |
//! | `validate_nak_eigensolve` | NAK-optimized eigensolve correctness | GPU |
//! | `validate_transport_gpu_only` | GPU-only transport: D* via GPU VACF | GPU |
//! | `validate_screened_coulomb` | Yukawa bound states (Murillo-Weisheit) | No |
//! | `validate_pure_gauge` | Pure gauge SU(3) lattice QCD | No |
//! | `validate_dynamical_qcd` | Dynamical fermion HMC (Paper 10) | No |
//! | `validate_abelian_higgs` | Abelian Higgs (1+1)D U(1)+scalar | No |
//! | `validate_npu_quantization` | NPU ESN quantization cascade | No |
//! | `validate_npu_beyond_sdk` | NPU beyond-SDK capabilities | No |
//! | `validate_npu_pipeline` | NPU physics pipeline math | No |
//! | `validate_lattice_npu` | Lattice QCD + NPU phase classification | No |
//! | `validate_hetero_monitor` | Heterogeneous real-time physics monitor | No |
//! | `validate_spectral` | Spectral theory: Anderson + almost-Mathieu | No |
//! | `validate_lanczos` | Lanczos + SpMV + 2D Anderson | No |
//! | `validate_anderson_3d` | 3D Anderson: mobility edge, dimensional hierarchy | No |
//! | `validate_hofstadter` | Hofstadter butterfly: band counting, spectral topology | No |
//! | `validate_barracuda_evolution` | CPU foundation: all domains, evolution evidence | No |
//! | `validate_gpu_spmv` | GPU CSR SpMV: CPU/GPU parity for spectral theory | GPU |
//! | `validate_gpu_lanczos` | GPU Lanczos eigensolve: GPU SpMV inner loop | GPU |
//! | `validate_gpu_dirac` | GPU staggered Dirac: SU(3) × color on GPU (Papers 9-12) | GPU |
//! | `validate_gpu_cg` | GPU CG solver: D†D x = b on GPU (Papers 9-12 complete) | GPU |
//! | `validate_pure_gpu_qcd` | Pure GPU workload: HMC + CG on thermalized configs | GPU |

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
