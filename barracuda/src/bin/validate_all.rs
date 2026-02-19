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
//! | `validate_stanton_murillo` | Transport coefficients D*, η*, λ* | No |
//! | `validate_pure_gauge` | Pure gauge SU(3) lattice QCD | No |

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
        name: "BarraCUDA HFB Pipeline",
        binary: "validate_barracuda_hfb",
        requires_gpu: true,
    },
    Suite {
        name: "BarraCUDA MD Pipeline",
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
        name: "Stanton-Murillo Transport (Paper 5)",
        binary: "validate_stanton_murillo",
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
