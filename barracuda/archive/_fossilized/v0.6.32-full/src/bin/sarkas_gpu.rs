// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sarkas GPU — Yukawa OCP Molecular Dynamics on Consumer GPU
//!
//! Full GPU MD simulation matching Sarkas PP Yukawa DSF study:
//!   9 cases: κ=1,2,3 × Γ=weak,medium,strong
//!   Velocity-Verlet, PBC, Berendsen thermostat
//!   All physics in f64 WGSL (`SHADER_F64`)
//!
//! Validates against Python Sarkas baseline:
//!   - Energy conservation (drift < 5%)
//!   - RDF: peak position, g(r)→1 tail
//!   - VACF: diffusion coefficient
//!   - SSF: S(k→0) compressibility
//!
//! Run:
//!   cargo run --release --bin `sarkas_gpu`              # quick validation (N=500)
//!   cargo run --release --bin `sarkas_gpu` -- --full    # full 9-case sweep (N=2000)
//!   cargo run --release --bin `sarkas_gpu` -- --long    # long run: 9 cases, 80k steps (~71 min)
//!   cargo run --release --bin `sarkas_gpu` -- --paper   # PAPER PARITY: 9 cases, N=10000, 80k steps
//!   cargo run --release --bin `sarkas_gpu` -- --paper-ext # Extended paper: N=10000, 100k steps
//!   cargo run --release --bin `sarkas_gpu` -- --nscale  # N-scaling: 500→20000, GPU-only (~2-3h)
//!   cargo run --release --bin `sarkas_gpu` -- --scale   # quick scaling test (N=500,2000, GPU+CPU)
//!   cargo run --release --bin `sarkas_gpu` -- --brain-sweep  # persistent brain across 9 cases
//!   cargo run --release --bin `sarkas_gpu` -- --brain-nscale # persistent brain across N sizes
//!   cargo run --release --bin `sarkas_gpu` -- --brain-skin   # skin sweep to teach optimal skin

use hotspring_barracuda::bench::{BenchReport, HardwareInventory};
use hotspring_barracuda::bin_helpers::sarkas_gpu::{
    run_brain_nscale, run_brain_skin_sweep, run_brain_sweep, run_full_sweep, run_long_sweep,
    run_n_scaling, run_paper_parity, run_quick_validation, run_scaling_test,
};
use hotspring_barracuda::validation::ValidationHarness;

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Sarkas GPU — Yukawa OCP Molecular Dynamics                ║");
    println!("║  f64 WGSL on Consumer GPU (SHADER_F64)                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("sarkas_gpu");
    let args: Vec<String> = std::env::args().collect();
    let full_sweep = args.iter().any(|a| a == "--full");
    let long_run = args.iter().any(|a| a == "--long");
    let paper_parity = args.iter().any(|a| a == "--paper");
    let paper_ext = args.iter().any(|a| a == "--paper-ext");
    let scale_test = args.iter().any(|a| a == "--scale");
    let nscale = args.iter().any(|a| a == "--nscale");
    let brain_sweep = args.iter().any(|a| a == "--brain-sweep");
    let brain_nscale = args.iter().any(|a| a == "--brain-nscale");
    let brain_skin = args.iter().any(|a| a == "--brain-skin");

    let hw = HardwareInventory::detect_local();
    println!("  Hardware: {} / {}", hw.cpu_model, hw.gpu_name);
    println!();

    let mut report = BenchReport::new(hw);

    if brain_sweep {
        run_brain_sweep(&mut report, &mut harness).await;
    } else if brain_nscale {
        run_brain_nscale(&mut report, &mut harness).await;
    } else if brain_skin {
        run_brain_skin_sweep(&mut report, &mut harness).await;
    } else if nscale {
        run_n_scaling(&mut report, &mut harness).await;
    } else if paper_parity {
        run_paper_parity(&mut report, &mut harness, false).await;
    } else if paper_ext {
        run_paper_parity(&mut report, &mut harness, true).await;
    } else if scale_test {
        run_scaling_test(&mut report, &mut harness).await;
    } else if long_run {
        run_long_sweep(&mut report, &mut harness).await;
    } else if full_sweep {
        run_full_sweep(&mut report, &mut harness).await;
    } else {
        run_quick_validation(&mut report, &mut harness).await;
    }

    report.save_and_print();
    println!();
    report.print_summary();
    harness.finish();
}
