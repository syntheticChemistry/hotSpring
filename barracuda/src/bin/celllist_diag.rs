// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cell-List Force Diagnostic
//!
//! Compares all-pairs vs cell-list Yukawa force kernels on identical particle
//! data at multiple N values. This isolates whether the cell-list energy
//! conservation bug is in the shader physics or the sort/rebuild infrastructure.
//!
//! Experiment 002: Cell-List Force Kernel Investigation
//!
//! Usage:
//!   cargo run --release --bin `celllist_diag`

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::validation::ValidationHarness;

#[path = "../bin_helpers/celllist_diag/mod.rs"]
mod celllist_diag;

use celllist_diag::compare_forces::compare_forces;
use celllist_diag::hybrid::test_hybrid;
use celllist_diag::phase1b_integrity::run_phase1b_integrity;
use celllist_diag::phase1c_pair_count::run_phase1c_pair_count;

#[tokio::main]
async fn main() {
    let mut harness = ValidationHarness::new("celllist_diag");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Cell-List Force Diagnostic                                 ║");
    println!("║  Experiment 002: All-Pairs vs Cell-List comparison          ║");
    println!("║  Same positions, same params → forces must match            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let gpu = GpuF64::new().await.expect("Failed to init GPU");
    gpu.print_info();
    println!();

    // Test at multiple N values to see how the bug scales
    // N=108 (3³×4 FCC): tiny, 3 cells/dim → cell-list degenerates to all-pairs
    // N=500: small, 1 cell/dim (forced to 3) → effectively all-pairs
    // N=2048: medium, 3 cells/dim → first real cell-list test
    // N=4000: medium, 4 cells/dim → 4×4×4 cells
    // N=8788: large, 5 cells/dim → matches the broken N=10k threshold
    // N=10976: actual N=10k FCC → the exact failure case

    let sizes = [108, 500, 2048, 4000, 8788, 10976];

    for &n in &sizes {
        compare_forces(&gpu, n, &mut harness);
    }

    run_phase1b_integrity(&mut harness);
    run_phase1c_pair_count(&gpu, &mut harness);

    // ── Phase 2: Hybrid test ──
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Phase 2: Hybrid Isolation Test                             ║");
    println!("║  All-pairs loop + cell-list bindings → isolate bug location ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    for &n in &[500, 2048, 10976] {
        test_hybrid(&gpu, n, &mut harness);
    }

    println!("════════════════════════════════════════════════════════════════");
    println!("  Diagnostic complete. See experiment 002 journal for analysis.");
    println!("════════════════════════════════════════════════════════════════");
    harness.finish();
}
