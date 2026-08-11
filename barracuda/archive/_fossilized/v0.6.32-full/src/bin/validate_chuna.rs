// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unified Chuna paper validation — Papers 43, 44, 45.

use hotspring_barracuda::bin_helpers::validate_chuna::{
    gpu_substrate_validation, paper_43_gradient_flow, paper_44_dielectric, paper_45_kinetic_fluid,
};
use hotspring_barracuda::lattice::measurement::RunManifest;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

fn main() {
    let mut harness = ValidationHarness::new("validate_chuna");
    harness.run_manifest = Some(RunManifest::capture("validate_chuna"));

    let args: Vec<String> = std::env::args().collect();
    let output_dir = args.windows(2).find_map(|pair| {
        if pair[0] == "--output" {
            Some(pair[1].clone())
        } else {
            pair[0]
                .strip_prefix("--output=")
                .map(std::string::ToString::to_string)
        }
    });

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Chuna Validation Suite — Papers 43, 44, 45                    ║");
    println!("║  guideStone: CPU baseline + GPU parity on every substrate      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!(
        "substrate: {} {}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!("engine:    cpu-native (pure Rust, GPU auto-detected)\n");

    let wall_start = Instant::now();

    let cpu_ref = paper_43_gradient_flow(&mut harness);
    paper_44_dielectric(&mut harness);
    paper_45_kinetic_fluid(&mut harness);

    let cpu_ms = wall_start.elapsed().as_millis() as u64;
    println!(
        "  CPU baseline: {}/{} checks in {:.1}s\n",
        harness.passed_count(),
        harness.total_count(),
        cpu_ms as f64 / 1000.0
    );

    gpu_substrate_validation(&mut harness, &cpu_ref);

    let total_ms = wall_start.elapsed().as_millis() as u64;

    if let Some(ref dir) = output_dir {
        harness.write_json(dir, total_ms);
    }

    harness.finish();
}
