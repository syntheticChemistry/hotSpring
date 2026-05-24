// SPDX-License-Identifier: AGPL-3.0-or-later

//! CLI for cazyme-fel validation module.
//! Matches the interface expected by lithoSpore's module registry.

use std::path::PathBuf;
use std::process;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let hills_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("Usage: cazyme-fel <HILLS> [--reference <fes.dat>] [--nbins N] [--json]");
        process::exit(1);
    });

    let mut reference_path: Option<PathBuf> = None;
    let mut nbins = 110;
    let mut json_output = false;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--reference" => {
                reference_path = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--nbins" => {
                nbins = args.get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(110);
                i += 2;
            }
            "--json" => {
                json_output = true;
                i += 1;
            }
            _ => { i += 1; }
        }
    }

    let result = cazyme_fel::run_validation(
        &hills_path,
        reference_path.as_deref(),
        nbins,
    );

    match result {
        Ok(validation) => {
            if json_output {
                println!("{}", serde_json::to_string_pretty(&validation).unwrap());
            } else {
                println!("CAZyme FEL Validation (Tier 2 Rust)");
                println!("{}", "=".repeat(50));
                println!("  Basins: {}", validation.basins.len());
                for b in &validation.basins {
                    println!("    {:20} θ={:6.1}°  E={:.2} kJ/mol", b.label, b.theta_deg, b.energy_kjmol);
                }
                println!("  Barriers: [{:.1}, {:.1}] kJ/mol",
                    validation.barrier_range_kjmol[0],
                    validation.barrier_range_kjmol[1]);
                println!("  Chairs: {}, Boat: {}",
                    validation.chair_basins_found,
                    validation.boat_basin_found);

                if let Some(ref p) = validation.parity {
                    println!("\n  Parity: {} (max dev {:.4} kJ/mol)",
                        p.status, p.max_deviation_kjmol);
                }
            }

            let exit_code = match &validation.parity {
                Some(p) if p.status == "MATCH" => 0,
                Some(_) => 1,
                None => 0,
            };
            process::exit(exit_code);
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            process::exit(2);
        }
    }
}
