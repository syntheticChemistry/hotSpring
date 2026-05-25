// SPDX-License-Identifier: AGPL-3.0-or-later

//! CLI for cazyme-fel validation module.
//! Matches the interface expected by lithoSpore's module registry.

use std::path::PathBuf;
use std::process;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.get(1).map(|a| a == "--help" || a == "-h").unwrap_or(false) {
        println!("cazyme-fel — Tier 2 FEL validation (Rust implementation)");
        println!();
        println!("Reconstructs Free Energy Landscapes from HILLS files and validates");
        println!("against reference outputs (plumed sum_hills). Reports RMSD parity.");
        println!();
        println!("USAGE:");
        println!("  cazyme-fel <HILLS> [OPTIONS]");
        println!();
        println!("OPTIONS:");
        println!("  --reference <fes.dat>  Reference FES file to validate against");
        println!("  --nbins N              Grid bins (default: 110)");
        println!("  --json                 Output results as JSON");
        println!("  --2d                   2D FES mode (expects 2-CV HILLS)");
        println!("  --periodic-y           Treat Y axis as periodic");
        println!("  --grid-min <x,y>       Grid minimum bounds");
        println!("  --grid-max <x,y>       Grid maximum bounds");
        println!("  --help, -h             Show this help");
        println!();
        println!("EXAMPLES:");
        println!("  cazyme-fel data/HILLS --reference outputs/fes_theta.dat --json");
        println!("  cazyme-fel data/HILLS_2d --2d --grid-min -0.12,-0.12 --grid-max 0.12,0.12");
        process::exit(0);
    }

    let hills_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("Usage: cazyme-fel <HILLS> [--reference <fes.dat>] [--nbins N] [--json] [--2d]");
        eprintln!("Try 'cazyme-fel --help' for more information.");
        process::exit(1);
    });

    let mut reference_path: Option<PathBuf> = None;
    let mut nbins = 110;
    let mut json_output = false;
    let mut mode_2d = false;
    let mut periodic_y = false;
    let mut grid_min_x: Option<f64> = None;
    let mut grid_max_x: Option<f64> = None;
    let mut grid_min_y: Option<f64> = None;
    let mut grid_max_y: Option<f64> = None;

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
            "--2d" => {
                mode_2d = true;
                i += 1;
            }
            "--periodic-y" => {
                periodic_y = true;
                i += 1;
            }
            "--grid-min" => {
                if let Some(s) = args.get(i + 1) {
                    let parts: Vec<&str> = s.split(',').collect();
                    grid_min_x = parts.first().and_then(|p| p.parse().ok());
                    grid_min_y = parts.get(1).and_then(|p| p.parse().ok());
                }
                i += 2;
            }
            "--grid-max" => {
                if let Some(s) = args.get(i + 1) {
                    let parts: Vec<&str> = s.split(',').collect();
                    grid_max_x = parts.first().and_then(|p| p.parse().ok());
                    grid_max_y = parts.get(1).and_then(|p| p.parse().ok());
                }
                i += 2;
            }
            _ => { i += 1; }
        }
    }

    if mode_2d {
        let bounds = match (grid_min_x, grid_max_x, grid_min_y, grid_max_y) {
            (Some(mnx), Some(mxx), Some(mny), Some(mxy)) => Some((mnx, mxx, mny, mxy)),
            _ => None,
        };
        run_2d(&hills_path, reference_path.as_deref(), nbins, periodic_y, json_output, bounds);
    } else {
        run_1d(&hills_path, reference_path.as_deref(), nbins, json_output);
    }
}

fn run_1d(hills_path: &PathBuf, reference_path: Option<&std::path::Path>, nbins: usize, json_output: bool) {
    let result = cazyme_fel::run_validation(hills_path, reference_path, nbins);

    match result {
        Ok(validation) => {
            if json_output {
                println!("{}", serde_json::to_string_pretty(&validation).unwrap());
            } else {
                println!("CAZyme FEL Validation (Tier 2 Rust) — 1D");
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

fn run_2d(hills_path: &PathBuf, reference_path: Option<&std::path::Path>, nbins: usize, periodic_y: bool, json_output: bool, bounds: Option<(f64, f64, f64, f64)>) {
    let result = if let Some((mnx, mxx, mny, mxy)) = bounds {
        cazyme_fel::run_validation_2d_with_bounds(hills_path, reference_path, nbins, nbins, periodic_y, mnx, mxx, mny, mxy)
    } else {
        cazyme_fel::run_validation_2d(hills_path, reference_path, nbins, nbins, periodic_y)
    };

    match result {
        Ok((fes, parity)) => {
            if json_output {
                let output = serde_json::json!({
                    "mode": "2D",
                    "nbins_x": fes.nbins_x,
                    "nbins_y": fes.nbins_y,
                    "grid_x_range": [fes.grid_x.first(), fes.grid_x.last()],
                    "grid_y_range": [fes.grid_y.first(), fes.grid_y.last()],
                    "energy_range": [
                        fes.free_energy.iter().flat_map(|r| r.iter()).cloned().fold(f64::INFINITY, f64::min),
                        fes.free_energy.iter().flat_map(|r| r.iter()).cloned().fold(f64::NEG_INFINITY, f64::max),
                    ],
                    "parity": parity,
                });
                println!("{}", serde_json::to_string_pretty(&output).unwrap());
            } else {
                println!("CAZyme FEL Validation (Tier 2 Rust) — 2D");
                println!("{}", "=".repeat(50));
                println!("  Grid: {}x{}", fes.nbins_x, fes.nbins_y);
                println!("  θ range: [{:.3}, {:.3}] rad", fes.grid_x.first().unwrap(), fes.grid_x.last().unwrap());
                println!("  φ range: [{:.3}, {:.3}] rad", fes.grid_y.first().unwrap(), fes.grid_y.last().unwrap());
                let e_min = fes.free_energy.iter().flat_map(|r| r.iter()).cloned().fold(f64::INFINITY, f64::min);
                let e_max = fes.free_energy.iter().flat_map(|r| r.iter()).cloned().fold(f64::NEG_INFINITY, f64::max);
                println!("  Energy: [{:.2}, {:.2}] kJ/mol", e_min, e_max);
                println!("  Periodic φ: {}", periodic_y);

                if let Some(ref p) = parity {
                    println!("\n  Parity: {} (max dev {:.4} kJ/mol)", p.status, p.max_deviation_kjmol);
                }
            }

            let exit_code = match &parity {
                Some(p) if p.status == "MATCH" => 0,
                Some(p) if p.status == "DIVERGENCE" => 1,
                _ => 0,
            };
            process::exit(exit_code);
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            process::exit(2);
        }
    }
}
