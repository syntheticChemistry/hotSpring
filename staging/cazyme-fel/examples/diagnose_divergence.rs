use cazyme_fel::{parse_hills, reconstruct_fes, parse_fes, check_parity, FesResult};
use std::path::Path;

fn main() {
    let hills_path = Path::new("../../control/gromacs_fel/guidestone_refresh/enzyme_bound_1d/HILLS");
    let plumed_fes_path = Path::new("../../control/gromacs_fel/guidestone_refresh/enzyme_bound_1d/fes_theta.dat");

    println!("=== Parsing HILLS ===");
    let hills = parse_hills(hills_path).unwrap();
    println!("  Gaussians: {}", hills.n_gaussians);
    println!("  Biasfactor: {}", hills.biasfactor);
    println!("  Center range: {:.4} to {:.4}",
        hills.centers.iter().cloned().fold(f64::INFINITY, f64::min),
        hills.centers.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    println!("  Sigma range: {:.6} to {:.6}",
        hills.sigmas.iter().cloned().fold(f64::INFINITY, f64::min),
        hills.sigmas.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    println!("  Height range: {:.6} to {:.6}",
        hills.heights.iter().cloned().fold(f64::INFINITY, f64::min),
        hills.heights.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    println!("\n=== PLUMED reference FES ===");
    let plumed_fes = parse_fes(plumed_fes_path).unwrap();
    println!("  Grid points: {}", plumed_fes.nbins);
    println!("  Grid range: {:.6} to {:.6}", plumed_fes.grid[0], plumed_fes.grid[plumed_fes.nbins - 1]);
    let plumed_min = plumed_fes.free_energy.iter().cloned().fold(f64::INFINITY, f64::min);
    let plumed_max = plumed_fes.free_energy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("  Energy range: {:.2} to {:.2} (span: {:.2} kJ/mol)", plumed_min, plumed_max, plumed_max - plumed_min);

    println!("\n=== Rust reconstruction (same grid bounds as PLUMED) ===");
    let rust_fes = reconstruct_fes(&hills, plumed_fes.grid[0], *plumed_fes.grid.last().unwrap(), plumed_fes.nbins);
    let rust_min = rust_fes.free_energy.iter().cloned().fold(f64::INFINITY, f64::min);
    let rust_max = rust_fes.free_energy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("  Energy range: {:.2} to {:.2} (span: {:.2} kJ/mol)", rust_min, rust_max, rust_max - rust_min);

    println!("\n=== Parity check (Rust vs PLUMED) ===");
    let parity = check_parity(&rust_fes, &plumed_fes, 1.0);
    println!("  RMSD: {:.4} kJ/mol", parity.rmsd_kjmol);
    println!("  Max deviation: {:.4} kJ/mol", parity.max_deviation_kjmol);
    println!("  Status: {}", parity.status);

    println!("\n=== Basin comparison (after min-zero shift) ===");
    println!("  {:>10} {:>12} {:>12} {:>10}", "theta", "PLUMED", "Rust", "Diff");
    let targets = [0.15, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    for &t in &targets {
        let p_idx = plumed_fes.grid.iter().position(|&x| (x - t).abs() < 0.05).unwrap_or(0);
        let r_idx = rust_fes.grid.iter().position(|&x| (x - t).abs() < 0.05).unwrap_or(0);
        let p_val = plumed_fes.free_energy[p_idx] - plumed_min;
        let r_val = rust_fes.free_energy[r_idx] - rust_min;
        println!("  {:>10.3} {:>12.2} {:>12.2} {:>10.2}", t, p_val, r_val, r_val - p_val);
    }

    // PLUMED well-tempered metadynamics FES recovery formula:
    //   F(s) = -(T + DeltaT)/DeltaT * V(s,t->inf)
    //        = -(gamma/(gamma-1)) * V(s,t->inf)
    // where gamma = biasfactor = (T + DeltaT)/T
    //
    // Simple kernel sum gives V(s) = sum_i h_i * exp(-(s-s_i)^2 / (2*sigma^2))
    // Our Rust code does: F(s) = -V(s)  [MISSING the gamma/(gamma-1) scaling!]
    //
    // This is analogous to the Kinoshita-Lee-Nauenberg cancellation in QCD:
    // a naive summation misses a renormalization factor that only matters
    // when comparing between systems with different sampling histories.
    let gamma = hills.biasfactor;
    let correction = gamma / (gamma - 1.0);
    println!("\n=== DIAGNOSIS ===");
    println!("  Biasfactor (gamma): {}", gamma);
    println!("  Well-tempered correction: gamma/(gamma-1) = {:.6}", correction);
    println!("  PLUMED applies: F(s) = -{:.4} * V(s,t->inf)", correction);
    println!("  Our Rust code:  F(s) = -1.0 * V(s)");
    println!("  Our barriers are {:.1}% too small ({:.4}x)", (1.0 - 1.0/correction) * 100.0, 1.0/correction);

    // Apply correction and recheck
    println!("\n=== With well-tempered correction applied ===");
    let mut corrected_energy: Vec<f64> = rust_fes.free_energy.iter()
        .map(|&e| e * correction)
        .collect();
    let corr_min = corrected_energy.iter().cloned().fold(f64::INFINITY, f64::min);
    for e in &mut corrected_energy {
        *e -= corr_min;
    }

    let corrected_fes = FesResult {
        grid: rust_fes.grid.clone(),
        free_energy: corrected_energy,
        nbins: rust_fes.nbins,
    };
    let parity2 = check_parity(&corrected_fes, &plumed_fes, 1.0);
    println!("  RMSD: {:.4} kJ/mol", parity2.rmsd_kjmol);
    println!("  Max deviation: {:.4} kJ/mol", parity2.max_deviation_kjmol);
    println!("  Status: {}", parity2.status);

    println!("\n=== Basin comparison (corrected) ===");
    println!("  {:>10} {:>12} {:>12} {:>10}", "theta", "PLUMED", "Corrected", "Diff");
    for &t in &targets {
        let p_idx = plumed_fes.grid.iter().position(|&x| (x - t).abs() < 0.05).unwrap_or(0);
        let r_idx = corrected_fes.grid.iter().position(|&x| (x - t).abs() < 0.05).unwrap_or(0);
        let p_val = plumed_fes.free_energy[p_idx] - plumed_min;
        let r_val = corrected_fes.free_energy[r_idx];
        println!("  {:>10.3} {:>12.2} {:>12.2} {:>10.2}", t, p_val, r_val, r_val - p_val);
    }
}
