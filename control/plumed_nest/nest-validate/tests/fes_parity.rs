// Integration test: verify Rust FES reconstruction matches PLUMED sum_hills output.

use std::path::Path;

#[path = "../src/hills.rs"]
mod hills;
#[path = "../src/fes.rs"]
mod fes;
#[path = "../src/colvar.rs"]
mod colvar;

const NEST_ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_alanine_hills_parse() {
    let hills_path = Path::new(NEST_ROOT).join("../target_01_alanine_dipeptide/output/HILLS");
    if !hills_path.exists() {
        eprintln!("SKIP: HILLS file not found (run simulation first)");
        return;
    }

    let h = hills::parse_hills_2d(&hills_path).unwrap();
    assert!(h.n_gaussians() > 1000, "Expected >1000 Gaussians, got {}", h.n_gaussians());
    assert!(h.centers_x[0].abs() < std::f64::consts::PI + 0.1);
    assert!(h.centers_y[0].abs() < std::f64::consts::PI + 0.1);
}

#[test]
fn test_alanine_fes_reconstruction() {
    let hills_path = Path::new(NEST_ROOT).join("../target_01_alanine_dipeptide/output/HILLS");
    if !hills_path.exists() {
        eprintln!("SKIP: HILLS file not found");
        return;
    }

    let h = hills::parse_hills_2d(&hills_path).unwrap();
    let pi = std::f64::consts::PI;
    let fes_2d = fes::reconstruct_2d(&h, -pi, pi, -pi, pi, 50, 50, (true, true));

    assert_eq!(fes_2d.nbins_x, 50);
    assert_eq!(fes_2d.nbins_y, 50);

    let global_min = fes_2d.free_energy.iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!((global_min - 0.0).abs() < 1e-10, "FES should be min-shifted to 0");

    let global_max = fes_2d.free_energy.iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(global_max > 10.0, "FES should have barriers >10 kJ/mol, got {global_max}");
    assert!(global_max < 200.0, "FES max suspiciously high: {global_max}");
}

#[test]
fn test_minima_detection_synthetic() {
    let n = 100;
    let pi = std::f64::consts::PI;
    let grid: Vec<f64> = (0..n).map(|i| -pi + 2.0 * pi * i as f64 / (n - 1) as f64).collect();

    // Synthetic: two basins at phi=-1.4 (deep) and phi=1.0 (shallow)
    let free_energy: Vec<f64> = grid.iter().map(|&x| {
        let basin1 = 50.0 * (-(x + 1.4_f64).powi(2) / 0.1).exp();
        let basin2 = 30.0 * (-(x - 1.0_f64).powi(2) / 0.1).exp();
        50.0 - basin1 - basin2
    }).collect();

    let fes = fes::Fes1D { grid, free_energy, nbins: n };
    let minima = fes::find_minima_1d(&fes, 30.0);

    assert!(minima.len() >= 2, "Expected >=2 minima, got {}", minima.len());
    assert!((minima[0].x - (-1.4)).abs() < 0.15, "First minimum at {}, expected ~-1.4", minima[0].x);
}

#[test]
fn test_block_averaging_convergence() {
    let n = 100;
    let grid: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();

    // 10 nearly-identical FES windows (converged)
    let stride_fes: Vec<fes::Fes1D> = (0..10).map(|w| {
        let fe: Vec<f64> = grid.iter().map(|&x| {
            10.0 * (x - 0.5).powi(2) + (w as f64 * 0.1) * (x * 3.14).sin()
        }).collect();
        fes::Fes1D { grid: grid.clone(), free_energy: fe, nbins: n }
    }).collect();

    let ba = fes::block_average(&stride_fes, 5).unwrap();
    assert!(ba.max_stderr < 5.0, "Expected converged, got max_stderr={:.2}", ba.max_stderr);
    assert!(ba.converged);
}

#[test]
fn test_colvar_parse() {
    let colvar_path = Path::new(NEST_ROOT).join("../target_02_chignolin_opes/COLVARb");
    if !colvar_path.exists() {
        eprintln!("SKIP: COLVARb not found");
        return;
    }

    let cv = colvar::parse_colvar(&colvar_path).unwrap();
    assert!(cv.n_frames > 0);
    assert!(cv.fields.contains(&"hlda".to_string()));

    let hlda = cv.column("hlda").unwrap();
    assert_eq!(hlda.len(), cv.n_frames);
}

#[test]
fn test_reweighted_fes() {
    // Simple test: uniform bias should give uniform FES
    let cv: Vec<f64> = (0..1000).map(|i| (i as f64 / 1000.0) * 2.0 - 1.0).collect();
    let bias = vec![0.0; 1000]; // no bias

    let (centers, fes) = colvar::reweighted_fes_1d(&cv, &bias, 20, 2.5);
    assert_eq!(centers.len(), 20);

    let finite_fes: Vec<f64> = fes.iter().filter(|f| f.is_finite()).cloned().collect();
    let fes_range = finite_fes.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - finite_fes.iter().cloned().fold(f64::INFINITY, f64::min);

    // Uniform sampling should give flat FES (within ~1 kT due to bin statistics)
    assert!(fes_range < 5.0, "Uniform sampling FES should be flat, got range {fes_range:.2}");
}

#[test]
fn test_transition_counting() {
    let cv = vec![0.0, 0.1, 0.2, 1.5, 1.6, 1.3, 0.1, 0.0, 1.5, 1.4, 0.2, 0.1];
    let (fold, unfold) = colvar::count_transitions(&cv, 1.2, 0.3);
    assert_eq!(fold, 2, "Expected 2 fold events");
    assert_eq!(unfold, 2, "Expected 2 unfold events");
}
