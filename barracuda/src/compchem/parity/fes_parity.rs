// SPDX-License-Identifier: AGPL-3.0-or-later
//! 1D FES parity: sovereign reconstruction vs GROMACS reference.
//!
//! Loads a PLUMED HILLS file and GROMACS `sum_hills --mintozero` FES output,
//! reconstructs the FES using our Gaussian kernel summation, and reports
//! RMSD, max deviation, and pass/fail verdict.

use crate::compchem::metadynamics::hills::GaussianHill;

/// FES parity checker.
pub struct FesParity;

/// Parity result between sovereign and GROMACS FES.
#[derive(Debug, Clone)]
pub struct FesParityResult {
    /// RMSD between sovereign and reference FES (kJ/mol)
    pub rmsd_kjmol: f64,
    /// Maximum absolute deviation (kJ/mol)
    pub max_deviation_kjmol: f64,
    /// Mean absolute deviation (kJ/mol)
    pub mean_deviation_kjmol: f64,
    /// Number of grid points compared
    pub n_points: usize,
    /// Sovereign FES grid values
    pub sovereign_grid: Vec<f64>,
    /// Sovereign FES free energy values
    pub sovereign_fes: Vec<f64>,
    /// Reference grid values
    pub reference_grid: Vec<f64>,
    /// Reference FES values
    pub reference_fes: Vec<f64>,
}

impl FesParityResult {
    /// Pass if RMSD < tolerance (default 2.0 kJ/mol from threshold_calibration.toml).
    pub fn is_pass(&self, tolerance: f64) -> bool {
        self.rmsd_kjmol < tolerance
    }
}

impl FesParity {
    /// Parse a PLUMED HILLS file (1D) into a vector of Gaussian hills.
    ///
    /// Format: `time cv sigma height biasfactor`
    pub fn parse_hills(content: &str) -> Vec<GaussianHill> {
        let mut hills = Vec::new();
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('@') {
                continue;
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 4 {
                continue;
            }
            let cv: f64 = match parts[1].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let sigma: f64 = match parts[2].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let height: f64 = match parts[3].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            hills.push(GaussianHill {
                center: [cv, 0.0],
                sigma: [sigma, 0.0],
                height,
                ndim: 1,
            });
        }
        hills
    }

    /// Parse a GROMACS FES reference file (2-column: cv, free_energy).
    pub fn parse_fes_reference(content: &str) -> (Vec<f64>, Vec<f64>) {
        let mut grid = Vec::new();
        let mut fes = Vec::new();
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('@') {
                continue;
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }
            if let (Ok(cv), Ok(f)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                grid.push(cv);
                fes.push(f);
            }
        }
        (grid, fes)
    }

    /// Run parity comparison between HILLS reconstruction and reference FES.
    ///
    /// Reconstructs FES on the reference grid points using Gaussian kernel
    /// summation. For well-tempered metadynamics, the heights in the HILLS
    /// file already encode the decay, so F(s) = -V(s) (no gamma factor).
    pub fn compare(
        hills: &[GaussianHill],
        _biasfactor: f64,
        ref_grid: &[f64],
        ref_fes: &[f64],
    ) -> FesParityResult {
        assert_eq!(ref_grid.len(), ref_fes.len());

        // F(s) = -V(s) for well-tempered HILLS (heights include decay)
        let mut sovereign_fes: Vec<f64> = ref_grid
            .iter()
            .map(|&s| {
                let mut v = 0.0f64;
                for h in hills {
                    let ds = s - h.center[0];
                    let exponent = ds * ds / (2.0 * h.sigma[0] * h.sigma[0]);
                    v += h.height * (-exponent).exp();
                }
                -v
            })
            .collect();

        // Shift to minimum = 0
        let min_sov = sovereign_fes.iter().copied().fold(f64::INFINITY, f64::min);
        for f in &mut sovereign_fes {
            *f -= min_sov;
        }

        let n = ref_grid.len();
        let mut sum_sq = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut max_dev = 0.0f64;

        for i in 0..n {
            let dev = (sovereign_fes[i] - ref_fes[i]).abs();
            sum_sq += dev * dev;
            sum_abs += dev;
            if dev > max_dev {
                max_dev = dev;
            }
        }

        FesParityResult {
            rmsd_kjmol: (sum_sq / n as f64).sqrt(),
            max_deviation_kjmol: max_dev,
            mean_deviation_kjmol: sum_abs / n as f64,
            n_points: n,
            sovereign_grid: ref_grid.to_vec(),
            sovereign_fes,
            reference_grid: ref_grid.to_vec(),
            reference_fes: ref_fes.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hills_format() {
        let content = r#"#! FIELDS time puck.theta sigma_puck.theta height biasf
#! SET multivariate false
      1.0      1.508      0.1      1.607      15
      2.0      1.620      0.1      1.570      15
      3.0      1.510      0.1      1.505      15
"#;
        let hills = FesParity::parse_hills(content);
        assert_eq!(hills.len(), 3);
        assert!((hills[0].center[0] - 1.508).abs() < 1e-3);
        assert!((hills[0].sigma[0] - 0.1).abs() < 1e-10);
        assert!((hills[0].height - 1.607).abs() < 1e-3);
    }

    #[test]
    fn test_parse_fes_reference() {
        let content = "-0.298467 155.241312\n-0.264188 154.675486\n";
        let (grid, fes) = FesParity::parse_fes_reference(content);
        assert_eq!(grid.len(), 2);
        assert!((grid[0] - (-0.298467)).abs() < 1e-6);
        assert!((fes[0] - 155.241312).abs() < 1e-3);
    }

    #[test]
    fn test_self_parity_perfect() {
        let hills = vec![
            GaussianHill {
                center: [0.5, 0.0],
                sigma: [0.1, 0.0],
                height: 1.0,
                ndim: 1,
            },
            GaussianHill {
                center: [0.7, 0.0],
                sigma: [0.1, 0.0],
                height: 0.8,
                ndim: 1,
            },
        ];

        let biasfactor = 10.0;

        // F(s) = -V(s) for well-tempered
        let grid: Vec<f64> = (0..50).map(|i| i as f64 * 0.02).collect();
        let ref_fes: Vec<f64> = grid
            .iter()
            .map(|&s| {
                let mut v = 0.0;
                for h in &hills {
                    let ds = s - h.center[0];
                    v += h.height * (-ds * ds / (2.0 * h.sigma[0] * h.sigma[0])).exp();
                }
                -v
            })
            .collect();

        let min_ref = ref_fes.iter().copied().fold(f64::INFINITY, f64::min);
        let ref_fes_shifted: Vec<f64> = ref_fes.iter().map(|f| f - min_ref).collect();

        let result = FesParity::compare(&hills, biasfactor, &grid, &ref_fes_shifted);
        assert!(
            result.rmsd_kjmol < 1e-10,
            "self-parity should be perfect, RMSD={}",
            result.rmsd_kjmol
        );
    }

    #[test]
    fn test_real_xylose_parity() {
        let hills_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../control/gromacs_fel/guidestone_refresh/free_xylose_1d/HILLS"
        );
        let fes_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../control/gromacs_fel/guidestone_refresh/free_xylose_1d/fes_theta.dat"
        );

        let Ok(hills_content) = std::fs::read_to_string(hills_path) else {
            eprintln!("Skipping: HILLS file not found at {hills_path}");
            return;
        };
        let Ok(fes_content) = std::fs::read_to_string(fes_path) else {
            eprintln!("Skipping: FES reference not found at {fes_path}");
            return;
        };

        let hills = FesParity::parse_hills(&hills_content);
        assert!(
            hills.len() > 1000,
            "should have >1000 hills, got {}",
            hills.len()
        );

        let (ref_grid, ref_fes) = FesParity::parse_fes_reference(&fes_content);
        assert!(ref_grid.len() > 50, "reference should have >50 points");

        let biasfactor = 15.0;
        let result = FesParity::compare(&hills, biasfactor, &ref_grid, &ref_fes);

        eprintln!(
            "Xylose 1D FES parity: RMSD={:.3} kJ/mol, max_dev={:.3} kJ/mol, n_points={}",
            result.rmsd_kjmol, result.max_deviation_kjmol, result.n_points
        );

        // Threshold from pseudoSpore derivations/threshold_calibration.toml
        assert!(
            result.rmsd_kjmol < 2.0,
            "RMSD {:.3} kJ/mol exceeds 2.0 kJ/mol threshold",
            result.rmsd_kjmol
        );
    }
}
