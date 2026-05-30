// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gaussian hill deposition and bias evaluation for well-tempered metadynamics.
//!
//! Implements the simulation-side metadynamics engine:
//! - Hill deposition with well-tempering factor
//! - Bias potential V(s) evaluation
//! - Bias force -dV/ds computation
//!
//! Compatible with PLUMED's METAD/OPES output format for parity validation.

use std::f64::consts::PI;

/// A single Gaussian hill in CV space.
#[derive(Debug, Clone, Copy)]
pub struct GaussianHill {
    /// CV center(s) — up to 2D. `center[0]` = theta, `center[1]` = phi (if 2D).
    pub center: [f64; 2],
    /// Gaussian width(s) sigma.
    pub sigma: [f64; 2],
    /// Height (kJ/mol), already well-tempered if applicable.
    pub height: f64,
    /// Dimensionality (1 or 2).
    pub ndim: usize,
}

/// Well-tempered metadynamics bias engine.
pub struct MetadynamicsBias {
    hills: Vec<GaussianHill>,
    /// Well-tempering bias factor gamma. gamma=inf is standard metadynamics.
    biasfactor: f64,
    /// Initial Gaussian height W0 (kJ/mol).
    initial_height: f64,
    /// Deposition stride (MD steps between hills).
    pace: usize,
    /// Dimensionality of CV space.
    ndim: usize,
    /// Whether CVs are periodic (e.g., dihedral angles).
    periodic: [bool; 2],
    /// Period for periodic CVs (typically 2pi).
    period: [f64; 2],
}

impl MetadynamicsBias {
    /// Create a new 1D metadynamics bias engine (e.g., for theta-only FEL).
    pub fn new_1d(initial_height: f64, biasfactor: f64, pace: usize) -> Self {
        Self {
            hills: Vec::new(),
            biasfactor,
            initial_height,
            pace,
            ndim: 1,
            periodic: [false, false],
            period: [0.0, 0.0],
        }
    }

    /// Create a new 2D metadynamics bias (theta, phi).
    pub fn new_2d(initial_height: f64, biasfactor: f64, pace: usize) -> Self {
        Self {
            hills: Vec::new(),
            biasfactor,
            initial_height,
            pace,
            ndim: 2,
            periodic: [false, true],
            period: [0.0, 2.0 * PI],
        }
    }

    /// Set periodicity for a CV dimension.
    pub fn set_periodic(&mut self, dim: usize, periodic: bool, period: f64) {
        if dim < 2 {
            self.periodic[dim] = periodic;
            self.period[dim] = period;
        }
    }

    /// Number of deposited hills.
    pub fn n_hills(&self) -> usize {
        self.hills.len()
    }

    /// Deposition pace.
    pub fn pace(&self) -> usize {
        self.pace
    }

    /// Deposit a new Gaussian hill at the current CV value.
    ///
    /// In well-tempered metadynamics, the height is rescaled:
    ///   w(t) = W0 * exp(-V(s,t) / (kBT * (gamma - 1)))
    ///
    /// For simplicity, we accept the pre-computed height directly
    /// (matching PLUMED's HILLS file format).
    pub fn deposit(&mut self, center: &[f64], sigma: &[f64], height: f64) {
        let mut c = [0.0f64; 2];
        let mut s = [0.0f64; 2];
        for i in 0..self.ndim.min(2) {
            c[i] = center[i];
            s[i] = sigma[i];
        }
        self.hills.push(GaussianHill {
            center: c,
            sigma: s,
            height,
            ndim: self.ndim,
        });
    }

    /// Deposit with well-tempering: automatically compute rescaled height.
    pub fn deposit_well_tempered(&mut self, center: &[f64], sigma: &[f64], kbt: f64) {
        let v_current = self.evaluate(center);
        let rescale = (-v_current / (kbt * (self.biasfactor - 1.0))).exp();
        let height = self.initial_height * rescale;
        self.deposit(center, sigma, height);
    }

    /// Evaluate the total bias potential V(s) at a given CV point.
    pub fn evaluate(&self, cv: &[f64]) -> f64 {
        let mut v = 0.0f64;
        for hill in &self.hills {
            v += self.evaluate_single_hill(hill, cv);
        }
        v
    }

    /// Evaluate the bias force -dV/ds (gradient of bias potential).
    ///
    /// Returns a vector of length `ndim`.
    pub fn bias_force(&self, cv: &[f64]) -> Vec<f64> {
        let mut force = vec![0.0f64; self.ndim];
        for hill in &self.hills {
            let (_, grad) = self.evaluate_single_hill_with_grad(hill, cv);
            for d in 0..self.ndim {
                force[d] -= grad[d];
            }
        }
        force
    }

    fn evaluate_single_hill(&self, hill: &GaussianHill, cv: &[f64]) -> f64 {
        let mut exponent = 0.0f64;
        for d in 0..self.ndim {
            let ds = self.periodic_distance(d, cv[d], hill.center[d]);
            exponent += ds * ds / (2.0 * hill.sigma[d] * hill.sigma[d]);
        }
        hill.height * (-exponent).exp()
    }

    fn evaluate_single_hill_with_grad(
        &self,
        hill: &GaussianHill,
        cv: &[f64],
    ) -> (f64, [f64; 2]) {
        let mut exponent = 0.0f64;
        let mut ds_vals = [0.0f64; 2];
        for d in 0..self.ndim {
            let ds = self.periodic_distance(d, cv[d], hill.center[d]);
            ds_vals[d] = ds;
            exponent += ds * ds / (2.0 * hill.sigma[d] * hill.sigma[d]);
        }
        let gauss = hill.height * (-exponent).exp();

        // dV/ds_d = -gauss * ds_d / sigma_d²
        let mut grad = [0.0f64; 2];
        for d in 0..self.ndim {
            grad[d] = -gauss * ds_vals[d] / (hill.sigma[d] * hill.sigma[d]);
        }
        (gauss, grad)
    }

    fn periodic_distance(&self, dim: usize, a: f64, b: f64) -> f64 {
        let mut d = a - b;
        if self.periodic[dim] {
            let p = self.period[dim];
            while d > p / 2.0 {
                d -= p;
            }
            while d < -p / 2.0 {
                d += p;
            }
        }
        d
    }

    /// Load hills from a PLUMED HILLS file content (1D).
    ///
    /// Format: `time cv sigma height biasfactor`
    pub fn load_hills_1d(content: &str) -> Option<Self> {
        let mut hills = Vec::new();
        let mut biasfactor = 10.0f64;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('#') || trimmed.starts_with("@") || trimmed.is_empty() {
                if trimmed.contains("biasfactor") {
                    if let Some(val) = extract_field_value(trimmed, "biasfactor") {
                        biasfactor = val;
                    }
                }
                continue;
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 4 {
                continue;
            }
            let cv: f64 = parts[1].parse().ok()?;
            let sigma: f64 = parts[2].parse().ok()?;
            let height: f64 = parts[3].parse().ok()?;

            hills.push(GaussianHill {
                center: [cv, 0.0],
                sigma: [sigma, 0.0],
                height,
                ndim: 1,
            });
        }

        if hills.is_empty() {
            return None;
        }

        let initial_height = hills[0].height;
        Some(Self {
            hills,
            biasfactor,
            initial_height,
            pace: 500,
            ndim: 1,
            periodic: [false, false],
            period: [0.0, 0.0],
        })
    }

    /// Reconstruct the free energy surface F(s) on a grid.
    ///
    /// F(s) = -(gamma/(gamma-1)) * V(s) + C
    ///
    /// Returns (grid_points, free_energy) shifted so minimum is 0.
    pub fn reconstruct_fes(&self, min: f64, max: f64, nbins: usize) -> (Vec<f64>, Vec<f64>) {
        let step = (max - min) / nbins as f64;
        let mut grid = Vec::with_capacity(nbins);
        let mut fes = Vec::with_capacity(nbins);

        let scale = if self.biasfactor > 1.0 {
            -(self.biasfactor) / (self.biasfactor - 1.0)
        } else {
            -1.0
        };

        for i in 0..nbins {
            let s = min + (i as f64 + 0.5) * step;
            grid.push(s);
            fes.push(scale * self.evaluate(&[s]));
        }

        let min_fes = fes.iter().copied().fold(f64::INFINITY, f64::min);
        for f in &mut fes {
            *f -= min_fes;
        }

        (grid, fes)
    }

    /// Access deposited hills (for serialization/inspection).
    pub fn hills(&self) -> &[GaussianHill] {
        &self.hills
    }
}

fn extract_field_value(line: &str, field: &str) -> Option<f64> {
    let idx = line.find(field)?;
    let rest = &line[idx + field.len()..];
    let rest = rest.trim_start_matches(|c: char| c == '=' || c.is_whitespace());
    rest.split_whitespace().next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_hill_evaluation() {
        let mut bias = MetadynamicsBias::new_1d(1.0, 10.0, 500);
        bias.deposit(&[0.5], &[0.1], 1.0);

        let v_at_center = bias.evaluate(&[0.5]);
        assert!(
            (v_at_center - 1.0).abs() < 1e-10,
            "V at hill center should be height, got {}",
            v_at_center
        );

        let v_at_sigma = bias.evaluate(&[0.6]);
        let expected = (-0.5f64).exp();
        assert!(
            (v_at_sigma - expected).abs() < 1e-10,
            "V at 1 sigma should be exp(-0.5), got {}",
            v_at_sigma
        );
    }

    #[test]
    fn test_bias_force_direction() {
        let mut bias = MetadynamicsBias::new_1d(1.0, 10.0, 500);
        bias.deposit(&[0.5], &[0.1], 1.0);

        let force_left = bias.bias_force(&[0.4]);
        let force_right = bias.bias_force(&[0.6]);

        assert!(force_left[0] < 0.0, "force left of hill should push further left");
        assert!(force_right[0] > 0.0, "force right of hill should push further right");

        let force_center = bias.bias_force(&[0.5]);
        assert!(
            force_center[0].abs() < 1e-10,
            "force at hill center should be ~0"
        );
    }

    #[test]
    fn test_multiple_hills_superposition() {
        let mut bias = MetadynamicsBias::new_1d(1.0, 10.0, 500);
        bias.deposit(&[0.5], &[0.1], 1.0);
        bias.deposit(&[0.7], &[0.1], 1.0);

        let v_sum = bias.evaluate(&[0.5]);
        assert!(v_sum > 1.0, "V should be > single hill at 0.5 due to overlap");
    }

    #[test]
    fn test_periodic_distance() {
        let mut bias = MetadynamicsBias::new_1d(1.0, 10.0, 500);
        bias.set_periodic(0, true, 2.0 * PI);

        let d1 = bias.periodic_distance(0, 0.1, 2.0 * PI - 0.1);
        assert!(
            (d1 - 0.2).abs() < 1e-10,
            "periodic distance should wrap, got {}",
            d1
        );
    }

    #[test]
    fn test_well_tempered_height_decay() {
        let mut bias = MetadynamicsBias::new_1d(1.0, 10.0, 500);
        let kbt = 2.494; // 300K in kJ/mol

        bias.deposit_well_tempered(&[0.5], &[0.1], kbt);
        let h1 = bias.hills().last().unwrap().height;
        assert!((h1 - 1.0).abs() < 1e-10, "first hill should have full height");

        bias.deposit_well_tempered(&[0.5], &[0.1], kbt);
        let h2 = bias.hills().last().unwrap().height;
        assert!(
            h2 < h1,
            "second hill at same position should be shorter: {} vs {}",
            h2,
            h1
        );
    }

    #[test]
    fn test_fes_reconstruction() {
        let mut bias = MetadynamicsBias::new_1d(1.0, 10.0, 500);
        bias.deposit(&[0.5], &[0.1], 1.0);

        let (grid, fes) = bias.reconstruct_fes(0.0, 1.0, 100);
        assert_eq!(grid.len(), 100);
        assert_eq!(fes.len(), 100);

        let min_fes = fes.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(min_fes.abs() < 1e-10, "FES minimum should be shifted to 0");
    }

    #[test]
    fn test_2d_bias() {
        let mut bias = MetadynamicsBias::new_2d(1.0, 10.0, 500);
        bias.deposit(&[0.5, 1.0], &[0.1, 0.15], 1.0);

        let v = bias.evaluate(&[0.5, 1.0]);
        assert!((v - 1.0).abs() < 1e-10);

        let force = bias.bias_force(&[0.5, 1.0]);
        assert_eq!(force.len(), 2);
        assert!(force[0].abs() < 1e-10);
        assert!(force[1].abs() < 1e-10);
    }
}
