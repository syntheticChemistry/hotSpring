// SPDX-License-Identifier: AGPL-3.0-only

//! Physics proxy pipeline — library functions for Anderson 3D and Z(3) Potts.
//!
//! These cheap physics models run concurrently on the CPU (or Titan V) while
//! the primary GPU performs dynamical HMC. Results feed the NPU's proxy heads
//! (Heads 11-13) for physics-informed CG prediction and phase classification.
//!
//! # Brain Architecture Layer 3
//!
//! The CPU cortex thread calls these functions and sends `ProxyFeatures` to
//! the NPU worker via an `mpsc` channel.

use crate::spectral::{
    anderson_3d, find_all_eigenvalues, lanczos, lanczos_eigenvalues, level_spacing_ratio,
};

/// Features extracted from a physics proxy model, sent to the NPU.
#[derive(Debug, Clone)]
pub struct ProxyFeatures {
    /// QCD β this proxy was evaluated for.
    pub beta: f64,
    /// Level spacing ratio ⟨r⟩ (GOE ≈ 0.53, Poisson ≈ 0.39).
    pub level_spacing_ratio: f64,
    /// Smallest absolute eigenvalue — predicts CG condition number.
    pub lambda_min: f64,
    /// Inverse participation ratio — extended vs. localized.
    pub ipr: f64,
    /// Spectral bandwidth.
    pub bandwidth: f64,
    /// Phase label: "extended", "localized", or "critical".
    pub phase: String,
    /// Proxy tier: 1=3D scalar, 2=4D scalar, 3=4D Wegner.
    pub tier: u8,
    /// Wall time for this proxy evaluation in milliseconds.
    pub wall_ms: f64,
}

/// Request sent from the main thread to the CPU cortex.
#[derive(Debug, Clone)]
pub struct CortexRequest {
    /// QCD β value to evaluate.
    pub beta: f64,
    /// Fermion mass.
    pub mass: f64,
    /// Lattice size (one dimension of the L⁴ lattice).
    pub lattice: usize,
    /// Current plaquette variance (maps to Anderson disorder W).
    pub plaq_var: f64,
}

/// Map QCD β and plaquette variance to Anderson disorder W.
///
/// Higher plaquette variance = more gauge fluctuation = more "disorder"
/// in the Anderson analogy. The mapping is:
///   W ≈ 4 + 200 * plaq_var  (calibrated from Exp 024 data)
///
/// At strong coupling (β ~ 4.3), plaq_var ~ 0.08 → W ~ 20 (localized).
/// At weak coupling (β ~ 6.1), plaq_var ~ 0.02 → W ~ 8 (extended).
fn qcd_to_anderson_disorder(plaq_var: f64) -> f64 {
    (4.0 + 200.0 * plaq_var).clamp(1.0, 30.0)
}

/// Run 3D Anderson proxy for a given QCD configuration.
///
/// Returns `ProxyFeatures` with level statistics that predict CG difficulty.
pub fn anderson_3d_proxy(req: &CortexRequest, seed: u64) -> ProxyFeatures {
    let t0 = std::time::Instant::now();

    let w = qcd_to_anderson_disorder(req.plaq_var);
    let l = req.lattice.min(12);
    let h = anderson_3d(l, l, l, w, seed);
    let n = l * l * l;

    let eigenvalues = if n <= 1000 {
        let tri = lanczos(&h, n.min(h.n), seed);
        find_all_eigenvalues(&tri.alpha, &tri.beta)
    } else {
        let k = 200.min(n);
        let tri = lanczos(&h, k, seed);
        lanczos_eigenvalues(&tri)
    };

    let r = level_spacing_ratio(&eigenvalues);
    let bandwidth = if eigenvalues.len() >= 2 {
        eigenvalues.last().unwrap_or(&0.0) - eigenvalues.first().unwrap_or(&0.0)
    } else {
        0.0
    };
    let lambda_min = eigenvalues.iter().map(|e| e.abs()).fold(f64::MAX, f64::min);
    let ipr = compute_ipr_from_stats(&eigenvalues);
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let phase = if r > 0.48 {
        "extended"
    } else if r < 0.42 {
        "localized"
    } else {
        "critical"
    };

    ProxyFeatures {
        beta: req.beta,
        level_spacing_ratio: r,
        lambda_min,
        ipr,
        bandwidth,
        phase: phase.to_string(),
        tier: 1,
        wall_ms,
    }
}

/// Run Z(3) Potts Monte Carlo proxy for a given QCD β.
///
/// Uses the Svetitsky-Yaffe mapping: β_QCD → β_Potts ≈ (β_QCD - 4.0) / 3.0.
/// Returns phase label and susceptibility for NPU phase classification.
pub fn potts_z3_proxy(req: &CortexRequest, seed: u64) -> ProxyFeatures {
    let t0 = std::time::Instant::now();

    let beta_potts = ((req.beta - 4.0) / 3.0).clamp(0.1, 1.5);
    let l = req.lattice.min(16);
    let (mag, chi, _energy, phase_label) = potts_z3_monte_carlo(l, beta_potts, 200, 100, seed);

    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    ProxyFeatures {
        beta: req.beta,
        level_spacing_ratio: mag,
        lambda_min: chi,
        ipr: 0.0,
        bandwidth: 0.0,
        phase: phase_label,
        tier: 1,
        wall_ms,
    }
}

/// Estimate IPR from eigenvalue density.
fn compute_ipr_from_stats(eigenvalues: &[f64]) -> f64 {
    if eigenvalues.len() < 3 {
        return 1.0;
    }
    let n = eigenvalues.len() as f64;
    let mut spacings: Vec<f64> = eigenvalues
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .collect();
    spacings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = spacings[spacings.len() / 2];
    if median < 1e-15 {
        return 1.0 / n;
    }
    let var = spacings.iter().map(|s| (s - median).powi(2)).sum::<f64>() / spacings.len() as f64;
    let cv = var.sqrt() / median;
    let ipr = (1.0 / n) + (1.0 - 1.0 / n) * (cv - 0.52).max(0.0) / 0.48;
    ipr.clamp(1.0 / n, 1.0)
}

/// 3D Z(3) Potts Monte Carlo on an L^3 cubic lattice.
fn potts_z3_monte_carlo(
    l: usize,
    beta: f64,
    n_therm: usize,
    n_meas: usize,
    seed: u64,
) -> (f64, f64, f64, String) {
    let n = l * l * l;
    let mut spins: Vec<u8> = vec![0; n];
    let mut rng = LcgRng::new(seed);

    for s in &mut spins {
        *s = (rng.next_u64() % 3) as u8;
    }

    let idx = |x: usize, y: usize, z: usize| -> usize { (x % l) + (y % l) * l + (z % l) * l * l };

    let neighbors = |site: usize| -> [usize; 6] {
        let x = site % l;
        let y = (site / l) % l;
        let z = site / (l * l);
        [
            idx((x + 1) % l, y, z),
            idx((x + l - 1) % l, y, z),
            idx(x, (y + 1) % l, z),
            idx(x, (y + l - 1) % l, z),
            idx(x, y, (z + 1) % l),
            idx(x, y, (z + l - 1) % l),
        ]
    };

    let mut mag_samples = Vec::with_capacity(n_meas);
    let mut energy_samples = Vec::with_capacity(n_meas);

    for sweep in 0..(n_therm + n_meas) {
        for site in 0..n {
            let old_spin = spins[site];
            let new_spin = ((old_spin as u64 + 1 + rng.next_u64() % 2) % 3) as u8;

            let nbrs = neighbors(site);
            let mut delta_e: i32 = 0;
            for &nb in &nbrs {
                if spins[nb] == new_spin {
                    delta_e -= 1;
                }
                if spins[nb] == old_spin {
                    delta_e += 1;
                }
            }

            let accept = if delta_e <= 0 {
                true
            } else {
                rng.uniform() < (-beta * delta_e as f64).exp()
            };

            if accept {
                spins[site] = new_spin;
            }
        }

        if sweep >= n_therm {
            let omega_re = [-0.5, -0.5, 1.0];
            let omega_im = [3.0_f64.sqrt() / 2.0, -(3.0_f64.sqrt()) / 2.0, 0.0];
            let mut m = [0.0f64; 2];
            for &s in &spins {
                m[0] += omega_re[s as usize];
                m[1] += omega_im[s as usize];
            }
            let mag = (m[0] * m[0] + m[1] * m[1]).sqrt() / n as f64;
            mag_samples.push(mag);

            let mut energy = 0.0f64;
            for site in 0..n {
                let nbrs = neighbors(site);
                for &nb in &nbrs {
                    if spins[nb] == spins[site] {
                        energy -= 1.0;
                    }
                }
            }
            energy /= 2.0 * n as f64;
            energy_samples.push(energy);
        }
    }

    let mean_mag = mag_samples.iter().sum::<f64>() / mag_samples.len().max(1) as f64;
    let mean_mag2 =
        mag_samples.iter().map(|m| m * m).sum::<f64>() / mag_samples.len().max(1) as f64;
    let chi = (mean_mag2 - mean_mag * mean_mag) * n as f64;
    let mean_energy = energy_samples.iter().sum::<f64>() / energy_samples.len().max(1) as f64;

    let phase = if mean_mag > 0.5 {
        "ordered".to_string()
    } else if mean_mag < 0.2 {
        "disordered".to_string()
    } else {
        "transition".to_string()
    };

    (mean_mag, chi, mean_energy, phase)
}

struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(6364136223846793005).wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn anderson_3d_proxy_returns_valid_features() {
        let req = CortexRequest {
            beta: 5.5,
            mass: 0.1,
            lattice: 4,
            plaq_var: 0.05,
        };
        let features = anderson_3d_proxy(&req, 42);
        assert!(features.beta > 0.0);
        assert!(features.level_spacing_ratio.is_finite());
        assert!(features.lambda_min.is_finite());
        assert!(features.wall_ms >= 0.0);
        assert!(features.tier == 1);
        assert!(
            features.phase == "extended"
                || features.phase == "localized"
                || features.phase == "critical"
        );
    }

    #[test]
    fn potts_z3_proxy_returns_valid_features() {
        let req = CortexRequest {
            beta: 5.0,
            mass: 0.1,
            lattice: 4,
            plaq_var: 0.03,
        };
        let features = potts_z3_proxy(&req, 123);
        assert!(features.beta > 0.0);
        assert!(features.lambda_min.is_finite());
        assert!(features.wall_ms >= 0.0);
        assert!(features.tier == 1);
        assert!(
            features.phase == "ordered"
                || features.phase == "disordered"
                || features.phase == "transition"
        );
    }

    #[test]
    fn anderson_3d_proxy_determinism() {
        let req = CortexRequest {
            beta: 5.5,
            mass: 0.1,
            lattice: 4,
            plaq_var: 0.05,
        };
        let a = anderson_3d_proxy(&req, 999);
        let b = anderson_3d_proxy(&req, 999);
        assert_eq!(
            a.level_spacing_ratio.to_bits(),
            b.level_spacing_ratio.to_bits()
        );
        assert_eq!(a.lambda_min.to_bits(), b.lambda_min.to_bits());
        assert_eq!(a.phase, b.phase);
    }

    #[test]
    fn potts_z3_proxy_determinism() {
        let req = CortexRequest {
            beta: 5.0,
            mass: 0.1,
            lattice: 4,
            plaq_var: 0.03,
        };
        let a = potts_z3_proxy(&req, 777);
        let b = potts_z3_proxy(&req, 777);
        assert_eq!(
            a.level_spacing_ratio.to_bits(),
            b.level_spacing_ratio.to_bits()
        );
        assert_eq!(a.lambda_min.to_bits(), b.lambda_min.to_bits());
        assert_eq!(a.phase, b.phase);
    }

    #[test]
    fn cortex_request_small_lattice() {
        let req = CortexRequest {
            beta: 6.0,
            mass: 0.05,
            lattice: 2,
            plaq_var: 0.02,
        };
        let features = anderson_3d_proxy(&req, 1);
        assert!(features.wall_ms >= 0.0);
        assert!(features.beta == 6.0);
    }
}
