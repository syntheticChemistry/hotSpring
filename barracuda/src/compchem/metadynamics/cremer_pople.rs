// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cremer-Pople ring puckering coordinates.
//!
//! Implements the general puckering analysis for 6-membered rings from:
//! Cremer & Pople, JACS 97, 1354 (1975).
//!
//! For a 6-membered ring (pyranose), the 3 puckering parameters are:
//! - Q (total puckering amplitude)
//! - θ (polar angle: 0° = ⁴C₁, 180° = ¹C₄)
//! - φ (azimuthal angle: distinguishes boat/twist-boat conformations)
//!
//! These CVs are used by PLUMED's `PUCKERING` function and are the natural
//! coordinates for carbohydrate FEL calculations.

use std::f64::consts::PI;

/// Cremer-Pople puckering coordinate calculator for 6-membered rings.
pub struct CremerPople;

/// Computed puckering coordinates.
#[derive(Debug, Clone, Copy)]
pub struct PuckeringCoords {
    /// Total puckering amplitude Q (Å or nm, same units as input)
    pub q_total: f64,
    /// Polar puckering angle θ (radians, [0, π])
    pub theta: f64,
    /// Azimuthal puckering angle φ (radians, [-π, π])
    pub phi: f64,
    /// q₂ component
    pub q2: f64,
    /// q₃ component (z-displacement of alternating atoms)
    pub q3: f64,
}

impl CremerPople {
    /// Compute Cremer-Pople puckering coordinates from 6 ring atom positions.
    ///
    /// `ring_positions` must contain exactly 6 atoms in ring order,
    /// each as `[x, y, z]` (18 f64 values in flat layout: `[x0,y0,z0, x1,y1,z1, ...]`).
    ///
    /// Returns `None` if the ring is degenerate (zero area).
    pub fn compute(ring_positions: &[f64]) -> Option<PuckeringCoords> {
        if ring_positions.len() != 18 {
            return None;
        }

        let n = 6usize;

        // Extract positions
        let r: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                [
                    ring_positions[i * 3],
                    ring_positions[i * 3 + 1],
                    ring_positions[i * 3 + 2],
                ]
            })
            .collect();

        // Step 1: Compute geometric center
        let center = [
            r.iter().map(|p| p[0]).sum::<f64>() / n as f64,
            r.iter().map(|p| p[1]).sum::<f64>() / n as f64,
            r.iter().map(|p| p[2]).sum::<f64>() / n as f64,
        ];

        // Step 2: Compute R' and R'' reference vectors
        let mut r_prime = [0.0f64; 3];
        let mut r_double_prime = [0.0f64; 3];
        for j in 0..n {
            let angle_p = 2.0 * PI * (j as f64) / (n as f64);
            let angle_dp = 2.0 * angle_p;
            let sin_p = angle_p.sin();
            let cos_p = angle_p.cos();
            let sin_dp = angle_dp.sin();
            let cos_dp = angle_dp.cos();
            for d in 0..3 {
                let rj = r[j][d] - center[d];
                r_prime[d] += rj * sin_p;
                r_double_prime[d] += rj * cos_dp;
                // Also accumulate for R'' with cos
                let _ = sin_dp; // used below
                let _ = cos_p;
            }
        }

        // Recompute properly: R' = Σ r_j sin(2πj/N), R'' = Σ r_j cos(2πj/N)
        r_prime = [0.0; 3];
        r_double_prime = [0.0; 3];
        for j in 0..n {
            let angle = 2.0 * PI * (j as f64) / (n as f64);
            for d in 0..3 {
                let rj = r[j][d] - center[d];
                r_prime[d] += rj * angle.sin();
                r_double_prime[d] += rj * angle.cos();
            }
        }

        // Step 3: Normal vector n = R' × R''
        let normal = cross(r_prime, r_double_prime);
        let normal_len = norm(normal);
        if normal_len < 1e-15 {
            return None;
        }
        let n_hat = [
            normal[0] / normal_len,
            normal[1] / normal_len,
            normal[2] / normal_len,
        ];

        // Step 4: Compute z_j = (r_j - center) · n_hat
        let z: Vec<f64> = (0..n)
            .map(|j| {
                let dx = r[j][0] - center[0];
                let dy = r[j][1] - center[1];
                let dz = r[j][2] - center[2];
                dx * n_hat[0] + dy * n_hat[1] + dz * n_hat[2]
            })
            .collect();

        // Step 5: Compute puckering coordinates for N=6
        // q₂ cos(φ) = (1/√3) Σ z_j cos(2π·2j/6)
        // q₂ sin(φ) = -(1/√3) Σ z_j sin(2π·2j/6)
        // q₃ = (1/√6) Σ z_j (-1)^j

        let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
        let inv_sqrt6 = 1.0 / 6.0_f64.sqrt();

        let mut q2_cos_phi = 0.0f64;
        let mut q2_sin_phi = 0.0f64;
        let mut q3_val = 0.0f64;

        for j in 0..n {
            let angle = 2.0 * PI * 2.0 * (j as f64) / (n as f64);
            q2_cos_phi += z[j] * angle.cos();
            q2_sin_phi += z[j] * angle.sin();
            q3_val += z[j] * if j % 2 == 0 { 1.0 } else { -1.0 };
        }

        q2_cos_phi *= inv_sqrt3;
        q2_sin_phi *= -inv_sqrt3;
        q3_val *= inv_sqrt6;

        let q2 = (q2_cos_phi * q2_cos_phi + q2_sin_phi * q2_sin_phi).sqrt();
        let phi = q2_sin_phi.atan2(q2_cos_phi);
        let q_total = (q2 * q2 + q3_val * q3_val).sqrt();

        // θ = arccos(q₃/Q): θ≈0 for ⁴C₁ (q₃>0), θ≈π for ¹C₄ (q₃<0)
        let theta = if q_total < 1e-15 {
            0.0
        } else {
            (q3_val / q_total).clamp(-1.0, 1.0).acos()
        };

        Some(PuckeringCoords {
            q_total,
            theta,
            phi,
            q2,
            q3: q3_val,
        })
    }

    /// Compute numerical gradient of θ w.r.t. ring atom positions.
    ///
    /// Returns `[dθ/dx₀, dθ/dy₀, dθ/dz₀, ..., dθ/dx₅, dθ/dy₅, dθ/dz₅]` (18 values).
    /// Used for applying metadynamics bias forces to atoms.
    pub fn gradient_theta(ring_positions: &[f64], epsilon: f64) -> Option<Vec<f64>> {
        if ring_positions.len() != 18 {
            return None;
        }
        let mut grad = Vec::with_capacity(18);
        for i in 0..18 {
            let mut pos_plus = ring_positions.to_vec();
            let mut pos_minus = ring_positions.to_vec();
            pos_plus[i] += epsilon;
            pos_minus[i] -= epsilon;

            let theta_plus = Self::compute(&pos_plus)?.theta;
            let theta_minus = Self::compute(&pos_minus)?.theta;
            grad.push((theta_plus - theta_minus) / (2.0 * epsilon));
        }
        Some(grad)
    }

    /// Compute numerical gradient of φ w.r.t. ring atom positions.
    pub fn gradient_phi(ring_positions: &[f64], epsilon: f64) -> Option<Vec<f64>> {
        if ring_positions.len() != 18 {
            return None;
        }
        let mut grad = Vec::with_capacity(18);
        for i in 0..18 {
            let mut pos_plus = ring_positions.to_vec();
            let mut pos_minus = ring_positions.to_vec();
            pos_plus[i] += epsilon;
            pos_minus[i] -= epsilon;

            let phi_plus = Self::compute(&pos_plus)?.phi;
            let phi_minus = Self::compute(&pos_minus)?.phi;

            // Handle periodic wrap-around for φ
            let mut dphi = phi_plus - phi_minus;
            if dphi > PI {
                dphi -= 2.0 * PI;
            }
            if dphi < -PI {
                dphi += 2.0 * PI;
            }
            grad.push(dphi / (2.0 * epsilon));
        }
        Some(grad)
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chair_4c1(r: f64, amplitude: f64) -> Vec<f64> {
        // Ideal ⁴C₁ chair: alternating z displacements
        // Atoms at hexagonal positions in xy, alternating +z/-z
        let mut positions = Vec::with_capacity(18);
        for j in 0..6 {
            let angle = 2.0 * PI * (j as f64) / 6.0;
            let x = r * angle.cos();
            let y = r * angle.sin();
            let z = amplitude * if j % 2 == 0 { 1.0 } else { -1.0 };
            positions.push(x);
            positions.push(y);
            positions.push(z);
        }
        positions
    }

    fn make_chair_1c4(r: f64, amplitude: f64) -> Vec<f64> {
        // ¹C₄: inverted z pattern vs ⁴C₁
        let mut positions = Vec::with_capacity(18);
        for j in 0..6 {
            let angle = 2.0 * PI * (j as f64) / 6.0;
            let x = r * angle.cos();
            let y = r * angle.sin();
            let z = amplitude * if j % 2 == 0 { -1.0 } else { 1.0 };
            positions.push(x);
            positions.push(y);
            positions.push(z);
        }
        positions
    }

    #[test]
    fn test_chairs_opposite_theta() {
        let pos_a = make_chair_4c1(0.15, 0.05);
        let pos_b = make_chair_1c4(0.15, 0.05);
        let cp_a = CremerPople::compute(&pos_a).unwrap();
        let cp_b = CremerPople::compute(&pos_b).unwrap();

        assert!(cp_a.q_total > 0.01, "Q should be nonzero for puckered ring");
        assert!(cp_b.q_total > 0.01, "Q should be nonzero for puckered ring");

        // The two chairs should be at opposite poles of θ
        let theta_sum = cp_a.theta + cp_b.theta;
        assert!(
            (theta_sum - PI).abs() < 0.3,
            "chair A (θ={:.2}°) + chair B (θ={:.2}°) should sum to ~180°, got {:.2}°",
            cp_a.theta.to_degrees(),
            cp_b.theta.to_degrees(),
            theta_sum.to_degrees()
        );

        // One should be near 0 and the other near π
        let (theta_min, theta_max) = if cp_a.theta < cp_b.theta {
            (cp_a.theta, cp_b.theta)
        } else {
            (cp_b.theta, cp_a.theta)
        };
        assert!(theta_min < 0.3, "one chair should have θ near 0, got {:.2}°", theta_min.to_degrees());
        assert!(theta_max > 2.8, "other chair should have θ near π, got {:.2}°", theta_max.to_degrees());
    }

    #[test]
    fn test_flat_ring_zero_puckering() {
        // All atoms in the same plane → Q ≈ 0
        let mut positions = Vec::with_capacity(18);
        for j in 0..6 {
            let angle = 2.0 * PI * (j as f64) / 6.0;
            positions.push(0.15 * angle.cos());
            positions.push(0.15 * angle.sin());
            positions.push(0.0);
        }
        let cp = CremerPople::compute(&positions).unwrap();
        assert!(
            cp.q_total < 1e-10,
            "flat ring should have Q ≈ 0, got {}",
            cp.q_total
        );
    }

    #[test]
    fn test_theta_gradient_finite_difference() {
        let pos = make_chair_4c1(0.15, 0.04);
        let grad = CremerPople::gradient_theta(&pos, 1e-6).unwrap();
        assert_eq!(grad.len(), 18);

        // At least some gradient components should be nonzero
        let max_grad = grad.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
        assert!(
            max_grad > 1e-6,
            "θ gradient should have nonzero components, max={}",
            max_grad
        );
    }

    #[test]
    fn test_wrong_input_length() {
        assert!(CremerPople::compute(&[0.0; 15]).is_none());
        assert!(CremerPople::compute(&[0.0; 21]).is_none());
    }
}
