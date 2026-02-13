//! Pre-screening cascade for nuclear EOS parameter space
//!
//! Reduces expensive L2 HFB evaluations by filtering unphysical parameter sets.
//!
//! **Three-tier cascade:**
//! - Tier 1 (algebraic, ~0 cost): NMP bounds check
//! - Tier 2 (L1 proxy, ~0.001s):  SEMF χ² threshold — if bad at L1, skip L2
//! - Tier 3 (learned, NPU/CPU):   Classifier trained on accumulated data
//!
//! **Heterogeneous compute:**
//! - Tier 1-2: CPU (instant)
//! - Tier 3:   NPU (Akida, ~1W) or CPU fallback
//! - Tier 4:   CPU parallel (rayon) for HFB that passes all screens

use crate::physics::{nuclear_matter_properties, NuclearMatterProps, semf_binding_energy};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// Tier 1: NMP Algebraic Pre-screen
// ═══════════════════════════════════════════════════════════════════

/// Physical constraints on nuclear matter properties.
/// Parameter sets producing NMP outside these bounds are rejected
/// before any expensive calculation.
#[derive(Debug, Clone)]
pub struct NMPConstraints {
    pub rho0_min: f64,   // Minimum saturation density (fm⁻³)
    pub rho0_max: f64,   // Maximum saturation density
    pub e_a_min: f64,    // Minimum binding energy (MeV), e.g. -20
    pub e_a_max: f64,    // Maximum binding energy, e.g. -5
    pub k_inf_min: f64,  // Minimum incompressibility (MeV)
    pub k_inf_max: f64,  // Maximum incompressibility
    pub m_eff_min: f64,  // Minimum effective mass ratio
    pub m_eff_max: f64,  // Maximum effective mass ratio
    pub j_min: f64,      // Minimum symmetry energy (MeV)
    pub j_max: f64,      // Maximum symmetry energy
}

impl Default for NMPConstraints {
    fn default() -> Self {
        // Tightened bounds from L1 Pareto analysis
        // Experiment: ρ₀≈0.16, E/A≈-16, K∞≈230, m*≈0.7, J≈32
        Self {
            rho0_min: 0.10,
            rho0_max: 0.22,
            e_a_min: -22.0,
            e_a_max: -8.0,
            k_inf_min: 100.0,
            k_inf_max: 500.0,
            m_eff_min: 0.2,
            m_eff_max: 2.0,
            j_min: 20.0,
            j_max: 45.0,
        }
    }
}

/// Result of Tier 1 NMP pre-screening
#[derive(Debug, Clone)]
pub enum NMPScreenResult {
    /// Passed — NMP within physical bounds
    Pass(NuclearMatterProps),
    /// Failed — specific reason for rejection
    Fail(String),
}

/// Tier 1: Check if Skyrme parameters produce physically reasonable NMP.
/// Cost: ~1μs (algebraic + bisection for ρ₀)
pub fn nmp_prescreen(params: &[f64], constraints: &NMPConstraints) -> NMPScreenResult {
    // Alpha sanity check
    if params.len() != 10 || params[8] <= 0.01 || params[8] > 1.0 {
        return NMPScreenResult::Fail("alpha out of range".to_string());
    }

    let nmp = match nuclear_matter_properties(params) {
        Some(nmp) => nmp,
        None => return NMPScreenResult::Fail("NMP calculation failed".to_string()),
    };

    if nmp.rho0_fm3 < constraints.rho0_min || nmp.rho0_fm3 > constraints.rho0_max {
        return NMPScreenResult::Fail(format!("ρ₀={:.4} out of [{}, {}]",
            nmp.rho0_fm3, constraints.rho0_min, constraints.rho0_max));
    }
    if nmp.e_a_mev < constraints.e_a_min || nmp.e_a_mev > constraints.e_a_max {
        return NMPScreenResult::Fail(format!("E/A={:.2} out of [{}, {}]",
            nmp.e_a_mev, constraints.e_a_min, constraints.e_a_max));
    }
    if nmp.k_inf_mev < constraints.k_inf_min || nmp.k_inf_mev > constraints.k_inf_max {
        return NMPScreenResult::Fail(format!("K∞={:.1} out of [{}, {}]",
            nmp.k_inf_mev, constraints.k_inf_min, constraints.k_inf_max));
    }
    if nmp.m_eff_ratio < constraints.m_eff_min || nmp.m_eff_ratio > constraints.m_eff_max {
        return NMPScreenResult::Fail(format!("m*/m={:.3} out of [{}, {}]",
            nmp.m_eff_ratio, constraints.m_eff_min, constraints.m_eff_max));
    }
    if nmp.j_mev < constraints.j_min || nmp.j_mev > constraints.j_max {
        return NMPScreenResult::Fail(format!("J={:.1} out of [{}, {}]",
            nmp.j_mev, constraints.j_min, constraints.j_max));
    }

    NMPScreenResult::Pass(nmp)
}

// ═══════════════════════════════════════════════════════════════════
// Tier 2: L1 (SEMF) Proxy Pre-screen
// ═══════════════════════════════════════════════════════════════════

/// Tier 2: Quick L1 SEMF χ²/datum check.
/// If a parameterization can't fit nuclei at the simple SEMF level,
/// it won't work with HFB either. Cost: ~0.1ms
pub fn l1_proxy_prescreen(
    params: &[f64],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    chi2_threshold: f64,
) -> Option<f64> {
    let mut chi2 = 0.0;
    let mut n_valid = 0;

    for (&(z, n), &(b_exp, _sigma)) in exp_data.iter() {
        let b_calc = semf_binding_energy(z, n, params);
        if b_calc > 0.0 {
            let sigma_theo = (0.01 * b_exp).max(2.0);
            chi2 += ((b_calc - b_exp) / sigma_theo).powi(2);
            n_valid += 1;
        }
    }

    if n_valid == 0 {
        return None; // reject
    }

    let chi2_per_datum = chi2 / n_valid as f64;
    if chi2_per_datum < chi2_threshold {
        Some(chi2_per_datum)
    } else {
        None // reject: too poor even for SEMF
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tier 3: Learned Classifier (NPU/CPU)
// ═══════════════════════════════════════════════════════════════════

/// Simple logistic regression classifier for parameter pre-screening.
///
/// Trained on accumulated L1/L2 data to predict whether a parameter set
/// will produce a reasonable L2 χ². Designed for NPU deployment
/// (small, fixed-size model), with CPU fallback.
///
/// Architecture: 10 inputs → normalize → linear(10→1) → sigmoid → P(promising)
#[derive(Debug, Clone)]
pub struct PreScreenClassifier {
    /// Weights [10] — one per Skyrme parameter
    pub weights: Vec<f64>,
    /// Bias term
    pub bias: f64,
    /// Normalization: (mean, std) per feature
    pub norm: Vec<(f64, f64)>,
    /// Decision threshold
    pub threshold: f64,
    /// Training statistics
    pub n_train: usize,
    pub n_positive: usize,
}

impl PreScreenClassifier {
    /// Train classifier on accumulated evaluation data.
    ///
    /// Training data: (params, f_value) pairs from previous runs.
    /// Labels: f_value < label_threshold → positive (promising)
    pub fn train(
        xs: &[Vec<f64>],
        ys: &[f64],
        label_threshold: f64,
    ) -> Self {
        let n = xs.len();
        let dim = if n > 0 { xs[0].len() } else { 10 };

        // Compute labels: y < threshold → 1.0 (promising), else 0.0
        let labels: Vec<f64> = ys.iter()
            .map(|&y| if y < label_threshold { 1.0 } else { 0.0 })
            .collect();

        let n_positive = labels.iter().filter(|&&l| l > 0.5).count();

        // Normalize features
        let mut means = vec![0.0; dim];
        let mut stds = vec![1.0; dim];

        if n > 0 {
            for i in 0..dim {
                let vals: Vec<f64> = xs.iter().map(|x| x[i]).collect();
                means[i] = vals.iter().sum::<f64>() / n as f64;
                let var = vals.iter().map(|v| (v - means[i]).powi(2)).sum::<f64>() / n as f64;
                stds[i] = var.sqrt().max(1e-10);
            }
        }

        let norm: Vec<(f64, f64)> = means.iter().zip(stds.iter())
            .map(|(&m, &s)| (m, s))
            .collect();

        // Train logistic regression with simple gradient descent
        let mut weights = vec![0.0; dim];
        let mut bias = 0.0;
        let lr = 0.01;
        let epochs = 200;

        for _epoch in 0..epochs {
            let mut dw = vec![0.0; dim];
            let mut db = 0.0;

            for (x, &label) in xs.iter().zip(labels.iter()) {
                // Normalize
                let x_norm: Vec<f64> = x.iter().enumerate()
                    .map(|(i, &v)| (v - norm[i].0) / norm[i].1)
                    .collect();

                // Forward: z = w·x + b, p = sigmoid(z)
                let z: f64 = x_norm.iter().zip(weights.iter())
                    .map(|(&xi, &wi)| xi * wi)
                    .sum::<f64>() + bias;
                let p = 1.0 / (1.0 + (-z).exp());

                // Gradient: d/dw = (p - label) * x
                let err = p - label;
                for i in 0..dim {
                    dw[i] += err * x_norm[i];
                }
                db += err;
            }

            // Update
            let scale = lr / n.max(1) as f64;
            for i in 0..dim {
                weights[i] -= scale * dw[i];
            }
            bias -= scale * db;
        }

        Self {
            weights,
            bias,
            norm,
            threshold: 0.5,
            n_train: n,
            n_positive,
        }
    }

    /// Predict probability of being "promising"
    pub fn predict_prob(&self, params: &[f64]) -> f64 {
        let x_norm: Vec<f64> = params.iter().enumerate()
            .map(|(i, &v)| (v - self.norm[i].0) / self.norm[i].1)
            .collect();

        let z: f64 = x_norm.iter().zip(self.weights.iter())
            .map(|(&xi, &wi)| xi * wi)
            .sum::<f64>() + self.bias;

        1.0 / (1.0 + (-z).exp())
    }

    /// Binary prediction: is this parameter set promising?
    pub fn predict(&self, params: &[f64]) -> bool {
        self.predict_prob(params) > self.threshold
    }

    /// Get model weights for NPU deployment
    pub fn export_weights(&self) -> (Vec<f64>, f64, Vec<(f64, f64)>) {
        (self.weights.clone(), self.bias, self.norm.clone())
    }
}

// ═══════════════════════════════════════════════════════════════════
// Full Pre-screening Cascade
// ═══════════════════════════════════════════════════════════════════

/// Pre-screening cascade statistics
#[derive(Debug, Default, Clone)]
pub struct CascadeStats {
    pub total_candidates: usize,
    pub tier1_rejected: usize,   // NMP out of bounds
    pub tier2_rejected: usize,   // L1 proxy too high
    pub tier3_rejected: usize,   // Classifier says no
    pub tier4_evaluated: usize,  // Passed all screens, HFB evaluated
}

impl CascadeStats {
    pub fn pass_rate(&self) -> f64 {
        if self.total_candidates == 0 { return 0.0; }
        self.tier4_evaluated as f64 / self.total_candidates as f64
    }

    pub fn print_summary(&self) {
        println!("  Pre-screening cascade:");
        println!("    Total candidates:     {}", self.total_candidates);
        println!("    Tier 1 rejected (NMP):  {} ({:.1}%)", self.tier1_rejected,
            100.0 * self.tier1_rejected as f64 / self.total_candidates.max(1) as f64);
        println!("    Tier 2 rejected (L1):   {} ({:.1}%)", self.tier2_rejected,
            100.0 * self.tier2_rejected as f64 / self.total_candidates.max(1) as f64);
        println!("    Tier 3 rejected (clf):  {} ({:.1}%)", self.tier3_rejected,
            100.0 * self.tier3_rejected as f64 / self.total_candidates.max(1) as f64);
        println!("    Tier 4 evaluated (HFB): {} ({:.1}% pass rate)", self.tier4_evaluated,
            100.0 * self.pass_rate());
    }
}

