// SPDX-License-Identifier: AGPL-3.0-or-later

//! ESN training data construction and beta-steering helpers.

use crate::md::reservoir::heads;
use crate::production::BetaResult;
use crate::production::trajectory_input::canonical_input;
use crate::provenance::KNOWN_BETA_C_SU3_NT4;

/// Estimate β_c from the susceptibility peak in results.
#[must_use]
pub(crate) fn estimate_beta_c(results: &[BetaResult]) -> f64 {
    if results.is_empty() {
        return KNOWN_BETA_C_SU3_NT4;
    }
    let max_chi_r = results
        .iter()
        .max_by(|a, b| a.susceptibility.total_cmp(&b.susceptibility));
    max_chi_r.map_or(KNOWN_BETA_C_SU3_NT4, |r| r.beta)
}

/// Find the n largest gaps in [min, max] among measured values; return midpoints.
#[must_use]
pub(crate) fn find_largest_gaps(measured: &[f64], min: f64, max: f64, n: usize) -> Vec<f64> {
    let mut sorted = measured.to_vec();
    sorted.push(min);
    sorted.push(max);
    sorted.sort_by(f64::total_cmp);
    sorted.dedup();

    let mut gaps: Vec<(f64, f64)> = sorted
        .windows(2)
        .map(|w| (w[1] - w[0], f64::midpoint(w[0], w[1])))
        .collect();
    gaps.sort_by(|a, b| b.0.total_cmp(&a.0));
    gaps.iter().take(n).map(|g| g.1).collect()
}

/// Build ESN training sequences and targets from accumulated β-scan results.
#[must_use]
pub(crate) fn build_training_data(
    results: &[BetaResult],
    lattice: usize,
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut seqs = Vec::new();
    let mut targets = Vec::new();
    let seq_len = 10;

    for r in results {
        let phase_continuous = r.polyakov.abs().clamp(0.0, 1.0);
        let delta_h_continuous = (-(r.acceptance - 0.5) * 4.0) / 10.0;

        let proximity = (-(r.beta - KNOWN_BETA_C_SU3_NT4).powi(2) / 0.1).exp();
        let anomaly_val = f64::from(r.acceptance < 0.05 || r.mean_cg_iters > 3000.0);
        let quality = (r.n_traj as f64 / 1000.0)
            .min(1.0)
            .mul_add(
                0.3,
                r.acceptance.mul_add(
                    0.4,
                    (1.0 - r.std_plaq / r.mean_plaq.abs().max(1e-10)).clamp(0.0, 1.0) * 0.3,
                ),
            )
            .clamp(0.0, 1.0);
        let cg_norm = r.mean_cg_iters / 500.0;
        let quenched_therm_target = f64::from(r.npu_quenched_early_exit);

        let anderson_phase = if r.beta > 5.5 {
            1.0
        } else if r.beta < 5.0 {
            0.0
        } else {
            0.5
        };
        let potts_phase = if r.beta > 5.8 {
            1.0
        } else if r.beta < 5.2 {
            0.0
        } else {
            0.5
        };
        let target_acc = 0.70;
        let acc_error = r.acceptance - target_acc;
        let optimal_dt = (r.dt_used * 0.5f64.mul_add(-acc_error, 1.0)).clamp(0.002, 0.02);
        let optimal_nmd = ((1.0 / optimal_dt).round() / 200.0).clamp(0.0, 1.0);

        let seq: Vec<Vec<f64>> = (0..seq_len)
            .map(|j| {
                let noise = 0.005 * (f64::from(j) * 0.7).sin();
                canonical_input(
                    r.beta,
                    r.mean_plaq + noise * r.std_plaq,
                    r.mass,
                    r.susceptibility,
                    r.acceptance,
                    lattice,
                )
            })
            .collect();
        seqs.push(seq);

        let mut t = vec![0.0; heads::NUM_HEADS];

        t[heads::A0_ANDERSON_CG_COST] = cg_norm;
        t[heads::A1_ANDERSON_PHASE] = anderson_phase;
        t[heads::A2_ANDERSON_LAMBDA_MIN] = (1.0 / (cg_norm + 0.01)).clamp(0.0, 1.0);
        t[heads::A3_ANDERSON_ANOMALY] = anomaly_val;
        t[heads::A4_ANDERSON_THERM] = quenched_therm_target;
        t[heads::A5_ANDERSON_PRIORITY] = proximity;

        t[heads::B0_QCD_CG_COST] = cg_norm;
        t[heads::B1_QCD_PHASE] = phase_continuous;
        t[heads::B2_QCD_ACCEPTANCE] = delta_h_continuous;
        t[heads::B3_QCD_ANOMALY] = anomaly_val;
        t[heads::B4_QCD_THERM] = if r.acceptance > 0.3 { 1.0 } else { 0.0 };
        t[heads::B5_QCD_PRIORITY] = proximity;

        t[heads::C0_POTTS_CG_COST] = cg_norm;
        t[heads::C1_POTTS_PHASE] = potts_phase;
        t[heads::C2_POTTS_BETA_C] = (r.beta - KNOWN_BETA_C_SU3_NT4).abs().min(1.0);
        t[heads::C3_POTTS_ANOMALY] = anomaly_val;
        t[heads::C4_POTTS_ORDER] = r.polyakov.abs().clamp(0.0, 1.0);
        t[heads::C5_POTTS_PRIORITY] = proximity;

        t[heads::D0_NEXT_BETA] = proximity;
        t[heads::D1_OPTIMAL_DT] = optimal_dt;
        t[heads::D2_OPTIMAL_NMD] = optimal_nmd;
        t[heads::D3_CHECK_INTERVAL] = (1.0 - cg_norm).clamp(0.0, 1.0);
        t[heads::D4_KILL_DECISION] = (r.mean_cg_iters / 500.0).clamp(0.0, 1.0);
        t[heads::D5_SKIP_DECISION] = (1.0 - quality).clamp(0.0, 1.0);

        t[heads::E0_RESIDUAL_ETA] = cg_norm;
        t[heads::E1_RESIDUAL_ANOMALY] = anomaly_val;
        t[heads::E2_CONVERGENCE_RATE] = (1.0 - cg_norm).clamp(0.0, 1.0);
        t[heads::E3_STALL_DETECTOR] = (r.mean_cg_iters / 400.0).clamp(0.0, 1.0);
        t[heads::E4_DIVERGENCE_DETECTOR] = anomaly_val.clamp(0.0, 1.0);
        t[heads::E5_QUALITY_FORECAST] = quality;

        t[heads::M0_CG_CONSENSUS] = cg_norm;
        t[heads::M1_PHASE_CONSENSUS] = phase_continuous;
        t[heads::M2_CG_UNCERTAINTY] = (anderson_phase - phase_continuous).abs();
        t[heads::M3_PHASE_UNCERTAINTY] = (potts_phase - phase_continuous).abs();
        let proxy_agrees = (anderson_phase - phase_continuous).abs() < 0.3;
        t[heads::M4_PROXY_TRUST] = if proxy_agrees { 0.8 } else { 0.3 };
        t[heads::M5_ATTENTION_LEVEL] = (anomaly_val * 0.6 + cg_norm * 0.4).clamp(0.0, 1.0);

        targets.push(t);
    }

    (seqs, targets)
}
