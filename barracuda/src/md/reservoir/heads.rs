// SPDX-License-Identifier: AGPL-3.0-only

// ─── Gen 1 aliases (backward-compatible) ───

/// Gen 1 alias: next β to explore (maps to `D0_NEXT_BETA`).
pub const BETA_PRIORITY: usize = D0_NEXT_BETA;
/// Gen 1 alias: suggested HMC dt (maps to `D1_OPTIMAL_DT`).
pub const PARAM_SUGGEST: usize = D1_OPTIMAL_DT;
/// Gen 1 alias: thermalization detector (maps to `B4_QCD_THERM`).
pub const THERM_DETECT: usize = B4_QCD_THERM;
/// Gen 1 alias: rejection predictor (maps to `B2_QCD_ACCEPTANCE`).
pub const REJECT_PREDICT: usize = B2_QCD_ACCEPTANCE;
/// Gen 1 alias: phase classifier (maps to `B1_QCD_PHASE`).
pub const PHASE_CLASSIFY: usize = B1_QCD_PHASE;
/// Gen 1 alias: CG iteration estimator (maps to `B0_QCD_CG_COST`).
pub const CG_ESTIMATE: usize = B0_QCD_CG_COST;
/// Gen 1 alias: trajectory quality (maps to `E5_QUALITY_FORECAST`).
pub const QUALITY_SCORE: usize = E5_QUALITY_FORECAST;
/// Gen 1 alias: anomaly detector (maps to `B3_QCD_ANOMALY`).
pub const ANOMALY_DETECT: usize = B3_QCD_ANOMALY;
/// Gen 1 alias: same as `BETA_PRIORITY`.
pub const NEXT_RUN_RECOMMEND: usize = D0_NEXT_BETA;
/// Gen 1 head: quenched run length predictor.
pub const QUENCHED_LENGTH: usize = 9;
/// Gen 1 head: quenched thermalization detector.
pub const QUENCHED_THERM: usize = 10;
/// Gen 1 head: RMT spectral classifier.
pub const RMT_SPECTRAL: usize = 11;
/// Gen 1 alias: Potts phase (maps to `C1_POTTS_PHASE`).
pub const POTTS_PHASE: usize = C1_POTTS_PHASE;
/// Gen 1 alias: Anderson CG cost (maps to `A0_ANDERSON_CG_COST`).
pub const ANDERSON_CG: usize = A0_ANDERSON_CG_COST;
/// Gen 1 alias: CG residual monitor (maps to `E0_RESIDUAL_ETA`).
pub const CG_RESIDUAL_MONITOR: usize = E0_RESIDUAL_ETA;

// ─── Group A: Anderson-informed (disorder → localization → spectral) ───

/// Anderson proxy: predicted CG iteration cost.
pub const A0_ANDERSON_CG_COST: usize = 0;
/// Anderson proxy: localization phase (0=extended, 1=localized).
pub const A1_ANDERSON_PHASE: usize = 1;
/// Anderson proxy: minimum eigenvalue (mobility edge).
pub const A2_ANDERSON_LAMBDA_MIN: usize = 2;
/// Anderson proxy: spectral anomaly score.
pub const A3_ANDERSON_ANOMALY: usize = 3;
/// Anderson proxy: thermalization progress.
pub const A4_ANDERSON_THERM: usize = 4;
/// Anderson proxy: exploration priority.
pub const A5_ANDERSON_PRIORITY: usize = 5;

// ─── Group B: QCD-empirical (pure HMC observables, no proxy) ───

/// QCD empirical: predicted CG iteration cost.
pub const B0_QCD_CG_COST: usize = 6;
/// QCD empirical: confinement phase (0=confined, 1=deconfined).
pub const B1_QCD_PHASE: usize = 7;
/// QCD empirical: HMC acceptance rate prediction.
pub const B2_QCD_ACCEPTANCE: usize = 8;
/// QCD empirical: trajectory anomaly score.
pub const B3_QCD_ANOMALY: usize = 9;
/// QCD empirical: thermalization progress.
pub const B4_QCD_THERM: usize = 10;
/// QCD empirical: exploration priority.
pub const B5_QCD_PRIORITY: usize = 11;

// ─── Group C: Potts-informed (Svetitsky-Yaffe universality) ───

/// Potts proxy: predicted CG iteration cost.
pub const C0_POTTS_CG_COST: usize = 12;
/// Potts proxy: order/disorder phase.
pub const C1_POTTS_PHASE: usize = 13;
/// Potts proxy: estimated critical β_c.
pub const C2_POTTS_BETA_C: usize = 14;
/// Potts proxy: phase anomaly score.
pub const C3_POTTS_ANOMALY: usize = 15;
/// Potts proxy: order parameter magnitude.
pub const C4_POTTS_ORDER: usize = 16;
/// Potts proxy: exploration priority.
pub const C5_POTTS_PRIORITY: usize = 17;

// ─── Group D: Steering/Control (action targets, not observable targets) ───

/// Steering: next β value to explore.
pub const D0_NEXT_BETA: usize = 18;
/// Steering: optimal HMC step size dt.
pub const D1_OPTIMAL_DT: usize = 19;
/// Steering: optimal MD integration steps.
pub const D2_OPTIMAL_NMD: usize = 20;
/// Steering: residual check interval.
pub const D3_CHECK_INTERVAL: usize = 21;
/// Steering: kill/abort decision signal.
pub const D4_KILL_DECISION: usize = 22;
/// Steering: skip/fast-forward decision.
pub const D5_SKIP_DECISION: usize = 23;

// ─── Group E: Brain/Monitor (real-time CG residual stream) ───

/// Monitor: estimated remaining CG iterations.
pub const E0_RESIDUAL_ETA: usize = 24;
/// Monitor: residual anomaly (unexpected convergence pattern).
pub const E1_RESIDUAL_ANOMALY: usize = 25;
/// Monitor: convergence rate (residual drop per iteration).
pub const E2_CONVERGENCE_RATE: usize = 26;
/// Monitor: CG stall detector.
pub const E3_STALL_DETECTOR: usize = 27;
/// Monitor: CG divergence detector.
pub const E4_DIVERGENCE_DETECTOR: usize = 28;
/// Monitor: trajectory quality forecast.
pub const E5_QUALITY_FORECAST: usize = 29;

// ─── Group M: Meta-mixer (cross-group agreement, proxy trust) ───

/// Meta: CG cost consensus across groups A/B/C.
pub const M0_CG_CONSENSUS: usize = 30;
/// Meta: phase consensus across groups A/B/C.
pub const M1_PHASE_CONSENSUS: usize = 31;
/// Meta: CG cost uncertainty (spread across groups).
pub const M2_CG_UNCERTAINTY: usize = 32;
/// Meta: phase uncertainty (spread across groups).
pub const M3_PHASE_UNCERTAINTY: usize = 33;
/// Meta: proxy trust score (Anderson/Potts agreement with QCD).
pub const M4_PROXY_TRUST: usize = 34;
/// Meta: attention escalation level.
pub const M5_ATTENTION_LEVEL: usize = 35;

/// Total number of heads (Gen 2).
pub const NUM_HEADS: usize = 36;

/// Number of heads per group.
pub const GROUP_SIZE: usize = 6;

/// Group A (Anderson) base index.
pub const GROUP_A: usize = 0;
/// Group B (QCD) base index.
pub const GROUP_B: usize = 6;
/// Group C (Potts) base index.
pub const GROUP_C: usize = 12;
/// Group D (Steering) base index.
pub const GROUP_D: usize = 18;
/// Group E (Monitor) base index.
pub const GROUP_E: usize = 24;
/// Group M (Meta) base index.
pub const GROUP_M: usize = 30;

/// CG cost heads across groups (for disagreement computation).
pub const CG_COST_HEADS: [usize; 3] = [A0_ANDERSON_CG_COST, B0_QCD_CG_COST, C0_POTTS_CG_COST];
/// Phase heads across groups.
pub const PHASE_HEADS: [usize; 3] = [A1_ANDERSON_PHASE, B1_QCD_PHASE, C1_POTTS_PHASE];
/// Anomaly heads across groups.
pub const ANOMALY_HEADS: [usize; 3] = [A3_ANDERSON_ANOMALY, B3_QCD_ANOMALY, C3_POTTS_ANOMALY];
/// Priority heads across groups.
pub const PRIORITY_HEADS: [usize; 3] = [A5_ANDERSON_PRIORITY, B5_QCD_PRIORITY, C5_POTTS_PRIORITY];

/// Cross-group disagreement signals (Gen 2 "Developed Organism").
///
/// Computed after each `predict_all_heads()` call by comparing overlapping
/// heads across Groups A (Anderson), B (QCD), C (Potts). Large disagreement
/// = high epistemic uncertainty = attention escalation.
#[derive(Debug, Clone, Default)]
pub struct HeadGroupDisagreement {
    /// `max(A0,B0,C0) - min(A0,B0,C0)` — CG cost prediction spread.
    pub delta_cg: f64,
    /// Number of groups disagreeing on phase label (0, 1, 2, or 3).
    pub delta_phase: f64,
    /// `max(A3,B3,C3) - min(A3,B3,C3)` — anomaly score spread.
    pub delta_anomaly: f64,
    /// `max(A5,B5,C5) - min(A5,B5,C5)` — priority score spread.
    pub delta_priority: f64,
}

impl HeadGroupDisagreement {
    /// Compute disagreement from a full head output vector (length >= NUM_HEADS).
    /// Returns default (all zeros) if the output is from a Gen 1 model (< 36 heads).
    #[must_use]
    pub fn from_outputs(outputs: &[f64]) -> Self {
        if outputs.len() < NUM_HEADS {
            return Self::default();
        }
        let spread = |indices: &[usize]| -> f64 {
            let vals: Vec<f64> = indices.iter().map(|&i| outputs[i]).collect();
            let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
            max - min
        };
        let phase_disagree = {
            let labels: Vec<i32> = PHASE_HEADS
                .iter()
                .map(|&i| {
                    if outputs[i] > 0.6 {
                        2
                    } else {
                        i32::from(outputs[i] >= 0.3)
                    }
                })
                .collect();
            let distinct = labels
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len();
            (distinct - 1) as f64
        };
        Self {
            delta_cg: spread(&CG_COST_HEADS),
            delta_phase: phase_disagree,
            delta_anomaly: spread(&ANOMALY_HEADS),
            delta_priority: spread(&PRIORITY_HEADS),
        }
    }

    /// Scalar urgency score for the attention state machine.
    /// 0.0 = full agreement, 1.0 = maximum disagreement.
    #[must_use]
    pub fn urgency(&self) -> f64 {
        (self.delta_cg * 0.4
            + self.delta_phase * 0.3
            + self.delta_anomaly * 0.2
            + self.delta_priority * 0.1)
            .clamp(0.0, 1.0)
    }
}
