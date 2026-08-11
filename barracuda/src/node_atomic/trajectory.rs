// SPDX-License-Identifier: AGPL-3.0-or-later

//! HMC trajectory runner using barraCuda's `GpuHmcTrajectory`.
//!
//! Orchestrates warmup, production, and adaptive step-size tuning.
//! All GPU physics is delegated — this module only handles scheduling
//! and result interpretation.

use barracuda::ops::lattice::gpu_hmc_types::GpuHmcResult;

use super::NodeAtomicQcd;

/// Result of a warmup or production campaign segment.
pub struct CampaignSegmentResult {
    pub trajectories: usize,
    pub accepted: usize,
    pub final_plaquette: f64,
    pub mean_delta_h: f64,
}

/// Trajectory runner for HMC campaigns.
pub struct TrajectoryRunner {
    pub warmup_count: usize,
    pub production_count: usize,
    pub target_acceptance: f64,
}

impl Default for TrajectoryRunner {
    fn default() -> Self {
        Self {
            warmup_count: 500,
            production_count: 200,
            target_acceptance: 0.70,
        }
    }
}

impl TrajectoryRunner {
    /// Run warmup trajectories, returning the segment result.
    ///
    /// # Errors
    ///
    /// Returns an error if any GPU dispatch fails.
    pub fn run_warmup(
        &self,
        qcd: &NodeAtomicQcd,
        report_every: usize,
        mut on_report: impl FnMut(usize, f64, f64),
    ) -> Result<CampaignSegmentResult, barracuda::error::BarracudaError> {
        let mut accepted = 0usize;
        let mut last_plaquette = 0.0;
        let mut sum_delta_h = 0.0;
        let beta = qcd.config.beta;
        let volume = qcd.volume();

        for i in 0..self.warmup_count {
            let result = qcd.run_trajectory()?;
            if result.accepted {
                accepted += 1;
            }
            sum_delta_h += result.delta_h.abs();

            let plaq = 1.0 - result.gauge_action / (6.0 * volume as f64 * beta);
            last_plaquette = plaq;

            if report_every > 0 && (i + 1) % report_every == 0 {
                let acc_rate = accepted as f64 / (i + 1) as f64;
                on_report(i + 1, plaq, acc_rate);
            }
        }

        Ok(CampaignSegmentResult {
            trajectories: self.warmup_count,
            accepted,
            final_plaquette: last_plaquette,
            mean_delta_h: sum_delta_h / self.warmup_count as f64,
        })
    }

    /// Run production trajectories, collecting plaquette measurements.
    ///
    /// # Errors
    ///
    /// Returns an error if any GPU dispatch fails.
    pub fn run_production(
        &self,
        qcd: &NodeAtomicQcd,
        measurements: &mut Vec<f64>,
    ) -> Result<CampaignSegmentResult, barracuda::error::BarracudaError> {
        let mut accepted = 0usize;
        let mut last_plaquette = 0.0;
        let mut sum_delta_h = 0.0;
        let beta = qcd.config.beta;
        let volume = qcd.volume();

        for _ in 0..self.production_count {
            let result = qcd.run_trajectory()?;
            if result.accepted {
                accepted += 1;
            }
            sum_delta_h += result.delta_h.abs();

            let plaq = 1.0 - result.gauge_action / (6.0 * volume as f64 * beta);
            last_plaquette = plaq;
            measurements.push(plaq);
        }

        Ok(CampaignSegmentResult {
            trajectories: self.production_count,
            accepted,
            final_plaquette: last_plaquette,
            mean_delta_h: sum_delta_h / self.production_count as f64,
        })
    }
}
