// SPDX-License-Identifier: AGPL-3.0-or-later

//! HMC trajectory runner using barraCuda's `GpuHmcTrajectory`.
//!
//! Orchestrates warmup, production, and adaptive step-size tuning.
//! All GPU physics is delegated — this module only handles scheduling,
//! result interpretation, gossip event emission, and toadStool
//! performance surface reporting.

use super::NodeAtomicQcd;

/// Result of a warmup or production campaign segment.
pub struct CampaignSegmentResult {
    pub trajectories: usize,
    pub accepted: usize,
    pub final_plaquette: f64,
    pub mean_delta_h: f64,
    /// Average wall-clock milliseconds per trajectory.
    pub ms_per_trajectory: f64,
    /// FP64 strategy used by the underlying GpuHmcTrajectory.
    pub fp64_strategy: String,
}

/// Trajectory runner for HMC campaigns with gossip integration.
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
    /// Run a full campaign: warmup + production with gossip events.
    ///
    /// Emits `campaign_started`, periodic `campaign_progress`, `config_complete`,
    /// and `campaign_complete` gossip events via swarmVine.
    ///
    /// # Errors
    ///
    /// Returns an error if any GPU dispatch fails.
    pub fn run_campaign(
        &self,
        qcd: &NodeAtomicQcd,
        warmup_report_every: usize,
        mut on_warmup_report: impl FnMut(usize, f64, f64),
    ) -> Result<(CampaignSegmentResult, Vec<f64>), barracuda::error::BarracudaError> {
        let volume_str = format!("{}x{}", qcd.config.nx, qcd.config.nt);
        let total = self.warmup_count + self.production_count;
        let t0 = std::time::Instant::now();

        crate::gossip::campaign_started(&volume_str, qcd.config.beta, total);

        let warmup = self.run_warmup(qcd, warmup_report_every, &mut on_warmup_report)?;

        crate::gossip::campaign_progress(self.warmup_count, total, &volume_str);

        let mut measurements = Vec::with_capacity(self.production_count);
        let production = self.run_production(qcd, &mut measurements)?;

        let acc_rate = production.accepted as f64 / production.trajectories as f64;
        crate::gossip::config_complete(
            &volume_str,
            qcd.config.beta,
            0,
            production.final_plaquette,
            acc_rate,
        );

        let wall_hours = t0.elapsed().as_secs_f64() / 3600.0;
        crate::gossip::campaign_complete(total, wall_hours);

        Ok((production, measurements))
    }

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
        let t0 = std::time::Instant::now();

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

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ms_per_traj = if self.warmup_count > 0 {
            elapsed_ms / self.warmup_count as f64
        } else {
            0.0
        };

        let strategy = format!("{:?}", qcd.trajectory.strategy());

        Ok(CampaignSegmentResult {
            trajectories: self.warmup_count,
            accepted,
            final_plaquette: last_plaquette,
            mean_delta_h: sum_delta_h / self.warmup_count.max(1) as f64,
            ms_per_trajectory: ms_per_traj,
            fp64_strategy: strategy,
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
        let t0 = std::time::Instant::now();

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

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ms_per_traj = if self.production_count > 0 {
            elapsed_ms / self.production_count as f64
        } else {
            0.0
        };

        let strategy = format!("{:?}", qcd.trajectory.strategy());

        Ok(CampaignSegmentResult {
            trajectories: self.production_count,
            accepted,
            final_plaquette: last_plaquette,
            mean_delta_h: sum_delta_h / self.production_count.max(1) as f64,
            ms_per_trajectory: ms_per_traj,
            fp64_strategy: strategy,
        })
    }
}
