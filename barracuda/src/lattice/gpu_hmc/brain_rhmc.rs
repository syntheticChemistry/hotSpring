// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dual-GPU brain-steered RHMC: NPU cortex drives GPU physics parameters.

use crate::error::HotSpringError;
use crate::lattice::rhmc::RhmcConfig;

use super::brain_config as brain_cfg;
use super::unidirectional_cortex::{TrajectoryResult, UnidirectionalRhmc};

use std::sync::mpsc;

#[derive(Debug, Clone)]
pub struct TrajectoryObservation {
    pub gpu_name: String,
    pub traj_idx: usize,
    pub accepted: bool,
    pub delta_h: f64,
    pub plaquette: f64,
    pub total_cg_iters: usize,
    pub elapsed_secs: f64,
    pub beta: f64,
    pub mass: f64,
    pub dt: f64,
    pub n_md_steps: usize,
    pub lattice_size: usize,
    pub silicon_tags: SiliconRoutingTags,
}

#[derive(Debug, Clone, Default)]
pub struct SiliconRoutingTags {
    pub tmu_prng: bool,
    pub subgroup_reduce: bool,
    pub rop_force_accum: bool,
    pub fp64_strategy_id: u8,
    pub has_native_f64: bool,
}

#[derive(Debug)]
pub struct BrainIterationResult {
    pub gpu_a: TrajectoryResult,
    pub gpu_b: TrajectoryResult,
    pub traj_idx: usize,
}

pub struct BrainRhmcRunner {
    observation_tx: mpsc::Sender<TrajectoryObservation>,
    config_live: brain_cfg::RhmcConfigLive,
    traj_count: usize,
    seed_a: u64,
    seed_b: u64,
}

impl BrainRhmcRunner {
    pub fn new(
        base_config: RhmcConfig,
        seed_a: u64,
        seed_b: u64,
    ) -> (
        Self,
        mpsc::Receiver<TrajectoryObservation>,
        mpsc::Sender<brain_cfg::RhmcParamSuggestion>,
    ) {
        let (obs_tx, obs_rx) = mpsc::channel();
        let (sug_tx, sug_rx) = mpsc::channel();

        let runner = Self {
            observation_tx: obs_tx,
            config_live: brain_cfg::RhmcConfigLive::new(base_config, sug_rx),
            traj_count: 0,
            seed_a,
            seed_b,
        };

        (runner, obs_rx, sug_tx)
    }

    pub fn run_iteration(
        &mut self,
        gpu_a: &mut UnidirectionalRhmc,
        gpu_b: &mut UnidirectionalRhmc,
    ) -> Result<BrainIterationResult, HotSpringError> {
        self.config_live.apply_pending(self.traj_count);

        let config_a = self.config_live.config_for_gpu(gpu_a.adapter_name());
        let config_b = self.config_live.config_for_gpu(gpu_b.adapter_name());

        let obs_beta_a = config_a.beta;
        let obs_mass_a = config_a.sectors.first().map_or(0.1, |s| s.mass);
        let obs_dt_a = config_a.dt;
        let obs_nmd_a = config_a.n_md_steps;
        let obs_beta_b = config_b.beta;
        let obs_mass_b = config_b.sectors.first().map_or(0.1, |s| s.mass);
        let obs_dt_b = config_b.dt;
        let obs_nmd_b = config_b.n_md_steps;

        let (result_a, result_b) = std::thread::scope(|scope| {
            let mut sa = self.seed_a;
            let mut sb = self.seed_b;
            let ga = &mut *gpu_a;
            let gb = &mut *gpu_b;

            let handle_a =
                scope.spawn(move || -> Result<(TrajectoryResult, u64), HotSpringError> {
                    let r = ga.run_trajectory(&config_a, &mut sa)?;
                    Ok((r, sa))
                });
            let handle_b =
                scope.spawn(move || -> Result<(TrajectoryResult, u64), HotSpringError> {
                    let r = gb.run_trajectory(&config_b, &mut sb)?;
                    Ok((r, sb))
                });

            let (ra, new_sa) = handle_a.join().map_err(|_| {
                HotSpringError::ThreadPanicked("GPU A trajectory thread panicked")
            })??;
            let (rb, new_sb) = handle_b.join().map_err(|_| {
                HotSpringError::ThreadPanicked("GPU B trajectory thread panicked")
            })??;
            self.seed_a = new_sa;
            self.seed_b = new_sb;
            Ok::<(TrajectoryResult, TrajectoryResult), HotSpringError>((ra, rb))
        })?;

        let lattice_size = gpu_a.state().gauge.gauge.dims[0];

        let obs_a = TrajectoryObservation {
            gpu_name: gpu_a.adapter_name().to_string(),
            traj_idx: self.traj_count,
            accepted: result_a.accepted,
            delta_h: result_a.delta_h,
            plaquette: result_a.plaquette,
            total_cg_iters: result_a.total_cg_iterations,
            elapsed_secs: result_a.elapsed_secs,
            beta: obs_beta_a,
            mass: obs_mass_a,
            dt: obs_dt_a,
            n_md_steps: obs_nmd_a,
            lattice_size,
            silicon_tags: gpu_a.silicon_routing_tags(),
        };
        let obs_b = TrajectoryObservation {
            gpu_name: gpu_b.adapter_name().to_string(),
            traj_idx: self.traj_count,
            accepted: result_b.accepted,
            delta_h: result_b.delta_h,
            plaquette: result_b.plaquette,
            total_cg_iters: result_b.total_cg_iterations,
            elapsed_secs: result_b.elapsed_secs,
            beta: obs_beta_b,
            mass: obs_mass_b,
            dt: obs_dt_b,
            n_md_steps: obs_nmd_b,
            lattice_size,
            silicon_tags: gpu_b.silicon_routing_tags(),
        };

        let _ = self.observation_tx.send(obs_a);
        let _ = self.observation_tx.send(obs_b);

        let idx = self.traj_count;
        self.traj_count += 1;

        Ok(BrainIterationResult {
            gpu_a: result_a,
            gpu_b: result_b,
            traj_idx: idx,
        })
    }

    pub fn traj_count(&self) -> usize {
        self.traj_count
    }

    pub fn config_live(&self) -> &brain_cfg::RhmcConfigLive {
        &self.config_live
    }
}

pub use super::brain_config::{
    AppliedSuggestion, RhmcConfigLive, RhmcParamSuggestion, SuggestionSource,
};
pub use super::brain_cortex::NpuCortex;
pub use super::brain_inference::{
    CrossGpuAgreement, NpuBackend, NpuInference, NpuInferenceMetrics, NpuRunStats,
};
