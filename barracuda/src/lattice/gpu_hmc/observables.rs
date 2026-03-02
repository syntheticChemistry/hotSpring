// SPDX-License-Identifier: AGPL-3.0-only

//! Three-substrate stream integration: observable streaming and NPU screening.

use super::dynamical::GpuDynHmcResult;
use super::resident_cg::gpu_dynamical_hmc_trajectory_resident;
use super::{
    gpu_polyakov_loop, GpuDynHmcState, GpuDynHmcStreamingPipelines, GpuF64, GpuResidentCgBuffers,
    GpuResidentCgPipelines,
};

/// Observable scalars for the readback stream.
///
/// 8-feature vector for NPU multi-output ESN monitoring:
/// [plaquette, plaquette_var, polyakov_mag, polyakov_phase,
///  action_density, acceptance_rate, delta_h_mag, cg_iterations]
#[derive(Clone, Debug)]
pub struct StreamObservables {
    /// Mean plaquette ⟨P⟩ = Re Tr(U□) / (3·N_plaq).
    pub plaquette: f64,
    /// Real part of the Polyakov loop (order parameter for deconfinement).
    pub polyakov_re: f64,
    /// Hamiltonian change ΔH = H_new − H_old (Metropolis test input).
    pub delta_h: f64,
    /// Number of CG iterations taken this trajectory.
    pub cg_iterations: usize,
    /// Metropolis accept/reject decision (true = accepted).
    pub accepted: bool,
    /// Plaquette variance over recent trajectory window.
    pub plaquette_var: f64,
    /// Polyakov loop phase angle (atan2 of imaginary/real).
    pub polyakov_phase: f64,
    /// Action density S / V_4.
    pub action_density: f64,
}

impl StreamObservables {
    /// Convert to 8-element feature vector for NPU inference.
    ///
    /// Features are ordered to match the multi-output ESN input convention:
    /// `[plaq, plaq_var, poly_mag, poly_phase, action_density, acc_rate, |ΔH|, cg_iter]`
    #[must_use]
    pub fn to_feature_vec(&self, running_acceptance_rate: f64) -> Vec<f64> {
        vec![
            self.plaquette,
            self.plaquette_var,
            self.polyakov_re,
            self.polyakov_phase,
            self.action_density,
            running_acceptance_rate,
            self.delta_h.abs(),
            self.cg_iterations as f64,
        ]
    }
}

impl Default for StreamObservables {
    fn default() -> Self {
        Self {
            plaquette: 0.0,
            polyakov_re: 0.0,
            delta_h: 0.0,
            cg_iterations: 0,
            accepted: false,
            plaquette_var: 0.0,
            polyakov_phase: 0.0,
            action_density: 0.0,
        }
    }
}

/// Bidirectional stream: GPU ↔ CPU with NPU screening branch.
///
/// The GPU runs trajectories continuously. The readback stream (10% bandwidth)
/// carries convergence scalars and observables back to CPU. The CPU makes
/// Metropolis decisions and rebatches parameters to the GPU. A separate
/// NPU branch screens trajectory quality asynchronously.
pub struct BidirectionalStream {
    npu_tx: Option<std::sync::mpsc::Sender<StreamObservables>>,
    npu_rx: Option<std::sync::mpsc::Receiver<bool>>,
    /// Running count of trajectories processed.
    pub trajectories: usize,
    /// Running count of accepted trajectories.
    pub accepted: usize,
    /// Accumulated CG iterations across all trajectories.
    pub total_cg: usize,
    plaquette_history: Vec<f64>,
}

impl BidirectionalStream {
    /// Create a new bidirectional stream (no NPU screening by default).
    #[must_use]
    pub fn new() -> Self {
        Self {
            npu_tx: None,
            npu_rx: None,
            trajectories: 0,
            accepted: 0,
            total_cg: 0,
            plaquette_history: Vec::with_capacity(32),
        }
    }

    /// Attach an NPU screening channel pair.
    pub fn attach_npu(
        &mut self,
        tx: std::sync::mpsc::Sender<StreamObservables>,
        rx: std::sync::mpsc::Receiver<bool>,
    ) {
        self.npu_tx = Some(tx);
        self.npu_rx = Some(rx);
    }

    /// Run one trajectory through the bidirectional stream.
    ///
    /// GPU-resident CG with async readback. Observables are sent to
    /// NPU for screening if attached.
    pub fn run_trajectory(
        &mut self,
        gpu: &GpuF64,
        streaming_pipelines: &GpuDynHmcStreamingPipelines,
        resident_pipelines: &GpuResidentCgPipelines,
        state: &GpuDynHmcState,
        cg_bufs: &GpuResidentCgBuffers,
        n_md_steps: usize,
        dt: f64,
        traj_id: u32,
        seed: &mut u64,
        check_interval: usize,
    ) -> GpuDynHmcResult {
        let result = gpu_dynamical_hmc_trajectory_resident(
            gpu,
            streaming_pipelines,
            resident_pipelines,
            state,
            cg_bufs,
            n_md_steps,
            dt,
            traj_id,
            seed,
            check_interval,
        );

        self.trajectories += 1;
        if result.accepted {
            self.accepted += 1;
        }
        self.total_cg += result.cg_iterations;

        self.plaquette_history.push(result.plaquette);
        if self.plaquette_history.len() > 32 {
            self.plaquette_history.remove(0);
        }

        if let Some(ref tx) = self.npu_tx {
            let (poly_mag, poly_phase) =
                gpu_polyakov_loop(gpu, &streaming_pipelines.dyn_hmc.gauge, &state.gauge);

            let plaq_var = if self.plaquette_history.len() > 1 {
                let mean = self.plaquette_history.iter().sum::<f64>()
                    / self.plaquette_history.len() as f64;
                self.plaquette_history
                    .iter()
                    .map(|p| (p - mean).powi(2))
                    .sum::<f64>()
                    / (self.plaquette_history.len() - 1) as f64
            } else {
                0.0
            };

            let obs = StreamObservables {
                plaquette: result.plaquette,
                polyakov_re: poly_mag,
                delta_h: result.delta_h,
                cg_iterations: result.cg_iterations,
                accepted: result.accepted,
                plaquette_var: plaq_var,
                polyakov_phase: poly_phase,
                action_density: 6.0 * (1.0 - result.plaquette),
            };
            let _ = tx.send(obs);
        }

        if let Some(ref rx) = self.npu_rx {
            if let Ok(_skip) = rx.try_recv() {
                // NPU screening can influence future trajectory scheduling
            }
        }

        result
    }

    /// Acceptance rate so far.
    pub fn acceptance_rate(&self) -> f64 {
        if self.trajectories == 0 {
            0.0
        } else {
            self.accepted as f64 / self.trajectories as f64
        }
    }
}

impl Default for BidirectionalStream {
    fn default() -> Self {
        Self::new()
    }
}
