// SPDX-License-Identifier: AGPL-3.0-only

//! Titan V pre-motor worker for quenched pre-thermalization.
//!
//! Runs quenched HMC on a secondary GPU (Titan V) in a background thread
//! to warm configurations for the next β point while the primary GPU
//! runs dynamical HMC. Avoids concurrent `wgpu::Instance` creation
//! which can deadlock the NVK/nouveau kernel driver.

use crate::gpu::GpuF64;
use crate::lattice::gpu_hmc::{
    flatten_links, gpu_hmc_trajectory_streaming, gpu_links_to_lattice, GpuHmcState,
    GpuHmcStreamingPipelines,
};
use crate::lattice::hmc::{self, HmcConfig, IntegratorType};
use crate::lattice::wilson::Lattice;
use std::sync::mpsc;
use std::time::Instant;

/// Request sent to the Titan V pre-motor thread.
pub enum TitanRequest {
    /// Run quenched pre-thermalization for a given β.
    PreThermalize {
        /// Inverse coupling β.
        beta: f64,
        #[allow(dead_code)]
        /// Fermion mass (unused in quenched).
        mass: f64,
        /// Lattice size (one dimension of L⁴).
        lattice: usize,
        /// Number of quenched trajectories.
        n_quenched: usize,
        /// RNG seed.
        seed: u64,
        /// MD step size.
        dt: f64,
        /// MD steps per trajectory.
        n_md: usize,
    },
    /// Shutdown the worker.
    Shutdown,
}

/// Response from the Titan V pre-motor thread.
pub enum TitanResponse {
    /// Warm configuration ready.
    WarmConfig {
        /// β value.
        beta: f64,
        /// Flattened gauge links.
        gauge_links: Vec<f64>,
        /// Average plaquette.
        plaquette: f64,
        /// Wall time in ms.
        wall_ms: f64,
    },
}

/// Handles for communicating with the Titan V worker thread.
pub struct TitanWorkerHandles {
    /// Send requests to the worker.
    pub titan_tx: mpsc::Sender<TitanRequest>,
    /// Receive responses from the worker.
    pub titan_rx: mpsc::Receiver<TitanResponse>,
}

/// Spawn the Titan V pre-motor worker.
///
/// Create the GPU on the *calling* thread, then move it into the worker.
/// This avoids concurrent `wgpu::Instance` creation which can deadlock
/// the NVK/nouveau kernel driver when two GPUs are opened from separate
/// threads simultaneously.
#[allow(clippy::expect_used)] // thread spawn failure is fatal (OOM, resource exhaustion)
pub fn spawn_titan_worker(titan_gpu: GpuF64) -> TitanWorkerHandles {
    let (req_tx, req_rx) = mpsc::channel::<TitanRequest>();
    let (resp_tx, resp_rx) = mpsc::channel::<TitanResponse>();

    let builder = std::thread::Builder::new().name("titan-premotor".into());
    builder
        .spawn(move || {
            let quenched_pipelines = GpuHmcStreamingPipelines::new(&titan_gpu);

            for req in req_rx {
                match req {
                    TitanRequest::PreThermalize {
                        beta,
                        mass: _,
                        lattice,
                        n_quenched,
                        seed,
                        dt,
                        n_md,
                    } => {
                        let t0 = Instant::now();
                        let dims = [lattice, lattice, lattice, lattice];
                        let mut lat = Lattice::hot_start(dims, beta, seed);

                        let mut cfg = HmcConfig {
                            n_md_steps: n_md,
                            dt,
                            seed: seed * 100,
                            integrator: IntegratorType::Omelyan,
                        };
                        for _ in 0..5 {
                            hmc::hmc_trajectory(&mut lat, &mut cfg);
                        }

                        let state = GpuHmcState::from_lattice(&titan_gpu, &lat, beta);
                        let mut titan_seed = seed * 200;

                        for i in 0..n_quenched {
                            let _ = gpu_hmc_trajectory_streaming(
                                &titan_gpu,
                                &quenched_pipelines,
                                &state,
                                n_md,
                                dt,
                                i as u32,
                                &mut titan_seed,
                            );
                        }

                        gpu_links_to_lattice(&titan_gpu, &state, &mut lat);
                        let plaq = lat.average_plaquette();
                        let links = flatten_links(&lat);
                        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

                        eprintln!(
                            "  [Titan] Pre-therm β={beta:.4}: {n_quenched} quenched trajs, P={plaq:.6}, {wall_ms:.0}ms"
                        );

                        resp_tx
                            .send(TitanResponse::WarmConfig {
                                beta,
                                gauge_links: links,
                                plaquette: plaq,
                                wall_ms,
                            })
                            .ok();
                    }
                    TitanRequest::Shutdown => break,
                }
            }
        })
        .expect("spawn titan-premotor thread");

    TitanWorkerHandles {
        titan_tx: req_tx,
        titan_rx: resp_rx,
    }
}
