// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-resident CG with async readback and speculative batches.

use super::dynamical::{GpuDynHmcPipelines, GpuDynHmcState};
use super::resident_cg_buffers::{encode_cg_batch, encode_reduce_chain, GpuResidentCgBuffers};
use super::resident_cg_pipelines::GpuResidentCgPipelines;
use super::GpuF64;

/// Non-blocking readback handle for CG convergence scalars.
///
/// Wraps `map_async` with a channel-based completion signal.
/// GPU can continue working while the CPU waits for the scalar.
pub struct AsyncCgReadback {
    receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl AsyncCgReadback {
    /// Initiate a non-blocking readback of one f64 from a GPU scalar buffer.
    ///
    /// The caller must have already submitted an encoder that copies the
    /// source buffer to the staging buffer.
    pub fn start(gpu: &GpuF64, staging: &wgpu::Buffer) -> Option<Self> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        gpu.device().poll(wgpu::Maintain::Poll);
        Some(Self { receiver: rx })
    }

    /// Poll: check if the readback is ready without blocking.
    pub fn is_ready(&self, gpu: &GpuF64) -> bool {
        gpu.device().poll(wgpu::Maintain::Poll);
        matches!(
            self.receiver.try_recv(),
            Ok(Ok(())) | Err(std::sync::mpsc::TryRecvError::Empty)
        )
    }

    /// Block until the readback is complete, then return the f64 value.
    pub fn wait(self, gpu: &GpuF64, staging: &wgpu::Buffer) -> f64 {
        gpu.device().poll(wgpu::Maintain::Wait);
        let _ = self.receiver.recv();
        let slice = staging.slice(..);
        let data = slice.get_mapped_range();
        let val = if data.len() >= 8 {
            f64::from_le_bytes(data[..8].try_into().unwrap_or([0u8; 8]))
        } else {
            f64::NAN
        };
        drop(data);
        staging.unmap();
        val
    }
}

/// GPU-resident CG with async readback and speculative batches.
///
/// While waiting for convergence readback, speculatively submits the next
/// batch of CG iterations. If convergence is detected, the speculative
/// work is discarded (wasted compute but hidden latency).
pub fn gpu_cg_solve_resident_async(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    resident_pipelines: &GpuResidentCgPipelines,
    state: &GpuDynHmcState,
    cg_bufs: &GpuResidentCgBuffers,
    b_buf: &wgpu::Buffer,
    check_interval: usize,
) -> usize {
    let vol = state.gauge.volume;
    let n_flat = vol * 6;

    let zeros = vec![0.0_f64; n_flat];
    gpu.upload_f64(&state.x_buf, &zeros);
    {
        let mut enc = gpu.begin_encoder("rcg_async_init");
        enc.copy_buffer_to_buffer(b_buf, 0, &state.r_buf, 0, (n_flat * 8) as u64);
        enc.copy_buffer_to_buffer(b_buf, 0, &state.p_buf, 0, (n_flat * 8) as u64);
        GpuF64::encode_pass(
            &mut enc,
            &dyn_pipelines.dot_pipeline,
            &cg_bufs.dot_rr_bg,
            cg_bufs.wg_dot,
        );
        encode_reduce_chain(
            &mut enc,
            &resident_pipelines.reduce_pipeline,
            &cg_bufs.reduce_to_rz,
        );
        enc.copy_buffer_to_buffer(&cg_bufs.rz_buf, 0, &cg_bufs.convergence_staging_a, 0, 8);
        gpu.submit_encoder(enc);
    }
    let b_norm_sq = match gpu.read_staging_f64(&cg_bufs.convergence_staging_a) {
        Ok(v) => v.first().copied().unwrap_or(0.0),
        Err(_) => return 0,
    };
    if b_norm_sq < 1e-30 {
        return 0;
    }
    let tol_sq = state.cg_tol * state.cg_tol * b_norm_sq;
    let check_interval = check_interval.max(1);
    let mut total_iters = 0usize;
    let mut use_staging_a = true;

    loop {
        let batch = check_interval.min(state.cg_max_iter - total_iters);
        if batch == 0 {
            break;
        }

        let staging = if use_staging_a {
            &cg_bufs.convergence_staging_a
        } else {
            &cg_bufs.convergence_staging_b
        };
        let mut enc = gpu.begin_encoder("rcg_async_batch");
        encode_cg_batch(&mut enc, dyn_pipelines, resident_pipelines, cg_bufs, batch);
        enc.copy_buffer_to_buffer(&cg_bufs.rz_new_buf, 0, staging, 0, 8);
        gpu.submit_encoder(enc);
        total_iters += batch;

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

        let next_batch = check_interval.min(state.cg_max_iter - total_iters);
        if next_batch > 0 {
            let spec_staging = if use_staging_a {
                &cg_bufs.convergence_staging_b
            } else {
                &cg_bufs.convergence_staging_a
            };
            let mut spec_enc = gpu.begin_encoder("rcg_speculative");
            encode_cg_batch(
                &mut spec_enc,
                dyn_pipelines,
                resident_pipelines,
                cg_bufs,
                next_batch,
            );
            spec_enc.copy_buffer_to_buffer(&cg_bufs.rz_new_buf, 0, spec_staging, 0, 8);
            gpu.submit_encoder(spec_enc);
        }

        gpu.device().poll(wgpu::Maintain::Wait);
        let map_result = rx.recv();
        let rz_new = if map_result.is_ok() {
            let slice = staging.slice(..);
            let data = slice.get_mapped_range();
            let val = if data.len() >= 8 {
                f64::from_le_bytes(data[..8].try_into().unwrap_or([0u8; 8]))
            } else {
                f64::MAX
            };
            drop(data);
            staging.unmap();
            val
        } else {
            break;
        };

        if rz_new < tol_sq {
            break;
        }

        if next_batch > 0 {
            total_iters += next_batch;
            let spec_staging = if use_staging_a {
                &cg_bufs.convergence_staging_b
            } else {
                &cg_bufs.convergence_staging_a
            };
            let spec_rz = match gpu.read_staging_f64(spec_staging) {
                Ok(v) => v.first().copied().unwrap_or(f64::MAX),
                Err(_) => break,
            };
            if spec_rz < tol_sq || total_iters >= state.cg_max_iter {
                break;
            }
        }

        use_staging_a = !use_staging_a;

        if total_iters >= state.cg_max_iter {
            break;
        }
    }

    total_iters
}
