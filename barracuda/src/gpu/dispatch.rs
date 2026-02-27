// SPDX-License-Identifier: AGPL-3.0-only

//! GPU dispatch and encoder management.
//!
//! Streaming dispatch pattern: pre-plan GPU work, submit as few command
//! buffers as possible, read back only at control points.
//!
//! ```text
//! begin_encoder()  → CommandEncoder
//!   ↕  encode N dispatches via compute passes
//! submit_encoder() → ONE GPU submission
//! read_staging_f64() → read back results
//! ```

use super::GpuF64;

/// Split workgroup count into (x, y, 1) for 2D dispatch when x > 65535.
/// Shaders must linearize via `gid.x + gid.y * num_workgroups.x * WG_SIZE`.
pub fn split_workgroups(total: u32) -> (u32, u32, u32) {
    if total <= 65535 {
        (total, 1, 1)
    } else {
        let y = total.div_ceil(65535);
        let x = total.div_ceil(y);
        (x, y, 1)
    }
}

impl GpuF64 {
    /// Create a bind group from a pipeline and ordered buffer slice.
    ///
    /// Each buffer is bound at binding index 0, 1, 2, ... in order.
    pub fn create_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf): (usize, &&wgpu::Buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group"),
            layout: &layout,
            entries: &entries,
        })
    }

    /// Dispatch a compute pipeline (single-shot submit — convenience only).
    ///
    /// **Prefer [`Self::begin_encoder`] + [`Self::submit_encoder`]** for MD loops
    /// or any multi-dispatch sequence. This method creates a new encoder and
    /// submits per call — one GPU round-trip per invocation.
    pub fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: u32,
    ) {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            let (wx, wy, wz) = split_workgroups(workgroups);
            pass.dispatch_workgroups(wx, wy, wz);
        }
        self.queue().submit(std::iter::once(encoder.finish()));
    }

    /// Begin a command encoder for streaming multiple dispatches.
    ///
    /// Encode as many compute passes / dispatches as needed, then call
    /// [`Self::submit_encoder`] to issue a single GPU submission.
    #[must_use]
    pub fn begin_encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    /// Submit a finished encoder to the GPU queue (single submission).
    pub fn submit_encoder(&self, encoder: wgpu::CommandEncoder) {
        self.queue().submit(std::iter::once(encoder.finish()));
    }

    /// Encode a compute pass into an existing encoder (no submit).
    ///
    /// Use with [`Self::begin_encoder`] to batch many dispatches into a single
    /// GPU submission, eliminating per-dispatch overhead.
    pub fn encode_pass(
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("streaming_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let (wx, wy, wz) = split_workgroups(workgroups);
        pass.dispatch_workgroups(wx, wy, wz);
    }

    /// Dispatch a compute pipeline and read back f64 results.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::HotSpringError::DeviceCreation`] if the GPU map
    /// callback fails or the channel is dropped.
    pub fn dispatch_and_read(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: u32,
        output_buffer: &wgpu::Buffer,
        output_count: usize,
    ) -> Result<Vec<f64>, crate::error::HotSpringError> {
        let staging = self.create_staging_buffer(output_count * 8, "staging");

        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            let (wx, wy, wz) = split_workgroups(workgroups);
            pass.dispatch_workgroups(wx, wy, wz);
        }

        encoder.copy_buffer_to_buffer(output_buffer, 0, &staging, 0, (output_count * 8) as u64);
        self.queue().submit(std::iter::once(encoder.finish()));

        self.read_staging_f64(&staging)
    }
}
