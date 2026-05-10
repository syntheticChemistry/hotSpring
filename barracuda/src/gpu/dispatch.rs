// SPDX-License-Identifier: AGPL-3.0-or-later

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
pub const fn split_workgroups(total: u32) -> (u32, u32, u32) {
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
    #[must_use]
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

// ── Pipeline creation ────────────────────────────────────────────────────────

use barracuda::shaders::precision::ShaderTemplate;
use log::{debug, error, warn};

impl GpuF64 {
    /// Create a compute pipeline with `WgslOptimizer` + driver-aware patching.
    ///
    /// Does NOT apply exp/log workarounds — use [`Self::create_pipeline_f64`]
    /// for shaders that call `exp()` or `log()` on f64 values.
    #[must_use]
    pub fn create_pipeline(&self, shader_source: &str, label: &str) -> wgpu::ComputePipeline {
        let optimized = ShaderTemplate::for_driver_auto(shader_source, false);
        self.build_pipeline_inner(&optimized, "main", label)
    }

    /// Create a compute pipeline with driver-aware f64 patching + sovereign compilation.
    #[must_use]
    pub fn create_pipeline_f64(&self, shader_source: &str, label: &str) -> wgpu::ComputePipeline {
        if self.full_df64_mode {
            return self.compile_full_df64_pipeline(shader_source, label);
        }
        let shader_module = self
            .wgpu_device
            .compile_shader_f64(shader_source, Some(label));
        self.validate_pipeline_inner(shader_module, "main", label)
    }

    /// DF64 pipeline for hand-written DF64 shaders (Hybrid mode).
    #[must_use]
    pub fn create_pipeline_df64(&self, shader_source: &str, label: &str) -> wgpu::ComputePipeline {
        let shader_module = self
            .wgpu_device
            .compile_shader_df64(shader_source, Some(label));
        self.validate_pipeline_inner(shader_module, "main", label)
    }

    /// Precision-preserving f64 pipeline (no FMA fusion).
    #[must_use]
    pub fn create_pipeline_f64_precise(
        &self,
        shader_source: &str,
        label: &str,
    ) -> wgpu::ComputePipeline {
        if self.full_df64_mode {
            return self.compile_full_df64_pipeline(shader_source, label);
        }
        let optimized = ShaderTemplate::for_driver_auto(
            shader_source,
            self.wgpu_device.needs_f64_exp_log_workaround(),
        );
        self.build_pipeline_inner(&optimized, "main", label)
    }

    /// Full DF64 pipeline: delegates to barraCuda's `compile_shader_universal`.
    pub fn compile_full_df64_pipeline(
        &self,
        shader_source: &str,
        label: &str,
    ) -> wgpu::ComputePipeline {
        let shader_module = self
            .wgpu_device
            .compile_shader_df64(shader_source, Some(label));
        self.validate_pipeline_inner(shader_module, "main", label)
    }

    /// Precise f64 pipeline with a named entry point (no FMA fusion).
    #[must_use]
    pub fn create_pipeline_f64_entry_precise(
        &self,
        shader_source: &str,
        entry_point: &str,
        label: &str,
    ) -> wgpu::ComputePipeline {
        if self.full_df64_mode {
            return self.compile_full_df64_pipeline(shader_source, label);
        }
        let optimized = ShaderTemplate::for_driver_auto(
            shader_source,
            self.wgpu_device.needs_f64_exp_log_workaround(),
        );
        self.build_pipeline_inner(&optimized, entry_point, label)
    }

    /// f64 pipeline with a named entry point.
    #[must_use]
    pub fn create_pipeline_f64_entry(
        &self,
        shader_source: &str,
        entry_point: &str,
        label: &str,
    ) -> wgpu::ComputePipeline {
        if self.full_df64_mode {
            return self.compile_full_df64_pipeline(shader_source, label);
        }
        let shader_module = self
            .wgpu_device
            .compile_shader_f64(shader_source, Some(label));
        self.validate_pipeline_inner(shader_module, entry_point, label)
    }

    /// Shared pipeline builder: validated path (sovereign / compiled shader module).
    ///
    /// Consolidates the former `validate_pipeline` + `validate_pipeline_entry` pair
    /// into a single function parameterised on `entry_point`.
    fn validate_pipeline_inner(
        &self,
        shader_module: wgpu::ShaderModule,
        entry_point: &str,
        label: &str,
    ) -> wgpu::ComputePipeline {
        let t0 = std::time::Instant::now();
        let gpu_tag = &self.adapter_name;
        let scope = self
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &shader_module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        use std::future::Future;
        let mut fut = std::pin::pin!(scope.pop());
        let _ = self.device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        match fut.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(Some(e)) => {
                error!("[pipeline:{gpu_tag}] {label}: PIPELINE ERROR: {e}");
            }
            std::task::Poll::Ready(None) => {
                debug!(
                    "[pipeline:{gpu_tag}] {label}: pipeline valid ({:?})",
                    t0.elapsed()
                );
            }
            std::task::Poll::Pending => {
                warn!(
                    "[pipeline:{gpu_tag}] {label}: pipeline status pending ({:?})",
                    t0.elapsed()
                );
            }
        }
        pipeline
    }

    /// Shared pipeline builder: WGSL-text path (skips sovereign compilation).
    ///
    /// Consolidates the former `build_pipeline` + `build_pipeline_entry` pair.
    fn build_pipeline_inner(&self, wgsl: &str, entry_point: &str, label: &str) -> wgpu::ComputePipeline {
        let shader_module = self
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        self.device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &shader_module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
    }
}
