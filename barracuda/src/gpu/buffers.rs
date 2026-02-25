// SPDX-License-Identifier: AGPL-3.0-only

//! GPU buffer creation, upload, and readback for f64/u32 science data.

use super::GpuF64;

impl GpuF64 {
    /// Create a storage buffer from f64 data (read-only)
    #[must_use]
    pub fn create_f64_buffer(&self, data: &[f64], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// Create a writable storage buffer for f64 output
    #[must_use]
    pub fn create_f64_output_buffer(&self, count: usize, label: &str) -> wgpu::Buffer {
        self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (count * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a staging buffer for reading results back to CPU
    #[must_use]
    pub fn create_staging_buffer(&self, size: usize, label: &str) -> wgpu::Buffer {
        self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a uniform buffer from raw bytes
    #[must_use]
    pub fn create_uniform_buffer(&self, data: &[u8], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Create a storage buffer from u32 data.
    ///
    /// Includes `COPY_DST` so cell-list buffers can be re-uploaded
    /// when the neighbor list is rebuilt on CPU.
    #[must_use]
    pub fn create_u32_buffer(&self, data: &[u32], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let bytes: &[u8] = bytemuck::cast_slice(data);
        self.device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Upload f64 data to a GPU storage buffer (overwrites from offset 0).
    pub fn upload_f64(&self, buffer: &wgpu::Buffer, data: &[f64]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.queue().write_buffer(buffer, 0, &bytes);
    }

    /// Read back f64 data from a GPU buffer via staging copy.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::HotSpringError::DeviceCreation`] if the GPU map
    /// callback fails or the channel is dropped.
    pub fn read_back_f64(
        &self,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f64>, crate::error::HotSpringError> {
        let staging = self.create_staging_buffer(count * 8, "readback");
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 8) as u64);
        self.queue().submit(std::iter::once(encoder.finish()));
        self.read_staging_f64_inner(&staging)
    }

    /// Read f64 data from a staging buffer after submit + poll.
    ///
    /// Call this after [`Self::submit_encoder`] when the encoder included a
    /// `copy_buffer_to_buffer` into the staging buffer.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::HotSpringError::DeviceCreation`] if the GPU map
    /// callback fails or the channel is dropped.
    pub fn read_staging_f64(
        &self,
        staging: &wgpu::Buffer,
    ) -> Result<Vec<f64>, crate::error::HotSpringError> {
        self.read_staging_f64_inner(staging)
    }

    /// Initiate a non-blocking readback from a staging buffer.
    ///
    /// Returns a channel receiver that signals when the map is complete.
    /// Call `device().poll(Maintain::Poll)` to drive progress without blocking,
    /// or `device().poll(Maintain::Wait)` to block.
    pub fn start_async_readback(
        &self,
        staging: &wgpu::Buffer,
    ) -> std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.device().poll(wgpu::Maintain::Poll);
        rx
    }

    /// Complete an async readback: block until ready, read f64 data, unmap.
    pub fn finish_async_readback_f64(
        &self,
        staging: &wgpu::Buffer,
        rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ) -> Result<Vec<f64>, crate::error::HotSpringError> {
        self.device().poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| {
                crate::error::HotSpringError::DeviceCreation(
                    "Async readback: channel recv failed".into(),
                )
            })?
            .map_err(|e| {
                crate::error::HotSpringError::DeviceCreation(format!("Async readback mapping: {e}"))
            })?;
        let data = staging.slice(..).get_mapped_range();
        let result = mapped_bytes_to_f64(&data);
        drop(data);
        staging.unmap();
        Ok(result)
    }

    fn read_staging_f64_inner(
        &self,
        staging: &wgpu::Buffer,
    ) -> Result<Vec<f64>, crate::error::HotSpringError> {
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device().poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| {
                crate::error::HotSpringError::DeviceCreation(
                    "GPU map callback: channel recv failed".into(),
                )
            })?
            .map_err(|e| {
                crate::error::HotSpringError::DeviceCreation(format!("GPU buffer mapping: {e}"))
            })?;

        let data = slice.get_mapped_range();
        let result = mapped_bytes_to_f64(&data);
        drop(data);
        staging.unmap();
        Ok(result)
    }
}

/// Convert mapped GPU buffer bytes to f64 values.
///
/// GPU mapped buffers are typically page-aligned, so `bytemuck::try_cast_slice`
/// will succeed. Falls back to manual byte conversion if alignment is wrong.
pub fn mapped_bytes_to_f64(data: &[u8]) -> Vec<f64> {
    bytemuck::try_cast_slice(data).map_or_else(
        |_| {
            data.chunks_exact(8)
                .map(|chunk| {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(chunk);
                    f64::from_le_bytes(b)
                })
                .collect()
        },
        <[f64]>::to_vec,
    )
}
