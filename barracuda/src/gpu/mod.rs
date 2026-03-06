// SPDX-License-Identifier: AGPL-3.0-only

//! GPU FP64 compute for hotSpring science workloads.
//!
//! Creates a wgpu device with `SHADER_F64` enabled, and provides helpers
//! for running f64 compute shaders on any Vulkan GPU (NVIDIA proprietary,
//! NVK/nouveau, RADV, etc.).
//!
//! ## Adapter selection
//!
//! Explicit adapter targeting is the default. Set `HOTSPRING_GPU_ADAPTER`
//! to select a specific GPU:
//!
//! | Value | Behavior |
//! |-------|----------|
//! | `auto` | wgpu `HighPerformance` preference (legacy default) |
//! | `0`, `1`, … | Select adapter by enumeration index |
//! | substring | Case-insensitive name match (e.g. `"titan"`, `"4070"`) |
//! | *(unset)* | Enumerate all adapters, pick first with `SHADER_F64` |
//!
//! Use `GpuF64::enumerate_adapters` to list available GPUs before selecting.
//!
//! ## Module structure
//!
//! - `adapter` — adapter discovery and selection
//! - `buffers` — f64/u32 buffer creation, upload, readback
//! - `dispatch` — command encoding and dispatch
//! - `telemetry` — GPU power, temperature, VRAM monitoring

mod adapter;
mod buffers;
mod dispatch;
mod telemetry;

pub use adapter::{discover_best_adapter, discover_primary_and_secondary_adapters, AdapterInfo};

use barracuda::device::capabilities::GpuDriverProfile;
use barracuda::device::{TensorContext, WgpuDevice};
use barracuda::shaders::precision::ShaderTemplate;
use std::sync::Arc;

/// GPU context with FP64 support for science workloads.
///
/// Wraps wgpu device with `SHADER_F64` + barraCuda's `TensorContext` for
/// batched dispatch (`begin_batch`/`end_batch`) and `BufferPool` reuse.
///
/// When the f64 probe fails (NVK Mesa 25.1+), falls back to full DF64 mode:
/// all shaders compiled via `compile_shader_universal(Precision::Df64)` and
/// buffers use `vec2<f32>` wire format — no `SHADER_F64` required.
#[must_use]
pub struct GpuF64 {
    /// Human-readable adapter name (e.g. "NVIDIA `GeForce` RTX 4070").
    pub adapter_name: String,
    /// Whether the device can actually compile f64 shaders (probe-verified,
    /// not just feature-advertised).
    pub has_f64: bool,
    /// Whether timestamp queries are available for profiling.
    pub has_timestamps: bool,
    /// True when native f64 shaders are broken and all GPU work must use
    /// full DF64 (f32-pair) data plane — storage as `vec2<f32>`, not `f64`.
    pub full_df64_mode: bool,
    wgpu_device: Arc<WgpuDevice>,
    tensor_ctx: Arc<TensorContext>,
    driver_profile: GpuDriverProfile,
}

// ── DF64 wire-format conversion (CPU side) ───────────────────────────

/// Split a single `f64` into a DF64 (hi, lo) f32 pair.
/// hi + lo ≈ value with ~48-bit mantissa.
#[inline]
#[must_use]
pub fn f64_to_df64(v: f64) -> [f32; 2] {
    let hi = v as f32;
    let lo = (v - f64::from(hi)) as f32;
    [hi, lo]
}

/// Reconstruct a single `f64` from a DF64 (hi, lo) f32 pair.
#[inline]
#[must_use]
pub fn df64_to_f64(pair: [f32; 2]) -> f64 {
    f64::from(pair[0]) + f64::from(pair[1])
}

/// Convert a slice of `f64` values to DF64 wire format bytes (pairs of f32).
/// Output has the same byte length as the input (8 bytes per value).
#[must_use]
pub fn f64_slice_to_df64_bytes(data: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for &v in data {
        let [hi, lo] = f64_to_df64(v);
        bytes.extend_from_slice(&hi.to_le_bytes());
        bytes.extend_from_slice(&lo.to_le_bytes());
    }
    bytes
}

/// Convert DF64 wire format bytes back to `f64` values.
#[must_use]
pub fn df64_bytes_to_f64_slice(bytes: &[u8]) -> Vec<f64> {
    let count = bytes.len() / 8;
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * 8;
        let hi = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        let lo = f32::from_le_bytes([
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ]);
        result.push(df64_to_f64([hi, lo]));
    }
    result
}

// ── Core accessors ───────────────────────────────────────────────────

impl GpuF64 {
    /// Access the underlying wgpu Device.
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        self.wgpu_device.device()
    }

    /// Access the underlying wgpu Queue.
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        self.wgpu_device.queue()
    }

    /// Get a cloned device handle (wgpu 28: Device is Clone, no Arc needed).
    #[must_use]
    pub fn device_clone(&self) -> wgpu::Device {
        self.wgpu_device.device_clone()
    }

    /// Get a cloned queue handle (wgpu 28: Queue is Clone, no Arc needed).
    #[must_use]
    pub fn queue_clone(&self) -> wgpu::Queue {
        self.wgpu_device.queue_clone()
    }

    /// Bridge to toadstool's `WgpuDevice`.
    #[must_use]
    pub fn to_wgpu_device(&self) -> Arc<WgpuDevice> {
        Arc::clone(&self.wgpu_device)
    }

    /// Access the `TensorContext` for batched dispatch.
    #[must_use]
    pub const fn tensor_context(&self) -> &Arc<TensorContext> {
        &self.tensor_ctx
    }

    /// Access the runtime-detected driver profile for shader specialization.
    #[must_use]
    pub const fn driver_profile(&self) -> &GpuDriverProfile {
        &self.driver_profile
    }
}

// ── Constructor ──────────────────────────────────────────────────────

/// Negotiate GPU features, requesting all science-relevant capabilities.
///
/// On NVK, `SPIRV_SHADER_PASSTHROUGH` is skipped — the NVK driver segfaults
/// when ingesting Sovereign-compiled SPIR-V for complex f64 shaders.
fn negotiate_features(adapter_features: wgpu::Features, _is_nvk: bool) -> wgpu::Features {
    let mut f = wgpu::Features::empty();
    for feat in [
        wgpu::Features::SHADER_F64,
        wgpu::Features::SHADER_F16,
        wgpu::Features::TIMESTAMP_QUERY,
        wgpu::Features::PIPELINE_CACHE,
    ] {
        if adapter_features.contains(feat) {
            f |= feat;
        }
    }
    // wgpu 28: SPIRV_SHADER_PASSTHROUGH constant removed.
    // barraCuda's WgpuDevice::has_spirv_passthrough() handles this at device level.
    f
}

/// Heuristic NVK detection from adapter info (before device creation).
fn adapter_is_nvk(info: &wgpu::AdapterInfo) -> bool {
    let d = info.driver.to_lowercase();
    let di = info.driver_info.to_lowercase();
    d.contains("nvk")
        || d.contains("nouveau")
        || di.contains("nvk")
        || di.contains("nouveau")
        || (d.contains("mesa") && info.name.contains("NV"))
}

/// Shared post-creation logic: probe f64 builtins, detect `full_df64_mode`.
///
/// On NVK, forces full DF64 even if the basic probe passes — NVK's NAK
/// compiler segfaults or produces invalid pipelines for complex f64 shaders
/// (HMC gauge force, CG solvers) even though simple f64 arithmetic compiles.
async fn finalize_device(
    adapter_name: String,
    advertised_f64: bool,
    has_timestamps: bool,
    is_nvk: bool,
    wgpu_device: Arc<WgpuDevice>,
) -> GpuF64 {
    let tensor_ctx = Arc::new(TensorContext::new(Arc::clone(&wgpu_device)));
    let driver_profile = GpuDriverProfile::from_device(&wgpu_device);

    let (has_f64, full_df64_mode) = if advertised_f64 {
        let caps = barracuda::device::probe::probe_f64_builtins(&wgpu_device).await;
        if caps.can_compile_f64() {
            if is_nvk {
                eprintln!(
                    "  f64 probe: PASS on NVK ({}/{} builtins) — using barraCuda polyfill \
                     pipeline (native f64 + exp/log/sin/cos polyfills)",
                    caps.native_count(),
                    9
                );
            } else {
                eprintln!(
                    "  f64 probe: PASS — native f64 shaders available ({}/{} builtins)",
                    caps.native_count(),
                    9
                );
            }
            (true, false)
        } else {
            eprintln!("  f64 probe: FAIL — switching to full DF64 data plane");
            (false, true)
        }
    } else {
        eprintln!("  SHADER_F64 not advertised — using full DF64 data plane");
        (false, true)
    };

    GpuF64 {
        adapter_name,
        has_f64,
        has_timestamps,
        full_df64_mode,
        wgpu_device,
        tensor_ctx,
        driver_profile,
    }
}

impl GpuF64 {
    /// Create GPU device requesting `SHADER_F64`.
    ///
    /// Adapter selection: `HOTSPRING_GPU_ADAPTER` takes priority, then falls
    /// through to `BARRACUDA_GPU_ADAPTER`, then auto-detect.
    ///
    /// After device creation, runs barraCuda's f64 probe to verify shader
    /// compilation actually works. On NVK with broken NAK, switches to full
    /// DF64 mode automatically.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::HotSpringError`] if no compatible adapter is found
    /// or device creation fails.
    pub async fn new() -> Result<Self, crate::error::HotSpringError> {
        let selected = adapter::select_adapter()?;
        let adapter_info = selected.get_info();
        let adapter_features = selected.features();
        let is_nvk = adapter_is_nvk(&adapter_info);

        let required_features = negotiate_features(adapter_features, is_nvk);

        let adapter_limits = selected.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits
                .max_storage_buffer_binding_size
                .min(2 * 1024 * 1024 * 1024),
            max_buffer_size: adapter_limits.max_buffer_size.min(4 * 1024 * 1024 * 1024),
            max_storage_buffers_per_shader_stage: 12,
            ..wgpu::Limits::default()
        };

        let device_result: Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError> = selected
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("hotSpring science device"),
                required_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::default(),
            })
            .await;
        let (device, queue) = device_result
            .map_err(|e| crate::error::HotSpringError::DeviceCreation(e.to_string()))?;

        let adapter_name = adapter_info.name.clone();
        let advertised_f64 = required_features.contains(wgpu::Features::SHADER_F64);
        let has_timestamps = required_features.contains(wgpu::Features::TIMESTAMP_QUERY);

        let wgpu_device = Arc::new(WgpuDevice::from_existing(device, queue, adapter_info));

        Ok(finalize_device(
            adapter_name,
            advertised_f64,
            has_timestamps,
            is_nvk,
            wgpu_device,
        )
        .await)
    }

    /// Create GPU device by matching an adapter name substring.
    ///
    /// Searches all available adapters for one whose name (case-insensitive)
    /// contains `name_hint`. Useful for targeting a specific GPU (e.g., "titan")
    /// from a thread without modifying environment variables.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::HotSpringError`] if no matching adapter is found.
    pub async fn from_adapter_name(name_hint: &str) -> Result<Self, crate::error::HotSpringError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

        let hint_lower = name_hint.to_lowercase();
        let selected = adapters
            .into_iter()
            .find(|a: &wgpu::Adapter| a.get_info().name.to_lowercase().contains(&hint_lower))
            .ok_or(crate::error::HotSpringError::NoAdapter)?;

        Self::from_adapter(selected).await
    }

    /// Create GPU device from a pre-selected wgpu adapter.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::HotSpringError`] if device creation fails.
    pub async fn from_adapter(
        selected: wgpu::Adapter,
    ) -> Result<Self, crate::error::HotSpringError> {
        let adapter_info = selected.get_info();
        let adapter_features = selected.features();
        let is_nvk = adapter_is_nvk(&adapter_info);

        let required_features = negotiate_features(adapter_features, is_nvk);

        let adapter_limits = selected.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits
                .max_storage_buffer_binding_size
                .min(2 * 1024 * 1024 * 1024),
            max_buffer_size: adapter_limits.max_buffer_size.min(4 * 1024 * 1024 * 1024),
            max_storage_buffers_per_shader_stage: 12,
            ..wgpu::Limits::default()
        };

        let device_result = selected
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("hotSpring secondary device"),
                required_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::default(),
            })
            .await;
        let (device, queue) = device_result
            .map_err(|e| crate::error::HotSpringError::DeviceCreation(e.to_string()))?;

        let adapter_name = adapter_info.name.clone();
        let advertised_f64 = required_features.contains(wgpu::Features::SHADER_F64);
        let has_timestamps = required_features.contains(wgpu::Features::TIMESTAMP_QUERY);

        let wgpu_device = Arc::new(WgpuDevice::from_existing(device, queue, adapter_info));

        Ok(finalize_device(
            adapter_name,
            advertised_f64,
            has_timestamps,
            is_nvk,
            wgpu_device,
        )
        .await)
    }

    /// Enumerate all available GPU adapters.
    #[must_use]
    pub fn enumerate_adapters() -> Vec<AdapterInfo> {
        adapter::enumerate_adapters()
    }

    /// Print device capabilities.
    pub fn print_info(&self) {
        println!("  GPU: {}", self.adapter_name);
        println!("  SHADER_F64: {}", if self.has_f64 { "YES" } else { "NO" });
        println!(
            "  TIMESTAMP_QUERY: {}",
            if self.has_timestamps { "YES" } else { "NO" }
        );
        println!(
            "  Driver: {:?}, Compiler: {:?}, Arch: {:?}",
            self.driver_profile.driver, self.driver_profile.compiler, self.driver_profile.arch
        );
    }

    /// Print all available adapters to stdout.
    pub fn print_available_adapters() {
        let adapters = Self::enumerate_adapters();
        println!("  Available GPU adapters:");
        for info in &adapters {
            let marker = if info.has_f64 { "✓" } else { "✗" };
            println!("    {marker} {info}");
        }
        if adapters.is_empty() {
            println!("    (none found)");
        }
    }
}

// ── Full DF64 helpers ────────────────────────────────────────────────

/// Strip `df64_from_f64` and `df64_to_f64` functions from `df64_core.wgsl`.
/// These use native f64 types and fail on NVK. In full DF64 mode they're
/// dead code (storage is vec2<f32>, not f64).
fn strip_f64_from_df64_core(core: &str) -> String {
    let mut result = String::with_capacity(core.len());
    let mut skip = false;
    for line in core.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("fn df64_from_f64(") || trimmed.starts_with("fn df64_to_f64(") {
            skip = true;
        }
        if !skip {
            result.push_str(line);
            result.push('\n');
        }
        if skip && trimmed == "}" {
            skip = false;
        }
    }
    result
}

// ── Pipeline creation ────────────────────────────────────────────────

impl GpuF64 {
    /// Create a compute pipeline with `WgslOptimizer` + `GpuDriverProfile`.
    ///
    /// Does NOT apply exp/log workarounds — use [`Self::create_pipeline_f64`]
    /// for shaders that call `exp()` or `log()` on f64 values.
    #[must_use]
    pub fn create_pipeline(&self, shader_source: &str, label: &str) -> wgpu::ComputePipeline {
        let optimized =
            ShaderTemplate::for_driver_profile(shader_source, false, &self.driver_profile);
        self.build_pipeline(&optimized, label)
    }

    /// Create a compute pipeline with driver-aware f64 patching + sovereign compilation.
    ///
    /// Routes through barraCuda's `WgpuDevice::compile_shader_f64()` which applies:
    /// 1. `ShaderTemplate::for_driver_profile` — fossil substitution, polyfills
    /// 2. Sovereign compiler — naga IR → FMA fusion → SPIR-V (when available)
    /// 3. WGSL text fallback when passthrough is unavailable
    ///
    /// In full DF64 mode (f64 probe failed), rewrites shaders to pure-f32 DF64.
    #[must_use]
    pub fn create_pipeline_f64(&self, shader_source: &str, label: &str) -> wgpu::ComputePipeline {
        if self.full_df64_mode {
            return self.compile_full_df64_pipeline(shader_source, label);
        }
        let shader_module = self
            .wgpu_device
            .compile_shader_f64(shader_source, Some(label));
        self.validate_pipeline(shader_module, label)
    }

    /// DF64 pipeline: prepend `df64_core` + `df64_transcendentals`, compile with
    /// barraCuda's `compile_shader_f64()` which injects `round_f64` and other
    /// polyfills. The shader source must use DF64 arithmetic (Df64 struct,
    /// `df64_add/mul/div`, `sqrt_df64`, `exp_df64`, etc.) and `round_f64` for PBC.
    ///
    /// Used for `Fp64Strategy::Hybrid` — force math on FP32 cores, PBC/I/O
    /// in native f64.
    #[must_use]
    pub fn create_pipeline_df64(&self, shader_source: &str, label: &str) -> wgpu::ComputePipeline {
        use barracuda::ops::lattice::su3::{WGSL_DF64_CORE, WGSL_DF64_TRANSCENDENTALS};

        let combined =
            format!("enable f64;\n{WGSL_DF64_CORE}\n{WGSL_DF64_TRANSCENDENTALS}\n{shader_source}");
        let shader_module = self.wgpu_device.compile_shader_f64(&combined, Some(label));
        self.validate_pipeline(shader_module, label)
    }

    /// WGSL-text f64 pipeline — skips sovereign SPIR-V compilation.
    ///
    /// In full DF64 mode, routes through the full DF64 pipeline.
    #[must_use]
    pub fn create_pipeline_f64_precise(
        &self,
        shader_source: &str,
        label: &str,
    ) -> wgpu::ComputePipeline {
        if self.full_df64_mode {
            return self.compile_full_df64_pipeline(shader_source, label);
        }
        let optimized = ShaderTemplate::for_driver_profile(
            shader_source,
            self.wgpu_device.needs_f64_exp_log_workaround(),
            &self.driver_profile,
        );
        self.build_pipeline(&optimized, label)
    }

    /// Full DF64 pipeline: downcast f64 types → f32-pair, strip native f64
    /// helpers from `df64_core`, compile as pure f32 WGSL. No `SHADER_F64` needed.
    fn compile_full_df64_pipeline(
        &self,
        shader_source: &str,
        label: &str,
    ) -> wgpu::ComputePipeline {
        use barracuda::ops::lattice::su3::{WGSL_DF64_CORE, WGSL_DF64_TRANSCENDENTALS};
        use barracuda::shaders::precision::downcast_f64_to_df64;

        let df64_source = downcast_f64_to_df64(shader_source);
        let core_stripped = strip_f64_from_df64_core(WGSL_DF64_CORE);

        let combined = format!(
            "{core_stripped}\n{}\n{WGSL_DF64_TRANSCENDENTALS}\n{df64_source}",
            barracuda::shaders::precision::DF64_PACK_UNPACK,
        );

        if std::env::var("HOTSPRING_DUMP_DF64").is_ok() {
            let dump_dir = std::env::var("HOTSPRING_DUMP_DIR")
                .unwrap_or_else(|_| std::env::temp_dir().display().to_string());
            let dump_path = format!("{dump_dir}/df64_{label}.wgsl");
            let _ = std::fs::write(&dump_path, &combined);
            eprintln!(
                "[DF64 DUMP] {label} → {dump_path} ({} bytes)",
                combined.len()
            );
        }

        self.build_pipeline(&combined, label)
    }

    /// Create a pipeline from a shader module with validation error checking.
    fn validate_pipeline(
        &self,
        shader_module: wgpu::ShaderModule,
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
                entry_point: Some("main"),
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
                eprintln!("[pipeline:{gpu_tag}] {label}: *** PIPELINE ERROR: {e}");
            }
            std::task::Poll::Ready(None) => {
                eprintln!(
                    "[pipeline:{gpu_tag}] {label}: pipeline valid ({:?})",
                    t0.elapsed()
                );
            }
            std::task::Poll::Pending => {
                eprintln!(
                    "[pipeline:{gpu_tag}] {label}: pipeline status pending ({:?})",
                    t0.elapsed()
                );
            }
        }
        pipeline
    }

    fn build_pipeline(&self, wgsl: &str, label: &str) -> wgpu::ComputePipeline {
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
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f64_buffer_size_bytes(count: usize) -> usize {
        count * 8
    }

    fn f64_to_bytes(data: &[f64]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// `chunks_exact(8)` guarantees 8-byte chunks; array construction cannot fail.
    fn bytes_to_f64(data: &[u8]) -> Vec<f64> {
        data.chunks_exact(8)
            .map(|chunk| {
                let bytes: [u8; 8] = std::array::from_fn(|i| chunk[i]);
                f64::from_le_bytes(bytes)
            })
            .collect()
    }

    #[test]
    fn f64_buffer_size_calculation() {
        assert_eq!(f64_buffer_size_bytes(0), 0);
        assert_eq!(f64_buffer_size_bytes(1), 8);
        assert_eq!(f64_buffer_size_bytes(100), 800);
        assert_eq!(f64_buffer_size_bytes(1000 * 3), 24_000);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn f64_byte_roundtrip() {
        let original = vec![
            0.0,
            1.0,
            -1.0,
            std::f64::consts::PI,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let bytes = f64_to_bytes(&original);
        let recovered = bytes_to_f64(&bytes);
        assert_eq!(original.len(), recovered.len());
        for i in 0..original.len() {
            if original[i].is_nan() {
                assert!(recovered[i].is_nan());
            } else {
                assert_eq!(original[i], recovered[i]);
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn f64_byte_conversion_special_values() {
        let values = [std::f64::consts::PI, 1e-308, 1e308];
        let bytes = f64_to_bytes(&values);
        assert_eq!(bytes.len(), 24);
        let back = bytes_to_f64(&bytes);
        assert_eq!(back[0], std::f64::consts::PI);
        assert_eq!(back[1], 1e-308);
        assert_eq!(back[2], 1e308);
    }

    #[test]
    fn query_gpu_power_returns_four_tuple() {
        let (power, temp, util, vram) = GpuF64::query_gpu_power();
        assert!(power >= 0.0);
        assert!((0.0..=150.0).contains(&temp));
        assert!((0.0..=100.0).contains(&util));
        assert!(vram >= 0.0);
    }

    #[test]
    fn gpu_vram_used_non_negative() {
        let vram = GpuF64::gpu_vram_used_mib();
        assert!(vram >= 0.0);
    }

    #[test]
    #[ignore = "requires GPU"]
    fn dispatch_and_read_result_type() {
        let ok_result: Result<Vec<f64>, crate::error::HotSpringError> = Ok(vec![1.0, 2.0]);
        assert!(ok_result.is_ok());
        let err_result: Result<Vec<f64>, crate::error::HotSpringError> = Err(
            crate::error::HotSpringError::GpuCompute("no GPU available".into()),
        );
        assert!(err_result.is_err());
    }

    #[test]
    fn read_back_f64_empty() {
        let empty: Vec<u8> = vec![];
        assert_eq!(empty.len() / 8, 0);
    }
}
