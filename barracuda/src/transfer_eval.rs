// SPDX-License-Identifier: AGPL-3.0-or-later

//! PCIe transfer cost profiler per GPU card.
//!
//! Measures actual CPU→GPU upload, GPU→CPU readback, dispatch overhead,
//! and reduce-scalar overhead at various buffer sizes. Produces a
//! `TransferProfile` that maps the real bandwidth curve for each card.

use crate::gpu::GpuF64;
use std::time::Instant;

const WARMUP_REPS: usize = 3;
const MEASURE_REPS: usize = 10;

/// Buffer sizes to probe (bytes): 1KB → 16MB.
const PROBE_SIZES: &[usize] = &[
    1_024, 8_192, 65_536, 262_144, 1_048_576, 4_194_304, 16_777_216,
];

/// Measured transfer rates for a single GPU adapter.
#[derive(Debug, Clone)]
pub struct TransferProfile {
    /// GPU adapter name.
    pub adapter_name: String,
    /// Upload bandwidth samples at various buffer sizes.
    pub upload_rates: Vec<BandwidthSample>,
    /// Readback bandwidth samples at various buffer sizes.
    pub readback_rates: Vec<BandwidthSample>,
    /// Empty-shader dispatch round-trip overhead in microseconds.
    pub dispatch_overhead_us: f64,
    /// Reduce-scalar (dispatch + readback) overhead in microseconds.
    pub reduce_scalar_us: f64,
}

/// A single (buffer_size, measured bandwidth) sample.
#[derive(Debug, Clone, Copy)]
pub struct BandwidthSample {
    /// Buffer size in bytes.
    pub bytes: usize,
    /// Measured bandwidth in GB/s.
    pub gbps: f64,
    /// Median transfer time in microseconds.
    pub median_us: f64,
}

/// PCIe transfer cost profiler.
pub struct TransferEval<'a> {
    gpu: &'a GpuF64,
}

impl<'a> TransferEval<'a> {
    /// Create a transfer evaluator for the given GPU.
    #[must_use]
    pub fn new(gpu: &'a GpuF64) -> Self {
        Self { gpu }
    }

    /// Run the full transfer profile suite.
    pub fn profile(&self) -> TransferProfile {
        let adapter_name = self.gpu.adapter_name.clone();

        let upload_rates = self.profile_uploads();
        let readback_rates = self.profile_readbacks();
        let dispatch_overhead_us = self.profile_dispatch_overhead();
        let reduce_scalar_us = self.profile_reduce_scalar();

        TransferProfile {
            adapter_name,
            upload_rates,
            readback_rates,
            dispatch_overhead_us,
            reduce_scalar_us,
        }
    }

    fn profile_uploads(&self) -> Vec<BandwidthSample> {
        PROBE_SIZES
            .iter()
            .map(|&size| {
                let n_f64 = size / 8;
                let data: Vec<f64> = (0..n_f64).map(|i| i as f64 * 0.001).collect();
                let buf = self.gpu.create_f64_output_buffer(n_f64, "transfer_probe");

                for _ in 0..WARMUP_REPS {
                    self.gpu.upload_f64(&buf, &data);
                }

                let mut times = Vec::with_capacity(MEASURE_REPS);
                for _ in 0..MEASURE_REPS {
                    let t0 = Instant::now();
                    self.gpu.upload_f64(&buf, &data);
                    let _ = self.gpu.device().poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    });
                    times.push(t0.elapsed().as_secs_f64() * 1e6);
                }

                let median_us = median_f64(&mut times);
                let gbps = if median_us > 0.0 {
                    (size as f64) / (median_us * 1e-6) / 1e9
                } else {
                    0.0
                };

                BandwidthSample {
                    bytes: size,
                    gbps,
                    median_us,
                }
            })
            .collect()
    }

    fn profile_readbacks(&self) -> Vec<BandwidthSample> {
        PROBE_SIZES
            .iter()
            .map(|&size| {
                let n_f64 = size / 8;
                let data: Vec<f64> = (0..n_f64).map(|i| i as f64 * 0.001).collect();
                let buf = self.gpu.create_f64_buffer(&data, "readback_probe");

                for _ in 0..WARMUP_REPS {
                    let _ = self.gpu.read_back_f64(&buf, n_f64);
                }

                let mut times = Vec::with_capacity(MEASURE_REPS);
                for _ in 0..MEASURE_REPS {
                    let t0 = Instant::now();
                    let _ = self.gpu.read_back_f64(&buf, n_f64);
                    times.push(t0.elapsed().as_secs_f64() * 1e6);
                }

                let median_us = median_f64(&mut times);
                let gbps = if median_us > 0.0 {
                    (size as f64) / (median_us * 1e-6) / 1e9
                } else {
                    0.0
                };

                BandwidthSample {
                    bytes: size,
                    gbps,
                    median_us,
                }
            })
            .collect()
    }

    fn profile_dispatch_overhead(&self) -> f64 {
        let empty_shader = r"
@compute @workgroup_size(1)
fn main() { }
";
        let pipeline = self.gpu.create_pipeline(empty_shader, "empty_dispatch");
        let noop_buf = self
            .gpu
            .create_f64_output_buffer(1, "dispatch_overhead_noop");
        let bind_group = self.gpu.create_bind_group(&pipeline, &[&noop_buf]);

        for _ in 0..WARMUP_REPS {
            self.gpu.dispatch(&pipeline, &bind_group, 1);
            let _ = self.gpu.device().poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
        }

        let mut times = Vec::with_capacity(MEASURE_REPS);
        for _ in 0..MEASURE_REPS {
            let t0 = Instant::now();
            self.gpu.dispatch(&pipeline, &bind_group, 1);
            let _ = self.gpu.device().poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            times.push(t0.elapsed().as_secs_f64() * 1e6);
        }

        median_f64(&mut times)
    }

    fn profile_reduce_scalar(&self) -> f64 {
        let n = 4096_usize;
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
        let buf = self.gpu.create_f64_buffer(&data, "reduce_input");

        let reduce_shader = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x == 0u {
        var sum: f64 = 0.0;
        let n = arrayLength(&input);
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            sum = sum + input[i];
        }
        output[0] = sum;
    }
}
";
        let pipeline = self.gpu.create_pipeline_f64(reduce_shader, "reduce_eval");
        let out_buf = self.gpu.create_f64_output_buffer(1, "reduce_out");
        let bind_group = self.gpu.create_bind_group(&pipeline, &[&buf, &out_buf]);

        for _ in 0..WARMUP_REPS {
            self.gpu.dispatch(&pipeline, &bind_group, 1);
            let _ = self.gpu.read_back_f64(&out_buf, 1);
        }

        let mut times = Vec::with_capacity(MEASURE_REPS);
        for _ in 0..MEASURE_REPS {
            let t0 = Instant::now();
            self.gpu.dispatch(&pipeline, &bind_group, 1);
            let _ = self.gpu.read_back_f64(&out_buf, 1);
            times.push(t0.elapsed().as_secs_f64() * 1e6);
        }

        median_f64(&mut times)
    }
}

fn median_f64(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.is_empty() {
        return 0.0;
    }
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        f64::midpoint(v[mid - 1], v[mid])
    } else {
        v[mid]
    }
}

impl TransferProfile {
    /// Pretty-print the transfer profile.
    pub fn print_report(&self) {
        println!("  {} Transfer Profile:", self.adapter_name);
        print!("    Upload:  ");
        for s in &self.upload_rates {
            print!(" {}={:.1}GB/s", format_bytes(s.bytes), s.gbps);
        }
        println!();
        print!("    Readback:");
        for s in &self.readback_rates {
            print!(" {}={:.1}GB/s", format_bytes(s.bytes), s.gbps);
        }
        println!();
        println!(
            "    Dispatch overhead: {:.0}us   Reduce scalar: {:.0}us",
            self.dispatch_overhead_us, self.reduce_scalar_us
        );
    }
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{}MB", bytes / 1_048_576)
    } else {
        format!("{}KB", bytes / 1_024)
    }
}
