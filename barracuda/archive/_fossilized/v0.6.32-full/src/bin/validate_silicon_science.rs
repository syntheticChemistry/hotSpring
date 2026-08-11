// SPDX-License-Identifier: AGPL-3.0-or-later

//! Silicon science experiments: map QCD operations to every GPU hardware unit.
//!
//! Following the exploration protocol from `GPU_FIXED_FUNCTION_SCIENCE_REPURPOSING.md`:
//! for each operation, test on shader cores (baseline) and on accessible fixed-function
//! units, measure throughput and accuracy, report to toadStool performance surface.
//!
//! ## Experiments
//!
//! 1. **TMU table lookup**: precomputed exp() in a texture vs compute shader exp().
//!    First non-shader-core silicon experiment — tests texture unit throughput for
//!    EOS table lookups (hotSpring assignment from wateringHole).
//!
//! 2. **QCD operation characterization**: Wilson plaquette, gauge force, CG dot product,
//!    DF64 arithmetic across all available GPUs with precision measurement.
//!
//! 3. **Silicon unit mapping**: for each QCD operation, report which silicon unit is
//!    optimal based on measured data → feeds toadStool performance surface.
//!
//! ## Silicon units tested
//!
//! - `shader_core`: compute shaders (baseline for all operations)
//! - `texture_unit`: TMU-accelerated table lookup (exp, EOS tables)
//! - Future: `rt_core` (neighbor search), `rop` (scatter-add), `depth_buffer` (Voronoi)

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::toadstool_report::{self, PerformanceMeasurement};

#[path = "../bin_helpers/silicon_science/mod.rs"]
mod silicon_science;

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Silicon Science Experiments");
    println!("  QCD operations × GPU silicon units × precision tiers");
    println!("  Protocol: GPU_FIXED_FUNCTION_SCIENCE_REPURPOSING.md");
    println!("═══════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    if adapters.is_empty() {
        eprintln!("No GPU adapters found.");
        std::process::exit(1);
    }

    let mut measurements: Vec<PerformanceMeasurement> = Vec::new();
    let ts = toadstool_report::epoch_now();
    let iterations = 100;
    let n_threads: u32 = 1024;
    let workgroups = n_threads / 256;
    let out_bytes = (n_threads as usize) * 4;
    let mut counters = silicon_science::ExpCounters { pass: 0, fail: 0 };

    for adapter in adapters {
        let info = adapter.get_info();
        let gpu_name = info.name.clone();

        let gpu = match GpuF64::from_adapter(adapter).await {
            Ok(g) => g,
            Err(e) => {
                println!("  Could not create device for {gpu_name}: {e}\n");
                continue;
            }
        };

        println!("━━━ {} ━━━\n", gpu.adapter_name);

        silicon_science::run_tmu_vs_compute_exp(
            &gpu,
            &mut measurements,
            ts,
            iterations,
            n_threads,
            workgroups,
            out_bytes,
            &mut counters,
        );
        silicon_science::run_tmu_scaling_exp(&gpu, &mut measurements, ts, &mut counters);
        silicon_science::run_qcd_proxy_exp(
            &gpu,
            &mut measurements,
            ts,
            iterations,
            n_threads,
            workgroups,
            out_bytes,
            &mut counters,
        );
        silicon_science::run_silicon_mapping();
    }

    println!(
        "── Reporting {} measurements to toadStool ──\n",
        measurements.len()
    );
    toadstool_report::report_to_toadstool(&measurements);
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("  TOTAL: {} pass, {} fail", counters.pass, counters.fail);
    println!("  Live silicon units tested: shader_core, texture_unit");
    println!("  Planned units (need sovereign dispatch): tensor_core,");
    println!("    rt_core, rop, rasterizer, depth_buffer, tessellator,");
    println!("    video_encoder");
    println!("═══════════════════════════════════════════════════════════");

    if counters.fail > 0 {
        std::process::exit(1);
    }
}
