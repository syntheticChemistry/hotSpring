// SPDX-License-Identifier: AGPL-3.0-or-later

//! Silicon saturation micro-experiments: find actual peak of each silicon unit.
//!
//! Six targeted experiments, each designed to saturate exactly one silicon unit
//! and measure achieved throughput vs theoretical peak. The ratio gives the
//! "efficiency floor" — any QCD kernel scoring below this ratio has headroom.
//!
//! ## Experiments
//!
//! 1. **Pure FMA chain** — shader ALU peak (FP32 and DF64)
//! 2. **Bandwidth sweep** — memory controller peak (sequential, strided)
//! 3. **Cache hierarchy** — L2/Infinity Cache boundary detection
//! 4. **TMU saturation** — texture unit peak (textureLoad throughput)
//! 5. **Workgroup reduce** — shared memory / LDS bandwidth
//! 6. **Atomic contention** — global atomicAdd throughput

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::toadstool_report::{self, PerformanceMeasurement};

#[path = "../bin_helpers/silicon_saturation/mod.rs"]
mod silicon_saturation;

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Silicon Saturation Micro-Experiments");
    println!("  Find actual peak of each silicon unit");
    println!("═══════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    if adapters.is_empty() {
        eprintln!("No GPU adapters found.");
        std::process::exit(1);
    }

    let mut measurements: Vec<PerformanceMeasurement> = Vec::new();
    let ts = toadstool_report::epoch_now();

    for adapter in adapters {
        let info = adapter.get_info();
        let gpu_name = info.name.clone();

        let gpu = match GpuF64::from_adapter(adapter).await {
            Ok(g) => g,
            Err(e) => {
                println!("  Skip {gpu_name}: {e}\n");
                continue;
            }
        };

        let device = gpu.device();
        let queue = gpu.queue();

        println!("━━━ {} ━━━\n", gpu.adapter_name);

        silicon_saturation::exp_fma::run(device, queue, &gpu.adapter_name, &mut measurements, ts);
        silicon_saturation::exp_bandwidth::run(
            device,
            queue,
            &gpu.adapter_name,
            &mut measurements,
            ts,
        );
        silicon_saturation::exp_cache::run(device, queue, &gpu.adapter_name, &mut measurements, ts);
        silicon_saturation::exp_tmu::run(device, queue, &gpu.adapter_name, &mut measurements, ts);
        silicon_saturation::exp_reduce::run(
            device,
            queue,
            &gpu.adapter_name,
            &mut measurements,
            ts,
        );
        silicon_saturation::exp_atomic::run(
            device,
            queue,
            &gpu.adapter_name,
            &mut measurements,
            ts,
        );
    }

    println!(
        "── Reporting {} measurements to toadStool ──\n",
        measurements.len()
    );
    toadstool_report::report_to_toadstool(&measurements);

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Silicon Saturation Complete");
    println!("═══════════════════════════════════════════════════════════");
}
