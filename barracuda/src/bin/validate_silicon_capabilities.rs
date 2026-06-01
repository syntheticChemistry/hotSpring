// SPDX-License-Identifier: AGPL-3.0-or-later

//! Silicon capability diagnostic: characterize precision tier behavior per GPU.
//!
//! Probes each adapter for:
//! - f32 FMA two_prod correctness (Dekker error-free product)
//! - f32 workgroup tree reduction (baseline barrier test)
//! - DF64 scalar arithmetic in storage (no workgroup memory)
//! - DF64 workgroup tree reduction with f32 storage (isolates DF64 pattern)
//! - DF64 workgroup tree reduction with f64 storage (production pattern)
//! - `ReduceScalarPipeline` end-to-end (barracuda's production path)
//!
//! Results form the empirical capability matrix that `PrecisionRoutingAdvice`
//! should be built from. No per-card if/else — just probe and route.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::toadstool_report;
use hotspring_barracuda::validation::ValidationHarness;

#[path = "../bin_helpers/silicon_capabilities/mod.rs"]
mod silicon_capabilities;

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Silicon Capability Diagnostic");
    println!("  ecoPrimals/hotSpring — precision tier characterization");
    println!("═══════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    if adapters.is_empty() {
        eprintln!("No GPU adapters found.");
        std::process::exit(1);
    }

    println!("Found {} adapter(s):", adapters.len());
    for a in &adapters {
        let info = a.get_info();
        let f64_mark = if a.features().contains(wgpu::Features::SHADER_F64) {
            "✓ f64"
        } else {
            "✗ f64"
        };
        println!("  [{f64_mark}] {} ({:?})", info.name, info.backend);
    }
    println!();

    let mut harness = ValidationHarness::new("silicon_capabilities");

    for adapter in adapters {
        let info = adapter.get_info();
        let tag = format!("{} ({:?})", info.name, info.backend);
        println!("━━━ {tag} ━━━\n");

        let gpu = match GpuF64::from_adapter(adapter).await {
            Ok(g) => g,
            Err(e) => {
                println!("  ⚠ Could not create device: {e}");
                harness.check_bool(&format!("{tag}: device_creation"), false);
                println!();
                continue;
            }
        };

        gpu.print_info();
        println!();

        silicon_capabilities::probe_f32_fma(&gpu, &tag, &mut harness).await;
        silicon_capabilities::probe_f32_workgroup_reduce(&gpu, &tag, &mut harness).await;
        silicon_capabilities::probe_df64_storage_arith(&gpu, &tag, &mut harness).await;
        silicon_capabilities::probe_df64_workgroup_reduce_f32(&gpu, &tag, &mut harness).await;

        if gpu.has_f64 {
            silicon_capabilities::probe_df64_workgroup_reduce_f64(&gpu, &tag, &mut harness).await;
        } else {
            println!("── DF64 workgroup reduce (f64 storage) ──");
            println!("  ⊘ SKIPPED (SHADER_F64 not functional)\n");
        }

        silicon_capabilities::probe_reduce_pipeline(&gpu, &tag, &mut harness).await;
    }

    println!("── Reporting to toadStool ──\n");
    let ts = toadstool_report::epoch_now();
    let measurements = silicon_capabilities::harness_to_measurements(&harness, ts);
    toadstool_report::report_to_toadstool(&measurements);
    println!();

    harness.finish();
}
