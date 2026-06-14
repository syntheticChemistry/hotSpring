// SPDX-License-Identifier: AGPL-3.0-or-later

//! QCD silicon benchmark: per-kernel profiling across GPUs, lattice sizes, and
//! precision tiers (FP32 proxy + DF64).

use hotspring_barracuda::bin_helpers::qcd_silicon::{
    KernelSpec, bench_kernel, classify_silicon_opportunity, df64_kernel_specs, fp32_kernel_specs,
    print_trajectory_cost_model,
};
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::toadstool_report::{self, PerformanceMeasurement};

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  QCD Silicon Benchmark v2");
    println!("  Quenched → Dynamical × FP32/DF64 × 4^4 → 32^4");
    println!("═══════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    if adapters.is_empty() {
        eprintln!("No GPU adapters found.");
        std::process::exit(1);
    }

    let fp32_kernels = fp32_kernel_specs();
    let df64_kernels = df64_kernel_specs();

    let lattice_sizes: &[(u32, &str)] = &[
        (256, "4^4"),
        (4096, "8^4"),
        (8192, "8^3x16"),
        (65536, "16^4"),
        (1_048_576, "32^4"),
    ];

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

        let is_software = gpu.adapter_name.to_lowercase().contains("llvmpipe");

        for (n_sites, vol_name) in lattice_sizes {
            if is_software && *n_sites > 65536 {
                println!("  Skipping {vol_name} on software renderer\n");
                continue;
            }

            let iterations = if *n_sites >= 1_048_576 {
                20
            } else if *n_sites >= 65536 {
                100
            } else {
                200
            };

            println!("  ── Volume {vol_name} ({n_sites} sites, {iterations} iters) ──\n");
            println!(
                "  {:<20} {:>10} {:>10} {:>10} {:>8}  Silicon",
                "Kernel", "sites/s", "GFLOP/s", "GB/s", "Phase"
            );
            println!("  {}", "─".repeat(90));

            for k in &fp32_kernels {
                run_kernel_bench(
                    device,
                    queue,
                    k,
                    *n_sites,
                    iterations,
                    &gpu.adapter_name,
                    ts,
                    &mut measurements,
                );
            }

            println!();
            for k in &df64_kernels {
                run_kernel_bench(
                    device,
                    queue,
                    k,
                    *n_sites,
                    iterations,
                    &gpu.adapter_name,
                    ts,
                    &mut measurements,
                );
            }
            println!();
        }

        println!("  ── Silicon Unit Opportunity Analysis ──\n");
        println!(
            "  {:<20} {:>8} {:>12} {:>12}  Opportunity",
            "Kernel", "F/B", "Bottleneck", "Target"
        );
        println!("  {}", "─".repeat(80));

        let all_kernels: Vec<&KernelSpec> =
            fp32_kernels.iter().chain(df64_kernels.iter()).collect();
        for k in &all_kernels {
            let intensity = f64::from(k.flops_per_site) / f64::from(k.bytes_per_site);
            let (bottleneck, target, opportunity) = classify_silicon_opportunity(k, intensity);
            println!(
                "  {:<20} {:>6.1} {:>12} {:>12}  {}",
                k.name, intensity, bottleneck, target, opportunity
            );
        }

        println!("\n  ── Estimated HMC Trajectory Cost (32^4, 40 MD steps) ──\n");
        print_trajectory_cost_model(&gpu.adapter_name);

        println!();
    }

    println!(
        "── Reporting {} measurements to toadStool ──\n",
        measurements.len()
    );
    toadstool_report::report_to_toadstool(&measurements);

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  QCD Silicon Benchmark v2 Complete");
    println!("═══════════════════════════════════════════════════════════");
}

fn run_kernel_bench(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    k: &KernelSpec,
    n_sites: u32,
    iterations: u32,
    adapter_name: &str,
    ts: u64,
    measurements: &mut Vec<PerformanceMeasurement>,
) {
    let elapsed = bench_kernel(
        device,
        queue,
        k.wgsl,
        n_sites,
        k.wg_size,
        iterations,
        k.out_bytes_per_site,
    );
    let sites_per_sec = f64::from(n_sites) * f64::from(iterations) / elapsed.as_secs_f64();
    let gflops = sites_per_sec * f64::from(k.flops_per_site) / 1e9;
    let gbps = sites_per_sec * f64::from(k.bytes_per_site) / 1e9;

    println!(
        "  {:<20} {:>8.1}M {:>8.2} {:>8.2} {:>8}  {}",
        k.name,
        sites_per_sec / 1e6,
        gflops,
        gbps,
        k.phase,
        k.silicon_note,
    );

    measurements.push(PerformanceMeasurement {
        operation: format!("{}.v{n_sites}", k.op),
        silicon_unit: "shader_core".into(),
        precision_mode: k.precision.into(),
        throughput_gflops: gflops,
        tolerance_achieved: 0.0,
        gpu_model: adapter_name.to_string(),
        measured_by: "hotSpring/bench_qcd_silicon".into(),
        timestamp: ts,
    });
}
