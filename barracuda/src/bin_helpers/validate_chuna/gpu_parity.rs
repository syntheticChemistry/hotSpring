// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU substrate discovery and CPU-GPU parity validation.

use super::paper_43::CpuReferenceValues;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_flow::{GpuFlowPipelines, GpuFlowState, gpu_gradient_flow};
use hotspring_barracuda::lattice::gradient_flow::FlowIntegrator;
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::precision_brain::PrecisionBrain;
use hotspring_barracuda::precision_routing::PhysicsDomain;
use hotspring_barracuda::toadstool_report::{PerformanceMeasurement, report_to_toadstool};
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::{HardwareProfile, ValidationHarness};
use std::time::Instant;

/// GPU substrate discovery, profiling, and CPU-GPU parity validation.
///
/// For each f64-capable GPU discovered via toadStool-compatible enumeration:
/// 1. Probe precision tiers (F32/F64/DF64/F64Precise)
/// 2. Build PrecisionBrain domain routing table
/// 3. Run gradient flow on GPU, compare plaquette against CPU reference
/// 4. Report hardware profile to toadStool if socket available
pub fn gpu_substrate_validation(harness: &mut ValidationHarness, cpu_ref: &CpuReferenceValues) {
    println!("━━━ GPU Substrate Discovery ━━━");

    let adapters = GpuF64::enumerate_adapters();
    let f64_adapters: Vec<_> = adapters.iter().filter(|a| a.has_f64).collect();

    println!(
        "  Discovered {} adapter(s), {} with f64 support",
        adapters.len(),
        f64_adapters.len()
    );
    for a in &adapters {
        let f64_tag = if a.has_f64 { "f64" } else { "f32-only" };
        println!(
            "    [{}] {} ({}, {}, {}MB)",
            a.index,
            a.name,
            a.driver,
            f64_tag,
            a.memory_bytes / (1024 * 1024)
        );
    }
    println!();

    if f64_adapters.is_empty() {
        println!("  No f64-capable GPU found — CPU-only validation is complete.\n");
        return;
    }

    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => {
            println!("  Could not create async runtime: {e} — skipping GPU validation.\n");
            return;
        }
    };

    let mut gpu_plaquettes: Vec<(String, f64)> = Vec::new();

    for adapter in &f64_adapters {
        let token = adapter.index.to_string();
        let adapter_name = &adapter.name;

        println!("━━━ GPU: {adapter_name} ━━━");

        let gpu = match rt.block_on(GpuF64::with_adapter(&token)) {
            Ok(g) => g,
            Err(e) => {
                println!("  Failed to open {adapter_name}: {e} — skipping.\n");
                continue;
            }
        };
        let _guard = rt.enter();

        harness.set_gpu(adapter_name);
        harness.set_substrate(adapter_name);

        // Probe precision tiers
        let t_probe = Instant::now();
        let brain = PrecisionBrain::new(&gpu);
        let cal = &brain.calibration;
        let probe_ms = t_probe.elapsed().as_millis();

        println!("  Calibration ({probe_ms}ms):");
        println!(
            "    f64: {}, df64: {}, transcendental_risk: {}",
            cal.has_any_f64, cal.df64_safe, cal.nvvm_transcendental_risk
        );
        for tier in &cal.tiers {
            if tier.compiles {
                println!(
                    "    {:?}: compile={:.0}μs, dispatch={:.0}μs, ulp={:.1}",
                    tier.tier, tier.compile_us, tier.dispatch_us, tier.probe_ulp
                );
            }
        }

        let vram = gpu.device().limits().max_buffer_size;
        let bytes_per_site: u64 = 4 * 18 * 8 * 4;
        let max_l = ((vram / bytes_per_site) as f64).powf(0.25).floor() as usize;
        println!("  VRAM: {}MB, max lattice L={max_l}", vram / (1024 * 1024));

        let domains = [
            (PhysicsDomain::GradientFlow, "gradient_flow"),
            (PhysicsDomain::Dielectric, "dielectric"),
            (PhysicsDomain::KineticFluid, "kinetic_fluid"),
            (PhysicsDomain::LatticeQcd, "lattice_qcd"),
        ];
        let domain_routing: Vec<(String, String)> = domains
            .iter()
            .map(|(d, name)| (name.to_string(), format!("{:?}", brain.route(*d))))
            .collect();

        println!("  Domain routing:");
        for (name, tier) in &domain_routing {
            println!("    {name} → {tier}");
        }

        harness.hardware_profiles.push(HardwareProfile {
            adapter: adapter_name.clone(),
            vram_bytes: vram,
            precision_tiers: cal
                .tiers
                .iter()
                .map(|t| {
                    (
                        format!("{:?}", t.tier),
                        t.compiles,
                        t.dispatch_us,
                        t.probe_ulp,
                    )
                })
                .collect(),
            domain_routing: domain_routing.clone(),
            max_lattice_l: max_l,
        });

        // GPU gradient flow parity check
        let t_flow = Instant::now();
        let seed = 42_u64;
        let beta = 6.0;
        let dims = [8, 8, 8, 8];
        let eps: f64 = 0.01;
        let meas = (0.05 / eps).max(1.0) as usize;

        let gpu_lat = Lattice::hot_start(dims, beta, seed);
        let pipelines = GpuFlowPipelines::new(&gpu);
        let state = GpuFlowState::from_lattice(&gpu, &gpu_lat, beta);
        let gpu_flow = gpu_gradient_flow(
            &gpu,
            &pipelines,
            &state,
            FlowIntegrator::Rk3Luscher,
            eps,
            2.0,
            meas,
        );
        let flow_ms = t_flow.elapsed().as_millis();

        let gpu_plaq = gpu_flow
            .measurements
            .last()
            .map_or(f64::NAN, |m| m.plaquette);
        let gpu_energy = gpu_flow
            .measurements
            .last()
            .map_or(f64::NAN, |m| m.energy_density);

        println!("  GPU flow ({flow_ms}ms): plaq={gpu_plaq:.10}, E={gpu_energy:.6}");

        // CPU-GPU parity: 200 RK3 steps with different summation order
        // (GPU parallel reduction vs CPU sequential) accumulates ~O(200 * N_sites * ε_mach)
        // difference. Empirically 2-3e-10 on 8^4 lattice, so 1e-9 is the correct bound.
        let plaq_diff = (gpu_plaq - cpu_ref.plaquette_8_rk3).abs();
        harness.check_upper(
            &format!("gpu_cpu_plaquette_parity_{}", sanitize_name(adapter_name)),
            plaq_diff,
            tolerances::GRADIENT_FLOW_GPU_CPU_PLAQUETTE_ABS,
        );
        harness.annotate(
            "cross_substrate",
            "guideStone Property 1",
            "plaquette_difference",
            "CPU-GPU plaquette within 1e-9: 200 RK3 steps × parallel vs sequential reduction order",
        );

        let energy_diff = (gpu_energy - cpu_ref.energy_8_rk3).abs();
        harness.check_upper(
            &format!("gpu_cpu_energy_parity_{}", sanitize_name(adapter_name)),
            energy_diff,
            tolerances::ITERATIVE_F64,
        );
        harness.annotate(
            "cross_substrate",
            "guideStone Property 1",
            "energy_density_difference",
            "CPU and GPU flow energy within 1e-8 (accumulated integration differences)",
        );

        // GPU flow should also smooth energy
        let gpu_e_start = gpu_flow
            .measurements
            .first()
            .map_or(f64::NAN, |m| m.energy_density);
        harness.check_bool(
            &format!("gpu_flow_energy_smoothing_{}", sanitize_name(adapter_name)),
            gpu_energy <= gpu_e_start,
        );
        harness.annotate(
            "lattice_qcd",
            "Bazavov & Chuna, arXiv:2101.05320",
            "energy_density",
            "GPU flow E(t_final) <= E(t_initial)",
        );

        gpu_plaquettes.push((adapter_name.clone(), gpu_plaq));

        // Report to toadStool if available
        let measurements = vec![PerformanceMeasurement {
            operation: "gradient_flow_rk3_8x8x8x8".into(),
            silicon_unit: "shader".into(),
            precision_mode: if cal.has_any_f64 { "f64" } else { "df64" }.into(),
            throughput_gflops: 0.0,
            tolerance_achieved: plaq_diff,
            gpu_model: adapter_name.clone(),
            measured_by: "validate_chuna".into(),
            timestamp: hotspring_barracuda::toadstool_report::epoch_now(),
        }];
        report_to_toadstool(&measurements);

        harness.clear_substrate();
        println!();
    }

    // Cross-GPU comparison (if multiple GPUs validated)
    if gpu_plaquettes.len() > 1 {
        println!("━━━ Cross-GPU Comparison ━━━");
        for i in 0..gpu_plaquettes.len() {
            for j in (i + 1)..gpu_plaquettes.len() {
                let (ref name_a, plaq_a) = gpu_plaquettes[i];
                let (ref name_b, plaq_b) = gpu_plaquettes[j];
                let diff = (plaq_a - plaq_b).abs();
                println!("  {name_a} vs {name_b}: |Δplaq| = {diff:.2e}");
                harness.check_upper(
                    &format!(
                        "cross_gpu_plaquette_{}_{}",
                        sanitize_name(name_a),
                        sanitize_name(name_b)
                    ),
                    diff,
                    tolerances::GRADIENT_FLOW_CROSS_GPU_PLAQUETTE_ABS,
                );
                harness.annotate(
                    "cross_substrate", "guideStone Property 1",
                    "plaquette_difference",
                    "Cross-GPU plaquette within 5e-10: both use parallel reduction, minor driver differences",
                );
            }
        }
        println!();
    }
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .replace("__", "_")
}
