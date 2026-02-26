// SPDX-License-Identifier: AGPL-3.0-only

//! Discover and print all compute substrates, dispatch routing, and
//! pipeline topologies on this machine.
//!
//! GPU discovery uses the same wgpu path that toadstool/barracuda uses.
//! NPU and CPU discovery are local probes.

use hotspring_forge::dispatch::{self, profiles, Workload};
use hotspring_forge::pipeline::topologies;
use hotspring_forge::substrate::{Capability, Fp64Strategy, SubstrateKind};

fn main() {
    let substrates = hotspring_forge::inventory::discover();
    hotspring_forge::inventory::print_inventory(&substrates);

    // ── FP64 Strategy per substrate ──
    println!();
    println!("═══ FP64 Strategy Selection ═══════════════════════════════");
    for s in &substrates {
        if s.kind == SubstrateKind::Gpu {
            let strategy = Fp64Strategy::for_properties(&s.properties);
            let rate = s
                .properties
                .fp64_rate
                .map_or("unknown".to_string(), |r| format!("{r:?}"));
            println!(
                "  {} — rate: {}, df64: {}, strategy: {:?}",
                s.identity.name, rate, s.properties.has_df64, strategy
            );
        }
    }

    // ── Dispatch Routing ──
    println!();
    println!("═══ Dispatch Routing ═══════════════════════════════════════");
    let workloads: Vec<Workload> = vec![
        profiles::md_force(),
        profiles::lattice_cg(),
        profiles::lattice_cg_df64(),
        profiles::hfb_eigensolve(),
        profiles::esn_npu_inference(),
        profiles::cpu_validation(),
        profiles::spectral_spmv(),
        profiles::streaming_compute(),
        profiles::validation_oracle(),
        Workload::new(
            "Phase classifier",
            vec![Capability::QuantizedInference { bits: 8 }],
        ),
    ];

    for work in &workloads {
        match dispatch::route(work, &substrates) {
            Some(d) => println!("  {:35} → {} ({:?})", work.name, d.substrate, d.reason),
            None => println!("  {:35} → NO CAPABLE SUBSTRATE", work.name),
        }
    }

    // ── Pipeline Topologies ──
    println!();
    println!("═══ Pipeline Topologies ════════════════════════════════════");

    let pipelines = [
        topologies::qcd_gpu_npu_oracle(),
        topologies::qcd_parallel_gpu(),
        topologies::qcd_npu_first(),
        topologies::qcd_cpu_baseline(),
        topologies::qcd_gpu_only(),
    ];

    for p in &pipelines {
        println!();
        print!("{p}");
    }

    // ── DF64 Extension Summary ──
    println!();
    println!("═══ DF64 Extension Summary ════════════════════════════════");
    let df64_gpus: Vec<_> = substrates
        .iter()
        .filter(|s| s.has(&Capability::DF64Compute))
        .collect();
    let f64_gpus: Vec<_> = substrates
        .iter()
        .filter(|s| s.has(&Capability::F64Compute) && s.kind == SubstrateKind::Gpu)
        .collect();

    println!("  Native f64 GPUs: {}", f64_gpus.len());
    for g in &f64_gpus {
        println!("    - {}", g.identity.name);
    }
    println!("  DF64-capable GPUs: {}", df64_gpus.len());
    for g in &df64_gpus {
        println!("    - {}", g.identity.name);
    }

    println!();
    println!("  Strategy: Saturate native FP64 units → overflow to DF64 on FP32 cores");
    println!("  Result: More aggregate f64 TFLOPS from the same silicon");
}
