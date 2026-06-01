// SPDX-License-Identifier: AGPL-3.0-or-later

//! QCD silicon benchmark reporting: kernel specs and opportunity analysis.

use super::kernels::{
    SHADER_CG_DOT_DF64, SHADER_CG_DOT_FP32, SHADER_DIRAC_STENCIL_FP32, SHADER_FORCE_DF64,
    SHADER_FORCE_FP32, SHADER_GRADIENT_FLOW_FP32, SHADER_LINK_UPDATE_FP32, SHADER_MOM_UPDATE_FP32,
    SHADER_PLAQUETTE_DF64, SHADER_PLAQUETTE_FP32, SHADER_POLYAKOV_FP32,
    SHADER_PRNG_BOXMULLER_FP32, SHADER_PSEUDOFERMION_FORCE_FP32, SHADER_SU3_MATMUL_FP32,
};

pub struct KernelSpec {
    pub name: &'static str,
    pub op: &'static str,
    pub wgsl: &'static str,
    pub wg_size: u32,
    pub flops_per_site: u32,
    pub bytes_per_site: u32,
    pub out_bytes_per_site: u32,
    pub precision: &'static str,
    pub phase: &'static str,
    pub silicon_note: &'static str,
}

pub fn fp32_kernel_specs() -> [KernelSpec; 11] {
    [
        KernelSpec {
            name: "gauge force",
            op: "qcd.force.su3",
            wgsl: SHADER_FORCE_FP32,
            wg_size: 64,
            flops_per_site: 864,
            bytes_per_site: 4 * 18 * 5,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "quenched",
            silicon_note: "ALU-bound — 4 staple matmuls",
        },
        KernelSpec {
            name: "plaquette",
            op: "qcd.plaquette.wilson",
            wgsl: SHADER_PLAQUETTE_FP32,
            wg_size: 64,
            flops_per_site: 1296,
            bytes_per_site: 4 * 18 * 4,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "quenched",
            silicon_note: "ALU-bound — 6 plane matmuls",
        },
        KernelSpec {
            name: "SU3 matmul",
            op: "qcd.matmul.su3",
            wgsl: SHADER_SU3_MATMUL_FP32,
            wg_size: 64,
            flops_per_site: 216,
            bytes_per_site: 4 * 18 * 2,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "quenched",
            silicon_note: "ALU-bound → tensor_core candidate (MMA-shaped)",
        },
        KernelSpec {
            name: "link update",
            op: "qcd.link_update.cayley",
            wgsl: SHADER_LINK_UPDATE_FP32,
            wg_size: 64,
            flops_per_site: 400,
            bytes_per_site: 4 * 18 * 2,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "quenched",
            silicon_note: "ALU-bound — Cayley exp + reunitarize (sqrt)",
        },
        KernelSpec {
            name: "mom update",
            op: "qcd.momentum_update",
            wgsl: SHADER_MOM_UPDATE_FP32,
            wg_size: 64,
            flops_per_site: 72,
            bytes_per_site: 4 * 18 * 3,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "quenched",
            silicon_note: "Memory-bound — P += dt*F",
        },
        KernelSpec {
            name: "CG dot+reduce",
            op: "qcd.cg.dot_reduce",
            wgsl: SHADER_CG_DOT_FP32,
            wg_size: 256,
            flops_per_site: 8,
            bytes_per_site: 4 * 2,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "dynamical",
            silicon_note: "Shared-mem reduce — CG bottleneck",
        },
        KernelSpec {
            name: "Dirac stencil",
            op: "qcd.dirac.staggered",
            wgsl: SHADER_DIRAC_STENCIL_FP32,
            wg_size: 64,
            flops_per_site: 288,
            bytes_per_site: 4 * (18 * 8 + 6),
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "dynamical",
            silicon_note: "Balanced (stencil + matvec) — heart of CG",
        },
        KernelSpec {
            name: "pf force",
            op: "qcd.pseudofermion_force",
            wgsl: SHADER_PSEUDOFERMION_FORCE_FP32,
            wg_size: 64,
            flops_per_site: 1000,
            bytes_per_site: 4 * 18 * 6,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "dynamical",
            silicon_note: "ALU-bound — most expensive dynamical kernel",
        },
        KernelSpec {
            name: "PRNG heat bath",
            op: "qcd.prng.box_muller",
            wgsl: SHADER_PRNG_BOXMULLER_FP32,
            wg_size: 64,
            flops_per_site: 360,
            bytes_per_site: 4,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "dynamical",
            silicon_note: "Transcendental-heavy — TMU LUT candidate",
        },
        KernelSpec {
            name: "Polyakov loop",
            op: "qcd.polyakov.loop",
            wgsl: SHADER_POLYAKOV_FP32,
            wg_size: 64,
            flops_per_site: 576,
            bytes_per_site: 4 * 18 * 16,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "observable",
            silicon_note: "Latency-bound — serial Nt chain",
        },
        KernelSpec {
            name: "grad flow acc",
            op: "qcd.flow.accumulate",
            wgsl: SHADER_GRADIENT_FLOW_FP32,
            wg_size: 64,
            flops_per_site: 288,
            bytes_per_site: 4 * 18 * 2,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "observable",
            silicon_note: "Balanced — algebra accumulation",
        },
    ]
}

pub fn df64_kernel_specs() -> [KernelSpec; 3] {
    [
        KernelSpec {
            name: "force (DF64)",
            op: "qcd.force.su3.df64",
            wgsl: SHADER_FORCE_DF64,
            wg_size: 64,
            flops_per_site: 864 * 4,
            bytes_per_site: 8 * 18 * 5,
            out_bytes_per_site: 8,
            precision: "df64",
            phase: "quenched",
            silicon_note: "DF64 ALU — AMD 1:16 advantage",
        },
        KernelSpec {
            name: "plaquette (DF64)",
            op: "qcd.plaquette.wilson.df64",
            wgsl: SHADER_PLAQUETTE_DF64,
            wg_size: 64,
            flops_per_site: 1296 * 4,
            bytes_per_site: 8 * 18 * 4,
            out_bytes_per_site: 8,
            precision: "df64",
            phase: "quenched",
            silicon_note: "DF64 ALU — higher precision trace",
        },
        KernelSpec {
            name: "CG dot (DF64)",
            op: "qcd.cg.dot_reduce.df64",
            wgsl: SHADER_CG_DOT_DF64,
            wg_size: 256,
            flops_per_site: 8 * 4,
            bytes_per_site: 8 * 2,
            out_bytes_per_site: 8,
            precision: "df64",
            phase: "dynamical",
            silicon_note: "DF64 reduce — error-compensated accumulation",
        },
    ]
}

pub fn classify_silicon_opportunity(
    k: &KernelSpec,
    intensity: f64,
) -> (&'static str, &'static str, &'static str) {
    if k.name.contains("PRNG") {
        return (
            "transcendental",
            "TMU + ALU",
            "TMU LUT for log/cos (1.9× measured on 3090)",
        );
    }
    if k.name.contains("Polyakov") {
        return (
            "latency",
            "shader_core",
            "ILP limited — serial dependency chain",
        );
    }
    if k.name.contains("CG dot") {
        return ("reduce", "shader LDS", "Workgroup shared memory throughput");
    }
    if k.name.contains("SU3 matmul") {
        return (
            "compute",
            "tensor_core",
            "MMA reshape via coralReef SASS (future)",
        );
    }
    if intensity > 3.0 {
        if k.precision == "df64" {
            (
                "compute",
                "shader_core",
                "AMD 1:16 FP64 advantage for error terms",
            )
        } else {
            ("compute", "shader_core", "Peak ALU utilization")
        }
    } else {
        ("memory", "cache/BW", "Infinity Cache advantage at ≤16^4")
    }
}

pub fn print_trajectory_cost_model(gpu_name: &str) {
    let n_md = 40u32;
    let n_sites = 1_048_576u64;
    let n_links = n_sites * 4;

    let force_flops = n_links * 864;
    let link_update_flops = n_links * 400;
    let mom_update_flops = n_links * 72;
    let step_flops = 3 * force_flops + 2 * link_update_flops + 3 * mom_update_flops;
    let traj_flops = n_md as u64 * step_flops;

    let cg_iters_per_force = 100u64;
    let dirac_flops = n_sites * 288;
    let cg_per_force = cg_iters_per_force * dirac_flops;
    let dynamical_overhead = 3 * n_md as u64 * cg_per_force;

    let quenched_tflops = traj_flops as f64 / 1e12;
    let dynamical_tflops = (traj_flops + dynamical_overhead) as f64 / 1e12;

    let is_3090 = gpu_name.to_lowercase().contains("3090");
    let peak_tflops = if is_3090 { 35.6 } else { 23.6 };
    let efficiency = 0.3;

    let quenched_time = quenched_tflops / (peak_tflops * efficiency);
    let dynamical_time = dynamical_tflops / (peak_tflops * efficiency);

    println!(
        "  Quenched:  {quenched_tflops:.2} TFLOP/traj → ~{quenched_time:.1}s at 30% efficiency on {gpu_name}"
    );
    println!(
        "  Dynamical: {dynamical_tflops:.2} TFLOP/traj → ~{dynamical_time:.1}s at 30% efficiency (Nf=4, ~100 CG iters)"
    );
    println!(
        "  Overnight (500 traj): ~{:.1}h quenched, ~{:.1}h dynamical",
        quenched_time * 500.0 / 3600.0,
        dynamical_time * 500.0 / 3600.0
    );
}
