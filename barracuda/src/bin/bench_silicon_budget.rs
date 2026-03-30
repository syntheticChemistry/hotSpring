// SPDX-License-Identifier: AGPL-3.0-only

//! Silicon budget calculator: theoretical peak throughput per GPU per silicon unit.
//!
//! For each detected GPU, computes the full theoretical hardware budget from
//! adapter info and known silicon specifications. This is the denominator for
//! "achieved / theoretical" efficiency metrics across all silicon experiments.
//!
//! ## What this computes
//!
//! Per GPU:
//! - FP32 shader TFLOPS (peak)
//! - DF64 TFLOPS (FP32 / ~4 for Dekker arithmetic)
//! - FP64 TFLOPS (native, from FP64:FP32 ratio)
//! - Memory bandwidth (GB/s)
//! - L2 cache size and effective bandwidth tiers
//! - TMU theoretical texel rate (GT/s)
//! - ROP theoretical pixel rate (GP/s)
//! - Working-set breakpoints: which lattice sizes fit in L2, Infinity Cache, VRAM
//!
//! ## Compound budget
//!
//! When multiple silicon units operate on different sub-problems simultaneously,
//! effective throughput can exceed any single unit's peak. This binary computes
//! the theoretical "compound ceiling" for QCD workloads.

use hotspring_barracuda::toadstool_report::{self, PerformanceMeasurement};

use std::fmt;

struct GpuSiliconBudget {
    name: String,
    vendor: GpuVendor,
    vram_bytes: u64,

    // Shader cores
    fp32_tflops: f64,
    fp64_ratio: f64,
    fp64_tflops: f64,
    df64_tflops: f64,

    // Memory hierarchy
    memory_bw_gbs: f64,
    l2_bytes: u64,
    infinity_cache_bytes: u64,

    // Fixed-function
    tmu_count: u32,
    rop_count: u32,
    boost_ghz: f64,
    tmu_gtexels: f64,
    rop_gpixels: f64,

    // Tensor cores (NVIDIA only via WGSL — not directly accessible)
    tensor_fp16_tflops: f64,
    tensor_tf32_tflops: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Software,
    Unknown,
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nvidia => write!(f, "NVIDIA"),
            Self::Amd => write!(f, "AMD"),
            Self::Intel => write!(f, "Intel"),
            Self::Software => write!(f, "Software"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

fn classify_vendor(info: &wgpu::AdapterInfo) -> GpuVendor {
    match info.vendor {
        0x10DE => GpuVendor::Nvidia,
        0x1002 => GpuVendor::Amd,
        0x8086 => GpuVendor::Intel,
        _ => {
            let name_lower = info.name.to_lowercase();
            if name_lower.contains("llvmpipe") || name_lower.contains("software") {
                GpuVendor::Software
            } else {
                GpuVendor::Unknown
            }
        }
    }
}

/// Known GPU silicon specifications.
///
/// These come from vendor data sheets and our measured benchmarks.
/// The budget calculator matches adapter name substrings to look up specs
/// that wgpu cannot discover at runtime (boost clock, TMU count, etc.).
fn lookup_silicon_specs(name: &str, vendor: GpuVendor) -> GpuSiliconBudget {
    let name_lower = name.to_lowercase();

    // RTX 3090 (GA102, Ampere)
    if name_lower.contains("3090") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor,
            vram_bytes: 24 * 1024 * 1024 * 1024,
            fp32_tflops: 35.6,
            fp64_ratio: 1.0 / 64.0,
            fp64_tflops: 0.556,
            df64_tflops: 3.24, // measured in hotSpring bench_fp64_ratio
            memory_bw_gbs: 936.0,
            l2_bytes: 6 * 1024 * 1024,
            infinity_cache_bytes: 0,
            tmu_count: 328,
            rop_count: 112,
            boost_ghz: 1.70,
            tmu_gtexels: 328.0 * 1.70,
            rop_gpixels: 112.0 * 1.70,
            tensor_fp16_tflops: 142.0,
            tensor_tf32_tflops: 71.0,
        };
    }

    // RTX 3080 Ti (GA102, Ampere)
    if name_lower.contains("3080 ti") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor,
            vram_bytes: 12 * 1024 * 1024 * 1024,
            fp32_tflops: 34.1,
            fp64_ratio: 1.0 / 64.0,
            fp64_tflops: 0.533,
            df64_tflops: 3.1,
            memory_bw_gbs: 912.0,
            l2_bytes: 6 * 1024 * 1024,
            infinity_cache_bytes: 0,
            tmu_count: 320,
            rop_count: 112,
            boost_ghz: 1.67,
            tmu_gtexels: 320.0 * 1.67,
            rop_gpixels: 112.0 * 1.67,
            tensor_fp16_tflops: 136.0,
            tensor_tf32_tflops: 68.0,
        };
    }

    // RTX 5090 (GB202, Blackwell)
    if name_lower.contains("5090") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor,
            vram_bytes: 32 * 1024 * 1024 * 1024,
            fp32_tflops: 104.8,
            fp64_ratio: 1.0 / 64.0,
            fp64_tflops: 1.638,
            df64_tflops: 10.0, // estimate: FP32/~10 for Dekker
            memory_bw_gbs: 1792.0,
            l2_bytes: 36 * 1024 * 1024,
            infinity_cache_bytes: 0,
            tmu_count: 512,
            rop_count: 176,
            boost_ghz: 2.41,
            tmu_gtexels: 512.0 * 2.41,
            rop_gpixels: 176.0 * 2.41,
            tensor_fp16_tflops: 838.0,
            tensor_tf32_tflops: 419.0,
        };
    }

    // RTX 4070 (AD104, Ada Lovelace)
    if name_lower.contains("4070") && !name_lower.contains("4070 ti") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor,
            vram_bytes: 12 * 1024 * 1024 * 1024,
            fp32_tflops: 29.15,
            fp64_ratio: 1.0 / 64.0,
            fp64_tflops: 0.456,
            df64_tflops: 2.6,
            memory_bw_gbs: 504.0,
            l2_bytes: 36 * 1024 * 1024,
            infinity_cache_bytes: 0,
            tmu_count: 184,
            rop_count: 80,
            boost_ghz: 2.48,
            tmu_gtexels: 184.0 * 2.48,
            rop_gpixels: 80.0 * 2.48,
            tensor_fp16_tflops: 233.0,
            tensor_tf32_tflops: 117.0,
        };
    }

    // RTX 2070 (TU106, Turing)
    if name_lower.contains("2070") && !name_lower.contains("2070 super") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor,
            vram_bytes: 8 * 1024 * 1024 * 1024,
            fp32_tflops: 7.46,
            fp64_ratio: 1.0 / 32.0,
            fp64_tflops: 0.233,
            df64_tflops: 0.7,
            memory_bw_gbs: 448.0,
            l2_bytes: 4 * 1024 * 1024,
            infinity_cache_bytes: 0,
            tmu_count: 144,
            rop_count: 64,
            boost_ghz: 1.62,
            tmu_gtexels: 144.0 * 1.62,
            rop_gpixels: 64.0 * 1.62,
            tensor_fp16_tflops: 59.7,
            tensor_tf32_tflops: 0.0, // Turing has no TF32
        };
    }

    // RX 6950 XT (Navi 21, RDNA 2)
    if name_lower.contains("6950") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor,
            vram_bytes: 16 * 1024 * 1024 * 1024,
            fp32_tflops: 23.65,
            fp64_ratio: 1.0 / 16.0,
            fp64_tflops: 1.478,
            df64_tflops: 5.9, // estimate
            memory_bw_gbs: 576.0,
            l2_bytes: 4 * 1024 * 1024,
            infinity_cache_bytes: 128 * 1024 * 1024,
            tmu_count: 320,
            rop_count: 128,
            boost_ghz: 2.31,
            tmu_gtexels: 320.0 * 2.31,
            rop_gpixels: 128.0 * 2.31,
            tensor_fp16_tflops: 0.0, // no tensor cores
            tensor_tf32_tflops: 0.0,
        };
    }

    // RX 6900 XT (Navi 21, RDNA 2)
    if name_lower.contains("6900") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor,
            vram_bytes: 16 * 1024 * 1024 * 1024,
            fp32_tflops: 23.04,
            fp64_ratio: 1.0 / 16.0,
            fp64_tflops: 1.44,
            df64_tflops: 5.7,
            memory_bw_gbs: 512.0,
            l2_bytes: 4 * 1024 * 1024,
            infinity_cache_bytes: 128 * 1024 * 1024,
            tmu_count: 320,
            rop_count: 128,
            boost_ghz: 2.25,
            tmu_gtexels: 320.0 * 2.25,
            rop_gpixels: 128.0 * 2.25,
            tensor_fp16_tflops: 0.0,
            tensor_tf32_tflops: 0.0,
        };
    }

    // MI50 (Vega 20, GCN 5.0)
    if name_lower.contains("mi50") || name_lower.contains("vega 20") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor,
            vram_bytes: 16 * 1024 * 1024 * 1024, // 16 GB HBM2
            fp32_tflops: 13.4,
            fp64_ratio: 1.0 / 2.0,
            fp64_tflops: 6.7, // MI50 has 1:2 FP64
            df64_tflops: 3.4,
            memory_bw_gbs: 1024.0, // HBM2
            l2_bytes: 4 * 1024 * 1024,
            infinity_cache_bytes: 0,
            tmu_count: 240,
            rop_count: 64,
            boost_ghz: 1.75,
            tmu_gtexels: 240.0 * 1.75,
            rop_gpixels: 64.0 * 1.75,
            tensor_fp16_tflops: 0.0,
            tensor_tf32_tflops: 0.0,
        };
    }

    // Tesla P80 / P100 (GP100, Pascal)
    if name_lower.contains("p80") || name_lower.contains("p100") || name_lower.contains("gp100") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor,
            vram_bytes: 16 * 1024 * 1024 * 1024, // 16 GB HBM2
            fp32_tflops: 9.3,
            fp64_ratio: 1.0 / 2.0,
            fp64_tflops: 4.7, // GP100 has 1:2 FP64
            df64_tflops: 2.3,
            memory_bw_gbs: 732.0, // HBM2
            l2_bytes: 4 * 1024 * 1024,
            infinity_cache_bytes: 0,
            tmu_count: 224,
            rop_count: 96,
            boost_ghz: 1.33,
            tmu_gtexels: 224.0 * 1.33,
            rop_gpixels: 96.0 * 1.33,
            tensor_fp16_tflops: 0.0, // Pascal predates tensor cores
            tensor_tf32_tflops: 0.0,
        };
    }

    // V100 / Tesla V100 (GV100, Volta — data center variant)
    if name_lower.contains("v100") && !name_lower.contains("titan v") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor,
            vram_bytes: 32 * 1024 * 1024 * 1024, // 32 GB HBM2 (SXM2)
            fp32_tflops: 15.7,
            fp64_ratio: 1.0 / 2.0,
            fp64_tflops: 7.8,
            df64_tflops: 3.9,
            memory_bw_gbs: 900.0, // HBM2, SXM2
            l2_bytes: 6 * 1024 * 1024,
            infinity_cache_bytes: 0,
            tmu_count: 320,
            rop_count: 128,
            boost_ghz: 1.53,
            tmu_gtexels: 320.0 * 1.53,
            rop_gpixels: 128.0 * 1.53,
            tensor_fp16_tflops: 125.0,
            tensor_tf32_tflops: 0.0, // Volta predates TF32
        };
    }

    // Titan V (GV100, Volta)
    if name_lower.contains("titan v") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor,
            vram_bytes: 12 * 1024 * 1024 * 1024,
            fp32_tflops: 14.9,
            fp64_ratio: 1.0 / 2.0,
            fp64_tflops: 7.45,
            df64_tflops: 3.7,
            memory_bw_gbs: 653.0,
            l2_bytes: 4608 * 1024, // 4.5 MB
            infinity_cache_bytes: 0,
            tmu_count: 320,
            rop_count: 96,
            boost_ghz: 1.46,
            tmu_gtexels: 320.0 * 1.46,
            rop_gpixels: 96.0 * 1.46,
            tensor_fp16_tflops: 110.0,
            tensor_tf32_tflops: 0.0, // Volta predates TF32
        };
    }

    // === HPC Reference Cards (not local — for comparison tables) ===

    // A100 SXM (GA100, Ampere — CERN/USQCD reference)
    if name_lower.contains("a100") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor,
            vram_bytes: 80 * 1024 * 1024 * 1024, // 80 GB HBM2e
            fp32_tflops: 19.5,
            fp64_ratio: 1.0 / 2.0,
            fp64_tflops: 9.7, // GA100: 1:2 FP64
            df64_tflops: 4.9,
            memory_bw_gbs: 2039.0,      // HBM2e SXM
            l2_bytes: 40 * 1024 * 1024, // 40 MB
            infinity_cache_bytes: 0,
            tmu_count: 432,
            rop_count: 160,
            boost_ghz: 1.41,
            tmu_gtexels: 432.0 * 1.41,
            rop_gpixels: 160.0 * 1.41,
            tensor_fp16_tflops: 312.0,
            tensor_tf32_tflops: 156.0,
        };
    }

    // H100 SXM (GH100, Hopper — current HPC flagship)
    if name_lower.contains("h100") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor,
            vram_bytes: 80 * 1024 * 1024 * 1024, // 80 GB HBM3
            fp32_tflops: 67.0,
            fp64_ratio: 1.0 / 2.0,
            fp64_tflops: 34.0,
            df64_tflops: 17.0,
            memory_bw_gbs: 3350.0,      // HBM3 SXM
            l2_bytes: 50 * 1024 * 1024, // 50 MB
            infinity_cache_bytes: 0,
            tmu_count: 528,
            rop_count: 176,
            boost_ghz: 1.83,
            tmu_gtexels: 528.0 * 1.83,
            rop_gpixels: 176.0 * 1.83,
            tensor_fp16_tflops: 990.0,
            tensor_tf32_tflops: 495.0,
        };
    }

    // MI250X (CDNA2, Aldebaran — AMD HPC flagship)
    if name_lower.contains("mi250") || name_lower.contains("aldebaran") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor,
            vram_bytes: 128 * 1024 * 1024 * 1024, // 128 GB HBM2e
            fp32_tflops: 47.9,
            fp64_ratio: 1.0, // CDNA2: full-rate 1:1 FP64
            fp64_tflops: 47.9,
            df64_tflops: 12.0,
            memory_bw_gbs: 3277.0,      // HBM2e (dual GCD)
            l2_bytes: 16 * 1024 * 1024, // 8 MB per GCD × 2
            infinity_cache_bytes: 0,
            tmu_count: 0, // CDNA has no TMUs (compute-only die)
            rop_count: 0,
            boost_ghz: 1.7,
            tmu_gtexels: 0.0,
            rop_gpixels: 0.0,
            tensor_fp16_tflops: 383.0, // Matrix cores
            tensor_tf32_tflops: 95.7,  // FP64 matrix = 95.7 TFLOPS
        };
    }

    // llvmpipe / software fallback
    if name_lower.contains("llvmpipe") {
        return GpuSiliconBudget {
            name: name.to_string(),
            vendor: GpuVendor::Software,
            vram_bytes: 0,
            fp32_tflops: 0.05,
            fp64_ratio: 1.0,
            fp64_tflops: 0.05,
            df64_tflops: 0.01,
            memory_bw_gbs: 50.0,
            l2_bytes: 0,
            infinity_cache_bytes: 0,
            tmu_count: 0,
            rop_count: 0,
            boost_ghz: 0.0,
            tmu_gtexels: 0.0,
            rop_gpixels: 0.0,
            tensor_fp16_tflops: 0.0,
            tensor_tf32_tflops: 0.0,
        };
    }

    // Unknown GPU — provide zeroes so the binary still runs
    GpuSiliconBudget {
        name: name.to_string(),
        vendor,
        vram_bytes: 0,
        fp32_tflops: 0.0,
        fp64_ratio: 0.0,
        fp64_tflops: 0.0,
        df64_tflops: 0.0,
        memory_bw_gbs: 0.0,
        l2_bytes: 0,
        infinity_cache_bytes: 0,
        tmu_count: 0,
        rop_count: 0,
        boost_ghz: 0.0,
        tmu_gtexels: 0.0,
        rop_gpixels: 0.0,
        tensor_fp16_tflops: 0.0,
        tensor_tf32_tflops: 0.0,
    }
}

/// SU(3) lattice QCD working-set estimate for a given lattice volume.
///
/// Returns (link_bytes, total_bytes) where total includes links, momenta,
/// force buffer, neighbor table, plaquette output, and CG vectors.
fn qcd_working_set(n_sites: u64) -> (u64, u64) {
    let n_links = n_sites * 4; // 4 directions
    let su3_bytes = 18 * 8; // 18 f64 = 144 bytes per link
    let link_bytes = n_links * su3_bytes;
    let momenta_bytes = n_links * su3_bytes;
    let force_bytes = n_links * su3_bytes;
    let nbr_bytes = n_sites * 8 * 4; // 8 neighbors × 4 bytes
    let plaq_bytes = n_sites * 8; // 1 f64 per site
    let cg_vectors = 5 * n_sites * 6 * 8; // 5 CG vecs × 6 complex per site × 8 bytes
    let total = link_bytes + momenta_bytes + force_bytes + nbr_bytes + plaq_bytes + cg_vectors;
    (link_bytes, total)
}

fn print_budget(budget: &GpuSiliconBudget) {
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!(
        "  │ {} ({})                          ",
        budget.name, budget.vendor
    );
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │ SHADER CORES                                           │");
    println!(
        "  │   FP32:  {:>8.2} TFLOPS                               │",
        budget.fp32_tflops
    );
    println!(
        "  │   DF64:  {:>8.2} TFLOPS (measured/estimated)          │",
        budget.df64_tflops
    );
    println!(
        "  │   FP64:  {:>8.3} TFLOPS (1:{} native)                │",
        budget.fp64_tflops,
        (1.0 / budget.fp64_ratio) as u32
    );
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │ TENSOR CORES                                           │");
    if budget.tensor_fp16_tflops > 0.0 {
        println!(
            "  │   FP16:  {:>8.1} TFLOPS (NOT accessible via WGSL)    │",
            budget.tensor_fp16_tflops
        );
        if budget.tensor_tf32_tflops > 0.0 {
            println!(
                "  │   TF32:  {:>8.1} TFLOPS (requires sovereign SASS)    │",
                budget.tensor_tf32_tflops
            );
        }
    } else {
        println!("  │   (none — AMD RDNA / GCN)                             │");
    }
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │ MEMORY                                                  │");
    println!(
        "  │   VRAM:  {:>6} MB                                      │",
        budget.vram_bytes / (1024 * 1024)
    );
    println!(
        "  │   BW:    {:>8.1} GB/s                                  │",
        budget.memory_bw_gbs
    );
    println!(
        "  │   L2:    {:>6} KB                                      │",
        budget.l2_bytes / 1024
    );
    if budget.infinity_cache_bytes > 0 {
        println!(
            "  │   IC:    {:>6} MB (AMD Infinity Cache)                │",
            budget.infinity_cache_bytes / (1024 * 1024)
        );
    }
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │ FIXED-FUNCTION UNITS                                    │");
    println!(
        "  │   TMU:   {:>4} units @ {:.2} GHz → {:.1} GT/s         │",
        budget.tmu_count, budget.boost_ghz, budget.tmu_gtexels
    );
    println!(
        "  │   ROP:   {:>4} units @ {:.2} GHz → {:.1} GP/s         │",
        budget.rop_count, budget.boost_ghz, budget.rop_gpixels
    );
    println!("  └─────────────────────────────────────────────────────────┘");
}

fn print_compound_budget(budget: &GpuSiliconBudget) {
    println!("\n  ── Compound Budget (parallel sub-problems) ──\n");

    let shader_equiv = budget.fp32_tflops;
    let tensor_equiv = if budget.tensor_tf32_tflops > 0.0 {
        budget.tensor_tf32_tflops * 0.3 // 30% utilization estimate for MMA reshape
    } else {
        0.0
    };
    // TMU: each texel fetch is ~2 FMA equivalent (interpolation + lookup)
    let tmu_equiv = budget.tmu_gtexels * 2.0 / 1000.0; // GT/s × 2 FLOP → TFLOPS
    // ROP: each atomic blend is ~1 FMA equivalent
    let rop_equiv = budget.rop_gpixels / 1000.0;

    let compound_low = shader_equiv + tmu_equiv;
    let compound_high = shader_equiv + tensor_equiv + tmu_equiv + rop_equiv;

    println!("  Shader cores alone:         {shader_equiv:>8.2} TFLOPS (FP32)");
    println!(
        "  + TMU table lookups:        {:>8.2} TFLOPS equiv ({:.1} GT/s × 2 FLOP/texel)",
        tmu_equiv, budget.tmu_gtexels
    );
    if tensor_equiv > 0.0 {
        println!(
            "  + Tensor MMA (30% util):    {:>8.2} TFLOPS equiv (of {:.1} TF32 peak)",
            tensor_equiv, budget.tensor_tf32_tflops
        );
    }
    println!(
        "  + ROP atomic blend:         {:>8.2} TFLOPS equiv ({:.1} GP/s)",
        rop_equiv, budget.rop_gpixels
    );
    println!();
    println!("  Conservative compound:      {compound_low:>8.2} TFLOPS (shader + TMU)");
    println!("  Optimistic compound:        {compound_high:>8.2} TFLOPS (all units parallel)");
    println!(
        "  Multiplier over shader:     {:.2}x – {:.2}x",
        compound_low / shader_equiv.max(0.001),
        compound_high / shader_equiv.max(0.001)
    );
}

fn print_working_set_analysis(budget: &GpuSiliconBudget) {
    println!("\n  ── QCD Working-Set Analysis ──\n");

    let lattice_sizes: &[(u64, &str)] = &[
        (256, "4^4"),
        (4096, "8^4"),
        (8192, "8^3×16"),
        (65536, "16^4"),
        (1_048_576, "32^4"),
        (16_777_216, "64^4"),
    ];

    println!(
        "  {:<10} {:>12} {:>12}  {:>10}  Cache fit",
        "Lattice", "Links (MB)", "Total (MB)", "VRAM %"
    );
    println!("  {}", "─".repeat(72));

    for (n_sites, label) in lattice_sizes {
        let (link_bytes, total_bytes) = qcd_working_set(*n_sites);
        let link_mb = link_bytes as f64 / (1024.0 * 1024.0);
        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        let vram_pct = if budget.vram_bytes > 0 {
            100.0 * total_bytes as f64 / budget.vram_bytes as f64
        } else {
            f64::INFINITY
        };

        let cache_fit = if total_bytes <= budget.l2_bytes {
            "L2 ✓"
        } else if budget.infinity_cache_bytes > 0 && total_bytes <= budget.infinity_cache_bytes {
            "Infinity Cache ✓"
        } else if budget.vram_bytes > 0 && total_bytes <= budget.vram_bytes {
            "VRAM only"
        } else {
            "EXCEEDS VRAM"
        };

        println!("  {label:<10} {link_mb:>10.2} {total_mb:>12.2}  {vram_pct:>8.1}%  {cache_fit}");
    }

    println!();

    if budget.infinity_cache_bytes > 0 {
        println!(
            "  AMD Infinity Cache advantage: {:.0} MB of effective L2+IC",
            (budget.l2_bytes + budget.infinity_cache_bytes) as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  At 16^4 ({:.1} MB), Infinity Cache holds the full working set",
            qcd_working_set(65536).1 as f64 / (1024.0 * 1024.0)
        );
        println!("  This gives AMD a bandwidth advantage for cache-resident lattice sizes");
    }
}

fn print_precision_tier_analysis(budget: &GpuSiliconBudget) {
    println!("\n  ── Precision Tier Throughput ──\n");

    let tiers: &[(&str, f64, &str)] = &[
        ("int2", budget.fp32_tflops * 16.0, "16× FP32 (bit-packed)"),
        ("int4", budget.fp32_tflops * 8.0, "8× FP32 (bit-packed)"),
        ("int8", budget.fp32_tflops * 4.0, "4× FP32 (byte ops)"),
        ("fp8", budget.fp32_tflops * 4.0, "4× FP32 (e4m3/e5m2)"),
        ("bf16", budget.fp32_tflops * 2.0, "2× FP32 (brain float)"),
        ("fp16", budget.fp32_tflops * 2.0, "2× FP32 (IEEE half)"),
        ("fp32", budget.fp32_tflops, "native FP32 ALU"),
        (
            "df64",
            budget.df64_tflops,
            "FP32 pairs (Dekker, ~48-bit mantissa)",
        ),
        ("fp64", budget.fp64_tflops, "native FP64 ALU"),
        (
            "df128",
            budget.df64_tflops / 4.0,
            "double-f64 or quad-f32 (~96-bit)",
        ),
        (
            "qf128",
            budget.fp64_tflops / 4.0,
            "quad-f64 (~208-bit, extreme)",
        ),
    ];

    println!("  {:<8} {:>10}  Implementation", "Tier", "TFLOPS");
    println!("  {}", "─".repeat(60));

    for (tier, tflops, desc) in tiers {
        if *tflops > 0.001 {
            println!("  {tier:<8} {tflops:>8.2}  {desc}");
        } else {
            println!("  {:<8} {:>8}  {}", tier, "N/A", desc);
        }
    }
}

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Silicon Budget Calculator");
    println!("  Theoretical peak throughput per GPU per silicon unit");
    println!("═══════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    if adapters.is_empty() {
        eprintln!("No GPU adapters found.");
        std::process::exit(1);
    }

    let mut measurements: Vec<PerformanceMeasurement> = Vec::new();
    let ts = toadstool_report::epoch_now();

    for adapter in &adapters {
        let info = adapter.get_info();
        let vendor = classify_vendor(&info);
        let budget = lookup_silicon_specs(&info.name, vendor);

        println!("━━━ {} ━━━\n", info.name);

        print_budget(&budget);
        print_compound_budget(&budget);
        print_working_set_analysis(&budget);
        print_precision_tier_analysis(&budget);

        // Report theoretical peaks to toadStool
        let units = [
            ("theoretical.shader_core.fp32", budget.fp32_tflops),
            ("theoretical.shader_core.df64", budget.df64_tflops),
            ("theoretical.shader_core.fp64", budget.fp64_tflops),
            ("theoretical.memory.bandwidth_gbs", budget.memory_bw_gbs),
            ("theoretical.tmu.gtexels", budget.tmu_gtexels),
            ("theoretical.rop.gpixels", budget.rop_gpixels),
            ("theoretical.tensor.fp16", budget.tensor_fp16_tflops),
            ("theoretical.tensor.tf32", budget.tensor_tf32_tflops),
        ];

        for (op, value) in &units {
            if *value > 0.0 {
                measurements.push(PerformanceMeasurement {
                    operation: (*op).to_string(),
                    silicon_unit: op.split('.').nth(1).unwrap_or("unknown").to_string(),
                    precision_mode: "theoretical_peak".into(),
                    throughput_gflops: *value * 1000.0, // TFLOPS → GFLOPS
                    tolerance_achieved: 0.0,
                    gpu_model: info.name.clone(),
                    measured_by: "hotSpring/bench_silicon_budget".into(),
                    timestamp: ts,
                });
            }
        }

        println!();
    }

    // Cross-GPU comparison
    if adapters.len() >= 2 {
        println!("═══════════════════════════════════════════════════════════");
        println!("  Cross-GPU Comparison");
        println!("═══════════════════════════════════════════════════════════\n");

        let budgets: Vec<GpuSiliconBudget> = adapters
            .iter()
            .map(|a| {
                let info = a.get_info();
                lookup_silicon_specs(&info.name, classify_vendor(&info))
            })
            .filter(|b| b.fp32_tflops > 0.0)
            .collect();

        if budgets.len() >= 2 {
            println!(
                "  {:<28} {:>10} {:>10} {:>10} {:>10}",
                "Metric",
                &budgets[0].name[..budgets[0].name.len().min(10)],
                &budgets[1].name[..budgets[1].name.len().min(10)],
                "Ratio",
                "Advantage"
            );
            println!("  {}", "─".repeat(72));

            let comparisons: &[(&str, f64, f64)] = &[
                (
                    "FP32 TFLOPS",
                    budgets[0].fp32_tflops,
                    budgets[1].fp32_tflops,
                ),
                (
                    "DF64 TFLOPS",
                    budgets[0].df64_tflops,
                    budgets[1].df64_tflops,
                ),
                (
                    "FP64 TFLOPS",
                    budgets[0].fp64_tflops,
                    budgets[1].fp64_tflops,
                ),
                (
                    "Memory GB/s",
                    budgets[0].memory_bw_gbs,
                    budgets[1].memory_bw_gbs,
                ),
                ("TMU GT/s", budgets[0].tmu_gtexels, budgets[1].tmu_gtexels),
                ("ROP GP/s", budgets[0].rop_gpixels, budgets[1].rop_gpixels),
                (
                    "L2+IC (MB)",
                    (budgets[0].l2_bytes + budgets[0].infinity_cache_bytes) as f64 / 1_048_576.0,
                    (budgets[1].l2_bytes + budgets[1].infinity_cache_bytes) as f64 / 1_048_576.0,
                ),
                (
                    "VRAM (GB)",
                    budgets[0].vram_bytes as f64 / 1e9,
                    budgets[1].vram_bytes as f64 / 1e9,
                ),
            ];

            for (metric, a, b) in comparisons {
                if *a > 0.0 && *b > 0.0 {
                    let ratio = a / b;
                    let advantage = if ratio > 1.05 {
                        &budgets[0].name
                    } else if ratio < 0.95 {
                        &budgets[1].name
                    } else {
                        "~parity"
                    };
                    println!("  {metric:<28} {a:>10.2} {b:>10.2} {ratio:>10.2}x  {advantage}");
                }
            }
        }
    }

    println!(
        "\n── Reporting {} measurements to toadStool ──\n",
        measurements.len()
    );
    toadstool_report::report_to_toadstool(&measurements);

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Silicon Budget Calculator Complete");
    println!("═══════════════════════════════════════════════════════════");
}
