// SPDX-License-Identifier: AGPL-3.0-or-later

//! Vendor spec-sheet lookups: classify GPU adapters and populate theoretical peaks.

use std::collections::BTreeMap;

use super::silicon_profile::{GpuVendorTag, SiliconProfile, SiliconUnit, UnitThroughput};

/// Classify vendor from a wgpu `AdapterInfo` vendor ID.
#[must_use]
pub fn classify_vendor(vendor_id: u32, name: &str) -> GpuVendorTag {
    match vendor_id {
        0x10DE => GpuVendorTag::Nvidia,
        0x1002 => GpuVendorTag::Amd,
        0x8086 => GpuVendorTag::Intel,
        _ => {
            let lower = name.to_lowercase();
            if lower.contains("llvmpipe") || lower.contains("software") {
                GpuVendorTag::Software
            } else {
                GpuVendorTag::Unknown
            }
        }
    }
}

/// Build a `SiliconProfile` from known spec-sheet data for an adapter.
///
/// All `measured_peak` fields start at 0.0. Run `bench_silicon_profile`
/// to fill them in with actual hardware measurements.
#[must_use]
pub fn from_spec_sheet(name: &str, vendor_id: u32) -> SiliconProfile {
    let vendor = classify_vendor(vendor_id, name);
    let lower = name.to_lowercase();
    let mut p = SiliconProfile {
        adapter_name: name.to_string(),
        vendor,
        vram_bytes: 0,
        boost_ghz: 0.0,
        units: BTreeMap::new(),
        compositions: Vec::new(),
        df64_tflops: 0.0,
        l2_bytes: 0,
        infinity_cache_bytes: 0,
        tmu_count: 0,
        rop_count: 0,
        subgroup_size: match vendor {
            GpuVendorTag::Nvidia => 32,
            GpuVendorTag::Amd => 64, // RDNA can do 32 or 64, wavefront64 is default in compute
            _ => 0,
        },
        measured_at: String::new(),
    };

    // RTX 3090 (GA102, Ampere)
    if lower.contains("3090") {
        p.vram_bytes = 24 * 1024 * 1024 * 1024;
        p.boost_ghz = 1.70;
        p.df64_tflops = 3.24;
        p.l2_bytes = 6 * 1024 * 1024;
        p.tmu_count = 328;
        p.rop_count = 112;
        insert_unit(&mut p, SiliconUnit::Fp32Alu, 35.6, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Fp64Alu, 0.556, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Tmu, 328.0 * 1.70, "GT/s");
        insert_unit(&mut p, SiliconUnit::Rop, 112.0 * 1.70, "GP/s");
        insert_unit(&mut p, SiliconUnit::TensorCore, 71.0, "TF32 TFLOPS");
        insert_unit(&mut p, SiliconUnit::MemoryBandwidth, 936.0, "GB/s");
        return p;
    }

    // RX 6950 XT (Navi 21, RDNA 2)
    if lower.contains("6950") {
        p.vram_bytes = 16 * 1024 * 1024 * 1024;
        p.boost_ghz = 2.31;
        p.df64_tflops = 5.9;
        p.l2_bytes = 4 * 1024 * 1024;
        p.infinity_cache_bytes = 128 * 1024 * 1024;
        p.tmu_count = 320;
        p.rop_count = 128;
        insert_unit(&mut p, SiliconUnit::Fp32Alu, 23.65, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Fp64Alu, 1.478, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Tmu, 320.0 * 2.31, "GT/s");
        insert_unit(&mut p, SiliconUnit::Rop, 128.0 * 2.31, "GP/s");
        insert_unit(&mut p, SiliconUnit::MemoryBandwidth, 576.0, "GB/s");
        return p;
    }

    // RTX 4070 (AD104, Ada Lovelace)
    if lower.contains("4070") && !lower.contains("4070 ti") {
        p.vram_bytes = 12 * 1024 * 1024 * 1024;
        p.boost_ghz = 2.48;
        p.df64_tflops = 2.6;
        p.l2_bytes = 36 * 1024 * 1024;
        p.tmu_count = 184;
        p.rop_count = 80;
        insert_unit(&mut p, SiliconUnit::Fp32Alu, 29.15, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Fp64Alu, 0.456, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Tmu, 184.0 * 2.48, "GT/s");
        insert_unit(&mut p, SiliconUnit::Rop, 80.0 * 2.48, "GP/s");
        insert_unit(&mut p, SiliconUnit::TensorCore, 117.0, "TF32 TFLOPS");
        insert_unit(&mut p, SiliconUnit::MemoryBandwidth, 504.0, "GB/s");
        return p;
    }

    // RTX 5090 (GB202, Blackwell)
    if lower.contains("5090") {
        p.vram_bytes = 32 * 1024 * 1024 * 1024;
        p.boost_ghz = 2.41;
        p.df64_tflops = 10.0;
        p.l2_bytes = 36 * 1024 * 1024;
        p.tmu_count = 512;
        p.rop_count = 176;
        insert_unit(&mut p, SiliconUnit::Fp32Alu, 104.8, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Fp64Alu, 1.638, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Tmu, 512.0 * 2.41, "GT/s");
        insert_unit(&mut p, SiliconUnit::Rop, 176.0 * 2.41, "GP/s");
        insert_unit(&mut p, SiliconUnit::TensorCore, 419.0, "TF32 TFLOPS");
        insert_unit(&mut p, SiliconUnit::MemoryBandwidth, 1792.0, "GB/s");
        return p;
    }

    // RTX 2070 (TU106, Turing)
    if lower.contains("2070") && !lower.contains("2070 super") {
        p.vram_bytes = 8 * 1024 * 1024 * 1024;
        p.boost_ghz = 1.62;
        p.df64_tflops = 0.7;
        p.l2_bytes = 4 * 1024 * 1024;
        p.tmu_count = 144;
        p.rop_count = 64;
        insert_unit(&mut p, SiliconUnit::Fp32Alu, 7.46, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Fp64Alu, 0.233, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Tmu, 144.0 * 1.62, "GT/s");
        insert_unit(&mut p, SiliconUnit::Rop, 64.0 * 1.62, "GP/s");
        insert_unit(&mut p, SiliconUnit::TensorCore, 59.7, "FP16 TFLOPS");
        insert_unit(&mut p, SiliconUnit::MemoryBandwidth, 448.0, "GB/s");
        return p;
    }

    // RX 6900 XT (Navi 21, RDNA 2)
    if lower.contains("6900") {
        p.vram_bytes = 16 * 1024 * 1024 * 1024;
        p.boost_ghz = 2.25;
        p.df64_tflops = 5.7;
        p.l2_bytes = 4 * 1024 * 1024;
        p.infinity_cache_bytes = 128 * 1024 * 1024;
        p.tmu_count = 320;
        p.rop_count = 128;
        insert_unit(&mut p, SiliconUnit::Fp32Alu, 23.04, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Fp64Alu, 1.44, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Tmu, 320.0 * 2.25, "GT/s");
        insert_unit(&mut p, SiliconUnit::Rop, 128.0 * 2.25, "GP/s");
        insert_unit(&mut p, SiliconUnit::MemoryBandwidth, 512.0, "GB/s");
        return p;
    }

    // A100 SXM (GA100, Ampere — HPC reference)
    if lower.contains("a100") {
        p.vram_bytes = 80 * 1024 * 1024 * 1024;
        p.boost_ghz = 1.41;
        p.df64_tflops = 4.9;
        p.l2_bytes = 40 * 1024 * 1024;
        p.tmu_count = 432;
        p.rop_count = 160;
        insert_unit(&mut p, SiliconUnit::Fp32Alu, 19.5, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Fp64Alu, 9.7, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Tmu, 432.0 * 1.41, "GT/s");
        insert_unit(&mut p, SiliconUnit::Rop, 160.0 * 1.41, "GP/s");
        insert_unit(&mut p, SiliconUnit::TensorCore, 156.0, "TF32 TFLOPS");
        insert_unit(&mut p, SiliconUnit::MemoryBandwidth, 2039.0, "GB/s");
        return p;
    }

    // H100 SXM (GH100, Hopper)
    if lower.contains("h100") {
        p.vram_bytes = 80 * 1024 * 1024 * 1024;
        p.boost_ghz = 1.83;
        p.df64_tflops = 17.0;
        p.l2_bytes = 50 * 1024 * 1024;
        p.tmu_count = 528;
        p.rop_count = 176;
        insert_unit(&mut p, SiliconUnit::Fp32Alu, 67.0, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Fp64Alu, 34.0, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Tmu, 528.0 * 1.83, "GT/s");
        insert_unit(&mut p, SiliconUnit::Rop, 176.0 * 1.83, "GP/s");
        insert_unit(&mut p, SiliconUnit::TensorCore, 495.0, "TF32 TFLOPS");
        insert_unit(&mut p, SiliconUnit::MemoryBandwidth, 3350.0, "GB/s");
        return p;
    }

    // MI250X (CDNA2 — no TMU/ROP, compute-only die)
    if lower.contains("mi250") || lower.contains("aldebaran") {
        p.vram_bytes = 128 * 1024 * 1024 * 1024;
        p.boost_ghz = 1.7;
        p.df64_tflops = 12.0;
        p.l2_bytes = 16 * 1024 * 1024;
        insert_unit(&mut p, SiliconUnit::Fp32Alu, 47.9, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Fp64Alu, 47.9, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::TensorCore, 95.7, "FP64 Matrix TFLOPS");
        insert_unit(&mut p, SiliconUnit::MemoryBandwidth, 3277.0, "GB/s");
        return p;
    }

    // V100 (GV100, Volta)
    if lower.contains("v100") {
        p.vram_bytes = 32 * 1024 * 1024 * 1024;
        p.boost_ghz = 1.53;
        p.df64_tflops = 3.9;
        p.l2_bytes = 6 * 1024 * 1024;
        p.tmu_count = 320;
        p.rop_count = 128;
        insert_unit(&mut p, SiliconUnit::Fp32Alu, 15.7, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Fp64Alu, 7.8, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Tmu, 320.0 * 1.53, "GT/s");
        insert_unit(&mut p, SiliconUnit::Rop, 128.0 * 1.53, "GP/s");
        insert_unit(&mut p, SiliconUnit::TensorCore, 125.0, "FP16 TFLOPS");
        insert_unit(&mut p, SiliconUnit::MemoryBandwidth, 900.0, "GB/s");
        return p;
    }

    // MI50 (Vega 20, GCN 5.0)
    if lower.contains("mi50") || lower.contains("vega 20") {
        p.vram_bytes = 16 * 1024 * 1024 * 1024;
        p.boost_ghz = 1.75;
        p.df64_tflops = 3.4;
        p.l2_bytes = 4 * 1024 * 1024;
        p.tmu_count = 240;
        p.rop_count = 64;
        insert_unit(&mut p, SiliconUnit::Fp32Alu, 13.4, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Fp64Alu, 6.7, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Tmu, 240.0 * 1.75, "GT/s");
        insert_unit(&mut p, SiliconUnit::Rop, 64.0 * 1.75, "GP/s");
        insert_unit(&mut p, SiliconUnit::MemoryBandwidth, 1024.0, "GB/s");
        return p;
    }

    // Titan V (GV100, Volta)
    if lower.contains("titan v") {
        p.vram_bytes = 12 * 1024 * 1024 * 1024;
        p.boost_ghz = 1.46;
        p.df64_tflops = 3.7;
        p.l2_bytes = 4608 * 1024;
        p.tmu_count = 320;
        p.rop_count = 96;
        insert_unit(&mut p, SiliconUnit::Fp32Alu, 14.9, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Fp64Alu, 7.45, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Tmu, 320.0 * 1.46, "GT/s");
        insert_unit(&mut p, SiliconUnit::Rop, 96.0 * 1.46, "GP/s");
        insert_unit(&mut p, SiliconUnit::TensorCore, 110.0, "FP16 TFLOPS");
        insert_unit(&mut p, SiliconUnit::MemoryBandwidth, 653.0, "GB/s");
        return p;
    }

    // Tesla P100 (GP100, Pascal)
    if lower.contains("p100") || lower.contains("p80") || lower.contains("gp100") {
        p.vram_bytes = 16 * 1024 * 1024 * 1024;
        p.boost_ghz = 1.33;
        p.df64_tflops = 2.3;
        p.l2_bytes = 4 * 1024 * 1024;
        p.tmu_count = 224;
        p.rop_count = 96;
        insert_unit(&mut p, SiliconUnit::Fp32Alu, 9.3, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Fp64Alu, 4.7, "TFLOPS");
        insert_unit(&mut p, SiliconUnit::Tmu, 224.0 * 1.33, "GT/s");
        insert_unit(&mut p, SiliconUnit::Rop, 96.0 * 1.33, "GP/s");
        insert_unit(&mut p, SiliconUnit::MemoryBandwidth, 732.0, "GB/s");
        return p;
    }

    p
}

fn insert_unit(profile: &mut SiliconProfile, unit: SiliconUnit, peak: f64, unit_name: &str) {
    profile.units.insert(
        unit,
        UnitThroughput {
            theoretical_peak: peak,
            measured_peak: 0.0,
            efficiency: 0.0,
            unit: unit_name.to_string(),
            idle_watts: 0.0,
            loaded_watts: 0.0,
            delta_watts: 0.0,
            ops_per_watt: 0.0,
        },
    );
}
