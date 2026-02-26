// SPDX-License-Identifier: AGPL-3.0-only

//! Hardware probing — GPU via wgpu/barracuda, NPU and CPU locally.
//!
//! GPU discovery leans on wgpu (which toadstool/barracuda uses). We get
//! adapter name, device type, driver, backend, and feature flags (`SHADER_F64`)
//! directly from the Vulkan/wgpu layer — no sysfs reimplementation needed.
//!
//! NPU discovery is local (probing `/dev/akida*`). This is evolution that
//! toadstool can absorb once NPU substrate support matures upstream.
//!
//! CPU discovery reads `/proc/cpuinfo` for model, core count, and SIMD flags.

use crate::substrate::{Capability, Fp64Rate, Identity, Properties, Substrate, SubstrateKind};
use std::fs;

/// Probe all GPU adapters via wgpu.
///
/// Uses the same wgpu instance/backend configuration that barracuda uses.
/// Each adapter becomes a substrate with capabilities derived from its
/// feature flags (`SHADER_F64` → `F64Compute`, etc.).
#[must_use]
pub fn probe_gpus() -> Vec<Substrate> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let mut gpus = Vec::new();

    for (idx, adapter) in adapters.into_iter().enumerate() {
        let info = adapter.get_info();
        let features = adapter.features();

        if info.device_type == wgpu::DeviceType::Cpu {
            continue;
        }

        let has_f64 = features.contains(wgpu::Features::SHADER_F64);
        let has_timestamps = features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let fp64_rate = classify_fp64_rate(&info.name);
        let supports_df64 = features.contains(wgpu::Features::SHADER_F16);

        let mut capabilities = vec![
            Capability::F32Compute,
            Capability::ShaderDispatch,
            Capability::StreamingStage,
        ];
        if has_f64 {
            capabilities.push(Capability::F64Compute);
            capabilities.push(Capability::ScalarReduce);
            capabilities.push(Capability::SparseSpMV);
            capabilities.push(Capability::Eigensolve);
            capabilities.push(Capability::ConjugateGradient);
        }
        if supports_df64 {
            capabilities.push(Capability::DF64Compute);
        }
        if has_timestamps {
            capabilities.push(Capability::TimestampQuery);
        }
        capabilities.push(Capability::PcieTransfer);

        let limits = adapter.limits();
        let max_buffer = limits.max_buffer_size;

        gpus.push(Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity {
                name: info.name.clone(),
                driver: Some(format!("{} ({})", info.driver, info.driver_info)),
                backend: Some(format!("{:?}", info.backend)),
                adapter_index: Some(idx),
                device_node: None,
                pci_id: None,
            },
            properties: Properties {
                memory_bytes: Some(max_buffer),
                has_f64,
                has_timestamps,
                fp64_rate: Some(fp64_rate),
                has_df64: supports_df64,
                ..Properties::default()
            },
            capabilities,
        });
    }

    gpus
}

/// Probe CPU via `/proc/cpuinfo` and `/proc/meminfo`.
#[must_use]
pub fn probe_cpu() -> Substrate {
    let (model, cores, threads, cache_kb, has_avx2) = parse_cpuinfo();
    let mem_bytes = parse_meminfo();

    let name = model.unwrap_or_else(|| String::from("Unknown CPU"));

    let mut capabilities = vec![
        Capability::F64Compute,
        Capability::F32Compute,
        Capability::SparseSpMV,
        Capability::Eigensolve,
        Capability::ConjugateGradient,
    ];
    if has_avx2 {
        capabilities.push(Capability::SimdVector);
    }

    Substrate {
        kind: SubstrateKind::Cpu,
        identity: Identity::named(name),
        properties: Properties {
            memory_bytes: mem_bytes,
            core_count: cores,
            thread_count: threads,
            cache_kb,
            ..Properties::default()
        },
        capabilities,
    }
}

/// Probe for NPU devices.
///
/// Discovers `BrainChip` AKD1000 via `/dev/akida0` and enriches properties
/// from PCIe sysfs (SRAM, vendor ID). This is local evolution — toadstool
/// doesn't have NPU substrate support yet, so we probe directly. Once
/// toadstool absorbs NPU dispatch, we lean on that.
///
/// # Absorption target: `toadstool::device::npu`
///
/// This probe should become `barracuda::device::npu::detect_akida_boards()`
/// once toadstool adds NPU substrate support.
#[must_use]
pub fn probe_npus() -> Vec<Substrate> {
    let mut npus = Vec::new();

    for idx in 0..4 {
        let dev_path = format!("/dev/akida{idx}");
        let path = std::path::Path::new(&dev_path);
        if !path.exists() {
            continue;
        }

        let pci_id = scan_akida_pci();

        npus.push(Substrate {
            kind: SubstrateKind::Npu,
            identity: Identity {
                name: String::from("BrainChip AKD1000"),
                device_node: Some(dev_path),
                pci_id,
                ..Identity::named("BrainChip AKD1000")
            },
            properties: Properties {
                memory_bytes: Some(AKIDA_SRAM_BYTES),
                ..Properties::default()
            },
            capabilities: vec![
                Capability::F32Compute,
                Capability::QuantizedInference { bits: 8 },
                Capability::QuantizedInference { bits: 4 },
                Capability::BatchInference { max_batch: 8 },
                Capability::WeightMutation,
                Capability::PcieTransfer,
                Capability::StreamingStage,
            ],
        });
    }

    npus
}

/// Scan PCIe bus for Akida vendor:device ID via sysfs.
fn scan_akida_pci() -> Option<String> {
    const BRAINCHIP_VENDOR: &str = "0x1e7c";
    let Ok(entries) = fs::read_dir("/sys/bus/pci/devices") else {
        return None;
    };
    for entry in entries.flatten() {
        let vendor_path = entry.path().join("vendor");
        if let Ok(vendor) = fs::read_to_string(&vendor_path) {
            if vendor.trim() == BRAINCHIP_VENDOR {
                let device_path = entry.path().join("device");
                let device = fs::read_to_string(device_path).unwrap_or_default();
                return Some(format!("{}:{}", BRAINCHIP_VENDOR, device.trim()));
            }
        }
    }
    None
}

/// AKD1000 has 8 MB SRAM (fixed architecture).
const AKIDA_SRAM_BYTES: u64 = 8 * 1024 * 1024;

/// Classify a GPU's native FP64:FP32 throughput ratio from its name.
///
/// Volta (Titan V, V100): 1:2. Datacenter (A100, H100): 1:1 or 1:2.
/// Consumer Ampere/Ada/Turing: 1:64. This heuristic covers the GPUs we
/// actually have; toadstool's `GpuDriverProfile` does deeper calibration.
fn classify_fp64_rate(name: &str) -> Fp64Rate {
    let name_lower = name.to_lowercase();
    if name_lower.contains("a100") || name_lower.contains("h100") {
        Fp64Rate::Full
    } else if name_lower.contains("titan v")
        || name_lower.contains("v100")
        || name_lower.contains("gv100")
    {
        Fp64Rate::Half
    } else {
        Fp64Rate::Narrow
    }
}

fn parse_cpuinfo() -> (Option<String>, Option<u32>, Option<u32>, Option<u32>, bool) {
    let Ok(content) = fs::read_to_string("/proc/cpuinfo") else {
        return (None, None, None, None, false);
    };

    let mut model = None;
    let mut cores = None;
    let mut siblings = None;
    let mut cache_kb = None;
    let mut has_avx2 = false;

    for line in content.lines() {
        if let Some((key, val)) = line.split_once(':') {
            let key = key.trim();
            let val = val.trim();
            match key {
                "model name" if model.is_none() => model = Some(val.to_string()),
                "cpu cores" if cores.is_none() => cores = val.parse().ok(),
                "siblings" if siblings.is_none() => siblings = val.parse().ok(),
                "cache size" if cache_kb.is_none() => {
                    cache_kb = val.trim_end_matches(" KB").parse().ok();
                }
                "flags" if !has_avx2 => {
                    has_avx2 = val.split_whitespace().any(|f| f == "avx2");
                }
                _ => {}
            }
        }
    }

    (model, cores, siblings, cache_kb, has_avx2)
}

fn parse_meminfo() -> Option<u64> {
    let content = fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb_str = rest.trim().trim_end_matches(" kB").trim();
            let kb: u64 = kb_str.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn cpu_always_discovered() {
        let cpu = probe_cpu();
        assert_eq!(cpu.kind, SubstrateKind::Cpu);
        assert!(cpu.has(&Capability::F64Compute));
        assert!(!cpu.identity.name.is_empty());
    }

    #[test]
    fn gpu_probe_uses_wgpu() {
        let gpus = probe_gpus();
        for gpu in &gpus {
            assert_eq!(gpu.kind, SubstrateKind::Gpu);
            assert!(gpu.has(&Capability::ShaderDispatch));
            assert!(gpu.identity.adapter_index.is_some());
            assert!(gpu.identity.driver.is_some());
        }
    }
}
