// SPDX-License-Identifier: AGPL-3.0-only

//! Hardware probing: CPU, memory, GPU detection, RAPL energy.
//!
//! Reads from Linux /proc, /sys, and nvidia-smi to build a hardware inventory.

use serde::{Deserialize, Serialize};
use std::process::Command;

/// Complete hardware description captured once at the start of a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInventory {
    pub gate_name: String,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub cpu_cache_kb: usize,
    pub ram_total_mb: usize,
    pub gpu_name: String,
    pub gpu_vram_mb: usize,
    pub gpu_driver: String,
    pub gpu_compute_cap: String,
    pub os_kernel: String,
    pub rust_version: String,
}

impl HardwareInventory {
    /// Auto-detect hardware from Linux sysfs / nvidia-smi.
    #[must_use]
    pub fn detect(gate_name: &str) -> Self {
        let (cpu_model, cpu_cores, cpu_threads, cpu_cache_kb) = read_cpuinfo();
        let ram_total_mb = read_meminfo();
        let (gpu_name, gpu_vram_mb, gpu_driver, gpu_compute_cap) = read_nvidia_smi_inventory();
        let os_kernel = read_stdout("uname", &["-r"]);
        let rust_version = String::new();

        Self {
            gate_name: gate_name.to_string(),
            cpu_model,
            cpu_cores,
            cpu_threads,
            cpu_cache_kb,
            ram_total_mb,
            gpu_name,
            gpu_vram_mb,
            gpu_driver,
            gpu_compute_cap,
            os_kernel,
            rust_version,
        }
    }

    /// Pretty-print the inventory block.
    pub fn print(&self) {
        let w = 52;
        println!("  ┌── Hardware ─{}┐", "─".repeat(w - 14));
        println!(
            "  │ {:<width$}│",
            format!("Gate:   {}", self.gate_name),
            width = w
        );
        println!(
            "  │ {:<width$}│",
            format!("CPU:    {}", self.cpu_model),
            width = w
        );
        println!(
            "  │ {:<width$}│",
            format!(
                "Cores:  {} ({} threads), L3 {} KB",
                self.cpu_cores, self.cpu_threads, self.cpu_cache_kb
            ),
            width = w
        );
        println!(
            "  │ {:<width$}│",
            format!("RAM:    {} MB", self.ram_total_mb),
            width = w
        );
        println!(
            "  │ {:<width$}│",
            format!("GPU:    {}", self.gpu_name),
            width = w
        );
        println!(
            "  │ {:<width$}│",
            format!(
                "VRAM:   {} MB, Driver {}, CC {}",
                self.gpu_vram_mb, self.gpu_driver, self.gpu_compute_cap
            ),
            width = w
        );
        println!(
            "  │ {:<width$}│",
            format!("Kernel: {}", self.os_kernel),
            width = w
        );
        println!("  └─{}┘", "─".repeat(w + 1));
    }
}

fn read_cpuinfo() -> (String, usize, usize, usize) {
    let content = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let mut model = String::from("unknown");
    let mut core_ids = std::collections::HashSet::new();
    let mut thread_count = 0usize;
    let mut cache_kb = 0usize;

    for line in content.lines() {
        if line.starts_with("model name") {
            if let Some(v) = line.split(':').nth(1) {
                model = v.trim().to_string();
            }
        } else if line.starts_with("core id") {
            if let Some(v) = line.split(':').nth(1) {
                core_ids.insert(v.trim().to_string());
            }
        } else if line.starts_with("processor") {
            thread_count += 1;
        } else if line.starts_with("cache size") {
            if let Some(v) = line.split(':').nth(1) {
                let v = v.trim().replace(" KB", "");
                cache_kb = v.parse().unwrap_or(0);
            }
        }
    }

    let cores = core_ids.len().max(1);
    (model, cores, thread_count, cache_kb)
}

fn read_meminfo() -> usize {
    let content = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<usize>() {
                    return kb / 1024;
                }
            }
        }
    }
    0
}

fn read_nvidia_smi_inventory() -> (String, usize, String, String) {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let parts: Vec<&str> = s.split(", ").collect();
            if parts.len() >= 4 {
                let name = parts[0].trim().to_string();
                let vram_mb = parts[1].trim().parse().unwrap_or(0);
                let driver = parts[2].trim().to_string();
                let cc = parts[3].trim().to_string();
                return (name, vram_mb, driver, cc);
            }
            (s, 0, String::new(), String::new())
        }
        _ => ("N/A".into(), 0, "N/A".into(), "N/A".into()),
    }
}

/// RAPL energy (microjoules). Used by `PowerMonitor`.
pub fn read_rapl_energy_uj() -> Option<u64> {
    // Note: RAPL energy_uj requires read access to /sys/class/powercap/.
    // If permission denied, run with: sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj
    // Or run the binary with CAP_DAC_READ_SEARCH capability.
    std::fs::read_to_string("/sys/class/powercap/intel-rapl:0/energy_uj")
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// RAPL max energy range (microjoules). Used for counter wrap handling.
pub fn read_rapl_max_energy_uj() -> Option<u64> {
    std::fs::read_to_string("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

fn read_stdout(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd).args(args).output().map_or_else(
        |_| "unknown".to_string(),
        |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn hardware_inventory_construction() {
        let hw = HardwareInventory {
            gate_name: "CI_Gate".to_string(),
            cpu_model: "Intel Xeon".to_string(),
            cpu_cores: 8,
            cpu_threads: 16,
            cpu_cache_kb: 16384,
            ram_total_mb: 65536,
            gpu_name: "NVIDIA A100".to_string(),
            gpu_vram_mb: 40960,
            gpu_driver: "535.0".to_string(),
            gpu_compute_cap: "8.0".to_string(),
            os_kernel: "6.5.0".to_string(),
            rust_version: "1.80".to_string(),
        };
        assert_eq!(hw.gate_name, "CI_Gate");
        assert_eq!(hw.cpu_cores, 8);
        assert_eq!(hw.cpu_threads, 16);
        assert_eq!(hw.cpu_cache_kb, 16384);
        assert_eq!(hw.ram_total_mb, 65536);
        assert_eq!(hw.gpu_name, "NVIDIA A100");
        assert_eq!(hw.gpu_vram_mb, 40960);
        assert_eq!(hw.gpu_driver, "535.0");
        assert_eq!(hw.gpu_compute_cap, "8.0");
        assert_eq!(hw.os_kernel, "6.5.0");
    }

    #[test]
    fn hardware_inventory_print_no_panic() {
        let hw = HardwareInventory {
            gate_name: "test".to_string(),
            cpu_model: "CPU".to_string(),
            cpu_cores: 2,
            cpu_threads: 4,
            cpu_cache_kb: 4096,
            ram_total_mb: 8192,
            gpu_name: "GPU".to_string(),
            gpu_vram_mb: 8192,
            gpu_driver: "500".to_string(),
            gpu_compute_cap: "8.0".to_string(),
            os_kernel: "6.x".to_string(),
            rust_version: String::new(),
        };
        hw.print(); // should not panic
    }

    #[test]
    fn hardware_inventory_serde_round_trip() {
        let hw = HardwareInventory {
            gate_name: "serde_test".to_string(),
            cpu_model: "test".to_string(),
            cpu_cores: 1,
            cpu_threads: 1,
            cpu_cache_kb: 0,
            ram_total_mb: 0,
            gpu_name: "none".to_string(),
            gpu_vram_mb: 0,
            gpu_driver: String::new(),
            gpu_compute_cap: String::new(),
            os_kernel: String::new(),
            rust_version: String::new(),
        };
        let json = serde_json::to_string(&hw).expect("serialize");
        let back: HardwareInventory = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.gate_name, hw.gate_name);
        assert_eq!(back.cpu_cores, hw.cpu_cores);
        assert_eq!(back.ram_total_mb, hw.ram_total_mb);
    }

    #[test]
    fn read_rapl_energy_uj_safe() {
        let _result = read_rapl_energy_uj();
    }

    #[test]
    fn read_rapl_max_energy_uj_safe() {
        let result = read_rapl_max_energy_uj();
        match result {
            None => {}
            Some(max) => assert!(max > 0, "RAPL max energy range should be positive"),
        }
    }

    #[test]
    fn hardware_inventory_detect_no_panic() {
        // Runs on any machine; may get "unknown"/0/N/A when hardware absent
        let hw = HardwareInventory::detect("test_gate");
        assert!(!hw.gate_name.is_empty());
        assert!(hw.cpu_cores >= 1);
        assert!(hw.cpu_threads >= 1);
        assert!(!hw.cpu_model.is_empty());
    }

    #[test]
    fn hardware_inventory_cpu_fields() {
        let hw = HardwareInventory {
            gate_name: "cpu_test".to_string(),
            cpu_model: "Intel Core i7".to_string(),
            cpu_cores: 4,
            cpu_threads: 8,
            cpu_cache_kb: 12288,
            ram_total_mb: 32768,
            gpu_name: "N/A".to_string(),
            gpu_vram_mb: 0,
            gpu_driver: "N/A".to_string(),
            gpu_compute_cap: "N/A".to_string(),
            os_kernel: String::new(),
            rust_version: String::new(),
        };
        assert_eq!(hw.cpu_model, "Intel Core i7");
        assert_eq!(hw.cpu_cores, 4);
        assert_eq!(hw.cpu_threads, 8);
        assert_eq!(hw.cpu_cache_kb, 12288);
    }

    #[test]
    fn hardware_inventory_gpu_default_like() {
        // Zero / N/A GPU-like construction (no GPU present)
        let hw = HardwareInventory {
            gate_name: "headless".to_string(),
            cpu_model: "unknown".to_string(),
            cpu_cores: 1,
            cpu_threads: 1,
            cpu_cache_kb: 0,
            ram_total_mb: 0,
            gpu_name: "N/A".to_string(),
            gpu_vram_mb: 0,
            gpu_driver: "N/A".to_string(),
            gpu_compute_cap: String::new(),
            os_kernel: String::new(),
            rust_version: String::new(),
        };
        assert_eq!(hw.gpu_name, "N/A");
        assert_eq!(hw.gpu_vram_mb, 0);
        assert_eq!(hw.gpu_driver, "N/A");
    }
}
