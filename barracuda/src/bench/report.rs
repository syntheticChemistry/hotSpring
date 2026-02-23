// SPDX-License-Identifier: AGPL-3.0-only

//! Benchmark report types, formatting, and JSON serialization.

use serde::{Deserialize, Serialize};

use super::hardware::HardwareInventory;
use super::power::EnergyReport;

/// Result from a single benchmark phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseResult {
    pub phase: String,
    pub substrate: String,
    pub wall_time_s: f64,
    pub per_eval_us: f64,
    pub n_evals: usize,
    pub energy: EnergyReport,
    pub peak_rss_mb: f64,
    pub chi2: f64,
    pub precision_mev: f64,
    pub notes: String,
}

/// Full benchmark report for a validation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub timestamp: String,
    pub hardware: HardwareInventory,
    pub phases: Vec<PhaseResult>,
}

impl BenchReport {
    /// Create a new report with hardware inventory.
    #[must_use]
    pub fn new(hw: HardwareInventory) -> Self {
        let timestamp = now_iso8601();
        Self {
            timestamp,
            hardware: hw,
            phases: Vec::new(),
        }
    }

    /// Add a phase result.
    pub fn add_phase(&mut self, phase: PhaseResult) {
        self.phases.push(phase);
    }

    /// Save to JSON file.  Returns the path written.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the directory cannot be created, the path cannot be
    /// written, or JSON serialization fails.
    pub fn save_json(&self, dir: &str) -> std::io::Result<String> {
        std::fs::create_dir_all(dir)?;
        let filename = format!(
            "{}_{}.json",
            self.hardware.gate_name.to_lowercase().replace(' ', "_"),
            self.timestamp.replace(':', "-").replace(' ', "_"),
        );
        let path = format!("{dir}/{filename}");
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(&path, json)?;
        Ok(path)
    }

    /// Print summary table to stdout.
    pub fn print_summary(&self) {
        println!();
        println!("══════════════════════════════════════════════════════════════════════════════════════════");
        println!(
            "  SUBSTRATE BENCHMARK REPORT — {} ({} / {})",
            self.hardware.gate_name, self.hardware.cpu_model, self.hardware.gpu_name
        );
        println!("══════════════════════════════════════════════════════════════════════════════════════════");
        println!();

        // Table header
        println!(
            "  {:<18} {:<14} {:>10} {:>10} {:>9} {:>9} {:>10} {:>10} {:>8}",
            "Phase",
            "Substrate",
            "Wall Time",
            "per-eval",
            "Energy J",
            "J/eval",
            "W (avg)",
            "W (peak)",
            "chi2"
        );
        println!("  {}", "─".repeat(100));

        for p in &self.phases {
            let wall_str = format_duration(p.wall_time_s);
            let eval_str = if p.per_eval_us > 0.0 {
                format_eval_time(p.per_eval_us)
            } else {
                String::from("—")
            };

            // Determine primary energy source by substrate
            let is_gpu_phase = p.substrate.contains("GPU") || p.substrate.contains("gpu");
            let primary_joules = if is_gpu_phase {
                p.energy.gpu_joules
            } else {
                // CPU phases: use RAPL (cpu_joules), or fall back to 0
                p.energy.cpu_joules
            };
            let primary_watts = if is_gpu_phase {
                p.energy.gpu_watts_avg
            } else {
                // CPU watts = cpu_joules / wall_time
                if p.energy.cpu_joules > 0.0 && p.wall_time_s > 0.0 {
                    p.energy.cpu_joules / p.wall_time_s
                } else {
                    0.0
                }
            };

            let energy_str = if primary_joules > 0.01 {
                format!("{primary_joules:.2}")
            } else if primary_joules > 0.0 {
                format!("{primary_joules:.4}")
            } else {
                String::from("—")
            };

            let j_per_eval = if primary_joules > 0.0 && p.n_evals > 0 {
                let j = primary_joules / p.n_evals as f64;
                if j > 0.01 {
                    format!("{j:.3}")
                } else if j > 0.0001 {
                    format!("{j:.1e}")
                } else {
                    format!("{j:.2e}")
                }
            } else {
                String::from("—")
            };

            let watts_str = if primary_watts > 0.1 {
                format!("{primary_watts:.0} W")
            } else {
                String::from("—")
            };

            let peak_watts = if is_gpu_phase {
                p.energy.gpu_watts_peak
            } else {
                // CPU peak ≈ avg (RAPL doesn't give instantaneous)
                primary_watts
            };
            let peak_watts_str = if peak_watts > 0.1 {
                format!("{peak_watts:.0} W")
            } else {
                String::from("—")
            };

            let chi2_str = if p.chi2 < 1e8 {
                format!("{:.2}", p.chi2)
            } else {
                String::from("—")
            };

            let sub_label = if is_gpu_phase {
                format!("{} [G]", p.substrate)
            } else {
                format!("{} [C]", p.substrate)
            };

            println!(
                "  {:<18} {:<14} {:>10} {:>10} {:>9} {:>9} {:>10} {:>10} {:>8}",
                p.phase,
                sub_label,
                wall_str,
                eval_str,
                energy_str,
                j_per_eval,
                watts_str,
                peak_watts_str,
                chi2_str
            );
        }
        println!("  {}", "─".repeat(100));
        println!(
            "  [C] = CPU energy (RAPL)  [G] = GPU energy (nvidia-smi, {}ms polling)",
            100
        );

        // Detailed power/thermal breakdown for GPU phases
        let gpu_phases: Vec<&PhaseResult> = self
            .phases
            .iter()
            .filter(|p| p.substrate.contains("GPU") || p.substrate.contains("gpu"))
            .collect();
        if !gpu_phases.is_empty() {
            println!();
            println!("  GPU Power Detail:");
            println!(
                "  {:<22} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
                "Phase", "W (avg)", "W (peak)", "Temp °C", "VRAM MB", "Samples", "Total J"
            );
            println!("  {}", "─".repeat(72));
            for p in &gpu_phases {
                println!(
                    "  {:<22} {:>7.1} {:>7.1} {:>7.0} {:>7.0} {:>8} {:>8.0}",
                    p.phase,
                    p.energy.gpu_watts_avg,
                    p.energy.gpu_watts_peak,
                    p.energy.gpu_temp_peak_c,
                    p.energy.gpu_vram_peak_mib,
                    p.energy.gpu_samples,
                    p.energy.gpu_joules
                );
            }
            println!("  {}", "─".repeat(72));
        }

        // Pairwise comparisons for matching phases across substrates
        println!();
        let mut seen = std::collections::HashSet::new();
        for p in &self.phases {
            if seen.contains(&p.phase) {
                continue;
            }
            seen.insert(p.phase.clone());

            let matching: Vec<&PhaseResult> =
                self.phases.iter().filter(|q| q.phase == p.phase).collect();
            if matching.len() < 2 {
                continue;
            }

            // Find fastest and slowest (matching.len() >= 2 from the guard above)
            let Some(fastest) = matching
                .iter()
                .min_by(|a, b| a.wall_time_s.total_cmp(&b.wall_time_s))
            else {
                continue;
            };
            let Some(slowest) = matching
                .iter()
                .max_by(|a, b| a.wall_time_s.total_cmp(&b.wall_time_s))
            else {
                continue;
            };

            if fastest.wall_time_s > 0.0 && slowest.wall_time_s > fastest.wall_time_s {
                let speedup = slowest.wall_time_s / fastest.wall_time_s;
                println!(
                    "  {} : {} is {:.1}x faster than {} ({} vs {})",
                    fastest.phase,
                    fastest.substrate,
                    speedup,
                    slowest.substrate,
                    format_duration(fastest.wall_time_s),
                    format_duration(slowest.wall_time_s)
                );

                // Energy comparison if both have primary energy > 0
                let fast_gpu =
                    fastest.substrate.contains("GPU") || fastest.substrate.contains("gpu");
                let slow_gpu =
                    slowest.substrate.contains("GPU") || slowest.substrate.contains("gpu");
                let fast_j = if fast_gpu {
                    fastest.energy.gpu_joules
                } else {
                    fastest.energy.cpu_joules
                };
                let slow_j = if slow_gpu {
                    slowest.energy.gpu_joules
                } else {
                    slowest.energy.cpu_joules
                };
                if fast_j > 0.0 && slow_j > 0.0 {
                    let ratio = slow_j / fast_j;
                    println!(
                        "           energy: {:.2}J ({}) vs {:.2}J ({}) — {:.1}x less",
                        fast_j,
                        if fast_gpu { "GPU" } else { "CPU" },
                        slow_j,
                        if slow_gpu { "GPU" } else { "CPU" },
                        ratio
                    );
                }
            }
        }
        println!();
    }
}

/// Read peak resident set size (`VmHWM`) in MB.
#[must_use]
pub fn peak_rss_mb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("VmHWM:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<f64>() {
                    return kb / 1024.0;
                }
            }
        }
    }
    0.0
}

/// Pure-Rust ISO 8601 timestamp.
pub fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let day_secs = (secs % 86400) as u32;
    let (hour, minute, second) = (day_secs / 3600, (day_secs % 3600) / 60, day_secs % 60);
    // Civil date from days since 1970-01-01 (Howard Hinnant, public domain)
    let z = (secs / 86400) as i64 + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = i64::from(yoe) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}")
}

/// Format duration for display.
pub fn format_duration(secs: f64) -> String {
    if secs < 0.001 {
        format!("{:.1} us", secs * 1e6)
    } else if secs < 1.0 {
        format!("{:.1} ms", secs * 1e3)
    } else if secs < 60.0 {
        format!("{secs:.2} s")
    } else {
        format!("{:.1} min", secs / 60.0)
    }
}

/// Format per-evaluation time for display.
pub fn format_eval_time(us: f64) -> String {
    if us < 1000.0 {
        format!("{us:.1} us")
    } else if us < 1_000_000.0 {
        format!("{:.2} ms", us / 1000.0)
    } else {
        format!("{:.2} s", us / 1_000_000.0)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    use crate::bench::{EnergyReport, HardwareInventory};

    #[test]
    fn format_duration_us() {
        let s = format_duration(0.0001);
        assert!(s.contains("us"));
        assert!(s.contains("100"));
    }

    #[test]
    fn format_duration_ms() {
        let s = format_duration(0.05);
        assert!(s.contains("ms"));
        assert!(s.contains("50"));
    }

    #[test]
    fn format_duration_seconds() {
        let s = format_duration(2.5);
        assert!(s.contains('s'));
        assert!(s.contains("2.50"));
        assert!(!s.contains("min"));
    }

    #[test]
    fn format_duration_minutes() {
        let s = format_duration(90.0);
        assert!(s.contains("min"));
        assert!(s.contains("1.5"));
    }

    #[test]
    fn format_duration_boundaries() {
        assert!(format_duration(0.0009).contains("us"));
        assert!(format_duration(0.001).contains("ms"));
        assert!(format_duration(0.999).contains("ms"));
        assert!(format_duration(1.0).contains('s'));
        assert!(format_duration(59.9).contains('s'));
        assert!(format_duration(60.0).contains("min"));
    }

    #[test]
    fn format_eval_time_us() {
        let s = format_eval_time(500.0);
        assert!(s.contains("us"));
        assert!(s.contains("500"));
    }

    #[test]
    fn format_eval_time_ms() {
        let s = format_eval_time(5_000.0);
        assert!(s.contains("ms"));
        assert!(s.contains("5.00"));
    }

    #[test]
    fn format_eval_time_seconds() {
        let s = format_eval_time(2_000_000.0);
        assert!(s.contains('s'));
        assert!(s.contains("2.00"));
    }

    #[test]
    fn format_eval_time_boundaries() {
        assert!(format_eval_time(999.0).contains("us"));
        assert!(format_eval_time(1000.0).contains("ms"));
        assert!(format_eval_time(999_999.0).contains("ms"));
        assert!(format_eval_time(1_000_000.0).contains('s'));
    }

    #[test]
    fn now_iso8601_valid_format() {
        let s = now_iso8601();
        // ISO 8601: YYYY-MM-DDTHH:MM:SS
        assert!(s.len() >= 19, "expected at least YYYY-MM-DDTHH:MM:SS");
        let parts: Vec<&str> = s.split('T').collect();
        assert_eq!(parts.len(), 2);
        let date: Vec<&str> = parts[0].split('-').collect();
        assert_eq!(date.len(), 3);
        assert_eq!(date[0].len(), 4);
        assert_eq!(date[1].len(), 2);
        assert_eq!(date[2].len(), 2);
        let time: Vec<&str> = parts[1].split(':').collect();
        assert_eq!(time.len(), 3);
        assert!(time[0].parse::<u8>().is_ok());
        assert!(time[1].parse::<u8>().is_ok());
        assert!(time[2].parse::<u8>().is_ok());
    }

    #[test]
    fn phase_result_construction() {
        let pr = PhaseResult {
            phase: "test_phase".to_string(),
            substrate: "BarraCuda CPU".to_string(),
            wall_time_s: 2.5,
            per_eval_us: 100.5,
            n_evals: 1_000,
            energy: EnergyReport::default(),
            peak_rss_mb: 256.0,
            chi2: 0.5,
            precision_mev: 0.05,
            notes: "notes".to_string(),
        };
        assert_eq!(pr.phase, "test_phase");
        assert!((pr.wall_time_s - 2.5).abs() < 1e-10);
        assert_eq!(pr.n_evals, 1_000);
    }

    #[test]
    fn phase_result_json_round_trip() {
        let pr = PhaseResult {
            phase: "round_trip".to_string(),
            substrate: "GPU".to_string(),
            wall_time_s: 1.0,
            per_eval_us: 50.0,
            n_evals: 500,
            energy: EnergyReport {
                cpu_joules: 10.0,
                gpu_joules: 25.0,
                gpu_watts_avg: 50.0,
                gpu_watts_peak: 75.0,
                gpu_temp_peak_c: 65.0,
                gpu_vram_peak_mib: 4096.0,
                gpu_samples: 100,
            },
            peak_rss_mb: 128.0,
            chi2: 1.5,
            precision_mev: 0.1,
            notes: "test".to_string(),
        };
        let json = serde_json::to_string(&pr).expect("serialize");
        let back: PhaseResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.phase, pr.phase);
        assert!((back.wall_time_s - pr.wall_time_s).abs() < 1e-10);
        assert!((back.energy.gpu_joules - pr.energy.gpu_joules).abs() < 1e-10);
    }

    fn make_test_hardware() -> HardwareInventory {
        HardwareInventory {
            gate_name: "test".to_string(),
            cpu_model: "x86".to_string(),
            cpu_cores: 4,
            cpu_threads: 8,
            cpu_cache_kb: 8192,
            ram_total_mb: 16384,
            gpu_name: "RTX 4070".to_string(),
            gpu_vram_mb: 12288,
            gpu_driver: "560".to_string(),
            gpu_compute_cap: "8.9".to_string(),
            os_kernel: "6.x".to_string(),
            rust_version: String::new(),
        }
    }

    #[test]
    fn bench_report_new_empty_phases() {
        let hw = make_test_hardware();
        let report = BenchReport::new(hw);
        assert!(report.phases.is_empty());
        assert!(!report.timestamp.is_empty());
        assert!(report.timestamp.contains('T'));
    }

    #[test]
    fn bench_report_add_phase() {
        let hw = make_test_hardware();
        let mut report = BenchReport::new(hw);
        report.add_phase(PhaseResult {
            phase: "p1".to_string(),
            substrate: "CPU".to_string(),
            wall_time_s: 1.0,
            per_eval_us: 10.0,
            n_evals: 100,
            energy: EnergyReport::default(),
            peak_rss_mb: 64.0,
            chi2: 0.1,
            precision_mev: 0.05,
            notes: String::new(),
        });
        assert_eq!(report.phases.len(), 1);
        report.add_phase(PhaseResult {
            phase: "p2".to_string(),
            substrate: "GPU".to_string(),
            wall_time_s: 0.5,
            per_eval_us: 5.0,
            n_evals: 100,
            energy: EnergyReport::default(),
            peak_rss_mb: 128.0,
            chi2: 0.2,
            precision_mev: 0.05,
            notes: String::new(),
        });
        assert_eq!(report.phases.len(), 2);
    }

    #[test]
    fn bench_report_print_summary_no_panic() {
        let hw = make_test_hardware();
        let mut report = BenchReport::new(hw);
        report.add_phase(PhaseResult {
            phase: "yukawa".to_string(),
            substrate: "GPU".to_string(),
            wall_time_s: 1.0,
            per_eval_us: 100.0,
            n_evals: 10_000,
            energy: EnergyReport {
                cpu_joules: 0.0,
                gpu_joules: 50.0,
                gpu_watts_avg: 50.0,
                gpu_watts_peak: 75.0,
                gpu_temp_peak_c: 65.0,
                gpu_vram_peak_mib: 4096.0,
                gpu_samples: 100,
            },
            peak_rss_mb: 64.0,
            chi2: 0.5,
            precision_mev: 0.1,
            notes: String::new(),
        });
        report.add_phase(PhaseResult {
            phase: "yukawa".to_string(),
            substrate: "CPU".to_string(),
            wall_time_s: 2.0,
            per_eval_us: 0.0, // exercises "—" branch
            n_evals: 10_000,
            energy: EnergyReport {
                cpu_joules: 30.0,
                gpu_joules: 0.0,
                gpu_watts_avg: 0.0,
                gpu_watts_peak: 0.0,
                gpu_temp_peak_c: 0.0,
                gpu_vram_peak_mib: 0.0,
                gpu_samples: 0,
            },
            peak_rss_mb: 128.0,
            chi2: 1e9, // exercises "—" for chi2 >= 1e8
            precision_mev: 0.1,
            notes: String::new(),
        });
        report.print_summary(); // should not panic
    }

    #[test]
    fn bench_report_print_summary_pairwise_speedup() {
        let hw = make_test_hardware();
        let mut report = BenchReport::new(hw);
        report.add_phase(PhaseResult {
            phase: "same_phase".to_string(),
            substrate: "GPU".to_string(),
            wall_time_s: 1.0,
            per_eval_us: 10.0,
            n_evals: 1000,
            energy: EnergyReport {
                cpu_joules: 0.0,
                gpu_joules: 50.0,
                gpu_watts_avg: 50.0,
                gpu_watts_peak: 60.0,
                gpu_temp_peak_c: 60.0,
                gpu_vram_peak_mib: 2048.0,
                gpu_samples: 50,
            },
            peak_rss_mb: 64.0,
            chi2: 0.1,
            precision_mev: 0.05,
            notes: String::new(),
        });
        report.add_phase(PhaseResult {
            phase: "same_phase".to_string(),
            substrate: "CPU".to_string(),
            wall_time_s: 3.0,
            per_eval_us: 20.0,
            n_evals: 1000,
            energy: EnergyReport {
                cpu_joules: 40.0,
                gpu_joules: 0.0,
                gpu_watts_avg: 0.0,
                gpu_watts_peak: 0.0,
                gpu_temp_peak_c: 0.0,
                gpu_vram_peak_mib: 0.0,
                gpu_samples: 0,
            },
            peak_rss_mb: 128.0,
            chi2: 0.2,
            precision_mev: 0.05,
            notes: String::new(),
        });
        report.print_summary(); // exercises pairwise comparison with energy
    }

    #[test]
    fn bench_report_save_json() {
        let hw = HardwareInventory {
            gate_name: "report_test".to_string(),
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
        let mut report = BenchReport::new(hw);
        report.add_phase(PhaseResult {
            phase: "save_test".to_string(),
            substrate: "CPU".to_string(),
            wall_time_s: 1.0,
            per_eval_us: 10.0,
            n_evals: 100,
            energy: EnergyReport::default(),
            peak_rss_mb: 32.0,
            chi2: 0.01,
            precision_mev: 0.1,
            notes: String::new(),
        });
        let dir = std::env::temp_dir().join("hotspring_report_test");
        let dir_str = dir.to_str().expect("temp path");
        let result = report.save_json(dir_str);
        assert!(result.is_ok(), "save_json should succeed");
        let path = result.expect("already asserted Ok");
        assert!(std::path::Path::new(&path).exists());
        let Ok(contents) = std::fs::read_to_string(&path) else {
            let _ = std::fs::remove_dir_all(&dir);
            panic!("could not read saved file");
        };
        let loaded: BenchReport = serde_json::from_str(&contents).expect("deserialize");
        assert_eq!(loaded.phases.len(), 1);
        assert_eq!(loaded.phases[0].phase, "save_test");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
