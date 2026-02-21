// SPDX-License-Identifier: AGPL-3.0-only

//! Benchmark harness for hotSpring validation runs.
//!
//! Captures hardware inventory, wall-clock time, CPU energy (Intel RAPL),
//! GPU power/temperature/VRAM (nvidia-smi), and process memory for every
//! validation phase.  Produces machine-readable JSON and human-readable
//! summary tables so that identical physics can be compared across
//! substrates (Python, BarraCUDA CPU, BarraCUDA GPU) and gates.
//!
//! See `benchmarks/PROTOCOL.md` for the full measurement specification.
//!
//! License: AGPL-3.0

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
//  Hardware Inventory
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
//  Energy / Power Report
// ═══════════════════════════════════════════════════════════════════

/// Energy and power measurements for a single benchmark phase.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnergyReport {
    /// CPU energy consumed (Joules) from Intel RAPL.
    pub cpu_joules: f64,
    /// GPU energy consumed (Joules) — integrated from nvidia-smi power samples.
    pub gpu_joules: f64,
    /// Average GPU power draw during the phase (Watts).
    pub gpu_watts_avg: f64,
    /// Peak GPU power draw (Watts).
    pub gpu_watts_peak: f64,
    /// Peak GPU temperature (Celsius).
    pub gpu_temp_peak_c: f64,
    /// Peak GPU VRAM usage (MiB).
    pub gpu_vram_peak_mib: f64,
    /// Number of nvidia-smi samples collected.
    pub gpu_samples: usize,
}

// ═══════════════════════════════════════════════════════════════════
//  Power Monitor (background thread)
// ═══════════════════════════════════════════════════════════════════

/// GPU power sample from nvidia-smi.
#[derive(Debug, Clone)]
struct GpuSample {
    watts: f64,
    temp_c: f64,
    vram_mib: f64,
    timestamp: Instant,
}

/// Background monitor that samples RAPL + nvidia-smi.
#[derive(Debug)]
pub struct PowerMonitor {
    /// RAPL energy at start (microjoules).
    rapl_start_uj: Option<u64>,
    /// Wall-clock start.
    wall_start: Instant,
    /// nvidia-smi child process.
    smi_child: Option<std::process::Child>,
    /// Collected GPU samples.
    gpu_samples: Arc<Mutex<Vec<GpuSample>>>,
    /// Reader thread handle.
    reader_handle: Option<std::thread::JoinHandle<()>>,
}

impl PowerMonitor {
    /// Begin monitoring.  Reads RAPL baseline and spawns nvidia-smi.
    pub fn start() -> Self {
        let rapl_start_uj = read_rapl_energy_uj();
        let wall_start = Instant::now();
        let gpu_samples: Arc<Mutex<Vec<GpuSample>>> = Arc::new(Mutex::new(Vec::new()));

        // Spawn nvidia-smi in loop mode (100 ms interval)
        let (smi_child, reader_handle) = spawn_nvidia_smi_poller(Arc::clone(&gpu_samples));

        Self {
            rapl_start_uj,
            wall_start,
            smi_child,
            gpu_samples,
            reader_handle,
        }
    }

    /// Stop monitoring and return the energy report.
    pub fn stop(mut self) -> EnergyReport {
        let wall_elapsed = self.wall_start.elapsed().as_secs_f64();

        // Kill nvidia-smi
        if let Some(ref mut child) = self.smi_child {
            let _ = child.kill();
            let _ = child.wait();
        }
        // Wait for reader thread
        if let Some(handle) = self.reader_handle.take() {
            let _ = handle.join();
        }

        // CPU energy
        let cpu_joules = match (self.rapl_start_uj, read_rapl_energy_uj()) {
            (Some(start), Some(end)) => {
                // RAPL counter wraps; handle it
                let delta = if end >= start {
                    end - start
                } else {
                    // max_energy_range_uj wrap
                    let max = read_rapl_max_energy_uj().unwrap_or(u64::MAX);
                    max - start + end
                };
                delta as f64 / 1_000_000.0 // µJ → J
            }
            _ => 0.0,
        };

        // GPU energy — integrate power samples over time
        let samples = self
            .gpu_samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let n = samples.len();
        if n == 0 {
            return EnergyReport {
                cpu_joules,
                ..Default::default()
            };
        }

        let mut gpu_joules = 0.0f64;
        let mut gpu_watts_sum = 0.0f64;
        let mut gpu_watts_peak = 0.0f64;
        let mut gpu_temp_peak = 0.0f64;
        let mut gpu_vram_peak = 0.0f64;

        for i in 0..n {
            gpu_watts_sum += samples[i].watts;
            if samples[i].watts > gpu_watts_peak {
                gpu_watts_peak = samples[i].watts;
            }
            if samples[i].temp_c > gpu_temp_peak {
                gpu_temp_peak = samples[i].temp_c;
            }
            if samples[i].vram_mib > gpu_vram_peak {
                gpu_vram_peak = samples[i].vram_mib;
            }

            // Trapezoidal integration
            if i > 0 {
                let dt = samples[i]
                    .timestamp
                    .duration_since(samples[i - 1].timestamp)
                    .as_secs_f64();
                let avg_w = f64::midpoint(samples[i].watts, samples[i - 1].watts);
                gpu_joules += avg_w * dt;
            }
        }
        // If only 1 sample, estimate energy from average power * wall time
        if n == 1 {
            gpu_joules = samples[0].watts * wall_elapsed;
        }

        let gpu_watts_avg = gpu_watts_sum / n as f64;

        EnergyReport {
            cpu_joules,
            gpu_joules,
            gpu_watts_avg,
            gpu_watts_peak,
            gpu_temp_peak_c: gpu_temp_peak,
            gpu_vram_peak_mib: gpu_vram_peak,
            gpu_samples: n,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Phase Result
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
//  Bench Report (top-level container)
// ═══════════════════════════════════════════════════════════════════

/// Full benchmark report for a validation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub timestamp: String,
    pub hardware: HardwareInventory,
    pub phases: Vec<PhaseResult>,
}

impl BenchReport {
    /// Create a new report with hardware inventory.
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
                "—".to_string()
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
                "—".to_string()
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
                "—".to_string()
            };

            let watts_str = if primary_watts > 0.1 {
                format!("{primary_watts:.0} W")
            } else {
                "—".to_string()
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
                "—".to_string()
            };

            let chi2_str = if p.chi2 < 1e8 {
                format!("{:.2}", p.chi2)
            } else {
                "—".to_string()
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

            // Find fastest and slowest
            #[allow(clippy::expect_used)] // SAFETY: matching.len() >= 2 guaranteed by prior check
            let fastest = matching
                .iter()
                .min_by(|a, b| a.wall_time_s.total_cmp(&b.wall_time_s))
                .expect("matching has >= 2 elements");
            #[allow(clippy::expect_used)] // SAFETY: matching.len() >= 2 guaranteed by prior check
            let slowest = matching
                .iter()
                .max_by(|a, b| a.wall_time_s.total_cmp(&b.wall_time_s))
                .expect("matching has >= 2 elements");

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

// ═══════════════════════════════════════════════════════════════════
//  Utility: read process RSS from /proc/self/status
// ═══════════════════════════════════════════════════════════════════

/// Read peak resident set size (VmHWM) in MB.
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

// ═══════════════════════════════════════════════════════════════════
//  Internal helpers
// ═══════════════════════════════════════════════════════════════════

fn read_cpuinfo() -> (String, usize, usize, usize) {
    let content = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let mut model = String::from("unknown");
    let mut physical_ids = std::collections::HashSet::new();
    let mut core_ids = std::collections::HashSet::new();
    let mut thread_count = 0usize;
    let mut cache_kb = 0usize;

    for line in content.lines() {
        if line.starts_with("model name") {
            if let Some(v) = line.split(':').nth(1) {
                model = v.trim().to_string();
            }
        } else if line.starts_with("physical id") {
            if let Some(v) = line.split(':').nth(1) {
                physical_ids.insert(v.trim().to_string());
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

fn read_rapl_energy_uj() -> Option<u64> {
    // Note: RAPL energy_uj requires read access to /sys/class/powercap/.
    // If permission denied, run with: sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj
    // Or run the binary with CAP_DAC_READ_SEARCH capability.
    std::fs::read_to_string("/sys/class/powercap/intel-rapl:0/energy_uj")
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

fn read_rapl_max_energy_uj() -> Option<u64> {
    std::fs::read_to_string("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

fn spawn_nvidia_smi_poller(
    samples: Arc<Mutex<Vec<GpuSample>>>,
) -> (
    Option<std::process::Child>,
    Option<std::thread::JoinHandle<()>>,
) {
    // Try to spawn nvidia-smi in continuous mode (100ms polling)
    let child = Command::new("nvidia-smi")
        .args([
            "--query-gpu=power.draw,temperature.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "-lms",
            "100",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();

    match child {
        Ok(mut child) => {
            let Some(stdout) = child.stdout.take() else {
                eprintln!("[nvidia-smi] stdout unavailable");
                return (None, None);
            };
            let handle = std::thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines() {
                    let Ok(line) = line else { break };
                    let line = line.trim().to_string();
                    if line.is_empty() {
                        continue;
                    }
                    // Parse: "85.23, 42, 601"
                    let parts: Vec<&str> = line.split(", ").collect();
                    if parts.len() >= 3 {
                        let watts = parts[0].trim().parse().unwrap_or(0.0);
                        let temp = parts[1].trim().parse().unwrap_or(0.0);
                        let vram = parts[2].trim().parse().unwrap_or(0.0);
                        if let Ok(mut v) = samples.lock() {
                            v.push(GpuSample {
                                watts,
                                temp_c: temp,
                                vram_mib: vram,
                                timestamp: Instant::now(),
                            });
                        }
                    }
                }
            });
            (Some(child), Some(handle))
        }
        Err(_) => (None, None),
    }
}

fn read_stdout(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd).args(args).output().map_or_else(
        |_| "unknown".to_string(),
        |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
    )
}

fn now_iso8601() -> String {
    // Pure-Rust ISO 8601 timestamp — no external `date` command dependency.
    // Uses Hinnant's civil_from_days algorithm for epoch → calendar conversion.
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

fn format_duration(secs: f64) -> String {
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

fn format_eval_time(us: f64) -> String {
    if us < 1000.0 {
        format!("{us:.1} us")
    } else if us < 1_000_000.0 {
        format!("{:.2} ms", us / 1000.0)
    } else {
        format!("{:.2} s", us / 1_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)] // exact known values (0.0)
    fn energy_report_default_values() {
        let r = EnergyReport::default();
        assert_eq!(r.cpu_joules, 0.0);
        assert_eq!(r.gpu_joules, 0.0);
        assert_eq!(r.gpu_watts_avg, 0.0);
        assert_eq!(r.gpu_watts_peak, 0.0);
        assert_eq!(r.gpu_temp_peak_c, 0.0);
        assert_eq!(r.gpu_vram_peak_mib, 0.0);
        assert_eq!(r.gpu_samples, 0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact known values
    fn phase_result_creation_and_fields() {
        let energy = EnergyReport {
            cpu_joules: 1.5,
            gpu_joules: 0.0,
            ..Default::default()
        };
        let pr = PhaseResult {
            phase: "yukawa".to_string(),
            substrate: "BarraCUDA GPU".to_string(),
            wall_time_s: 2.5,
            per_eval_us: 0.42,
            n_evals: 10_000,
            energy,
            peak_rss_mb: 128.0,
            chi2: 0.03,
            precision_mev: 0.1,
            notes: "smoke test".to_string(),
        };
        assert_eq!(pr.phase, "yukawa");
        assert_eq!(pr.substrate, "BarraCUDA GPU");
        assert!((pr.wall_time_s - 2.5).abs() < 1e-9);
        assert_eq!(pr.n_evals, 10_000);
        assert_eq!(pr.energy.cpu_joules, 1.5);
        assert_eq!(pr.peak_rss_mb, 128.0);
    }

    #[test]
    fn format_duration_sub_millisecond() {
        assert!(format_duration(0.0001).contains("us"));
    }

    #[test]
    fn format_duration_milliseconds() {
        let s = format_duration(0.05);
        assert!(s.contains("ms"));
    }

    #[test]
    fn format_duration_seconds() {
        let s = format_duration(1.5);
        assert!(s.contains('s'));
        assert!(!s.contains("min"));
    }

    #[test]
    fn format_duration_minutes() {
        let s = format_duration(90.0);
        assert!(s.contains("min"));
    }

    #[test]
    fn format_eval_time_microseconds() {
        let s = format_eval_time(500.0);
        assert!(s.contains("us"));
    }

    #[test]
    fn format_eval_time_milliseconds() {
        let s = format_eval_time(5_000.0);
        assert!(s.contains("ms"));
    }

    #[test]
    fn format_eval_time_seconds() {
        let s = format_eval_time(2_000_000.0);
        assert!(s.contains('s'));
    }

    #[test]
    fn now_iso8601_format() {
        let s = now_iso8601();
        // Expect YYYY-MM-DDTHH:MM:SS
        let parts: Vec<&str> = s.split('T').collect();
        assert_eq!(parts.len(), 2, "expected YYYY-MM-DDTHH:MM:SS format");
        let date: Vec<&str> = parts[0].split('-').collect();
        assert_eq!(date.len(), 3);
        assert_eq!(date[0].len(), 4);
        assert_eq!(date[1].len(), 2);
        assert_eq!(date[2].len(), 2);
    }

    #[test]
    fn peak_rss_mb_non_negative() {
        let rss = peak_rss_mb();
        assert!(rss >= 0.0, "peak_rss_mb should be non-negative");
    }

    #[test]
    #[ignore = "requires nvidia-smi and RAPL"]
    fn power_monitor_start_stop() {
        let monitor = PowerMonitor::start();
        std::thread::sleep(std::time::Duration::from_millis(50));
        let _ = monitor.stop();
    }
}
