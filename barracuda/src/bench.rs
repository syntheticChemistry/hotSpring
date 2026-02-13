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
        let rust_version = env!("CARGO_PKG_RUST_VERSION", "unknown").to_string();

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
        println!("  │ {:<width$}│", format!("Gate:   {}", self.gate_name), width = w);
        println!("  │ {:<width$}│", format!("CPU:    {}", self.cpu_model), width = w);
        println!("  │ {:<width$}│", format!("Cores:  {} ({} threads), L3 {} KB",
                 self.cpu_cores, self.cpu_threads, self.cpu_cache_kb), width = w);
        println!("  │ {:<width$}│", format!("RAM:    {} MB", self.ram_total_mb), width = w);
        println!("  │ {:<width$}│", format!("GPU:    {}", self.gpu_name), width = w);
        println!("  │ {:<width$}│", format!("VRAM:   {} MB, Driver {}, CC {}",
                 self.gpu_vram_mb, self.gpu_driver, self.gpu_compute_cap), width = w);
        println!("  │ {:<width$}│", format!("Kernel: {}", self.os_kernel), width = w);
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
        let (smi_child, reader_handle) = spawn_nvidia_smi_poller(gpu_samples.clone());

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
                let delta = if end >= start { end - start } else {
                    // max_energy_range_uj wrap
                    let max = read_rapl_max_energy_uj().unwrap_or(u64::MAX);
                    max - start + end
                };
                delta as f64 / 1_000_000.0  // µJ → J
            }
            _ => 0.0,
        };

        // GPU energy — integrate power samples over time
        let samples = self.gpu_samples.lock().unwrap();
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
            if samples[i].watts > gpu_watts_peak { gpu_watts_peak = samples[i].watts; }
            if samples[i].temp_c > gpu_temp_peak { gpu_temp_peak = samples[i].temp_c; }
            if samples[i].vram_mib > gpu_vram_peak { gpu_vram_peak = samples[i].vram_mib; }

            // Trapezoidal integration
            if i > 0 {
                let dt = samples[i].timestamp.duration_since(samples[i - 1].timestamp).as_secs_f64();
                let avg_w = (samples[i].watts + samples[i - 1].watts) / 2.0;
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
        let path = format!("{}/{}", dir, filename);
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&path, json)?;
        Ok(path)
    }

    /// Print summary table to stdout.
    pub fn print_summary(&self) {
        println!();
        println!("══════════════════════════════════════════════════════════════════════════════════════════");
        println!("  SUBSTRATE BENCHMARK REPORT — {} ({} / {})",
                 self.hardware.gate_name, self.hardware.cpu_model, self.hardware.gpu_name);
        println!("══════════════════════════════════════════════════════════════════════════════════════════");
        println!();

        // Table header
        println!("  {:<18} {:<14} {:>10} {:>10} {:>9} {:>9} {:>10} {:>8}",
                 "Phase", "Substrate", "Wall Time", "per-eval", "Energy J", "J/eval", "W (avg)", "chi2");
        println!("  {}", "─".repeat(90));

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
                format!("{:.2}", primary_joules)
            } else if primary_joules > 0.0 {
                format!("{:.4}", primary_joules)
            } else {
                "—".to_string()
            };

            let j_per_eval = if primary_joules > 0.0 && p.n_evals > 0 {
                let j = primary_joules / p.n_evals as f64;
                if j > 0.01 { format!("{:.3}", j) }
                else if j > 0.0001 { format!("{:.1e}", j) }
                else { format!("{:.2e}", j) }
            } else {
                "—".to_string()
            };

            let watts_str = if primary_watts > 0.1 {
                format!("{:.0} W", primary_watts)
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

            println!("  {:<18} {:<14} {:>10} {:>10} {:>9} {:>9} {:>10} {:>8}",
                     p.phase, sub_label, wall_str, eval_str, energy_str, j_per_eval, watts_str, chi2_str);
        }
        println!("  {}", "─".repeat(90));
        println!("  [C] = CPU energy (RAPL)  [G] = GPU energy (nvidia-smi)");

        // Pairwise comparisons for matching phases across substrates
        println!();
        let mut seen = std::collections::HashSet::new();
        for p in &self.phases {
            if seen.contains(&p.phase) { continue; }
            seen.insert(p.phase.clone());

            let matching: Vec<&PhaseResult> = self.phases.iter()
                .filter(|q| q.phase == p.phase)
                .collect();
            if matching.len() < 2 { continue; }

            // Find fastest and slowest
            let fastest = matching.iter().min_by(|a, b|
                a.wall_time_s.partial_cmp(&b.wall_time_s).unwrap()).unwrap();
            let slowest = matching.iter().max_by(|a, b|
                a.wall_time_s.partial_cmp(&b.wall_time_s).unwrap()).unwrap();

            if fastest.wall_time_s > 0.0 && slowest.wall_time_s > fastest.wall_time_s {
                let speedup = slowest.wall_time_s / fastest.wall_time_s;
                println!("  {} : {} is {:.1}x faster than {} ({} vs {})",
                         fastest.phase, fastest.substrate, speedup, slowest.substrate,
                         format_duration(fastest.wall_time_s), format_duration(slowest.wall_time_s));

                // Energy comparison if both have primary energy > 0
                let fast_gpu = fastest.substrate.contains("GPU") || fastest.substrate.contains("gpu");
                let slow_gpu = slowest.substrate.contains("GPU") || slowest.substrate.contains("gpu");
                let fast_j = if fast_gpu { fastest.energy.gpu_joules } else { fastest.energy.cpu_joules };
                let slow_j = if slow_gpu { slowest.energy.gpu_joules } else { slowest.energy.cpu_joules };
                if fast_j > 0.0 && slow_j > 0.0 {
                    let ratio = slow_j / fast_j;
                    println!("           energy: {:.2}J ({}) vs {:.2}J ({}) — {:.1}x less",
                             fast_j, if fast_gpu {"GPU"} else {"CPU"},
                             slow_j, if slow_gpu {"GPU"} else {"CPU"},
                             ratio);
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
        .args(["--query-gpu=name,memory.total,driver_version,compute_cap",
               "--format=csv,noheader,nounits"])
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
) -> (Option<std::process::Child>, Option<std::thread::JoinHandle<()>>) {
    // Try to spawn nvidia-smi in continuous mode (100ms polling)
    let child = Command::new("nvidia-smi")
        .args([
            "--query-gpu=power.draw,temperature.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "-lms", "100",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();

    match child {
        Ok(mut child) => {
            let stdout = child.stdout.take().expect("nvidia-smi stdout");
            let handle = std::thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines() {
                    let Ok(line) = line else { break };
                    let line = line.trim().to_string();
                    if line.is_empty() { continue; }
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
    Command::new(cmd)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

fn now_iso8601() -> String {
    // Simple ISO 8601 timestamp from system time (no chrono dependency)
    let output = Command::new("date")
        .args(["+%Y-%m-%dT%H:%M:%S"])
        .output();
    match output {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "unknown".to_string(),
    }
}

fn format_duration(secs: f64) -> String {
    if secs < 0.001 {
        format!("{:.1} us", secs * 1e6)
    } else if secs < 1.0 {
        format!("{:.1} ms", secs * 1e3)
    } else if secs < 60.0 {
        format!("{:.2} s", secs)
    } else {
        format!("{:.1} min", secs / 60.0)
    }
}

fn format_eval_time(us: f64) -> String {
    if us < 1000.0 {
        format!("{:.1} us", us)
    } else if us < 1_000_000.0 {
        format!("{:.2} ms", us / 1000.0)
    } else {
        format!("{:.2} s", us / 1_000_000.0)
    }
}
