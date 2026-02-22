// SPDX-License-Identifier: AGPL-3.0-only

//! Power monitoring: RAPL energy, nvidia-smi GPU sampling.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::hardware;

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
    #[must_use]
    pub fn start() -> Self {
        let rapl_start_uj = hardware::read_rapl_energy_uj();
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
        let cpu_joules = match (self.rapl_start_uj, hardware::read_rapl_energy_uj()) {
            (Some(start), Some(end)) => {
                // RAPL counter wraps; handle it
                let delta = if end >= start {
                    end - start
                } else {
                    // max_energy_range_uj wrap
                    let max = hardware::read_rapl_max_energy_uj().unwrap_or(u64::MAX);
                    max - start + end
                };
                delta as f64 / 1_000_000.0 // µJ → J
            }
            _ => 0.0,
        };

        // GPU energy — snapshot samples then release the mutex immediately
        let samples: Vec<GpuSample> = self
            .gpu_samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
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

            if i > 0 {
                let dt = samples[i]
                    .timestamp
                    .duration_since(samples[i - 1].timestamp)
                    .as_secs_f64();
                let avg_w = f64::midpoint(samples[i].watts, samples[i - 1].watts);
                gpu_joules += avg_w * dt;
            }
        }
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
