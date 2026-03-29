// SPDX-License-Identifier: AGPL-3.0-only

//! Live GPU telemetry — non-blocking background poller for hardware signals.
//!
//! Provides real-time GPU utilization, power, temperature, and memory stats
//! via a background thread. Supports both NVIDIA (nvidia-smi) and AMD (sysfs).
//! Snapshots are lock-free reads; the poller never stalls the compute path.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Point-in-time GPU hardware snapshot.
#[derive(Debug, Clone, Default)]
pub struct GpuSnapshot {
    /// Shader/compute unit utilization (0-100%).
    pub gpu_util_pct: f64,
    /// Memory controller utilization (0-100%).
    pub mem_util_pct: f64,
    /// Current power draw (Watts).
    pub power_w: f64,
    /// GPU die temperature (Celsius).
    pub temp_c: f64,
    /// VRAM in use (MiB).
    pub vram_used_mib: f64,
    /// Total VRAM (MiB).
    pub vram_total_mib: f64,
    /// Fan speed (0-100%), if available.
    pub fan_pct: f64,
    /// Monotonic sample counter.
    pub sample_id: u64,
}

impl GpuSnapshot {
    /// VRAM utilization as a percentage.
    pub fn vram_pct(&self) -> f64 {
        if self.vram_total_mib > 0.0 {
            self.vram_used_mib / self.vram_total_mib * 100.0
        } else {
            0.0
        }
    }

    /// One-line summary for per-trajectory printing.
    pub fn status_line(&self) -> String {
        format!(
            "GPU:{:.0}% MEM:{:.0}% {:.0}W {:.0}C {:.0}/{:.0}MB",
            self.gpu_util_pct,
            self.mem_util_pct,
            self.power_w,
            self.temp_c,
            self.vram_used_mib,
            self.vram_total_mib,
        )
    }
}

/// Shared atomic state for lock-free snapshot reads.
struct SharedState {
    gpu_util_pct: AtomicU64,
    mem_util_pct: AtomicU64,
    power_w: AtomicU64,
    temp_c: AtomicU64,
    vram_used_mib: AtomicU64,
    vram_total_mib: AtomicU64,
    fan_pct: AtomicU64,
    sample_id: AtomicU64,
    running: AtomicBool,
}

impl SharedState {
    fn new() -> Self {
        Self {
            gpu_util_pct: AtomicU64::new(0),
            mem_util_pct: AtomicU64::new(0),
            power_w: AtomicU64::new(0),
            temp_c: AtomicU64::new(0),
            vram_used_mib: AtomicU64::new(0),
            vram_total_mib: AtomicU64::new(0),
            fan_pct: AtomicU64::new(0),
            sample_id: AtomicU64::new(0),
            running: AtomicBool::new(true),
        }
    }

    fn store_snapshot(&self, snap: &GpuSnapshot) {
        self.gpu_util_pct
            .store(snap.gpu_util_pct.to_bits(), Ordering::Relaxed);
        self.mem_util_pct
            .store(snap.mem_util_pct.to_bits(), Ordering::Relaxed);
        self.power_w
            .store(snap.power_w.to_bits(), Ordering::Relaxed);
        self.temp_c.store(snap.temp_c.to_bits(), Ordering::Relaxed);
        self.vram_used_mib
            .store(snap.vram_used_mib.to_bits(), Ordering::Relaxed);
        self.vram_total_mib
            .store(snap.vram_total_mib.to_bits(), Ordering::Relaxed);
        self.fan_pct
            .store(snap.fan_pct.to_bits(), Ordering::Relaxed);
        self.sample_id.store(snap.sample_id, Ordering::Release);
    }

    fn load_snapshot(&self) -> GpuSnapshot {
        GpuSnapshot {
            gpu_util_pct: f64::from_bits(self.gpu_util_pct.load(Ordering::Relaxed)),
            mem_util_pct: f64::from_bits(self.mem_util_pct.load(Ordering::Relaxed)),
            power_w: f64::from_bits(self.power_w.load(Ordering::Relaxed)),
            temp_c: f64::from_bits(self.temp_c.load(Ordering::Relaxed)),
            vram_used_mib: f64::from_bits(self.vram_used_mib.load(Ordering::Relaxed)),
            vram_total_mib: f64::from_bits(self.vram_total_mib.load(Ordering::Relaxed)),
            fan_pct: f64::from_bits(self.fan_pct.load(Ordering::Relaxed)),
            sample_id: self.sample_id.load(Ordering::Acquire),
        }
    }
}

/// Live GPU telemetry handle. Drop to stop the background poller.
pub struct GpuTelemetry {
    state: Arc<SharedState>,
    _poller: Option<std::thread::JoinHandle<()>>,
    smi_child: Option<std::process::Child>,
    pub backend: TelemetryBackend,
    start: Instant,
}

/// Which hardware polling backend is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryBackend {
    NvidiaSmi,
    AmdSysfs,
    None,
}

impl std::fmt::Display for TelemetryBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NvidiaSmi => write!(f, "nvidia-smi"),
            Self::AmdSysfs => write!(f, "amdgpu-sysfs"),
            Self::None => write!(f, "none"),
        }
    }
}

impl GpuTelemetry {
    /// Start live telemetry for the GPU matching `adapter_name`.
    ///
    /// Auto-detects NVIDIA vs AMD from the adapter string. Falls back
    /// gracefully if neither nvidia-smi nor amdgpu sysfs is available.
    pub fn start(adapter_name: &str) -> Self {
        let state = Arc::new(SharedState::new());
        let name_lower = adapter_name.to_lowercase();

        if name_lower.contains("nvidia") || name_lower.contains("geforce") {
            Self::start_nvidia(state, adapter_name)
        } else if name_lower.contains("amd")
            || name_lower.contains("radeon")
            || name_lower.contains("radv")
        {
            Self::start_amd(state)
        } else {
            Self {
                state,
                _poller: None,
                smi_child: None,
                backend: TelemetryBackend::None,
                start: Instant::now(),
            }
        }
    }

    /// Latest hardware snapshot (lock-free, never blocks).
    pub fn snapshot(&self) -> GpuSnapshot {
        self.state.load_snapshot()
    }

    /// Seconds since telemetry started.
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn start_nvidia(state: Arc<SharedState>, _adapter: &str) -> Self {
        let child = Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu,memory.used,memory.total,fan.speed",
                "--format=csv,noheader,nounits",
                "-lms",
                "200",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn();

        match child {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let poll_state = Arc::clone(&state);
                let handle = std::thread::Builder::new()
                    .name("gpu-telemetry-nvidia".into())
                    .spawn(move || {
                        let Some(stdout) = stdout else { return };
                        let reader = BufReader::new(stdout);
                        let mut sample_id = 0u64;
                        for line in reader.lines() {
                            if !poll_state.running.load(Ordering::Relaxed) {
                                break;
                            }
                            let Ok(line) = line else { break };
                            let parts: Vec<&str> = line.split(',').map(str::trim).collect();
                            if parts.len() >= 6 {
                                sample_id += 1;
                                let snap = GpuSnapshot {
                                    gpu_util_pct: parts[0].parse().unwrap_or(0.0),
                                    mem_util_pct: parts[1].parse().unwrap_or(0.0),
                                    power_w: parts[2].parse().unwrap_or(0.0),
                                    temp_c: parts[3].parse().unwrap_or(0.0),
                                    vram_used_mib: parts[4].parse().unwrap_or(0.0),
                                    vram_total_mib: parts[5].parse().unwrap_or(0.0),
                                    fan_pct: parts
                                        .get(6)
                                        .and_then(|s| s.parse().ok())
                                        .unwrap_or(0.0),
                                    sample_id,
                                };
                                poll_state.store_snapshot(&snap);
                            }
                        }
                    })
                    .ok();

                Self {
                    state,
                    _poller: handle,
                    smi_child: Some(child),
                    backend: TelemetryBackend::NvidiaSmi,
                    start: Instant::now(),
                }
            }
            Err(_) => Self {
                state,
                _poller: None,
                smi_child: None,
                backend: TelemetryBackend::None,
                start: Instant::now(),
            },
        }
    }

    fn start_amd(state: Arc<SharedState>) -> Self {
        let hwmon = find_amdgpu_hwmon();
        let drm_card = find_amdgpu_drm();
        let has_hwmon = hwmon.is_some();
        let has_drm = drm_card.is_some();
        let vram_total = drm_card
            .as_ref()
            .and_then(|p| read_sysfs_u64(&format!("{p}/device/mem_info_vram_total")))
            .map(|b| b as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);

        let poll_state = Arc::clone(&state);
        let handle = std::thread::Builder::new()
            .name("gpu-telemetry-amd".into())
            .spawn(move || {
                let mut sample_id = 0u64;
                while poll_state.running.load(Ordering::Relaxed) {
                    sample_id += 1;
                    let mut snap = GpuSnapshot {
                        vram_total_mib: vram_total,
                        sample_id,
                        ..Default::default()
                    };

                    if let Some(ref path) = hwmon {
                        snap.power_w = read_sysfs_f64(&format!("{path}/power1_average"))
                            .map(|uw| uw / 1_000_000.0)
                            .unwrap_or(0.0);
                        snap.temp_c = read_sysfs_f64(&format!("{path}/temp1_input"))
                            .map(|mc| mc / 1000.0)
                            .unwrap_or(0.0);
                        snap.fan_pct = read_sysfs_f64(&format!("{path}/pwm1"))
                            .map(|v| v / 255.0 * 100.0)
                            .unwrap_or(0.0);
                    }

                    if let Some(ref path) = drm_card {
                        snap.gpu_util_pct =
                            read_sysfs_f64(&format!("{path}/device/gpu_busy_percent"))
                                .unwrap_or(0.0);
                        snap.vram_used_mib =
                            read_sysfs_u64(&format!("{path}/device/mem_info_vram_used"))
                                .map(|b| b as f64 / (1024.0 * 1024.0))
                                .unwrap_or(0.0);
                    }

                    poll_state.store_snapshot(&snap);
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
            })
            .ok();

        let backend = if has_hwmon || has_drm {
            TelemetryBackend::AmdSysfs
        } else {
            TelemetryBackend::None
        };

        Self {
            state,
            _poller: handle,
            smi_child: None,
            backend,
            start: Instant::now(),
        }
    }
}

impl Drop for GpuTelemetry {
    fn drop(&mut self) {
        self.state.running.store(false, Ordering::Relaxed);
        if let Some(ref mut child) = self.smi_child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn find_amdgpu_hwmon() -> Option<String> {
    for entry in std::fs::read_dir("/sys/class/hwmon").ok()? {
        let entry = entry.ok()?;
        let name_path = entry.path().join("name");
        if let Ok(name) = std::fs::read_to_string(&name_path) {
            if name.trim() == "amdgpu" {
                return Some(entry.path().to_string_lossy().to_string());
            }
        }
    }
    None
}

fn find_amdgpu_drm() -> Option<String> {
    for entry in std::fs::read_dir("/sys/class/drm").ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let vendor_path = entry.path().join("device/vendor");
        if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
            // AMD vendor ID = 0x1002
            if vendor.trim() == "0x1002" {
                return Some(entry.path().to_string_lossy().to_string());
            }
        }
    }
    None
}

fn read_sysfs_f64(path: &str) -> Option<f64> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

fn read_sysfs_u64(path: &str) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_default_values() {
        let s = GpuSnapshot::default();
        assert_eq!(s.gpu_util_pct, 0.0);
        assert_eq!(s.vram_pct(), 0.0);
    }

    #[test]
    fn snapshot_status_line_format() {
        let s = GpuSnapshot {
            gpu_util_pct: 95.0,
            mem_util_pct: 60.0,
            power_w: 186.0,
            temp_c: 72.0,
            vram_used_mib: 4096.0,
            vram_total_mib: 16384.0,
            ..Default::default()
        };
        let line = s.status_line();
        assert!(line.contains("95%"));
        assert!(line.contains("186W"));
    }

    #[test]
    fn shared_state_round_trip() {
        let state = SharedState::new();
        let snap = GpuSnapshot {
            gpu_util_pct: 42.5,
            power_w: 186.3,
            temp_c: 68.0,
            vram_used_mib: 2048.0,
            vram_total_mib: 16384.0,
            sample_id: 7,
            ..Default::default()
        };
        state.store_snapshot(&snap);
        let loaded = state.load_snapshot();
        assert_eq!(loaded.sample_id, 7);
        assert!((loaded.gpu_util_pct - 42.5).abs() < 1e-10);
        assert!((loaded.power_w - 186.3).abs() < 1e-10);
    }
}
