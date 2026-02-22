// SPDX-License-Identifier: AGPL-3.0-only

//! Benchmark harness for hotSpring validation runs.
//!
//! Captures hardware inventory, wall-clock time, CPU energy (Intel RAPL),
//! GPU power/temperature/VRAM (nvidia-smi), and process memory for every
//! validation phase.  Produces machine-readable JSON and human-readable
//! summary tables so that identical physics can be compared across
//! substrates (Python, `BarraCuda` CPU, `BarraCuda` GPU) and gates.
//!
//! See `benchmarks/PROTOCOL.md` for the full measurement specification.
//!
//! License: AGPL-3.0

mod hardware;
mod power;
mod report;

// Public API — unchanged from original bench.rs
pub use hardware::HardwareInventory;
pub use power::{EnergyReport, PowerMonitor};
pub use report::{peak_rss_mb, BenchReport, PhaseResult};

#[cfg(test)]
mod tests {
    use super::*;
    use report::{format_duration, format_eval_time, now_iso8601};

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
            substrate: "BarraCuda GPU".to_string(),
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
        assert_eq!(pr.substrate, "BarraCuda GPU");
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

    #[test]
    fn bench_report_new_and_add_phase() {
        let hw = HardwareInventory {
            gate_name: "test".to_string(),
            cpu_model: "x86".to_string(),
            cpu_cores: 4,
            cpu_threads: 8,
            cpu_cache_kb: 8192,
            ram_total_mb: 16384,
            gpu_name: "RTX 4070".to_string(),
            gpu_vram_mb: 12288,
            gpu_driver: "560.0".to_string(),
            gpu_compute_cap: "8.9".to_string(),
            os_kernel: "6.x".to_string(),
            rust_version: "1.80".to_string(),
        };
        let mut report = BenchReport::new(hw);
        assert!(report.phases.is_empty());
        assert!(!report.timestamp.is_empty());

        let phase = PhaseResult {
            phase: "yukawa".to_string(),
            substrate: "GPU".to_string(),
            wall_time_s: 1.0,
            per_eval_us: 100.0,
            n_evals: 10_000,
            energy: EnergyReport::default(),
            peak_rss_mb: 64.0,
            chi2: 0.01,
            precision_mev: 0.05,
            notes: String::new(),
        };
        report.add_phase(phase);
        assert_eq!(report.phases.len(), 1);
        assert_eq!(report.phases[0].phase, "yukawa");
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn bench_report_save_json_round_trip() {
        let hw = HardwareInventory {
            gate_name: "CI_Test".to_string(),
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
            phase: "semf".to_string(),
            substrate: "CPU".to_string(),
            wall_time_s: 0.5,
            per_eval_us: 50.0,
            n_evals: 100,
            energy: EnergyReport::default(),
            peak_rss_mb: 32.0,
            chi2: 0.02,
            precision_mev: 0.1,
            notes: "test".to_string(),
        });

        let dir = std::env::temp_dir().join("hotspring_bench_test");
        let dir_str = dir.to_str().expect("temp path");
        let path = report.save_json(dir_str).expect("save_json");
        assert!(std::path::Path::new(&path).exists());

        let contents = std::fs::read_to_string(&path).expect("read json");
        let loaded: BenchReport = serde_json::from_str(&contents).expect("deserialize");
        assert_eq!(loaded.phases.len(), 1);
        assert_eq!(loaded.phases[0].phase, "semf");
        assert_eq!(loaded.hardware.gate_name, "CI_Test");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn bench_report_serialize_deserialize() {
        let hw = HardwareInventory {
            gate_name: "serde_test".to_string(),
            cpu_model: "x86".to_string(),
            cpu_cores: 2,
            cpu_threads: 4,
            cpu_cache_kb: 4096,
            ram_total_mb: 8192,
            gpu_name: "none".to_string(),
            gpu_vram_mb: 0,
            gpu_driver: String::new(),
            gpu_compute_cap: String::new(),
            os_kernel: String::new(),
            rust_version: String::new(),
        };
        let report = BenchReport::new(hw);
        let json = serde_json::to_string(&report).expect("serialize");
        let back: BenchReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.hardware.gate_name, "serde_test");
        assert_eq!(back.hardware.cpu_cores, 2);
        assert!(back.phases.is_empty());
    }
}
