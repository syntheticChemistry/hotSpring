// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSONL telemetry writer for structured validation output.
//!
//! Writes one JSON object per line to a sidecar file. Each event has a
//! timestamp, section, observable name, and value. Designed for consumption
//! by petalTongue or any JSONL-aware tool.

use std::io::Write;

/// JSONL telemetry writer for structured validation output.
pub struct TelemetryWriter {
    file: Option<std::io::BufWriter<std::fs::File>>,
    start: std::time::Instant,
    substrate: Option<String>,
}

impl TelemetryWriter {
    /// Create a new telemetry writer. If the file cannot be opened, logging
    /// is silently disabled (validation still runs).
    pub fn new(path: &str) -> Self {
        let file = std::fs::File::create(path)
            .ok()
            .map(std::io::BufWriter::new);
        if file.is_some() {
            log::info!("Telemetry → {path}");
        }
        Self {
            file,
            start: std::time::Instant::now(),
            substrate: None,
        }
    }

    /// Create a telemetry writer using discovery-based path resolution.
    pub fn discover(filename: &str) -> Self {
        let path = crate::discovery::telemetry_path(filename);
        let display = path.display().to_string();
        let file = std::fs::File::create(&path)
            .ok()
            .map(std::io::BufWriter::new);
        if file.is_some() {
            log::info!("Telemetry → {display}");
        }
        Self {
            file,
            start: std::time::Instant::now(),
            substrate: None,
        }
    }

    /// Create a no-op writer (for when telemetry is disabled).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            file: None,
            start: std::time::Instant::now(),
            substrate: None,
        }
    }

    /// Tag all subsequent telemetry events with the GPU/substrate name.
    #[must_use]
    pub fn with_substrate(mut self, name: String) -> Self {
        self.substrate = Some(name);
        self
    }

    /// Log a telemetry event. Fields with NaN/Inf are written as null.
    pub fn log(&mut self, section: &str, observable: &str, value: f64) {
        let sub = substrate_fragment(self.substrate.as_deref());
        let Some(ref mut f) = self.file else { return };
        let t = self.start.elapsed().as_secs_f64();
        let val = if value.is_finite() {
            format!("{value:.6e}")
        } else {
            "null".to_string()
        };
        let _ = writeln!(
            f,
            r#"{{"t":{t:.3},"section":"{section}","obs":"{observable}","val":{val}{sub}}}"#
        );
        let _ = f.flush();
    }

    /// Log a telemetry event with multiple key-value pairs.
    pub fn log_map(&mut self, section: &str, fields: &[(&str, f64)]) {
        let sub = substrate_fragment(self.substrate.as_deref());
        let Some(ref mut f) = self.file else { return };
        let t = self.start.elapsed().as_secs_f64();
        let pairs: Vec<String> = fields
            .iter()
            .map(|(k, v)| {
                let val = if v.is_finite() {
                    format!("{v:.6e}")
                } else {
                    "null".to_string()
                };
                format!(r#""{k}":{val}"#)
            })
            .collect();
        let _ = writeln!(
            f,
            r#"{{"t":{t:.3},"section":"{section}",{}{sub}}}"#,
            pairs.join(",")
        );
        let _ = f.flush();
    }
}

fn substrate_fragment(substrate: Option<&str>) -> String {
    match substrate {
        Some(s) => format!(r#","substrate":"{s}""#),
        None => String::new(),
    }
}
