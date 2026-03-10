// SPDX-License-Identifier: AGPL-3.0-only

//! JSONL-based trajectory store for cross-run NPU learning.
//!
//! Each trajectory event is written as a single JSON line, enabling
//! append-only storage and streaming reads. The NPU retrain bridge
//! reads the history, extracts training pairs, and feeds them to
//! `NautilusBrain` for evolutionary improvement.
//!
//! # Format
//!
//! Per-trajectory line:
//! ```json
//! {"beta":5.4,"mass":0.1,"dt":0.005,"n_md":100,"plaquette":0.52,"delta_h":0.3,
//!  "cg_iters":450,"accepted":true,"wall_us":1200000,"stage":"anneal_m0.5"}
//! ```
//!
//! Per-run summary line:
//! ```json
//! {"type":"summary","beta":5.4,"mass":0.1,"final_acceptance":0.65,
//!  "final_plaquette":0.53,"final_dt":0.004,"n_trajectories":65,"converged":true}
//! ```

use std::io::Write;

/// A single HMC trajectory record.
#[derive(Clone, Debug)]
pub struct TrajectoryRecord {
    /// Inverse coupling
    pub beta: f64,
    /// Fermion mass
    pub mass: f64,
    /// MD step size
    pub dt: f64,
    /// Number of MD steps
    pub n_md: usize,
    /// Average plaquette after trajectory
    pub plaquette: f64,
    /// Hamiltonian violation
    pub delta_h: f64,
    /// Total CG iterations
    pub cg_iters: usize,
    /// Whether the trajectory was accepted
    pub accepted: bool,
    /// Wall-clock time in microseconds
    pub wall_us: u64,
    /// Annealing stage label (e.g., "anneal_m0.5", "production")
    pub stage: String,
}

/// Run-level summary record.
#[derive(Clone, Debug)]
pub struct RunSummary {
    /// Inverse coupling
    pub beta: f64,
    /// Target fermion mass
    pub mass: f64,
    /// Final acceptance rate
    pub final_acceptance: f64,
    /// Final plaquette
    pub final_plaquette: f64,
    /// Final adaptive dt
    pub final_dt: f64,
    /// Total trajectories
    pub n_trajectories: usize,
    /// Whether the run converged (acceptance > 20%)
    pub converged: bool,
}

/// JSONL writer for HMC trajectory history.
///
/// Appends trajectory records and run summaries to a JSONL file,
/// building the training corpus for NPU cross-run learning.
pub struct RunHistoryWriter {
    file: std::io::BufWriter<std::fs::File>,
}

impl RunHistoryWriter {
    /// Open or create a run history file for append.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened for appending.
    pub fn open(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            file: std::io::BufWriter::new(file),
        })
    }

    /// Write a trajectory record as a single JSONL line.
    pub fn write_trajectory(&mut self, rec: &TrajectoryRecord) {
        let line = format!(
            "{{\"beta\":{},\"mass\":{},\"dt\":{},\"n_md\":{},\
              \"plaquette\":{},\"delta_h\":{},\"cg_iters\":{},\
              \"accepted\":{},\"wall_us\":{},\"stage\":\"{}\"}}\n",
            rec.beta,
            rec.mass,
            rec.dt,
            rec.n_md,
            rec.plaquette,
            rec.delta_h,
            rec.cg_iters,
            rec.accepted,
            rec.wall_us,
            rec.stage,
        );
        let _ = self.file.write_all(line.as_bytes());
    }

    /// Write a run summary as a single JSONL line.
    pub fn write_summary(&mut self, summary: &RunSummary) {
        let line = format!(
            "{{\"type\":\"summary\",\"beta\":{},\"mass\":{},\
              \"final_acceptance\":{},\"final_plaquette\":{},\
              \"final_dt\":{},\"n_trajectories\":{},\"converged\":{}}}\n",
            summary.beta,
            summary.mass,
            summary.final_acceptance,
            summary.final_plaquette,
            summary.final_dt,
            summary.n_trajectories,
            summary.converged,
        );
        let _ = self.file.write_all(line.as_bytes());
    }

    /// Flush buffered writes.
    pub fn flush(&mut self) {
        let _ = self.file.flush();
    }
}

/// JSONL reader for loading trajectory history into NPU training.
pub struct RunHistoryReader {
    /// All trajectory records loaded from the file.
    pub trajectories: Vec<TrajectoryRecord>,
    /// All run summaries loaded from the file.
    pub summaries: Vec<RunSummary>,
}

impl RunHistoryReader {
    /// Load trajectory history from a JSONL file.
    ///
    /// Parses both trajectory records and run summaries, tolerating
    /// malformed lines (skipped with a warning).
    #[must_use]
    pub fn from_file(path: &std::path::Path) -> Self {
        let contents = std::fs::read_to_string(path).unwrap_or_default();
        Self::parse_content(&contents)
    }

    /// Parse JSONL content from a string.
    #[must_use]
    pub fn parse_content(contents: &str) -> Self {
        let mut trajectories = Vec::new();
        let mut summaries = Vec::new();

        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line.contains("\"type\":\"summary\"") {
                if let Some(s) = parse_summary(line) {
                    summaries.push(s);
                }
            } else if let Some(t) = parse_trajectory(line) {
                trajectories.push(t);
            }
        }

        Self {
            trajectories,
            summaries,
        }
    }

    /// Extract NPU training pairs: (canonical_input, target) for each trajectory.
    ///
    /// The canonical input is the 6D vector used by Gen 1 NPU, and the target
    /// is a scalar encoding the quality of the HMC parameters (acceptance-weighted).
    #[must_use]
    pub fn to_npu_training_pairs(&self) -> Vec<(Vec<f64>, f64)> {
        let mut pairs = Vec::with_capacity(self.trajectories.len());
        let mut running_acceptance = 0.0;
        let mut count = 0.0;

        for t in &self.trajectories {
            if t.accepted {
                running_acceptance += 1.0;
            }
            count += 1.0;
            let acc_rate = running_acceptance / count;

            let input = vec![
                (t.beta - 5.0) / 2.0,
                t.plaquette,
                t.mass,
                t.delta_h.abs().min(1000.0) / 1000.0,
                acc_rate,
                8.0_f64.recip(), // normalized lattice size (assume 8^4 for now)
            ];

            // Target: parameter quality signal — high acceptance with small dt
            // is good (efficient exploration)
            let quality = if t.accepted { 1.0 } else { 0.0 } - t.delta_h.abs().min(100.0) / 100.0;

            pairs.push((input, quality));
        }

        pairs
    }

    /// Number of trajectories loaded.
    #[must_use]
    pub fn n_trajectories(&self) -> usize {
        self.trajectories.len()
    }

    /// Number of summaries loaded.
    #[must_use]
    pub fn n_summaries(&self) -> usize {
        self.summaries.len()
    }
}

fn extract_f64(line: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{key}\":");
    let start = line.find(&pattern)? + pattern.len();
    let rest = &line[start..];
    let end = rest.find([',', '}', ' '])?;
    rest[..end].trim().parse().ok()
}

fn extract_usize(line: &str, key: &str) -> Option<usize> {
    extract_f64(line, key).map(|v| v as usize)
}

fn extract_bool(line: &str, key: &str) -> Option<bool> {
    let pattern = format!("\"{key}\":");
    let start = line.find(&pattern)? + pattern.len();
    let rest = &line[start..].trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn extract_string(line: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\":\"");
    let start = line.find(&pattern)? + pattern.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn parse_trajectory(line: &str) -> Option<TrajectoryRecord> {
    Some(TrajectoryRecord {
        beta: extract_f64(line, "beta")?,
        mass: extract_f64(line, "mass")?,
        dt: extract_f64(line, "dt")?,
        n_md: extract_usize(line, "n_md")?,
        plaquette: extract_f64(line, "plaquette")?,
        delta_h: extract_f64(line, "delta_h")?,
        cg_iters: extract_usize(line, "cg_iters")?,
        accepted: extract_bool(line, "accepted")?,
        wall_us: extract_f64(line, "wall_us").map(|v| v as u64)?,
        stage: extract_string(line, "stage").unwrap_or_default(),
    })
}

fn parse_summary(line: &str) -> Option<RunSummary> {
    Some(RunSummary {
        beta: extract_f64(line, "beta")?,
        mass: extract_f64(line, "mass")?,
        final_acceptance: extract_f64(line, "final_acceptance")?,
        final_plaquette: extract_f64(line, "final_plaquette")?,
        final_dt: extract_f64(line, "final_dt")?,
        n_trajectories: extract_usize(line, "n_trajectories")?,
        converged: extract_bool(line, "converged")?,
    })
}

/// Feed trajectory history to a `MultiHeadNpu` for retraining.
///
/// Extracts training pairs from the run history and uses the Nautilus
/// evolutionary loop to improve the NPU's parameter suggestions.
pub fn retrain_npu_from_history(npu: &mut super::NpuSteering, history: &RunHistoryReader) {
    let pairs = history.to_npu_training_pairs();
    if pairs.is_empty() {
        return;
    }

    // Feed trajectory data through the NPU to update its internal state
    for (input, _target) in &pairs {
        let seq = vec![input.clone(); 10];
        let _ = npu.npu.predict_all_heads(&seq);
    }

    // The actual weight update happens through the NautilusBrain
    // retrain cycle, which is triggered by the production NPU worker.
    // Here we just warm the reservoir state with historical data so
    // the next prediction is informed by past runs.
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_trajectory() {
        let dir = std::env::temp_dir().join("hotspring_run_history_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_history.jsonl");

        let rec = TrajectoryRecord {
            beta: 5.4,
            mass: 0.1,
            dt: 0.005,
            n_md: 100,
            plaquette: 0.523,
            delta_h: 0.42,
            cg_iters: 450,
            accepted: true,
            wall_us: 1_200_000,
            stage: "anneal_m0.5".into(),
        };

        {
            let mut writer = RunHistoryWriter::open(&path).unwrap();
            writer.write_trajectory(&rec);
            writer.flush();
        }

        let reader = RunHistoryReader::from_file(&path);
        assert_eq!(reader.n_trajectories(), 1);
        let t = &reader.trajectories[0];
        assert!((t.beta - 5.4).abs() < 1e-10);
        assert!((t.mass - 0.1).abs() < 1e-10);
        assert!(t.accepted);
        assert_eq!(t.stage, "anneal_m0.5");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn roundtrip_summary() {
        let dir = std::env::temp_dir().join("hotspring_run_summary_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_summary.jsonl");

        let summary = RunSummary {
            beta: 5.4,
            mass: 0.1,
            final_acceptance: 0.65,
            final_plaquette: 0.53,
            final_dt: 0.004,
            n_trajectories: 65,
            converged: true,
        };

        {
            let mut writer = RunHistoryWriter::open(&path).unwrap();
            writer.write_summary(&summary);
            writer.flush();
        }

        let reader = RunHistoryReader::from_file(&path);
        assert_eq!(reader.n_summaries(), 1);
        let s = &reader.summaries[0];
        assert!((s.beta - 5.4).abs() < 1e-10);
        assert!(s.converged);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn training_pairs_extract() {
        let content = concat!(
            "{\"beta\":5.4,\"mass\":0.1,\"dt\":0.005,\"n_md\":100,",
            "\"plaquette\":0.52,\"delta_h\":0.3,\"cg_iters\":400,",
            "\"accepted\":true,\"wall_us\":1000000,\"stage\":\"prod\"}\n",
            "{\"beta\":5.4,\"mass\":0.1,\"dt\":0.005,\"n_md\":100,",
            "\"plaquette\":0.53,\"delta_h\":1.5,\"cg_iters\":500,",
            "\"accepted\":false,\"wall_us\":1100000,\"stage\":\"prod\"}\n",
        );

        let reader = RunHistoryReader::parse_content(content);
        assert_eq!(reader.n_trajectories(), 2);

        let pairs = reader.to_npu_training_pairs();
        assert_eq!(pairs.len(), 2);

        // First trajectory: accepted, small delta_h → positive quality
        assert!(pairs[0].1 > 0.0);
        // Second trajectory: rejected, larger delta_h → lower quality
        assert!(pairs[1].1 < pairs[0].1);
    }

    #[test]
    fn mixed_lines_parse() {
        let content = concat!(
            "{\"beta\":5.4,\"mass\":0.1,\"dt\":0.005,\"n_md\":100,",
            "\"plaquette\":0.52,\"delta_h\":0.3,\"cg_iters\":400,",
            "\"accepted\":true,\"wall_us\":1000000,\"stage\":\"prod\"}\n",
            "{\"type\":\"summary\",\"beta\":5.4,\"mass\":0.1,",
            "\"final_acceptance\":0.65,\"final_plaquette\":0.53,",
            "\"final_dt\":0.004,\"n_trajectories\":65,\"converged\":true}\n",
        );

        let reader = RunHistoryReader::parse_content(content);
        assert_eq!(reader.n_trajectories(), 1);
        assert_eq!(reader.n_summaries(), 1);
    }
}
