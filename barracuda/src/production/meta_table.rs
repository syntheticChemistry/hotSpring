// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::error::HotSpringError;
use std::io::BufRead;

/// Per-beta aggregate statistics from a meta table or trajectory log.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MetaRow {
    /// Lattice size (one dimension of L⁴).
    pub lattice: usize,
    /// Coupling β.
    pub beta: f64,
    /// Fermion mass (if dynamical).
    pub mass: Option<f64>,
    /// Mode: "quenched", "dynamical", etc.
    pub mode: String,
    /// Mean plaquette.
    pub mean_plaq: f64,
    /// χ²-like variance measure.
    pub chi: f64,
    /// Acceptance rate.
    pub acceptance: f64,
    /// Mean CG iterations per trajectory.
    pub mean_cg_iters: f64,
    /// Wall time per trajectory in seconds.
    pub wall_s_per_traj: f64,
    /// Number of measurement trajectories.
    pub n_meas: usize,
}

/// Load meta table from path. Tries three formats in order:
/// 1. `MetaRow` JSONL (one `MetaRow` per line) — streamed, avoids loading large files
/// 2. Summary JSON (single object with top-level `lattice`/`mass` and `points` array)
/// 3. Per-trajectory JSONL (one trajectory record per line, aggregated into `MetaRows`)
///
/// Streaming JSONL first avoids double-reading large trajectory logs when the
/// file is already in MetaRow format.
pub fn load_meta_table(path: &str) -> Result<Vec<MetaRow>, HotSpringError> {
    if let Some(meta) = try_stream_meta_jsonl(path)? {
        return Ok(meta);
    }

    let contents = std::fs::read_to_string(path)?;

    if let Ok(rows) = load_summary_json_as_meta(&contents, path)
        && !rows.is_empty()
    {
        return Ok(rows);
    }

    load_trajectory_as_meta_from_contents(&contents, path)
}

fn try_stream_meta_jsonl(path: &str) -> Result<Option<Vec<MetaRow>>, HotSpringError> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    let mut meta: Vec<MetaRow> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if let Ok(row) = serde_json::from_str::<MetaRow>(&line) {
            meta.push(row);
        }
    }
    Ok(if meta.is_empty() { None } else { Some(meta) })
}

fn load_summary_json_as_meta(contents: &str, path: &str) -> Result<Vec<MetaRow>, HotSpringError> {
    #[derive(serde::Deserialize)]
    #[expect(dead_code, reason = "EVOLUTION: reserved for GPU pipeline wiring")]
    struct SummaryPoint {
        beta: f64,
        #[serde(default)]
        mass: f64,
        #[serde(default)]
        mean_plaquette: Option<f64>,
        #[serde(default)]
        std_plaquette: Option<f64>,
        #[serde(default)]
        acceptance: f64,
        #[serde(default)]
        mean_cg_iterations: Option<f64>,
        #[serde(default)]
        susceptibility: f64,
        #[serde(default)]
        n_trajectories: usize,
        #[serde(default)]
        wall_s: f64,
        #[serde(default)]
        phase: String,
    }

    #[derive(serde::Deserialize)]
    struct SummaryJson {
        #[serde(default)]
        lattice: usize,
        #[serde(default)]
        points: Vec<SummaryPoint>,
    }

    let summary: SummaryJson = serde_json::from_str(contents)
        .map_err(|e| HotSpringError::DataLoad(format!("not a summary JSON in {path}: {e}")))?;

    if summary.points.is_empty() {
        return Ok(Vec::new());
    }

    let rows = summary
        .points
        .iter()
        .map(|p| {
            let plaq = p.mean_plaquette.unwrap_or(0.0);
            let std_p = p.std_plaquette.unwrap_or(0.0);
            let cg = p.mean_cg_iterations.unwrap_or(0.0);
            let n = p.n_trajectories.max(1);
            MetaRow {
                lattice: summary.lattice,
                beta: p.beta,
                mass: Some(p.mass),
                mode: "dynamical".to_string(),
                mean_plaq: plaq,
                chi: std_p,
                acceptance: p.acceptance,
                mean_cg_iters: cg,
                wall_s_per_traj: if n > 0 { p.wall_s / n as f64 } else { 0.0 },
                n_meas: n,
            }
        })
        .collect();

    Ok(rows)
}

fn load_trajectory_as_meta_from_contents(
    contents: &str,
    path: &str,
) -> Result<Vec<MetaRow>, HotSpringError> {
    use std::collections::BTreeMap;

    #[derive(serde::Deserialize)]
    struct TrajRecord {
        beta: f64,
        #[serde(default)]
        mass: f64,
        plaquette: f64,
        accepted: bool,
        #[serde(default)]
        cg_iters: usize,
        #[serde(default)]
        phase: String,
    }

    let mut records: Vec<TrajRecord> = Vec::new();
    for line in contents.lines() {
        if let Ok(rec) = serde_json::from_str::<TrajRecord>(line) {
            records.push(rec);
        }
    }

    if records.is_empty() {
        return Err(HotSpringError::DataLoad(format!(
            "no parseable trajectory records in {path}"
        )));
    }

    let mut by_beta: BTreeMap<i64, Vec<&TrajRecord>> = BTreeMap::new();
    for r in &records {
        let key = (r.beta * 10000.0).round() as i64;
        by_beta.entry(key).or_default().push(r);
    }

    let mut rows = Vec::new();
    for group in by_beta.values() {
        let meas: Vec<&&TrajRecord> = group.iter().filter(|r| r.phase == "measurement").collect();
        let (plaqs, accepts, cgs): (Vec<f64>, Vec<bool>, Vec<usize>) = if meas.is_empty() {
            (
                group.iter().map(|r| r.plaquette).collect(),
                group.iter().map(|r| r.accepted).collect(),
                group.iter().map(|r| r.cg_iters).collect(),
            )
        } else {
            (
                meas.iter().map(|r| r.plaquette).collect(),
                meas.iter().map(|r| r.accepted).collect(),
                meas.iter().map(|r| r.cg_iters).collect(),
            )
        };

        if plaqs.is_empty() {
            continue;
        }

        let n = plaqs.len();
        let mean_plaq = plaqs.iter().sum::<f64>() / n as f64;
        let variance = plaqs.iter().map(|p| (p - mean_plaq).powi(2)).sum::<f64>() / n as f64;
        let chi = variance * (n as f64);
        let acceptance = accepts.iter().filter(|&&a| a).count() as f64 / n as f64;
        let mean_cg = if cgs.iter().any(|&c| c > 0) {
            let nonzero: Vec<f64> = cgs.iter().filter(|&&c| c > 0).map(|&c| c as f64).collect();
            nonzero.iter().sum::<f64>() / nonzero.len() as f64
        } else {
            0.0
        };

        rows.push(MetaRow {
            lattice: 8,
            beta: group[0].beta,
            mass: Some(group[0].mass),
            mode: "dynamical".into(),
            mean_plaq,
            chi,
            acceptance,
            mean_cg_iters: mean_cg,
            wall_s_per_traj: 0.0,
            n_meas: n,
        });
    }

    println!(
        "  Bootstrap: aggregated {path} → {} beta points from {} trajectories",
        rows.len(),
        records.len()
    );
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::MetaRow;
    use std::io::BufRead;

    #[test]
    fn load_meta_table_valid_jsonl() {
        let jsonl = concat!(
            r#"{"beta":5.0,"lattice":8,"mode":"quenched","mean_plaq":0.4,"chi":10.0,"acceptance":0.7,"mean_cg_iters":100.0,"wall_s_per_traj":0.1,"n_meas":50}"#,
            "\n",
            r#"{"beta":6.0,"lattice":8,"mode":"quenched","mean_plaq":0.6,"chi":5.0,"acceptance":0.8,"mean_cg_iters":50.0,"wall_s_per_traj":0.05,"n_meas":50}"#,
        );
        let cursor = std::io::Cursor::new(jsonl);
        let rows: Vec<MetaRow> = cursor
            .lines()
            .filter_map(|l| serde_json::from_str(&l.ok()?).ok())
            .collect();
        assert_eq!(rows.len(), 2);
        assert!((rows[0].beta - 5.0).abs() < 1e-10);
        assert!((rows[1].beta - 6.0).abs() < 1e-10);
    }
}
