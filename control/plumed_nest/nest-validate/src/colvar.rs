// SPDX-License-Identifier: AGPL-3.0-or-later

//! COLVAR file parser — handles PLUMED collective variable output.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Colvar {
    pub fields: Vec<String>,
    pub data: Vec<Vec<f64>>,
    pub n_frames: usize,
}

impl Colvar {
    pub fn column(&self, name: &str) -> Option<Vec<f64>> {
        let idx = self.fields.iter().position(|f| f == name)?;
        Some(self.data.iter().map(|row| row[idx]).collect())
    }

    pub fn time(&self) -> Vec<f64> {
        self.column("time").unwrap_or_else(|| {
            self.data.iter().map(|row| row[0]).collect()
        })
    }

    pub fn total_time_ps(&self) -> f64 {
        self.data.last().map(|row| row[0]).unwrap_or(0.0)
    }

    pub fn total_time_ns(&self) -> f64 {
        self.total_time_ps() / 1000.0
    }

    pub fn column_stats(&self, name: &str) -> Option<ColumnStats> {
        let col = self.column(name)?;
        let n = col.len() as f64;
        let mean = col.iter().sum::<f64>() / n;
        let variance = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Some(ColumnStats { mean, std: variance.sqrt(), min, max })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

pub fn parse_colvar(path: &Path) -> Result<Colvar, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read COLVAR: {e}"))?;

    let mut fields = Vec::new();
    let mut data = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("#! FIELDS") {
            fields = line.split_whitespace().skip(2).map(|s| s.to_string()).collect();
        } else if line.starts_with('#') || line.starts_with('@') || line.is_empty() {
            continue;
        } else {
            let row: Result<Vec<f64>, _> = line.split_whitespace()
                .map(|s| s.parse::<f64>())
                .collect();
            match row {
                Ok(r) if !fields.is_empty() && r.len() == fields.len() => data.push(r),
                Ok(r) if fields.is_empty() && r.len() >= 2 => data.push(r),
                _ => continue,
            }
        }
    }

    let n_frames = data.len();
    Ok(Colvar { fields, data, n_frames })
}

/// Compute 1D free energy surface from a CV column using bias reweighting.
///
/// F(s) = -kT * ln(P_reweighted(s))
pub fn reweighted_fes_1d(
    cv: &[f64],
    bias: &[f64],
    n_bins: usize,
    kt: f64,
) -> (Vec<f64>, Vec<f64>) {
    let weights: Vec<f64> = bias.iter().map(|b| (b / kt).exp()).collect();
    let w_sum: f64 = weights.iter().sum();

    let cv_min = cv.iter().cloned().fold(f64::INFINITY, f64::min);
    let cv_max = cv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let margin = (cv_max - cv_min) * 0.01;
    let bin_min = cv_min - margin;
    let bin_max = cv_max + margin;
    let bin_width = (bin_max - bin_min) / n_bins as f64;

    let mut histogram = vec![0.0_f64; n_bins];
    for (i, &s) in cv.iter().enumerate() {
        let bin = ((s - bin_min) / bin_width) as usize;
        let bin = bin.min(n_bins - 1);
        histogram[bin] += weights[i] / w_sum;
    }

    let centers: Vec<f64> = (0..n_bins)
        .map(|i| bin_min + (i as f64 + 0.5) * bin_width)
        .collect();

    let fes: Vec<f64> = histogram
        .iter()
        .map(|&p| {
            if p > 0.0 {
                -kt * p.ln()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    // Shift min to zero
    let fes_min = fes.iter().cloned().filter(|x| x.is_finite()).fold(f64::INFINITY, f64::min);
    let fes: Vec<f64> = fes.iter().map(|f| if f.is_finite() { f - fes_min } else { *f }).collect();

    (centers, fes)
}

/// Count transitions between two states defined by thresholds on a CV.
pub fn count_transitions(cv: &[f64], folded_thresh: f64, unfolded_thresh: f64) -> (usize, usize) {
    let mut state: Option<bool> = None; // true = folded
    let mut fold_events = 0;
    let mut unfold_events = 0;

    for &val in cv {
        if val > folded_thresh {
            if state == Some(false) {
                fold_events += 1;
            }
            state = Some(true);
        } else if val < unfolded_thresh {
            if state == Some(true) {
                unfold_events += 1;
            }
            state = Some(false);
        }
    }

    (fold_events, unfold_events)
}
