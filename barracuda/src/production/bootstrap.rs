// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::error::HotSpringError;
use crate::md::reservoir::NpuSimulator;
use crate::provenance::KNOWN_BETA_C_SU3_NT4 as KNOWN_BETA_C;
use std::io::BufRead;

use super::esn_heuristics::predict_beta_c;
use super::types::BetaResult;

/// Bootstrap ESN from a previous run's trajectory log.
/// Streams JSONL lines, extracts per-beta aggregates, trains ESN.
pub fn bootstrap_esn_from_trajectory_log<F>(
    path: &str,
    make_esn: &F,
    npu: &mut Option<NpuSimulator>,
) -> Result<(usize, f64), HotSpringError>
where
    F: Fn(u64, &[BetaResult]) -> Option<NpuSimulator>,
{
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    let mut beta_data: std::collections::BTreeMap<String, Vec<(f64, bool)>> =
        std::collections::BTreeMap::new();
    let mut n_lines = 0usize;

    for line in reader.lines() {
        let line = line?;
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        if v.get("is_therm") == Some(&serde_json::Value::Bool(true)) {
            continue;
        }
        let Some(beta) = v.get("beta").and_then(serde_json::Value::as_f64) else {
            continue;
        };
        let Some(plaq) = v.get("plaquette").and_then(serde_json::Value::as_f64) else {
            continue;
        };
        let accepted = v
            .get("accepted")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        let key = format!("{beta:.4}");
        beta_data.entry(key).or_default().push((plaq, accepted));
        n_lines += 1;
    }

    if beta_data.is_empty() {
        return Ok((0, KNOWN_BETA_C));
    }

    let results: Vec<BetaResult> = beta_data
        .into_iter()
        .map(|(key, entries)| {
            let beta: f64 = key.parse().unwrap_or(KNOWN_BETA_C);
            let n = entries.len();
            let plaqs: Vec<f64> = entries.iter().map(|(p, _)| *p).collect();
            let mean_plaq = plaqs.iter().sum::<f64>() / n as f64;
            let var_plaq =
                plaqs.iter().map(|p| (p - mean_plaq).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
            let n_accepted = entries.iter().filter(|(_, a)| *a).count();

            BetaResult {
                beta,
                mean_plaq,
                std_plaq: var_plaq.sqrt(),
                polyakov: 0.0,
                susceptibility: var_plaq * 1048576.0,
                action_density: 6.0 * (1.0 - mean_plaq),
                acceptance: n_accepted as f64 / n as f64,
                n_traj: n,
                phase: if beta < KNOWN_BETA_C - 0.1 {
                    "confined"
                } else if beta > KNOWN_BETA_C + 0.1 {
                    "deconfined"
                } else {
                    "transition"
                },
                ..Default::default()
            }
        })
        .collect();

    let n_betas = results.len();
    if let Some(new_npu) = make_esn(42, &results) {
        *npu = Some(new_npu);
    }

    let beta_c = if let Some(n) = npu.as_mut() {
        predict_beta_c(n)
    } else {
        KNOWN_BETA_C
    };

    Ok((n_betas * n_lines / n_betas.max(1), beta_c))
}
