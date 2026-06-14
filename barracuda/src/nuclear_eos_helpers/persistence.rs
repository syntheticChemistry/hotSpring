// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSON persistence for nuclear EOS L2 heterogeneous pipeline results.

use std::path::Path;

use barracuda::optimize::EvaluationCache;
use hotspring_barracuda::data;
use hotspring_barracuda::prescreen::CascadeStats;

pub fn save_results(
    base: &Path,
    best_params: &[f64],
    best_f: f64,
    cache: &EvaluationCache,
    stats: &CascadeStats,
    total_time: f64,
    plain_chi2: Option<f64>,
    plain_time: Option<f64>,
    plain_evals: Option<usize>,
) {
    let chi2 = best_f.exp_m1();
    let mut result_json = serde_json::json!({
        "level": 2,
        "engine": "hotspring::heterogeneous_pipeline",
        "architecture": {
            "tier1": "NMP algebraic pre-screen (CPU)",
            "tier2": "L1 SEMF proxy (CPU)",
            "tier3": "logistic regression classifier (NPU-ready, CPU fallback)",
            "tier4": "spherical HFB (CPU parallel, rayon)",
            "surrogate": "barracuda::RBFSurrogate (GPU WGSL cdist)",
        },
        "chi2_per_datum": chi2,
        "log_chi2": best_f,
        "total_hfb_evals": cache.len(),
        "time_seconds": total_time,
        "cascade_stats": {
            "total_candidates": stats.total_candidates,
            "tier1_rejected": stats.tier1_rejected,
            "tier2_rejected": stats.tier2_rejected,
            "tier3_rejected": stats.tier3_rejected,
            "tier4_evaluated": stats.tier4_evaluated,
            "pass_rate": stats.pass_rate(),
        },
        "best_params": best_params,
    });

    if let (Some(pc), Some(pt), Some(pe)) = (plain_chi2, plain_time, plain_evals) {
        result_json["comparison"] = serde_json::json!({
            "plain_chi2_per_datum": pc,
            "plain_time_s": pt,
            "plain_evals": pe,
            "hetero_speedup": pt / total_time,
            "hetero_chi2_improvement": (pc - chi2) / pc,
        });
    }

    data::save_json_to_results(base, "barracuda_l2_hetero.json", &result_json);
}
