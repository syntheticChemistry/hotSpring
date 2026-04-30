// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::classifier::PreScreenClassifier;
use super::l1_proxy::l1_proxy_prescreen;
use super::nmp::{NMPConstraints, NMPScreenResult, nmp_prescreen};

/// Run candidates through NMP (Tier 1), L1 proxy (Tier 2), and classifier (Tier 3).
///
/// Returns surviving candidates and updates `cascade_stats`.
pub fn cascade_filter<S: std::hash::BuildHasher>(
    candidates: &[Vec<f64>],
    constraints: &NMPConstraints,
    exp_data: &HashMap<(usize, usize), (f64, f64), S>,
    use_classifier: bool,
    classifier: &PreScreenClassifier,
    cascade_stats: &Arc<Mutex<CascadeStats>>,
) -> Vec<Vec<f64>> {
    let mut survivors = Vec::new();
    for params in candidates {
        let Ok(mut stats) = cascade_stats.lock() else {
            continue;
        };
        stats.total_candidates += 1;

        match nmp_prescreen(params, constraints) {
            NMPScreenResult::Fail(_) => {
                stats.tier1_rejected += 1;
                continue;
            }
            NMPScreenResult::Pass(_) => {}
        }
        if l1_proxy_prescreen(params, exp_data, 200.0).is_none() {
            stats.tier2_rejected += 1;
            continue;
        }
        if use_classifier && !classifier.predict(params) {
            stats.tier3_rejected += 1;
            continue;
        }
        stats.tier4_evaluated += 1;
        drop(stats);

        survivors.push(params.clone());
    }
    survivors
}

/// Pre-screening cascade statistics.
#[derive(Debug, Default, Clone)]
#[must_use]
pub struct CascadeStats {
    /// Total parameter sets considered.
    pub total_candidates: usize,
    /// Rejected by Tier 1 (NMP out of bounds).
    pub tier1_rejected: usize,
    /// Rejected by Tier 2 (L1 proxy too high).
    pub tier2_rejected: usize,
    /// Rejected by Tier 3 (classifier says no).
    pub tier3_rejected: usize,
    /// Passed all screens, HFB evaluated.
    pub tier4_evaluated: usize,
}

impl CascadeStats {
    /// Fraction of candidates that passed all pre-screening tiers.
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        if self.total_candidates == 0 {
            return 0.0;
        }
        self.tier4_evaluated as f64 / self.total_candidates as f64
    }

    /// Print a human-readable summary of pre-screening rejection rates.
    pub fn print_summary(&self) {
        println!("  Pre-screening cascade:");
        println!("    Total candidates:     {}", self.total_candidates);
        println!(
            "    Tier 1 rejected (NMP):  {} ({:.1}%)",
            self.tier1_rejected,
            100.0 * self.tier1_rejected as f64 / self.total_candidates.max(1) as f64
        );
        println!(
            "    Tier 2 rejected (L1):   {} ({:.1}%)",
            self.tier2_rejected,
            100.0 * self.tier2_rejected as f64 / self.total_candidates.max(1) as f64
        );
        println!(
            "    Tier 3 rejected (clf):  {} ({:.1}%)",
            self.tier3_rejected,
            100.0 * self.tier3_rejected as f64 / self.total_candidates.max(1) as f64
        );
        println!(
            "    Tier 4 evaluated (HFB): {} ({:.1}% pass rate)",
            self.tier4_evaluated,
            100.0 * self.pass_rate()
        );
    }
}
