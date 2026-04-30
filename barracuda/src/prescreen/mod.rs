// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pre-screening cascade for nuclear EOS parameter space
//!
//! Reduces expensive L2 HFB evaluations by filtering unphysical parameter sets.
//!
//! **Three-tier cascade:**
//! - Tier 1 (algebraic, ~0 cost): NMP bounds check
//! - Tier 2 (L1 proxy, ~0.001s):  SEMF χ² threshold — if bad at L1, skip L2
//! - Tier 3 (learned, NPU/CPU):   Classifier trained on accumulated data
//!
//! **Heterogeneous compute:**
//! - Tier 1-2: CPU (instant)
//! - Tier 3:   NPU (Akida, ~1W) or CPU fallback
//! - Tier 4:   CPU parallel (rayon) for HFB that passes all screens

mod classifier;
mod cascade;
mod l1_proxy;
mod nmp;
mod objectives;

pub use cascade::{CascadeStats, cascade_filter};
pub use classifier::PreScreenClassifier;
pub use l1_proxy::l1_proxy_prescreen;
pub use nmp::{NMPConstraints, NMPScreenResult, nmp_prescreen};
pub use objectives::{nmp_objective_penalty, perturb_params};

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "prescreen tests use expect on classifiers and fixtures"
)]
mod tests;
