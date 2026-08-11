// SPDX-License-Identifier: AGPL-3.0-or-later

//! Provenance and attestation for QCD campaigns.
//!
//! Produces BLAKE3-hashed witness chains for each measurement, suitable for
//! pseudoSpore publication and arXiv reproducibility.

use serde::{Deserialize, Serialize};

/// A single measurement with provenance metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenancedMeasurement {
    pub observable: String,
    pub value: f64,
    pub uncertainty: Option<f64>,
    pub trajectory_index: usize,
    pub config_hash: String,
    pub witness_hash: String,
}

/// Campaign provenance record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignProvenance {
    pub campaign_id: String,
    pub lattice_dims: [u32; 4],
    pub beta: f64,
    pub seed: u32,
    pub n_warmup: usize,
    pub n_production: usize,
    pub git_commit: Option<String>,
    pub measurements: Vec<ProvenancedMeasurement>,
}

impl CampaignProvenance {
    /// Create a new campaign provenance record.
    #[must_use]
    pub fn new(dims: [u32; 4], beta: f64, seed: u32) -> Self {
        let campaign_id = format!(
            "su3_{}x{}x{}x{}_b{:.2}_s{}",
            dims[0], dims[1], dims[2], dims[3], beta, seed
        );
        Self {
            campaign_id,
            lattice_dims: dims,
            beta,
            seed,
            n_warmup: 0,
            n_production: 0,
            git_commit: None,
            measurements: Vec::new(),
        }
    }

    /// Add a plaquette measurement with BLAKE3 witness.
    pub fn add_plaquette(&mut self, value: f64, traj_idx: usize) {
        let config_bytes = format!(
            "{}:{}:{}:{}",
            self.campaign_id, traj_idx, value, self.seed
        );
        let config_hash = blake3::hash(config_bytes.as_bytes()).to_hex().to_string();
        let witness_hash = if let Some(last) = self.measurements.last() {
            let chain = format!("{}:{}", last.witness_hash, config_hash);
            blake3::hash(chain.as_bytes()).to_hex().to_string()
        } else {
            config_hash.clone()
        };

        self.measurements.push(ProvenancedMeasurement {
            observable: "plaquette".into(),
            value,
            uncertainty: None,
            trajectory_index: traj_idx,
            config_hash,
            witness_hash,
        });
    }
}
