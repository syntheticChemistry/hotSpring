// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ensemble-level manifest: action parameters, provenance, and config catalog.

use serde::{Deserialize, Serialize};

use super::provenance::RunManifest;
use super::time_host::{hostname_best_effort, iso8601_now};

/// Ensemble-level manifest — one per production run.
///
/// Written alongside the ILDG config files as `ensemble.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnsembleManifest {
    /// Schema version for forward compatibility.
    pub schema_version: String,
    /// Unique ensemble identifier (matches ILDG metadata).
    pub ensemble_id: String,
    /// ISO 8601 timestamp of ensemble creation.
    pub created: String,

    // Action parameters
    /// Gauge action type ("Wilson", "Symanzik", etc.).
    pub gauge_action: String,
    /// Fermion action ("none", "staggered", "HISQ").
    pub fermion_action: String,
    /// Inverse bare coupling.
    pub beta: f64,
    /// Quark mass (0.0 for quenched).
    pub mass: f64,
    /// Number of dynamical flavors.
    pub nf: usize,
    /// Lattice dimensions `[Nx, Ny, Nz, Nt]`.
    pub dims: [usize; 4],

    /// Hardware and software provenance.
    pub provenance: Provenance,
    /// Algorithm parameters used for generation.
    pub algorithm: AlgorithmParams,
    /// Run manifest with full invocation metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<RunManifest>,
    /// List of configurations in this ensemble.
    pub configs: Vec<ConfigEntry>,
}

/// Hardware + software provenance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Provenance {
    /// Engine name and version.
    pub engine: String,
    /// GPU adapter name (if GPU was used).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    /// CPU architecture.
    pub arch: String,
    /// Operating system.
    pub os: String,
    /// Hostname.
    pub hostname: String,
}

/// HMC/RHMC algorithm parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlgorithmParams {
    /// Algorithm type ("HMC", "RHMC").
    pub algorithm: String,
    /// Integrator type ("Leapfrog", "Omelyan").
    pub integrator: String,
    /// MD step size.
    pub dt: f64,
    /// Number of MD steps per trajectory.
    pub n_md_steps: usize,
    /// Number of thermalization trajectories.
    pub n_therm: usize,
    /// Measurement interval (trajectories between saved configs).
    pub meas_interval: usize,
    /// CG solver tolerance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cg_tol: Option<f64>,
    /// Random seed.
    pub seed: u64,
}

/// Entry for a single configuration within an ensemble.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfigEntry {
    /// HMC trajectory index.
    pub trajectory: usize,
    /// Filename of the ILDG binary file.
    pub filename: String,
    /// ILDG logical file name.
    pub ildg_lfn: String,
    /// CRC32 checksum of the ILDG file (Ethernet CRC32, legacy).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum_crc32: Option<String>,
    /// ILDG-standard CRC (POSIX cksum / GNU cksum algorithm).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum_ildg_crc: Option<u32>,
    /// Average plaquette at save time.
    pub plaquette: f64,
}

impl EnsembleManifest {
    /// Create a new manifest with defaults filled in.
    pub fn new(ensemble_id: &str, dims: [usize; 4], beta: f64) -> Self {
        Self {
            schema_version: "1.0".to_string(),
            ensemble_id: ensemble_id.to_string(),
            created: iso8601_now(),
            gauge_action: "Wilson".to_string(),
            fermion_action: "none".to_string(),
            beta,
            mass: 0.0,
            nf: 0,
            dims,
            provenance: Provenance {
                engine: format!("hotSpring-barracuda {}", env!("CARGO_PKG_VERSION")),
                gpu: None,
                arch: std::env::consts::ARCH.to_string(),
                os: std::env::consts::OS.to_string(),
                hostname: hostname_best_effort(),
            },
            algorithm: AlgorithmParams {
                algorithm: "HMC".to_string(),
                integrator: "Omelyan".to_string(),
                dt: 0.01,
                n_md_steps: 20,
                n_therm: 200,
                meas_interval: 10,
                cg_tol: None,
                seed: 42,
            },
            run: None,
            configs: Vec::new(),
        }
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
    }
}
