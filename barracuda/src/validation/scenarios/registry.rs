// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario registry — taxonomy, metadata, and collection.
//!
//! Follows the primalSpring `ScenarioMeta` pattern for provenance,
//! track classification, and two-tier filtering.

use std::fmt;

/// Validation tier for scenario filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Pure Rust library validation — no IPC, safe for CI.
    Rust,
    /// Live NUCLEUS validation — requires deployed primals.
    Live,
    /// Exercises both tiers.
    Both,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rust => write!(f, "rust"),
            Self::Live => write!(f, "live"),
            Self::Both => write!(f, "both"),
        }
    }
}

/// Validation track — domain grouping for scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Track {
    NuclearPhysics,
    LatticeQcd,
    MolecularDynamics,
    SpectralTheory,
    GpuCompute,
    CompositionParity,
    TransportCoefficients,
    DomainScience,
}

impl Track {
    /// Loose string matching for CLI `--track` filter.
    #[must_use]
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s {
            "nuclear" | "nuclear-physics" => Some(Self::NuclearPhysics),
            "lattice" | "lattice-qcd" | "qcd" => Some(Self::LatticeQcd),
            "md" | "molecular-dynamics" => Some(Self::MolecularDynamics),
            "spectral" | "spectral-theory" => Some(Self::SpectralTheory),
            "gpu" | "gpu-compute" => Some(Self::GpuCompute),
            "composition" | "composition-parity" => Some(Self::CompositionParity),
            "transport" | "transport-coefficients" => Some(Self::TransportCoefficients),
            "domain" | "domain-science" => Some(Self::DomainScience),
            _ => None,
        }
    }
}

impl fmt::Display for Track {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NuclearPhysics => write!(f, "nuclear-physics"),
            Self::LatticeQcd => write!(f, "lattice-qcd"),
            Self::MolecularDynamics => write!(f, "molecular-dynamics"),
            Self::SpectralTheory => write!(f, "spectral-theory"),
            Self::GpuCompute => write!(f, "gpu-compute"),
            Self::CompositionParity => write!(f, "composition-parity"),
            Self::TransportCoefficients => write!(f, "transport-coefficients"),
            Self::DomainScience => write!(f, "domain-science"),
        }
    }
}

/// Scenario metadata — provenance, classification, and description.
#[derive(Debug, Clone)]
pub struct ScenarioMeta {
    /// Unique scenario identifier (e.g. `"semf-parity"`).
    pub id: &'static str,
    /// Which track this scenario belongs to.
    pub track: Track,
    /// Which validation tier this scenario exercises.
    pub tier: Tier,
    /// Original experiment binary name for provenance.
    pub provenance_crate: &'static str,
    /// Date of last significant update.
    pub provenance_date: &'static str,
    /// One-line description.
    pub description: &'static str,
}

/// A registered scenario: metadata + entry point function.
pub struct Scenario {
    pub meta: ScenarioMeta,
    pub run: fn(&mut crate::validation::ValidationHarness),
}

/// Collection of registered scenarios with filtering.
pub struct ScenarioRegistry {
    scenarios: Vec<Scenario>,
}

impl ScenarioRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scenarios: Vec::new(),
        }
    }

    pub fn register(&mut self, scenario: Scenario) {
        self.scenarios.push(scenario);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.scenarios.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scenarios.is_empty()
    }

    pub fn all(&self) -> &[Scenario] {
        &self.scenarios
    }
}

impl Default for ScenarioRegistry {
    fn default() -> Self {
        Self::new()
    }
}
