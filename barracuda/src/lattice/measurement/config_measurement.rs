// SPDX-License-Identifier: AGPL-3.0-or-later

//! Per-configuration measurement record: gauge, flow, topology, fermions, HVP.

use serde::{Deserialize, Serialize};

use super::provenance::{ImplementationInfo, RunManifest};

/// Per-configuration measurement results.
///
/// Written as `measurements/conf_NNNNNN.json` alongside the ILDG files.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfigMeasurement {
    /// Schema version.
    pub schema_version: String,
    /// Ensemble this config belongs to.
    pub ensemble_id: String,
    /// ILDG logical file name.
    pub ildg_lfn: String,
    /// HMC trajectory index.
    pub trajectory: usize,

    /// Gauge observables.
    pub gauge: GaugeObservables,
    /// Gradient flow results (if measured).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flow: Option<FlowResults>,
    /// Topological charge (if measured).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topology: Option<TopologyResults>,
    /// Wilson loop measurements (if measured).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wilson_loops: Option<Vec<WilsonLoopEntry>>,
    /// Fermion observables (if measured).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fermion: Option<FermionObservables>,
    /// HMC diagnostics for this trajectory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<HmcDiagnostics>,
    /// Implementation provenance for this measurement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub implementation: Option<ImplementationInfo>,
    /// Run manifest with full invocation metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<RunManifest>,
    /// Scale-setting context (lattice spacing in physical units).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale_setting: Option<ScaleSetting>,
    /// HVP (hadronic vacuum polarization) correlator and integral.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hvp: Option<HvpResults>,

    /// Wall-clock time for all measurements on this config (seconds).
    pub wall_seconds: f64,
}

/// Basic gauge observables measured on every configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaugeObservables {
    /// Average plaquette.
    pub plaquette: f64,
    /// Polyakov loop magnitude (spatial average).
    pub polyakov_abs: f64,
    /// Polyakov loop (Re, Im) spatial average.
    pub polyakov_re: f64,
    /// Polyakov loop imaginary part.
    pub polyakov_im: f64,
    /// Wilson action density = 6(1 - P).
    pub action_density: f64,
}

/// Gradient flow measurement results.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlowResults {
    /// Flow integrator used ("Rk3Luscher", "Lscfrk3w7", "Lscfrk4ck").
    pub integrator: String,
    /// Flow step size epsilon.
    pub epsilon: f64,
    /// Maximum flow time.
    pub t_max: f64,
    /// Scale t0 (from t^2 E(t) = 0.3), or null if not found.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub t0: Option<f64>,
    /// Scale w0, or null if not found.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub w0: Option<f64>,
    /// Flow curve: list of (t, E(t), t^2 E(t)) measurements.
    pub flow_curve: Vec<FlowPoint>,
}

/// Single point on the gradient flow curve.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlowPoint {
    /// Flow time.
    pub t: f64,
    /// Energy density E(t).
    pub energy_density: f64,
    /// t^2 E(t).
    pub t2_e: f64,
}

/// Topological charge measurement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopologyResults {
    /// Topological charge Q (measured on flowed config).
    pub charge: f64,
    /// Flow time at which Q was measured.
    pub flow_time: f64,
}

/// Single Wilson loop measurement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WilsonLoopEntry {
    /// Spatial extent R.
    pub r: usize,
    /// Temporal extent T.
    pub t: usize,
    /// Average (1/3) Re Tr W(R,T).
    pub value: f64,
}

/// Fermion observables.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FermionObservables {
    /// Chiral condensate \<psi-bar psi\>.
    pub chiral_condensate: f64,
    /// Statistical error on condensate.
    pub chiral_condensate_error: f64,
    /// Number of stochastic sources.
    pub n_sources: usize,
    /// Mass used for the measurement.
    pub mass: f64,
}

/// HVP (hadronic vacuum polarization) correlator and integral.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HvpResults {
    /// Quark mass used for the propagator inversion.
    pub mass: f64,
    /// CG solver residual achieved.
    pub cg_residual: f64,
    /// Number of CG iterations.
    pub cg_iterations: usize,
    /// Time-slice correlator C(t) = Σ_x |G(x,t)|².
    pub correlator: Vec<f64>,
    /// Lattice-units HVP integral: a_μ^HVP ∝ Σ_t K(t) C(t).
    pub hvp_integral: f64,
}

/// HMC trajectory diagnostics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HmcDiagnostics {
    /// Whether this trajectory was accepted.
    pub accepted: bool,
    /// Hamiltonian violation delta_H.
    pub delta_h: f64,
    /// Total CG iterations (for dynamical fermions).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cg_iterations: Option<usize>,
    /// Wall time for this trajectory in seconds.
    pub trajectory_seconds: f64,
}

/// Scale-setting context: lattice spacing in physical units.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScaleSetting {
    /// Lattice spacing `a` in fm (from t0 or w0 scale).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub a_fm: Option<f64>,
    /// Method used for scale setting ("t0", "w0", "r0", "string_tension").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// Physical value of the reference scale used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_value: Option<f64>,
    /// Reference: paper or value used (e.g. "BMW 2012: sqrt(t0) = 0.1465 fm").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
}

impl ConfigMeasurement {
    /// Create a minimal measurement record.
    pub fn new(ensemble_id: &str, trajectory: usize, lfn: &str) -> Self {
        Self {
            schema_version: "1.0".to_string(),
            ensemble_id: ensemble_id.to_string(),
            ildg_lfn: lfn.to_string(),
            trajectory,
            gauge: GaugeObservables {
                plaquette: 0.0,
                polyakov_abs: 0.0,
                polyakov_re: 0.0,
                polyakov_im: 0.0,
                action_density: 0.0,
            },
            flow: None,
            topology: None,
            wilson_loops: None,
            fermion: None,
            diagnostics: None,
            implementation: None,
            run: None,
            scale_setting: None,
            hvp: None,
            wall_seconds: 0.0,
        }
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
    }
}
