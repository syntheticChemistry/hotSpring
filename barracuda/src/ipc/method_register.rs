// SPDX-License-Identifier: AGPL-3.0-or-later

//! biomeOS `method.register` IPC client (v3.51).
//!
//! Registers hotSpring's physics methods with biomeOS dynamically,
//! eliminating the need for manual biomeOS configuration.

use crate::primal_bridge::send_jsonrpc;
use serde::{Deserialize, Serialize};

/// Method registration request for biomeOS v3.51 `method.register`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodRegistration {
    /// Dotted method name (e.g., "physics.nuclear_eos").
    pub method: String,
    /// The primal providing this method.
    pub provider: String,
    /// Human-readable description.
    pub description: String,
}

/// Response from biomeOS `method.register`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrationResult {
    pub registered: bool,
    pub method: String,
    #[serde(default)]
    pub message: String,
}

/// hotSpring's physics methods for dynamic registration.
pub const HOTSPRING_METHODS: &[(&str, &str)] = &[
    (
        "physics.lattice_qcd",
        "Lattice QCD: SU(3) gauge, Wilson action, HMC",
    ),
    (
        "physics.lattice_gauge_update",
        "SU(3) gauge link update via HMC trajectory",
    ),
    (
        "physics.hmc_trajectory",
        "Hybrid Monte Carlo trajectory (leapfrog/Omelyan)",
    ),
    (
        "physics.wilson_dirac",
        "Staggered Dirac operator + CG solver",
    ),
    (
        "physics.molecular_dynamics",
        "GPU molecular dynamics (Yukawa OCP)",
    ),
    ("physics.fluid", "Kinetic-fluid coupling (BGK/Euler)"),
    ("physics.nuclear_eos", "Nuclear EOS: SEMF L1, HFB L2/L3"),
    ("physics.thermal", "Two-temperature model (laser-plasma)"),
    ("physics.radiation", "Dense plasma dielectric response"),
    (
        "physics.semf",
        "Semi-empirical mass formula (Bethe-Weizsacker)",
    ),
    ("physics.hfb_spherical", "Spherical Hartree-Fock-Bogoliubov"),
    ("physics.hfb_deformed", "Deformed HFB in axial symmetry"),
    ("physics.nuclear_matter", "Nuclear matter properties (NMP)"),
    (
        "physics.transport_coefficients",
        "Green-Kubo transport coefficients",
    ),
    (
        "physics.yukawa_ocp",
        "Yukawa OCP simulation (Sarkas-validated)",
    ),
    ("physics.dielectric", "BGK dielectric functions (Mermin)"),
    (
        "physics.bgk_relaxation",
        "BGK relaxation for kinetic plasma",
    ),
    (
        "physics.kinetic_fluid",
        "Multi-species kinetic-fluid coupling",
    ),
    ("physics.plasma_dispersion", "Plasma dispersion function"),
    ("physics.gradient_flow", "Wilson gradient flow on SU(3)"),
    (
        "compute.df64",
        "Double-precision GPU compute (WGSL SHADER_F64)",
    ),
    ("compute.cg_solver", "Conjugate gradient solver (GPU)"),
    (
        "compute.gradient_flow",
        "Gradient flow computation dispatch",
    ),
    ("compute.f64", "f64 precision GPU math"),
];

/// Resolve the biomeOS socket via capability discovery, then env var,
/// then conventional socket-dir fallback.
fn biomeos_socket() -> Option<std::path::PathBuf> {
    let nucleus = crate::primal_bridge::NucleusContext::detect();
    if let Some(ep) = nucleus.by_domain("composition") {
        return Some(ep.socket.clone().into());
    }
    if let Ok(p) = std::env::var("BIOMEOS_SOCKET") {
        let path = std::path::PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    crate::niche::socket_dirs()
        .into_iter()
        .map(|d| d.join("biomeos/biomeos.sock"))
        .find(|p| p.exists())
}

/// Register all hotSpring methods with biomeOS via `method.register`.
///
/// Returns the count of successfully registered methods.
pub fn register_all_methods() -> usize {
    let Some(socket) = biomeos_socket() else {
        return 0;
    };

    let mut registered = 0;
    for &(method, description) in HOTSPRING_METHODS {
        let params = serde_json::json!({
            "method": method,
            "provider": "hotspring",
            "description": description,
        });

        if let Ok(resp) = send_jsonrpc(&socket, "method.register", &params) {
            if let Ok(result) = serde_json::from_value::<RegistrationResult>(resp) {
                if result.registered {
                    registered += 1;
                }
            }
        }
    }
    registered
}

/// Register a single method with biomeOS.
pub fn register_method(method: &str, description: &str) -> Option<RegistrationResult> {
    let socket = biomeos_socket()?;

    let params = serde_json::json!({
        "method": method,
        "provider": "hotspring",
        "description": description,
    });

    let resp = send_jsonrpc(&socket, "method.register", &params).ok()?;
    serde_json::from_value(resp).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hotspring_methods_are_non_empty() {
        assert!(HOTSPRING_METHODS.len() >= 20);
    }

    #[test]
    fn all_methods_have_dotted_names() {
        for &(method, _) in HOTSPRING_METHODS {
            assert!(
                method.contains('.'),
                "method '{method}' should use dotted notation"
            );
        }
    }

    #[test]
    fn register_all_returns_zero_when_biomeos_not_running() {
        assert_eq!(register_all_methods(), 0);
    }
}
