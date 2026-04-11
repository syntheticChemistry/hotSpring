// SPDX-License-Identifier: AGPL-3.0-or-later

//! Formalized measurement output schema for MILC/Bazavov ecosystem.
//!
//! Defines structured JSON schemas for gauge ensemble manifests and
//! per-configuration measurements. These "receipts" pair with ILDG binary
//! configs to produce complete, documented datasets.
//!
//! # Schema overview
//!
//! ```text
//! EnsembleManifest (one per ensemble)
//!   ├── action parameters (beta, mass, Nf, action type)
//!   ├── hardware info (GPU, CPU, engine version)
//!   ├── algorithm parameters (integrator, dt, n_md, CG tol)
//!   └── configs[] → list of ConfigEntry (trajectory, filename, checksum)
//!
//! ConfigMeasurement (one per measured configuration)
//!   ├── config identity (trajectory, ensemble_id, ildg_lfn)
//!   ├── gauge observables (plaquette, Polyakov, action density)
//!   ├── flow results (t0, w0, flow_curves[])
//!   ├── topology (Q, Q_density_rms)
//!   ├── wilson_loops (W(R,T) grid)
//!   ├── fermion observables (chiral condensate, correlators)
//!   └── diagnostics (acceptance, delta_h, CG iters, wall time)
//! ```

mod cli;
mod config_measurement;
mod ensemble;
mod provenance;
mod shader_manifest;
mod stats;
mod time_host;

pub use cli::{format_dims, format_dims_id, min_spatial_dim, parse_dims_from_args};
pub use config_measurement::{
    ConfigMeasurement, FermionObservables, FlowPoint, FlowResults, GaugeObservables,
    HmcDiagnostics, HvpResults, ScaleSetting, TopologyResults, WilsonLoopEntry,
};
pub use ensemble::{AlgorithmParams, ConfigEntry, EnsembleManifest, Provenance};
pub use provenance::{ImplementationInfo, NucleusManifest, RunManifest};
pub use shader_manifest::{
    PaperReference, PathValidation, ShaderBufferLayout, ShaderManifest, ShaderReferenceValue,
    ShaderValidationResult,
};
pub use stats::{ObservableSummary, StatisticalAnalysis, estimate_tau_int, jackknife_error};
pub use time_host::iso8601_now;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn manifest_serializes() {
        let manifest = EnsembleManifest::new("test_ensemble", [8, 8, 8, 8], 6.0);
        let json = manifest.to_json();
        assert!(json.contains("\"ensemble_id\": \"test_ensemble\""));
        assert!(json.contains("\"beta\": 6.0"));
        assert!(json.contains("\"schema_version\": \"1.0\""));
    }

    #[test]
    fn measurement_serializes() {
        let mut meas = ConfigMeasurement::new("test", 42, "/test/lfn");
        meas.gauge.plaquette = 0.598;
        meas.flow = Some(FlowResults {
            integrator: "Lscfrk3w7".to_string(),
            epsilon: 0.01,
            t_max: 4.0,
            t0: Some(1.234),
            w0: Some(0.987),
            flow_curve: vec![FlowPoint {
                t: 0.1,
                energy_density: 0.5,
                t2_e: 0.005,
            }],
        });
        let json = meas.to_json();
        assert!(json.contains("\"plaquette\": 0.598"));
        assert!(json.contains("\"t0\": 1.234"));
        assert!(json.contains("\"integrator\": \"Lscfrk3w7\""));
    }

    #[test]
    fn manifest_roundtrip() {
        let manifest = EnsembleManifest::new("rt", [16, 16, 16, 32], 5.8);
        let json = manifest.to_json();
        let parsed: EnsembleManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.ensemble_id, "rt");
        assert_eq!(parsed.dims, [16, 16, 16, 32]);
        assert!((parsed.beta - 5.8).abs() < 1e-10);
    }

    #[test]
    fn jackknife_constant_data() {
        let data = vec![1.0; 10];
        let (mean, err) = jackknife_error(&data);
        assert!((mean - 1.0).abs() < 1e-12);
        assert!(err < 1e-12);
    }

    #[test]
    fn jackknife_known_values() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, err) = jackknife_error(&data);
        assert!((mean - 3.0).abs() < 1e-12);
        assert!(err > 0.0);
    }

    #[test]
    fn tau_int_uncorrelated() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let (tau, _) = estimate_tau_int(&data);
        assert!(tau >= 1.0, "tau_int should be >= 1: got {tau}");
    }

    #[test]
    fn implementation_auto_detect() {
        let info = ImplementationInfo::auto_detect();
        assert_eq!(info.code_name, "hotSpring-barracuda");
        assert!(!info.code_version.is_empty());
    }

    #[test]
    fn measurement_with_new_fields() {
        let mut meas = ConfigMeasurement::new("test", 1, "/test");
        meas.implementation = Some(ImplementationInfo::auto_detect());
        meas.scale_setting = Some(ScaleSetting {
            a_fm: Some(0.1),
            method: Some("t0".to_string()),
            reference_value: Some(0.1465),
            reference: Some("BMW 2012".to_string()),
        });
        let json = meas.to_json();
        assert!(json.contains("\"code_name\": \"hotSpring-barracuda\""));
        assert!(json.contains("\"a_fm\": 0.1"));
        assert!(json.contains("\"method\": \"t0\""));
    }
}
