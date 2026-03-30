// SPDX-License-Identifier: AGPL-3.0-only

//! Integration tests: physics proxy pipeline (Anderson 3D, Z(3) Potts) public API.
//!
//! Validates proxy module end-to-end: Anderson and Potts proxies return
//! valid features and are deterministic. Uses `combined_proxy` for Potts
//! tests since it returns the full `ProxyFeatures` struct.

use hotspring_barracuda::proxy::{CortexRequest, anderson_3d_proxy, combined_proxy};

#[test]
fn proxy_anderson_3d_returns_features() {
    let req = CortexRequest {
        beta: 5.5,
        mass: 0.1,
        lattice: 6,
        plaq_var: 0.05,
    };
    let f = anderson_3d_proxy(&req, 42);
    assert!((f.beta - 5.5).abs() < f64::EPSILON);
    assert!(f.level_spacing_ratio.is_finite());
    assert!(f.lambda_min.is_finite());
    assert!(f.wall_ms >= 0.0);
    assert!(f.tier == 1);
}

#[test]
fn proxy_combined_returns_potts_features() {
    let req = CortexRequest {
        beta: 5.0,
        mass: 0.1,
        lattice: 6,
        plaq_var: 0.03,
    };
    let f = combined_proxy(&req, 123);
    assert!((f.beta - 5.0).abs() < f64::EPSILON);
    assert!(f.lambda_min.is_finite());
    assert!(f.wall_ms >= 0.0);
    assert!(f.tier == 1);
    assert!(f.potts_magnetization.is_finite());
    assert!(f.potts_susceptibility.is_finite());
    assert!(
        f.potts_phase == "ordered"
            || f.potts_phase == "disordered"
            || f.potts_phase == "transition"
    );
}

#[test]
fn proxy_anderson_deterministic() {
    let req = CortexRequest {
        beta: 5.5,
        mass: 0.1,
        lattice: 4,
        plaq_var: 0.05,
    };
    let a = anderson_3d_proxy(&req, 999);
    let b = anderson_3d_proxy(&req, 999);
    assert_eq!(
        a.level_spacing_ratio.to_bits(),
        b.level_spacing_ratio.to_bits()
    );
}

#[test]
fn proxy_combined_deterministic() {
    let req = CortexRequest {
        beta: 5.0,
        mass: 0.1,
        lattice: 4,
        plaq_var: 0.03,
    };
    let a = combined_proxy(&req, 777);
    let b = combined_proxy(&req, 777);
    assert_eq!(
        a.level_spacing_ratio.to_bits(),
        b.level_spacing_ratio.to_bits()
    );
    assert_eq!(
        a.potts_magnetization.to_bits(),
        b.potts_magnetization.to_bits()
    );
    assert_eq!(a.potts_phase, b.potts_phase);
}
