// SPDX-License-Identifier: AGPL-3.0-only
#![allow(clippy::unwrap_used)]

//! Integration tests: physics proxy pipeline (Anderson 3D, Z(3) Potts) public API.
//!
//! Validates proxy module end-to-end: Anderson and Potts proxies return
//! valid features and are deterministic.

use hotspring_barracuda::proxy::{anderson_3d_proxy, potts_z3_proxy, CortexRequest};

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
fn proxy_potts_z3_returns_features() {
    let req = CortexRequest {
        beta: 5.0,
        mass: 0.1,
        lattice: 6,
        plaq_var: 0.03,
    };
    let f = potts_z3_proxy(&req, 123);
    assert!((f.beta - 5.0).abs() < f64::EPSILON);
    assert!(f.lambda_min.is_finite());
    assert!(f.wall_ms >= 0.0);
    assert!(f.tier == 1);
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
fn proxy_potts_deterministic() {
    let req = CortexRequest {
        beta: 5.0,
        mass: 0.1,
        lattice: 4,
        plaq_var: 0.03,
    };
    let a = potts_z3_proxy(&req, 777);
    let b = potts_z3_proxy(&req, 777);
    assert_eq!(
        a.level_spacing_ratio.to_bits(),
        b.level_spacing_ratio.to_bits()
    );
}
