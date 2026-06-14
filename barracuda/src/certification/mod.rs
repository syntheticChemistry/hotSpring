// SPDX-License-Identifier: AGPL-3.0-or-later

//! Composition certification engine — absorbed guideStone organelle.
//!
//! Proves hotSpring NUCLEUS composition correctness through layered validation,
//! following the primalSpring L0–L5 certification model adapted for physics
//! domain validation.
//!
//! | Layer | Name | Description |
//! |-------|------|-------------|
//! | 0     | Bare Properties | Deterministic, traceable, self-verifying, agnostic, tolerance-documented |
//! | 1     | Discovery | NUCLEUS primals reachable via CompositionContext |
//! | 2     | Scalar Parity | stats.mean IPC matches Rust baseline |
//! | 3     | Vector Parity | tensor.matmul IPC matches Rust baseline |
//! | 4     | Domain Science | SEMF end-to-end via IPC |
//! | 5     | Full Composition | Crypto witness + compute dispatch |
//! | 6     | NUCLEUS Deployment | Deploy graph validation, biomeOS status, method registration |
//!
//! Originally the `hotspring_guidestone` binary. Endosymbiosed into the
//! library at the interstadial transition (May 2026).

pub mod bare;
pub mod composition_probes;
pub mod deployment;

use primalspring::composition::{CompositionContext, validate_liveness};
use primalspring::validation::ValidationResult;

/// Maximum certification layer (inclusive).
pub const MAX_LAYER: u8 = 6;

/// Run the full certification engine up to the specified layer.
///
/// Returns the `ValidationResult` after all layers complete. Callers
/// can inspect `exit_code()` for pass/fail/bare-only status.
///
/// # Exit semantics
///
/// - `0` — all layers passed (certified)
/// - `1` — one or more layers failed
/// - `2` — bare-only mode (no primals discovered, structural checks only)
#[must_use]
pub fn certify(max_layer: u8) -> ValidationResult {
    let mut v = ValidationResult::new("hotSpring Certification — Physics Composition");

    ValidationResult::print_banner("hotSpring Certification — Physics Composition");

    v.section("Layer 0: Bare Properties");
    bare::validate_bare_properties(&mut v);

    if max_layer == 0 {
        v.finish();
        return v;
    }

    v.section("Layer 1: Discovery + Liveness");
    let mut ctx = CompositionContext::from_live_discovery_with_fallback();
    let alive = validate_liveness(&mut ctx, &mut v, &["math", "crypto", "compute"]);

    if alive == 0 {
        log::warn!("[certify] No NUCLEUS primals discovered — bare certification only.");
        log::warn!("  Deploy from plasmidBin and rerun for full certification.");
        v.finish();
        return v;
    }

    if max_layer < 2 {
        v.finish();
        return v;
    }

    v.section("Layer 2: Scalar Parity");
    composition_probes::validate_scalar_parity(&mut ctx, &mut v);

    if max_layer < 3 {
        v.finish();
        return v;
    }

    v.section("Layer 3: Vector Parity");
    composition_probes::validate_vector_parity(&mut ctx, &mut v);

    if max_layer < 4 {
        v.finish();
        return v;
    }

    v.section("Layer 4: Domain Science — SEMF End-to-End");
    composition_probes::validate_semf_e2e(&mut ctx, &mut v);

    if max_layer < 5 {
        v.finish();
        return v;
    }

    v.section("Layer 5: Full Composition — Crypto + Compute");
    composition_probes::validate_provenance_witness(&mut ctx, &mut v);
    composition_probes::validate_compute_dispatch(&mut ctx, &mut v);

    if max_layer < 6 {
        v.finish();
        return v;
    }

    v.section("Layer 6: NUCLEUS Deployment Validation");
    deployment::validate_deployment(&mut v);

    v.finish();
    v
}
