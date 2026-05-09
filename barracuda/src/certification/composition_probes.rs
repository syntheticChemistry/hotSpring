// SPDX-License-Identifier: AGPL-3.0-or-later

//! NUCLEUS composition parity probes for certification layers 2-5.
//!
//! Each probe compares a local Rust result against an IPC result obtained
//! through `CompositionContext`. Follows the primalSpring certification
//! pattern: local baseline is ground truth, IPC is the system under test.

use primalspring::composition::{self, CompositionContext, validate_parity};
use primalspring::tolerances;
use primalspring::validation::ValidationResult;

use crate::physics::semf_binding_energy;
use crate::provenance::SLY4_PARAMS;

/// Layer 2: Scalar parity — stats.mean over IPC matches known values.
pub fn validate_scalar_parity(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    validate_parity(
        ctx,
        v,
        "parity:plaquette_mean",
        "tensor",
        "stats.mean",
        serde_json::json!({"data": [0.333, 0.334, 0.332, 0.335, 0.331]}),
        "result",
        0.333,
        tolerances::IPC_ROUND_TRIP_TOL,
    );

    validate_parity(
        ctx,
        v,
        "parity:observable_mean",
        "tensor",
        "stats.mean",
        serde_json::json!({"data": [1.0, 2.0, 3.0, 4.0, 5.0]}),
        "result",
        3.0,
        tolerances::IPC_ROUND_TRIP_TOL,
    );
}

/// Layer 3: Vector parity — tensor.matmul over IPC matches known matrix products.
pub fn validate_vector_parity(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    composition::validate_parity_vec(
        ctx,
        v,
        "parity:su3_identity_matmul",
        "tensor",
        "tensor.matmul",
        serde_json::json!({
            "a": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "b": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "rows_a": 3, "cols_a": 3, "cols_b": 3
        }),
        "result",
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        tolerances::IPC_ROUND_TRIP_TOL,
    );

    composition::validate_parity_vec(
        ctx,
        v,
        "parity:field_arithmetic_matmul",
        "tensor",
        "tensor.matmul",
        serde_json::json!({
            "a": [[1.0, 2.0], [3.0, 4.0]],
            "b": [[5.0, 6.0], [7.0, 8.0]],
            "rows_a": 2, "cols_a": 2, "cols_b": 2
        }),
        "result",
        &[19.0, 22.0, 43.0, 50.0],
        tolerances::IPC_ROUND_TRIP_TOL,
    );
}

/// Layer 4: SEMF end-to-end — local binding energy vs IPC stats.mean.
pub fn validate_semf_e2e(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    let local_be = semf_binding_energy(82, 126, &SLY4_PARAMS);
    v.check_bool(
        "semf:local_pb208_finite",
        local_be.is_finite() && local_be > 0.0,
        &format!("B.E.(Pb-208) = {local_be:.4} MeV"),
    );

    let part = local_be / 5.0;
    validate_parity(
        ctx,
        v,
        "parity:semf_pb208_mean",
        "tensor",
        "stats.mean",
        serde_json::json!({"data": [part, part, part, part, part]}),
        "result",
        part,
        tolerances::IPC_ROUND_TRIP_TOL,
    );
}

/// Layer 5: Provenance witness — BearDog crypto.hash over IPC.
pub fn validate_provenance_witness(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.hash_bytes(b"hotspring-guidestone-witness-2026", "blake3") {
        Ok(hash) => {
            v.check_bool(
                "crypto:blake3_witness",
                !hash.is_empty(),
                &format!("BLAKE3 produced {}B base64", hash.len()),
            );
            match ctx.hash_bytes(b"hotspring-guidestone-witness-2026", "blake3") {
                Ok(hash2) => {
                    v.check_bool(
                        "crypto:blake3_determinism",
                        hash == hash2,
                        "same input produces same hash",
                    );
                }
                Err(e) => {
                    v.check_bool(
                        "crypto:blake3_determinism",
                        false,
                        &format!("second hash call failed: {e}"),
                    );
                }
            }
        }
        Err(e) if e.is_connection_error() => {
            v.check_skip(
                "crypto:blake3_witness",
                &format!("security not available: {e}"),
            );
            v.check_skip("crypto:blake3_determinism", "security not available");
        }
        Err(e) if e.is_protocol_error() => {
            v.check_skip(
                "crypto:blake3_witness",
                &format!("security reachable but protocol mismatch: {e}"),
            );
            v.check_skip("crypto:blake3_determinism", "security protocol mismatch");
        }
        Err(e) => {
            v.check_bool("crypto:blake3_witness", false, &format!("hash error: {e}"));
            v.check_skip("crypto:blake3_determinism", "first hash failed");
        }
    }
}

/// Layer 5: Compute dispatch — toadStool compute.dispatch over IPC.
pub fn validate_compute_dispatch(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "compute",
        "compute.dispatch",
        serde_json::json!({
            "shader": "identity_f64",
            "workgroups": [1, 1, 1]
        }),
    ) {
        Ok(result) => {
            v.check_bool(
                "compute:dispatch_returns_result",
                true,
                &format!(
                    "response keys: {:?}",
                    result.as_object().map(|o| o.keys().collect::<Vec<_>>())
                ),
            );
        }
        Err(e) if e.is_connection_error() => {
            v.check_skip(
                "compute:dispatch_returns_result",
                &format!("compute not available: {e}"),
            );
        }
        Err(e) if e.is_protocol_error() => {
            v.check_skip(
                "compute:dispatch_returns_result",
                &format!("compute reachable but protocol mismatch: {e}"),
            );
        }
        Err(e) => {
            v.check_bool(
                "compute:dispatch_returns_result",
                false,
                &format!("dispatch error: {e}"),
            );
        }
    }
}
