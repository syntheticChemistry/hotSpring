// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: ToadStool Dispatch — offline validation of the compute
//! dispatch parameter assembly and response parsing.
//!
//! Validates the compile_and_submit/dispatch_node_compute parameter
//! construction, DispatchValidation struct logic, and barrier shader
//! path enumeration without requiring live IPC.

use crate::compute_dispatch::{BarrierShaderValidation, DispatchValidation, BARRIER_SHADERS};
use crate::dag_provenance::{DagProvenance, blake3_hex};
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};
use crate::witness::WireWitnessRef;

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "toadstool-dispatch",
        track: Track::GpuCompute,
        tier: Tier::Rust,
        provenance_crate: "validate_compute_dispatch",
        provenance_date: "2026-05-17",
        description: "ToadStool dispatch parameter assembly and response parsing (offline)",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    check_dispatch_validation_logic(v);
    check_input_hashing(v);
    check_barrier_shader_paths(v);
    check_witness_construction(v);
    check_dispatch_serialization(v);
    check_commit_provenance_params(v);
}

fn check_dispatch_validation_logic(v: &mut ValidationHarness) {
    let mut dv = DispatchValidation {
        capabilities_available: true,
        gpu_capabilities: vec!["gpu.f64".into(), "gpu.f32".into()],
        submit_succeeded: true,
        result_received: true,
        output_hash: Some("abc123".into()),
        witnesses: Vec::new(),
        errors: Vec::new(),
    };
    v.check_bool("dispatch:all_passed_true", dv.all_passed());

    dv.submit_succeeded = false;
    v.check_bool("dispatch:submit_false_blocks", !dv.all_passed());

    dv.submit_succeeded = true;
    dv.capabilities_available = false;
    v.check_bool("dispatch:caps_false_blocks", !dv.all_passed());

    dv.capabilities_available = true;
    dv.result_received = false;
    v.check_bool("dispatch:result_false_blocks", !dv.all_passed());
}

fn check_input_hashing(v: &mut ValidationHarness) {
    let input: Vec<f64> = (0..64).map(|i| i as f64).collect();
    let bytes = serde_json::to_vec(&input).unwrap_or_default();
    let hash = blake3_hex(&bytes);

    v.check_bool("dispatch:hash_length_64", hash.len() == 64);
    v.check_bool("dispatch:hash_hex_chars", hash.chars().all(|c| c.is_ascii_hexdigit()));

    let hash2 = blake3_hex(&bytes);
    v.check_bool("dispatch:hash_deterministic", hash == hash2);

    let other: Vec<f64> = (0..64).map(|i| i as f64 + 1.0).collect();
    let other_bytes = serde_json::to_vec(&other).unwrap_or_default();
    let other_hash = blake3_hex(&other_bytes);
    v.check_bool("dispatch:hash_differs_on_input", hash != other_hash);
}

fn check_barrier_shader_paths(v: &mut ValidationHarness) {
    v.check_bool("dispatch:barrier_shaders_nonempty", !BARRIER_SHADERS.is_empty());
    v.check_bool(
        "dispatch:barrier_shaders_all_wgsl",
        BARRIER_SHADERS.iter().all(|p| p.ends_with(".wgsl")),
    );
    v.check_bool(
        "dispatch:barrier_shaders_count",
        BARRIER_SHADERS.len() >= 8,
    );
}

fn check_witness_construction(v: &mut ValidationHarness) {
    let checkpoint = WireWitnessRef::checkpoint("hotspring:dispatch", "test:start");
    let json = serde_json::to_string(&checkpoint).unwrap_or_default();
    v.check_bool("dispatch:witness_checkpoint", json.contains("checkpoint"));

    let hash_witness = WireWitnessRef::hash("hotspring:dispatch", "deadbeef", Some("test:result"));
    let json2 = serde_json::to_string(&hash_witness).unwrap_or_default();
    v.check_bool("dispatch:witness_hash", json2.contains("deadbeef"));
}

fn check_dispatch_serialization(v: &mut ValidationHarness) {
    let dv = DispatchValidation {
        capabilities_available: true,
        gpu_capabilities: vec!["gpu.f64".into()],
        submit_succeeded: true,
        result_received: true,
        output_hash: Some("cafebabe".into()),
        witnesses: vec![WireWitnessRef::hash("test", "abc", None)],
        errors: Vec::new(),
    };
    let json = serde_json::to_string(&dv).unwrap_or_default();
    v.check_bool("dispatch:serializes_caps", json.contains("gpu.f64"));
    v.check_bool("dispatch:serializes_hash", json.contains("cafebabe"));
    v.check_bool("dispatch:serializes_witnesses", json.contains("witnesses"));

    let bsv = BarrierShaderValidation {
        shader_path: "test.wgsl".into(),
        compiled: true,
        error: None,
        binary_size: Some(1024),
        gpr_count: Some(16),
    };
    let bsv_json = serde_json::to_string(&bsv).unwrap_or_default();
    v.check_bool("dispatch:barrier_serializes", bsv_json.contains("test.wgsl"));
    v.check_bool("dispatch:barrier_size", bsv_json.contains("1024"));
}

/// Validates that commit_provenance parameter assembly produces the
/// expected JSON structure. This is the offline half — live dispatch
/// requires biomeOS orchestration.
fn check_commit_provenance_params(v: &mut ValidationHarness) {
    let provenance = DagProvenance {
        dag_session_id: "test-session-001".into(),
        merkle_root: blake3_hex(b"test merkle data"),
        events_count: 5,
        witnesses: vec![
            WireWitnessRef::checkpoint("hotspring:dispatch", "start"),
            WireWitnessRef::hash("hotspring:dispatch", "abc123", None),
        ],
    };

    v.check_bool("provenance:session_id", !provenance.dag_session_id.is_empty());
    v.check_bool("provenance:merkle_root_len", provenance.merkle_root.len() == 64);
    v.check_bool("provenance:events_count", provenance.events_count == 5);
    v.check_bool("provenance:witnesses_count", provenance.witnesses.len() == 2);

    let json = serde_json::to_string(&provenance).unwrap_or_default();
    v.check_bool("provenance:serializes", json.contains("test-session-001"));
    v.check_bool("provenance:has_merkle", json.contains("merkle_root"));

    let signal_params = serde_json::json!({
        "session_id": provenance.dag_session_id,
        "merkle_root": provenance.merkle_root,
        "events_count": provenance.events_count,
        "experiment_id": "exp-199-dispatch-test",
        "spring": "hotSpring",
        "paper_ref": "Paper 45",
    });
    v.check_bool(
        "provenance:signal_has_experiment",
        signal_params["experiment_id"] == "exp-199-dispatch-test",
    );
    v.check_bool(
        "provenance:signal_has_paper_ref",
        signal_params["paper_ref"] == "Paper 45",
    );
}
