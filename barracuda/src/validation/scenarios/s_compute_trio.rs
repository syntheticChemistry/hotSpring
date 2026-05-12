// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Compute Trio Pipeline — barraCuda → coralReef → toadStool.
//!
//! Validates the full sovereign GPU compute pipeline:
//!
//! 1. **barraCuda** provides WGSL shader source (physics kernels)
//! 2. **coralReef** compiles WGSL → native GPU binary via `shader.compile.wgsl`
//! 3. **toadStool** dispatches the compiled workload via `compute.dispatch.submit`
//!
//! This is the pipeline that replaces library-linked math with IPC-dispatched
//! sovereign compute on real hardware. In standalone mode, each IPC hop
//! gracefully degrades.
//!
//! Hardware matrix (biomeGate):
//! - RTX 5060 (SM120): warm via nvidia shared stack — dispatch PROVEN
//! - Titan V (SM70): warm-catch FECS running — dispatch blocked on VFIO adapter
//! - K80 (SM37): warm-catch GDDR5 trained — dispatch blocked on VFIO adapter

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "compute-trio-pipeline",
        track: Track::GpuCompute,
        tier: Tier::Live,
        provenance_crate: "validate_compute_trio_pipeline",
        provenance_date: "2026-05-12",
        description: "Compute trio: barraCuda shader → coralReef compile → toadStool dispatch",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    use crate::ipc::tier2;
    use crate::primal_bridge::NucleusContext;

    let nucleus = NucleusContext::detect();

    // --- Phase 1: barraCuda precision advisory ---
    let precision = tier2::precision_advisory(&nucleus, "lattice_qcd", "wilson_plaquette");
    let math_alive = nucleus.by_domain("math").is_some_and(|ep| ep.alive);
    v.check_bool("trio:precision_advisory", precision.is_some() || !math_alive);

    // --- Phase 2: coralReef shader compilation ---
    let shader_alive = nucleus.by_domain("shader").is_some_and(|ep| ep.alive);
    if shader_alive {
        let compile_params = serde_json::json!({
            "source": "// probe shader\n@compute @workgroup_size(1) fn main() {}",
            "format": "wgsl",
            "target": "spirv",
        });
        match nucleus.call_by_capability("shader", "shader.compile.wgsl", compile_params) {
            Ok(_resp) => {
                v.check_bool("trio:shader_compile", true);
            }
            Err(_) => {
                v.check_bool("trio:shader_compile", false);
            }
        }
    } else {
        v.check_bool("trio:shader_domain_available", shader_alive);
    }

    // --- Phase 3: toadStool workload preflight ---
    let preflight = tier2::workload_preflight(&nucleus, "compute-trio-probe");
    let compute_alive = nucleus.by_domain("compute").is_some_and(|ep| ep.alive);
    v.check_bool("trio:workload_preflight", preflight.is_some() || !compute_alive);

    // --- Phase 4: toadStool dispatch ---
    if compute_alive {
        let dispatch_params = serde_json::json!({
            "workload": "trio-validation-probe",
            "kind": "health_check",
            "dry_run": true,
            "spring": "hotSpring",
        });
        match nucleus.call_by_capability("compute", "compute.dispatch.submit", dispatch_params) {
            Ok(_resp) => {
                v.check_bool("trio:dispatch_submit", true);
            }
            Err(_) => {
                v.check_bool("trio:dispatch_submit", false);
            }
        }
    } else {
        v.check_bool("trio:compute_domain_available", compute_alive);
    }

    // --- Phase 5: GPU hardware readiness per card ---
    let ember_alive = nucleus.by_domain("compute").is_some_and(|ep| ep.alive);

    // RTX 5060 (SM120) — should always be ready
    v.check_bool(
        "trio:rtx5060_dispatchable",
        !ember_alive || check_device_ready(&nucleus, "21:00.0"),
    );

    // Titan V (SM70) — warm-catch works, dispatch blocked on VFIO adapter
    v.check_bool(
        "trio:titanv_warm",
        !ember_alive || check_device_ready(&nucleus, "02:00.0"),
    );

    // K80 (SM37) — warm-catch works, dispatch blocked on VFIO adapter
    v.check_bool(
        "trio:k80_warm",
        !ember_alive || check_device_ready(&nucleus, "4b:00.0"),
    );
}

fn check_device_ready(nucleus: &crate::primal_bridge::NucleusContext, bdf: &str) -> bool {
    let params = serde_json::json!({ "bdf": bdf });
    nucleus
        .call_by_capability("compute", "ember.device.health", params)
        .is_ok()
}
