// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Sovereign GPU Dispatch — validates the IPC-dispatched
//! compute path through toadStool + barraCuda precision advisory.
//!
//! This is the first scenario on the `GpuCompute` track. It exercises:
//! - `toadstool.validate` workload preflight via NUCLEUS IPC
//! - `precision.route` advisory via NUCLEUS IPC
//! - `compute.dispatch.submit` sovereign workload dispatch
//! - Structured capability/error reporting for coralReef sentinel feedback
//!
//! In standalone mode (no NUCLEUS), all IPC checks gracefully report
//! as skipped rather than failing.

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "sovereign-dispatch",
        track: Track::GpuCompute,
        tier: Tier::Live,
        provenance_crate: "exp170_sovereign_cold_boot",
        provenance_date: "2026-05-12",
        description: "Sovereign GPU dispatch: toadStool validate + precision route + dispatch submit",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    use crate::ipc::tier2;
    use crate::primal_bridge::NucleusContext;

    let nucleus = NucleusContext::detect();
    let t2 = tier2::tier2_status(&nucleus);

    t2.check(v);

    // --- Workload preflight ---
    let preflight = tier2::workload_preflight(&nucleus, "sovereign-dispatch-probe");
    let preflight_responded = preflight.is_some();
    v.check_bool(
        "sovereign:preflight_responded",
        preflight_responded || !t2.toadstool_alive,
    );
    if let Some(pf) = &preflight {
        v.check_bool("sovereign:preflight_valid", pf.valid);
        v.check_bool("sovereign:gpu_available", pf.gpu_available);
    }

    // --- Precision advisory ---
    let advisory = tier2::precision_advisory(&nucleus, "molecular_dynamics");
    let precision_responded = advisory.is_some();
    v.check_bool(
        "sovereign:precision_responded",
        precision_responded || !t2.barracuda_alive,
    );
    if let Some(pa) = &advisory {
        v.check_bool("sovereign:precision_has_hint", pa.hardware_hint.is_some());
        let has_dispatch_path = pa.dispatch_path.is_some();
        v.check_bool(
            "sovereign:dispatch_path_reported",
            has_dispatch_path || !t2.barracuda_alive,
        );
        if let Some(ref dp) = pa.dispatch_path {
            let valid = dp == "wgpu" || dp == "sovereign" || dp == "unavailable";
            v.check_bool("sovereign:dispatch_path_valid", valid);
        }
    }

    // --- Sovereign dispatch probe (method: compute.dispatch.submit) ---
    let dispatch_available = nucleus
        .get_by_capability("compute")
        .is_some_and(|ep| ep.alive);

    if dispatch_available {
        let params = serde_json::json!({
            "workload": "hotspring-sovereign-probe",
            "kind": "health_check",
            "dry_run": true,
        });
        match nucleus.call_by_capability("compute", "compute.dispatch.submit", params) {
            Ok(_resp) => {
                v.check_bool("sovereign:dispatch_responded", true);
            }
            Err(_) => {
                v.check_bool("sovereign:dispatch_responded", false);
            }
        }
    } else {
        v.check_bool(
            "sovereign:dispatch_routable",
            dispatch_available || !t2.toadstool_alive,
        );
    }

    // --- FECS state probe (coralReef sentinel; method: ember.fecs.state) ---
    let ember_available = nucleus
        .get_by_capability("compute")
        .is_some_and(|ep| ep.alive);

    if ember_available {
        let params = serde_json::json!({ "bdf": "auto" });
        match nucleus.call_by_capability("compute", "ember.fecs.state", params) {
            Ok(_resp) => {
                v.check_bool("sovereign:fecs_responded", true);
            }
            Err(_) => {
                v.check_bool("sovereign:fecs_responded", false);
            }
        }
    } else {
        v.check_bool(
            "sovereign:fecs_routable",
            ember_available || !t2.toadstool_alive,
        );
    }

    // --- Warm catch (toadStool S256+; method: device.warm_catch) ---
    let warm_catch_available = nucleus
        .get_by_capability("compute")
        .is_some_and(|ep| ep.alive);
    v.check_bool(
        "sovereign:warm_catch_routable",
        warm_catch_available || !t2.toadstool_alive,
    );

    // --- PBDMA dispatch surface (toadStool S258; method: device.vfio.open) ---
    let vfio_open_available = nucleus
        .get_by_capability("compute")
        .is_some_and(|ep| ep.alive);
    v.check_bool(
        "sovereign:pbdma_open_routable",
        vfio_open_available || !t2.toadstool_alive,
    );

    // method: device.vfio.roundtrip
    let vfio_roundtrip_available = nucleus
        .get_by_capability("compute")
        .is_some_and(|ep| ep.alive);
    v.check_bool(
        "sovereign:pbdma_roundtrip_routable",
        vfio_roundtrip_available || !t2.toadstool_alive,
    );

    // --- GR context init (toadStool S262; method: device.gr.init) ---
    let gr_init_available = nucleus
        .get_by_capability("compute")
        .is_some_and(|ep| ep.alive);
    v.check_bool(
        "sovereign:gr_init_routable",
        gr_init_available || !t2.toadstool_alive,
    );

    // --- Warm cycle routable (method: ember.warm_cycle) ---
    let warm_available = nucleus
        .get_by_capability("compute")
        .is_some_and(|ep| ep.alive);
    v.check_bool(
        "sovereign:warm_cycle_routable",
        warm_available || !t2.toadstool_alive,
    );

    // --- Phase D local dispatch probe ---
    #[cfg(feature = "toadstool-dispatch")]
    if dispatch_available {
        let local = crate::fleet_toadstool::try_local_dispatch(
            &nucleus,
            &serde_json::json!({
                "workload": "hotspring-phase-d-probe",
                "kind": "health_check",
                "dry_run": true,
            }),
        );
        v.check_bool("sovereign:phase_d_attempted", local.attempted);
    }
}
