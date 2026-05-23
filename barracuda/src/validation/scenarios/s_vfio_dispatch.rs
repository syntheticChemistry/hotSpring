// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: VFIO Sovereign Dispatch — validates VFIO-bound GPU detection,
//! warm FECS state, and toadStool IPC dispatch on sovereign hardware.
//!
//! ## Evolution (May 13, 2026)
//!
//! toadStool S258 wired PBDMA dispatch through `NvVfioComputeDevice`:
//! GPFIFO submission, DMA alloc/upload/readback, doorbell + sync.
//! `probe_warm_fecs()` (S256) probes BAR0 for warm-preserved FECS state.
//! The full pipeline: warm probe → `open_vfio()` → alloc/upload → dispatch → sync → readback.
//!
//! Exercises:
//! - VFIO GPU detection via sysfs probing
//! - VFIO-pci driver binding verification
//! - toadStool FECS state probe via `ember.fecs.state` IPC
//! - toadStool warm catch via `device.warm_catch` IPC
//! - Phase D dispatch probe via `compute.dispatch.submit` with `local_dispatch` flag
//!
//! Target hardware (biomeGate compute trio):
//!   - Titan V (GV100, SM70) at BDF 02:00.0
//!   - Tesla K80 (GK210, SM37) at BDF 4b:00.0
//!   - RTX 5060 (GB206, SM120) at env `HOTSPRING_RTX5060_BDF` (optional)

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "vfio-dispatch",
        track: Track::GpuCompute,
        tier: Tier::Live,
        provenance_crate: "embedded_vfio_dispatch_scenario",
        provenance_date: "2026-05-13",
        description: "VFIO sovereign dispatch: sysfs detection + FECS warm probe + toadStool Phase D dispatch",
    },
    run,
};

#[derive(Clone)]
struct VfioTarget {
    name: String,
    bdf: String,
    sm: u32,
}

fn discover_vfio_targets() -> Vec<VfioTarget> {
    let mut targets = Vec::new();

    let titan_bdf =
        std::env::var("HOTSPRING_TITAN_V_BDF").unwrap_or_else(|_| "0000:02:00.0".to_string());
    targets.push(VfioTarget {
        name: "titan-v".into(),
        bdf: titan_bdf,
        sm: 70,
    });

    let k80_bdf = std::env::var("HOTSPRING_K80_BDF").map_or_else(
        |_| "0000:4b:00.0".to_string(),
        |b| {
            if b.contains(':') && b.len() > 7 {
                b
            } else {
                format!("0000:{b}")
            }
        },
    );
    targets.push(VfioTarget {
        name: "k80-die0".into(),
        bdf: k80_bdf,
        sm: 37,
    });

    if let Ok(bdf_5060) = std::env::var("HOTSPRING_RTX5060_BDF") {
        let bdf = if bdf_5060.contains(':') && bdf_5060.len() > 7 {
            bdf_5060
        } else {
            format!("0000:{bdf_5060}")
        };
        targets.push(VfioTarget {
            name: "rtx-5060".into(),
            bdf,
            sm: 120,
        });
    }

    targets
}

pub fn run(v: &mut ValidationHarness) {
    use crate::primal_bridge::NucleusContext;

    let vfio_driver = std::path::Path::new("/sys/bus/pci/drivers/vfio-pci");
    let vfio_loaded = vfio_driver.exists();
    v.check_bool("vfio:driver_present", vfio_loaded);
    if !vfio_loaded {
        return;
    }

    let nucleus = NucleusContext::detect();
    let toadstool_alive = nucleus.by_domain("compute").is_some_and(|ep| ep.alive);

    let targets = discover_vfio_targets();
    for target in &targets {
        let prefix = format!("vfio:{}:", target.name);

        let device_sysfs = format!("/sys/bus/pci/devices/{}", target.bdf);
        let sysfs_present = std::path::Path::new(&device_sysfs).exists();
        v.check_bool(&format!("{prefix}sysfs_present"), sysfs_present);
        if !sysfs_present {
            continue;
        }

        let driver_link = format!("{device_sysfs}/driver");
        let bound_to_vfio = std::fs::read_link(&driver_link)
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .is_some_and(|name| name == "vfio-pci");
        v.check_bool(&format!("{prefix}bound_to_vfio"), bound_to_vfio);
        if !bound_to_vfio {
            continue;
        }

        // --- FECS state probe via toadStool IPC ---
        if toadstool_alive {
            let fecs_params = serde_json::json!({ "bdf": target.bdf });
            match nucleus.call_by_capability("compute", "ember.fecs.state", fecs_params) {
                Ok(resp) => {
                    let fecs_ready = resp
                        .get("fecs_ready")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false);
                    v.check_bool(&format!("{prefix}fecs_state_responded"), true);
                    v.check_bool(&format!("{prefix}fecs_ready"), fecs_ready);
                }
                Err(_) => {
                    v.check_bool(&format!("{prefix}fecs_state_responded"), false);
                }
            }

            // --- Warm catch probe ---
            let warm_params = serde_json::json!({ "bdf": target.bdf, "expected_sm": target.sm });
            let warm_fecs = if let Ok(resp) =
                nucleus.call_by_capability("compute", "device.warm_catch", warm_params)
            {
                let warm_ready = resp
                    .get("fecs_ready")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let vfio_open = resp
                    .get("vfio_open")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let channel_id = resp
                    .get("channel_id")
                    .and_then(serde_json::Value::as_u64)
                    .map(|c| c as u32);
                v.check_bool(&format!("{prefix}warm_catch_responded"), true);
                v.check_bool(&format!("{prefix}warm_catch_fecs_ready"), warm_ready);
                v.check_bool(&format!("{prefix}warm_catch_vfio_open"), vfio_open);
                if let Some(ch) = channel_id {
                    log::info!("[{prefix}] PBDMA channel ID: {ch}");
                }
                warm_ready
            } else {
                v.check_bool(&format!("{prefix}warm_catch_responded"), false);
                false
            };

            // --- S258 PBDMA DMA roundtrip probe ---
            if warm_fecs {
                let open_params = serde_json::json!({ "bdf": target.bdf });
                let vfio_opened = nucleus
                    .call_by_capability("compute", "device.vfio.open", open_params)
                    .is_ok();
                v.check_bool(&format!("{prefix}pbdma_vfio_open"), vfio_opened);

                // --- S262 GR context init probe ---
                if vfio_opened {
                    let gr_init_routable = nucleus
                        .get_by_capability("device.gr.init")
                        .is_some_and(|ep| ep.alive);
                    v.check_bool(&format!("{prefix}gr_init_routable"), gr_init_routable);
                }

                if vfio_opened {
                    let roundtrip_params = serde_json::json!({
                        "bdf": target.bdf,
                        "data_b64": crate::base64_encode::encode(b"hotspring-pbdma-probe"),
                    });
                    match nucleus.call_by_capability(
                        "compute",
                        "device.vfio.roundtrip",
                        roundtrip_params,
                    ) {
                        Ok(resp) => {
                            let ok = resp
                                .get("ok")
                                .and_then(serde_json::Value::as_bool)
                                .unwrap_or(false);
                            v.check_bool(&format!("{prefix}pbdma_dma_roundtrip"), ok);
                        }
                        Err(e) => {
                            log::warn!("[{prefix}] DMA roundtrip failed: {e}");
                            v.check_bool(&format!("{prefix}pbdma_dma_roundtrip"), false);
                        }
                    }
                }
            }

            // --- Phase D local dispatch probe ---
            #[cfg(feature = "toadstool-dispatch")]
            {
                let dispatch_params = serde_json::json!({
                    "workload": format!("hotspring-vfio-probe-{}", target.name),
                    "bdf": target.bdf,
                    "kind": "health_check",
                    "dry_run": true,
                });
                let local = crate::fleet_toadstool::try_local_dispatch(&nucleus, &dispatch_params);
                v.check_bool(&format!("{prefix}phase_d_attempted"), local.attempted);
            }

            #[cfg(not(feature = "sovereign-dispatch"))]
            {
                v.check_bool(&format!("{prefix}sovereign_dispatch_feature"), false);
            }
        } else {
            v.check_bool(&format!("{prefix}toadstool_available"), false);
        }
    }
}
