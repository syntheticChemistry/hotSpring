// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fleet-mode integration tests (mock JSON on disk; no live ember processes).

#![allow(clippy::unwrap_used)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use hotspring_barracuda::fleet_client::{
    DOMAIN_LATTICE_QCD, EmberClient, FleetDiscovery, FleetEmberHub, FleetFile, FleetRouter,
    ResilientRoute,
};

fn unique_tmp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("{prefix}_{nanos}"));
    fs::create_dir_all(&p).unwrap();
    p
}

fn write_fleet(path: &Path, json: &str) {
    fs::write(path, json).unwrap();
}

#[test]
fn fleet_discovery_parses_mock_file_and_enumerates_devices() {
    let dir = unique_tmp_dir("hotspring_fleet_disc");
    let fleet_path = dir.join("coral-ember-fleet.json");
    write_fleet(
        &fleet_path,
        r#"{
            "mode": "fleet",
            "routes": {
                "0000:03:00.0": "/run/a.sock",
                "0000:04:00.0": "/run/b.sock"
            },
            "devices": [
                { "bdf": "0000:03:00.0", "vendor": "NVIDIA" },
                { "bdf": "0000:04:00.0", "vendor": "NVIDIA" }
            ]
        }"#,
    );

    let disc = FleetDiscovery::load(&fleet_path).unwrap();
    assert_eq!(disc.path(), fleet_path);
    let file = disc.file();
    assert_eq!(file.mode.as_deref(), Some("fleet"));
    assert_eq!(file.routes.len(), 2);
    let router = FleetRouter::from_fleet_file(file);
    assert_eq!(router.devices().len(), 2);
    let bdfs: Vec<&str> = router.devices().iter().map(|d| d.bdf.as_str()).collect();
    assert!(bdfs.contains(&"0000:03:00.0"));
    assert!(bdfs.contains(&"0000:04:00.0"));
}

#[test]
fn fleet_standby_adoption_selects_clean_hot_standby_when_primary_needs_warm_cycle() {
    let f: FleetFile = serde_json::from_str(
        r#"{
            "mode": "fleet",
            "routes": {
                "0000:03:00.0": "/tmp/primary.sock",
                "0000:04:00.0": "/tmp/standby.sock"
            },
            "devices": [
                {
                    "bdf": "0000:03:00.0",
                    "physics_domains": ["lattice_qcd"],
                    "needs_warm_cycle": true
                },
                {
                    "bdf": "0000:04:00.0",
                    "hot_standby_of": "0000:03:00.0",
                    "physics_domains": ["lattice_qcd"]
                }
            ]
        }"#,
    )
    .unwrap();
    let router = FleetRouter::from_fleet_file(&f);
    let res = router.route_resilient(DOMAIN_LATTICE_QCD);
    match res {
        ResilientRoute::Routed {
            device,
            adopted_from_faulted_primary,
        } => {
            assert_eq!(device.bdf, "0000:04:00.0");
            assert!(adopted_from_faulted_primary);
        }
        _ => panic!("expected standby adoption"),
    }
}

#[test]
fn fleet_fault_informed_skips_dirty_device_or_requests_warm() {
    let f_dirty: FleetFile = serde_json::from_str(
        r#"{
            "mode": "fleet",
            "routes": { "0000:01:00.0": "/tmp/x.sock" },
            "devices": [{
                "bdf": "0000:01:00.0",
                "experiment_dirty": true,
                "needs_warm_cycle": true,
                "physics_domains": ["lattice_qcd"]
            }]
        }"#,
    )
    .unwrap();
    let router = FleetRouter::from_fleet_file(&f_dirty);
    assert!(matches!(
        router.route_resilient(DOMAIN_LATTICE_QCD),
        ResilientRoute::NoEligibleDevice
    ));

    let f_warm: FleetFile = serde_json::from_str(
        r#"{
            "mode": "fleet",
            "routes": { "0000:02:00.0": "/tmp/y.sock" },
            "devices": [{
                "bdf": "0000:02:00.0",
                "needs_warm_cycle": true,
                "physics_domains": ["lattice_qcd"]
            }]
        }"#,
    )
    .unwrap();
    let router = FleetRouter::from_fleet_file(&f_warm);
    match router.route_resilient(DOMAIN_LATTICE_QCD) {
        ResilientRoute::WarmCycleRequired { device } => {
            assert_eq!(device.bdf, "0000:02:00.0");
        }
        other => panic!("expected warm cycle requirement, got {other:?}"),
    }
}

#[test]
fn fleet_per_device_isolation_distinct_ember_clients_per_socket() {
    let f = FleetFile {
        mode: Some("fleet".into()),
        routes: HashMap::from([
            (String::from("0000:01:00.0"), String::from("/tmp/iso-a.sock")),
            (String::from("0000:02:00.0"), String::from("/tmp/iso-b.sock")),
        ]),
        standby_count: None,
        devices: vec![],
    };
    let router = FleetRouter::from_fleet_file(&f);
    let mut hub = FleetEmberHub::default();
    let d0 = router.route_by_bdf("0000:01:00.0").unwrap();
    let d1 = router.route_by_bdf("0000:02:00.0").unwrap();
    let p0 = hub.client_for_route(d0).socket_path().to_path_buf();
    let p1 = hub.client_for_route(d1).socket_path().to_path_buf();
    assert_ne!(p0, p1);
    assert_eq!(hub.len(), 2);
    let warm0 = EmberClient::warm_cycle_request(&d0.bdf);
    let warm1 = EmberClient::warm_cycle_request(&d1.bdf);
    assert_ne!(warm0["params"]["bdf"], warm1["params"]["bdf"]);
}

#[test]
fn fleet_file_missing_graceful_degradation() {
    let dir = unique_tmp_dir("hotspring_fleet_missing");
    let absent = dir.join("no-fleet-here.json");
    assert!(FleetDiscovery::load_if_present(&absent).unwrap().is_none());
}
