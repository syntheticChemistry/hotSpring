// SPDX-License-Identifier: AGPL-3.0-or-later

//! Gossip injection for hotSpring science pipeline.
//!
//! Fires events into the swarmVine epidemic mesh via local UDS.
//! Pattern follows rhizoCrypt/loamSpine/lithoSpore injection (conditional, zero-cost when absent).
//!
//! Events are fire-and-forget: if swarmVine is not reachable, the event is silently dropped.
//! Science computation never blocks on gossip delivery.

use serde::Serialize;
use std::os::unix::net::UnixStream;
use std::io::Write;
use std::sync::OnceLock;
use std::path::PathBuf;

static SWARMVINE_SOCKET: OnceLock<Option<PathBuf>> = OnceLock::new();

fn swarmvine_socket() -> &'static Option<PathBuf> {
    SWARMVINE_SOCKET.get_or_init(|| {
        let candidates = [
            PathBuf::from("/run/swarmvine/swarmvine.sock"),
            PathBuf::from("/tmp/swarmvine.sock"),
            dirs::runtime_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join("swarmvine.sock"),
        ];
        candidates.into_iter().find(|p| p.exists())
    })
}

#[derive(Serialize)]
struct GossipInjectRpc<'a> {
    jsonrpc: &'a str,
    method: &'a str,
    params: GossipInjectParams<'a>,
}

#[derive(Serialize)]
struct GossipInjectParams<'a> {
    topic: &'a str,
    payload: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<u32>,
}

fn inject(topic: &str, payload: serde_json::Value) {
    let Some(socket_path) = swarmvine_socket() else {
        return;
    };

    let rpc = GossipInjectRpc {
        jsonrpc: "2.0",
        method: "gossip.inject",
        params: GossipInjectParams {
            topic,
            payload,
            ttl: Some(300),
        },
    };

    let Ok(data) = serde_json::to_vec(&rpc) else { return };

    if let Ok(mut stream) = UnixStream::connect(socket_path) {
        stream.set_write_timeout(Some(std::time::Duration::from_millis(100))).ok();
        let _ = stream.write_all(&data);
    }
}

// ── Science Pipeline Events ──────────────────────────────────────────────────

/// Campaign started: new production run beginning
pub fn campaign_started(volume: &str, beta: f64, total_configs: usize) {
    inject(
        "science.campaign.started:strandGate:hotspring",
        serde_json::json!({
            "volume": volume,
            "beta": beta,
            "total_configs": total_configs,
            "binary": "arxiv_production_campaign",
        }),
    );
}

/// Config complete: one (volume, beta, seed) point finished
pub fn config_complete(volume: &str, beta: f64, seed: u64, plaquette: f64, acceptance: f64) {
    inject(
        "science.config.complete:strandGate:hotspring",
        serde_json::json!({
            "volume": volume,
            "beta": beta,
            "seed": seed,
            "plaquette": plaquette,
            "acceptance_rate": acceptance,
        }),
    );
}

/// Campaign progress: periodic update
pub fn campaign_progress(completed: usize, total: usize, current_volume: &str) {
    inject(
        "science.campaign.progress:strandGate:hotspring",
        serde_json::json!({
            "completed": completed,
            "total": total,
            "current_volume": current_volume,
            "percent": (completed as f64 / total as f64 * 100.0) as u32,
        }),
    );
}

/// Campaign complete: all configs finished
pub fn campaign_complete(total_configs: usize, wall_clock_hours: f64) {
    inject(
        "science.campaign.complete:strandGate:hotspring",
        serde_json::json!({
            "total_configs": total_configs,
            "wall_clock_hours": wall_clock_hours,
            "status": "complete",
        }),
    );
}

/// pseudoSpore manifest generated
pub fn pseudospore_manifest_generated(name: &str, version: &str, n_files: usize) {
    inject(
        "science.pseudospore.manifest:strandGate:hotspring",
        serde_json::json!({
            "name": name,
            "version": version,
            "n_files": n_files,
            "stage": "manifest",
        }),
    );
}

/// pseudoSpore bundle created
pub fn pseudospore_bundled(name: &str, version: &str, size_bytes: u64) {
    inject(
        "science.pseudospore.bundled:strandGate:hotspring",
        serde_json::json!({
            "name": name,
            "version": version,
            "size_bytes": size_bytes,
            "stage": "bundled",
        }),
    );
}

/// pseudoSpore signed by bearDog
pub fn pseudospore_signed(name: &str, version: &str, root_hash: &str) {
    inject(
        "science.pseudospore.signed:strandGate:hotspring",
        serde_json::json!({
            "name": name,
            "version": version,
            "root_hash": root_hash,
            "stage": "signed",
        }),
    );
}

/// pseudoSpore registered on ironGate
pub fn pseudospore_registered(name: &str, version: &str, nft_id: &str, cas_address: &str) {
    inject(
        "science.pseudospore.registered:strandGate:hotspring",
        serde_json::json!({
            "name": name,
            "version": version,
            "nft_id": nft_id,
            "cas_address": cas_address,
            "stage": "registered",
        }),
    );
}

/// Analysis complete: jackknife stats computed
pub fn analysis_complete(n_grid_points: usize, n_configs: usize) {
    inject(
        "science.analysis.complete:strandGate:hotspring",
        serde_json::json!({
            "n_grid_points": n_grid_points,
            "n_configs": n_configs,
            "binary": "arxiv_analysis",
        }),
    );
}

/// Validation pass/fail
pub fn validation_result(passed: bool, n_verified: usize, n_total: usize) {
    inject(
        "science.validation.result:strandGate:hotspring",
        serde_json::json!({
            "passed": passed,
            "verified": n_verified,
            "total": n_total,
        }),
    );
}
