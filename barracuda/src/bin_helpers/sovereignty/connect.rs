// SPDX-License-Identifier: AGPL-3.0-or-later

use hotspring_barracuda::fleet_client::{
    EmberClient, FleetDiscovery, discover_diesel_ember_socket, ember_socket_candidates,
};
use hotspring_barracuda::glowplug_client::GlowplugClient;
use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::validation::ValidationHarness;

/// Try to connect to ember for the given BDF. Returns None if no socket found.
pub fn try_connect_ember(bdf: &str) -> Option<EmberClient> {
    if let Some(sock) = discover_diesel_ember_socket(bdf) {
        return Some(EmberClient::connect(sock.to_string_lossy().as_ref()));
    }
    if let Ok(disc) = FleetDiscovery::load_default() {
        if let Some(sock) = disc.file().routes.get(bdf) {
            return Some(EmberClient::connect(sock));
        }
    }
    for candidate in ember_socket_candidates(bdf) {
        if candidate.exists() {
            return Some(EmberClient::connect(candidate.to_string_lossy().as_ref()));
        }
    }
    None
}

/// Try to connect to ember with a liveness probe (mmio read at offset 0).
pub fn try_connect_ember_probed(bdf: &str) -> Option<EmberClient> {
    if let Some(sock) = discover_diesel_ember_socket(bdf) {
        let client = EmberClient::connect(sock.to_string_lossy().as_ref());
        if client.mmio_read(bdf, 0).is_ok() {
            return Some(client);
        }
    }
    for candidate in ember_socket_candidates(bdf) {
        if candidate.exists() {
            let client = EmberClient::connect(candidate.to_string_lossy().as_ref());
            if client.mmio_read(bdf, 0).is_ok() {
                return Some(client);
            }
        }
    }
    let fleet_path = FleetDiscovery::resolve_path();
    if let Ok(fleet) = FleetDiscovery::load(&fleet_path) {
        for dev in &fleet.file().devices {
            if dev.bdf == bdf {
                if let Some(sock) = &dev.socket {
                    return Some(EmberClient::connect(sock));
                }
            }
        }
    }
    None
}

/// Connect to ember, with optional socket override. Exits on failure.
pub fn connect_ember(bdf: &str, override_socket: Option<&str>) -> EmberClient {
    if let Some(sock) = override_socket {
        return EmberClient::connect(sock);
    }
    if let Some(sock) = discover_diesel_ember_socket(bdf) {
        eprintln!("  diesel engine: found ember at {}", sock.display());
        return EmberClient::connect(sock.to_string_lossy().as_ref());
    }
    if let Ok(disc) = FleetDiscovery::load_default() {
        if let Some(sock) = disc.file().routes.get(bdf) {
            return EmberClient::connect(sock);
        }
    }
    for candidate in ember_socket_candidates(bdf) {
        if candidate.exists() {
            return EmberClient::connect(candidate.to_string_lossy().as_ref());
        }
    }
    eprintln!("FATAL: no ember socket found for {bdf}");
    std::process::exit(1);
}

/// Connect to ember or register failure in harness and exit.
pub fn require_ember(bdf: &str, harness: &mut ValidationHarness) -> EmberClient {
    match try_connect_ember(bdf) {
        Some(e) => {
            harness.check_bool("ember reachable", true);
            e
        }
        None => {
            eprintln!("  FATAL: ember not reachable for {bdf}");
            harness.check_bool("ember reachable", false);
            harness.finish();
        }
    }
}

/// Try to connect to glowplug via NUCLEUS discovery.
pub fn try_connect_glowplug() -> Option<GlowplugClient> {
    let nucleus = NucleusContext::detect();
    GlowplugClient::from_nucleus(&nucleus).ok()
}

/// Connect to glowplug or exit with error.
pub fn connect_glowplug() -> GlowplugClient {
    let nucleus = NucleusContext::detect();
    match GlowplugClient::from_nucleus(&nucleus) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("FATAL: glowplug not reachable — {e}");
            std::process::exit(1);
        }
    }
}

/// Connect to glowplug or register failure in harness and exit.
pub fn require_glowplug(harness: &mut ValidationHarness) -> GlowplugClient {
    match try_connect_glowplug() {
        Some(g) => {
            harness.check_bool("glowplug reachable", true);
            g
        }
        None => {
            eprintln!("  FATAL: glowplug not reachable");
            harness.check_bool("glowplug reachable", false);
            harness.finish();
        }
    }
}

/// Check coralReef (shader compiler) liveness via NUCLEUS.
pub fn check_coralreef_liveness() -> bool {
    let nucleus = NucleusContext::detect();
    match nucleus.call_by_capability("shader", "shader.compile.capabilities", serde_json::json!({}))
    {
        Ok(resp) => {
            if let Some(result) = resp.get("result") {
                eprintln!(
                    "  coralReef: alive (capabilities: {})",
                    result
                        .get("formats")
                        .and_then(|f| f.as_array())
                        .map_or(0, |a| a.len())
                );
                true
            } else {
                eprintln!("  coralReef: responded but no result");
                false
            }
        }
        Err(e) => {
            eprintln!("  coralReef: not reachable — {e}");
            false
        }
    }
}

/// Extract a CLI argument value by flag name (e.g. `--bdf`).
pub fn extract_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

/// Check if `--dry-run` was passed on the command line.
pub fn is_dry_run(args: &[String]) -> bool {
    args.iter().any(|a| a == "--dry-run")
}

/// Resolve target PCI BDF for experiment binaries.
///
/// Precedence: `--bdf` CLI flag → `HOTSPRING_BARRACUDA_TARGET_BDF` env →
/// legacy `HOTSPRING_BDF` env → fleet discovery (`device.list`) → empty.
pub fn resolve_target_bdf(args: &[String], default_idx: usize) -> String {
    if let Some(bdf) = extract_arg(args, "--bdf") {
        return bdf;
    }
    if let Ok(bdf) = std::env::var("HOTSPRING_BARRACUDA_TARGET_BDF") {
        return bdf;
    }
    if let Ok(bdf) = std::env::var("HOTSPRING_BDF") {
        return bdf;
    }
    if let Some(client) = try_connect_glowplug() {
        if let Ok(devices) = client.list_devices() {
            if let Some(dev) = devices.get(default_idx) {
                eprintln!("  Fleet discovery: using BDF {} from device.list", dev.bdf);
                return dev.bdf.clone();
            }
        }
    }
    eprintln!("WARNING: no target BDF from CLI, env, or fleet — see --help");
    String::new()
}
