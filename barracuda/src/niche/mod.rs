// SPDX-License-Identifier: AGPL-3.0-or-later

//! Niche deployment self-knowledge for hotSpring.
//!
//! A Spring is a niche validation domain — not a primal. It proves that
//! scientific Python baselines can be faithfully ported to sovereign
//! Rust + GPU compute using the ecoPrimals stack. The niche deploys as
//! a biomeOS graph (defined in `downstream_manifest.toml`) that composes
//! real primals (BearDog, Songbird, ToadStool, barraCuda, etc.).
//!
//! # Module structure
//!
//! - **`tables`** — Static capability tables, dependency declarations,
//!   semantic mappings, and cost estimates. Compile-time self-knowledge.
//! - **This module** — Runtime logic: socket resolution, biomeOS registration
//!   (`primal.announce` with legacy fallback), standalone mode.
//!
//! # Standalone Mode
//!
//! Set `HOTSPRING_NO_NUCLEUS=1` to run without biomeOS or NUCLEUS primals.
//! Registration is skipped and IPC degrades gracefully.

mod tables;

pub use tables::*;

use std::path::Path;

use log::{info, warn};

/// Conventional directory name for ecosystem IPC sockets.
pub const ECOSYSTEM_SOCKET_DIR: &str = "biomeos";

/// Default registration target — discovered at runtime, not hardcoded.
/// Override via `BIOMEOS_PRIMAL` env var for non-standard deployments.
const REGISTRATION_TARGET: &str = "biomeos";

// ── Family ID ───────────────────────────────────────────────────────

static FAMILY_ID_OVERRIDE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Set the family ID programmatically (thread-safe, first-write-wins).
///
/// Call before any socket resolution or primal discovery. This avoids
/// `unsafe { std::env::set_var }` in Edition 2024.
pub fn set_family_id(id: impl Into<String>) {
    FAMILY_ID_OVERRIDE.set(id.into()).ok();
}

/// Resolve the biomeOS family ID.
///
/// Priority: `set_family_id()` override -> `FAMILY_ID` env -> `BIOMEOS_FAMILY_ID` env -> `"default"`.
#[must_use]
pub fn family_id() -> String {
    if let Some(id) = FAMILY_ID_OVERRIDE.get() {
        return id.clone();
    }
    std::env::var("FAMILY_ID")
        .or_else(|_| std::env::var("BIOMEOS_FAMILY_ID"))
        .unwrap_or_else(|_| "default".to_string())
}

/// Look up the canonical primal name for a capability domain.
#[must_use]
pub fn primal_name_for_domain(domain: &str) -> Option<&'static str> {
    DEPENDENCIES
        .iter()
        .find(|d| d.capability_domain == domain)
        .map(|d| d.name)
}

// ── Socket resolution ───────────────────────────────────────────────

/// Socket directories in XDG-compliant priority order.
#[must_use]
pub fn socket_dirs() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;

    let mut dirs = Vec::new();

    // Primary: colon-separated explicit search paths (`/a/biomeos:/c/biomeos`).
    if let Ok(raw) = std::env::var("BIOMEOS_SOCKET_DIRS") {
        for d in raw.split(':').filter(|s| !s.is_empty()) {
            let p = PathBuf::from(d);
            if !dirs.contains(&p) {
                dirs.push(p);
            }
        }
    }

    if let Ok(d) = std::env::var("BIOMEOS_SOCKET_DIR") {
        let p = PathBuf::from(d);
        if !dirs.contains(&p) {
            dirs.push(p);
        }
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(xdg).join(ECOSYSTEM_SOCKET_DIR);
        if !dirs.contains(&p) {
            dirs.push(p);
        }
    }

    // Fallback for NUCLEUS-less environments: any primal runtime dir under `/run/`.
    if let Ok(entries) = std::fs::read_dir("/run") {
        for entry in entries.flatten() {
            let candidate = entry.path().join(ECOSYSTEM_SOCKET_DIR);
            if candidate.is_dir() && !dirs.contains(&candidate) {
                dirs.push(candidate);
            }
        }
    }
    let run_biomeos = PathBuf::from("/run").join(ECOSYSTEM_SOCKET_DIR);
    if run_biomeos.is_dir() && !dirs.contains(&run_biomeos) {
        dirs.push(run_biomeos);
    }

    let user = std::env::var("USER").unwrap_or_else(|_| "unknown".to_string());
    dirs.push(std::env::temp_dir().join(format!("{ECOSYSTEM_SOCKET_DIR}-{user}")));
    dirs.push(std::env::temp_dir());

    dirs
}

/// Resolve the socket path for this niche's IPC server.
#[must_use]
pub fn resolve_server_socket() -> std::path::PathBuf {
    use std::path::PathBuf;

    if let Ok(explicit) = std::env::var("HOTSPRING_SOCKET") {
        return PathBuf::from(explicit);
    }
    if let Ok(explicit) = std::env::var("PRIMAL_SOCKET") {
        return PathBuf::from(explicit);
    }

    let fid = family_id();
    let sock_name = format!("hotspring-physics-{fid}.sock");

    for dir in socket_dirs() {
        if dir.is_dir() || std::fs::create_dir_all(&dir).is_ok() {
            return dir.join(&sock_name);
        }
    }

    std::env::temp_dir().join(sock_name)
}

/// Resolve the Neural API socket path (discovered by convention).
#[must_use]
pub fn resolve_neural_api_socket() -> Option<std::path::PathBuf> {
    if let Ok(explicit) = std::env::var("NEURAL_API_SOCKET") {
        let p = std::path::PathBuf::from(&explicit);
        if p.exists() {
            return Some(p);
        }
    }

    let fid = family_id();
    let sock_name = format!("neural-api-{fid}.sock");

    for dir in socket_dirs() {
        let p = dir.join(&sock_name);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

/// Best-effort hostname for benchmark provenance and telemetry.
#[must_use]
pub fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .or_else(|_| std::fs::read_to_string("/etc/hostname").map(|s| s.trim().to_string()))
        .unwrap_or_else(|_| "unknown".into())
}

/// Whether standalone mode is requested (no NUCLEUS / biomeOS).
#[must_use]
pub fn standalone_mode() -> bool {
    std::env::var("HOTSPRING_NO_NUCLEUS").is_ok()
}

// ── Registration ────────────────────────────────────────────────────

/// Register this niche's capabilities with biomeOS.
///
/// Tries **`primal.announce`** (Wave 17 signal API) first — a single
/// atomic call carrying all methods, capabilities, semantic mappings,
/// and signal tiers. Falls back to the legacy multi-call pattern
/// (`lifecycle.register` + N × `capability.register`) for older biomeOS.
///
/// Degrades gracefully if biomeOS is unreachable or if standalone mode
/// is active (`HOTSPRING_NO_NUCLEUS=1`). Physics must never depend on
/// registration success.
pub fn register_with_target(our_socket: &Path) {
    if standalone_mode() {
        info!(target: "niche", "HOTSPRING_NO_NUCLEUS set — skipping registration");
        return;
    }

    let Some(biomeos_path) = discover_biomeos_socket() else {
        info!(
            target: "niche",
            "biomeOS socket not discovered — registration deferred"
        );
        return;
    };

    let sock_str = our_socket.to_string_lossy().to_string();

    if try_primal_announce(&biomeos_path, &sock_str) {
        return;
    }

    legacy_register(&biomeos_path, &sock_str);
}

/// Attempt `primal.announce` — the Wave 17 single-call registration.
/// Returns `true` if biomeOS accepted the announce.
fn try_primal_announce(biomeos_path: &Path, sock_str: &str) -> bool {
    let methods: Vec<&str> = LOCAL_CAPABILITIES.to_vec();
    let routed: Vec<serde_json::Value> = ROUTED_CAPABILITIES
        .iter()
        .map(|(cap, provider)| {
            serde_json::json!({
                "method": cap,
                "canonical_provider": provider,
            })
        })
        .collect();

    let announce_payload = serde_json::json!({
        "primal": NICHE_NAME,
        "socket": sock_str,
        "pid": std::process::id(),
        "version": NICHE_VERSION,
        "capabilities": ["physics", "composition"],
        "methods": methods,
        "routed_methods": routed,
        "semantic_mappings": tables::physics_semantic_mappings(),
        "signal_tiers": ["node", "nest"],
    });

    match send_registration(biomeos_path, "primal.announce", &announce_payload) {
        Ok(()) => {
            let total = LOCAL_CAPABILITIES.len() + ROUTED_CAPABILITIES.len();
            info!(
                target: "biomeos",
                "primal.announce accepted: {total} capabilities ({} local, {} routed), signal tiers: [node, nest]",
                LOCAL_CAPABILITIES.len(),
                ROUTED_CAPABILITIES.len(),
            );
            true
        }
        Err(e) => {
            info!(
                target: "biomeos",
                "primal.announce not available ({e}) — falling back to legacy registration"
            );
            false
        }
    }
}

/// Legacy multi-call registration for biomeOS that doesn't support
/// `primal.announce` yet.
fn legacy_register(biomeos_path: &Path, sock_str: &str) {
    let reg_payload = serde_json::json!({
        "name": NICHE_NAME,
        "socket_path": sock_str,
        "pid": std::process::id(),
        "domain": PRIMAL_DOMAIN,
        "version": NICHE_VERSION,
    });

    match send_registration(biomeos_path, "lifecycle.register", &reg_payload) {
        Ok(()) => info!(target: "biomeos", "registered with lifecycle manager"),
        Err(e) => warn!(target: "biomeos", "lifecycle.register failed (non-fatal): {e}"),
    }

    let physics_mappings = tables::physics_semantic_mappings();
    let domains: &[(&str, serde_json::Value)] = &[
        ("physics", physics_mappings),
        (
            "composition",
            serde_json::json!({ "health": "composition.health" }),
        ),
    ];

    for (domain, mappings) in domains {
        let mut payload = serde_json::json!({
            "primal": NICHE_NAME,
            "capability": domain,
            "socket": sock_str,
            "semantic_mappings": mappings,
        });
        if *domain == "physics" {
            payload["operation_dependencies"] = operation_dependencies();
            payload["cost_estimates"] = cost_estimates();
        }
        if let Err(e) = send_registration(biomeos_path, "capability.register", &payload) {
            warn!("niche registration failed: {e}");
        }
    }

    let mut registered = 0u32;
    for cap in LOCAL_CAPABILITIES {
        let params = serde_json::json!({
            "primal": NICHE_NAME,
            "capability": cap,
            "socket": sock_str,
            "served_locally": true,
        });
        if send_registration(biomeos_path, "capability.register", &params).is_ok() {
            registered += 1;
        }
    }

    for (cap, provider) in ROUTED_CAPABILITIES {
        let params = serde_json::json!({
            "primal": NICHE_NAME,
            "capability": cap,
            "socket": sock_str,
            "served_locally": false,
            "canonical_provider": provider,
        });
        if let Err(e) = send_registration(biomeos_path, "capability.register", &params) {
            warn!("niche registration failed: {e}");
        }
    }

    let total = LOCAL_CAPABILITIES.len() + ROUTED_CAPABILITIES.len();
    info!(
        target: "biomeos",
        "capabilities registered: {registered} local, {} routed, {total} total, {} domains",
        ROUTED_CAPABILITIES.len(),
        domains.len(),
    );
}

/// Discover the biomeOS socket at runtime.
fn discover_biomeos_socket() -> Option<std::path::PathBuf> {
    let target = std::env::var("BIOMEOS_PRIMAL").unwrap_or_else(|_| REGISTRATION_TARGET.to_owned());
    for dir in socket_dirs() {
        let fid = family_id();
        let candidate = dir.join(format!("{target}-{fid}.sock"));
        if candidate.exists() {
            return Some(candidate);
        }
        let candidate = dir.join(format!("neural-api-{fid}.sock"));
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn send_registration(
    socket_path: &std::path::Path,
    method: &str,
    params: &serde_json::Value,
) -> Result<(), String> {
    crate::primal_bridge::send_jsonrpc(socket_path, method, params)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "niche manifest tests use expect on fixtures"
)]
mod tests {
    use super::*;

    #[test]
    fn niche_constants_non_empty() {
        assert!(!NICHE_NAME.is_empty());
        assert!(!NICHE_DESCRIPTION.is_empty());
        assert!(!PROTO_NUCLEATE.is_empty());
        assert!(!LOCAL_CAPABILITIES.is_empty());
        assert!(!ROUTED_CAPABILITIES.is_empty());
        assert!(!DEPENDENCIES.is_empty());
        assert!(!FRAGMENTS.is_empty());
        assert!(!SEMANTIC_MAPPINGS.is_empty());
    }

    #[test]
    fn all_capabilities_follow_semantic_naming() {
        let all = all_capabilities();
        for cap in &all {
            assert!(
                cap.contains('.'),
                "capability '{cap}' should follow domain.operation format"
            );
        }
    }

    #[test]
    fn no_duplicate_capabilities() {
        let mut seen = std::collections::HashSet::new();
        let all = all_capabilities();
        for cap in &all {
            assert!(seen.insert(cap), "duplicate capability: {cap}");
        }
    }

    #[test]
    fn local_and_routed_are_disjoint() {
        for (routed_method, _) in ROUTED_CAPABILITIES {
            assert!(
                !LOCAL_CAPABILITIES.contains(routed_method),
                "'{routed_method}' is in both LOCAL and ROUTED — pick one"
            );
        }
    }

    #[test]
    fn routed_capabilities_have_providers() {
        for (method, provider) in ROUTED_CAPABILITIES {
            assert!(
                !provider.is_empty(),
                "routed capability '{method}' has empty provider"
            );
        }
    }

    #[test]
    fn capabilities_include_health_and_discovery() {
        assert!(LOCAL_CAPABILITIES.contains(&"health.liveness"));
        assert!(LOCAL_CAPABILITIES.contains(&"health.readiness"));
        assert!(LOCAL_CAPABILITIES.contains(&"capabilities.list"));
    }

    #[test]
    fn dependencies_include_core_primals() {
        let names: Vec<&str> = DEPENDENCIES.iter().map(|d| d.name).collect();
        assert!(names.contains(&"beardog"));
        assert!(names.contains(&"barracuda"));
        assert!(names.contains(&"toadstool"));
    }

    #[test]
    fn semantic_mappings_cover_local_capabilities() {
        for (_, full) in SEMANTIC_MAPPINGS {
            assert!(
                LOCAL_CAPABILITIES.contains(full),
                "semantic mapping target '{full}' not in LOCAL_CAPABILITIES"
            );
        }
    }

    #[test]
    fn operation_dependencies_covers_all_local_capabilities() {
        let deps = operation_dependencies();
        for cap in LOCAL_CAPABILITIES {
            assert!(
                deps.get(cap).is_some(),
                "missing dependency entry for {cap}"
            );
        }
    }

    #[test]
    fn cost_estimates_covers_all_local_capabilities() {
        let costs = cost_estimates();
        for cap in LOCAL_CAPABILITIES {
            assert!(costs.get(cap).is_some(), "missing cost entry for {cap}");
        }
    }

    #[test]
    fn cost_estimates_have_latency_and_memory() {
        let costs = cost_estimates();
        let map = costs.as_object().expect("costs should be object");
        for (key, val) in map {
            assert!(
                val.get("latency_ms").is_some(),
                "cost estimate for '{key}' missing latency_ms"
            );
            assert!(
                val.get("memory_bytes").is_some(),
                "cost estimate for '{key}' missing memory_bytes"
            );
        }
    }

    #[test]
    fn socket_dirs_never_empty() {
        let dirs = socket_dirs();
        assert!(!dirs.is_empty(), "should always resolve at least one dir");
    }

    #[test]
    fn family_id_has_default() {
        let fid = family_id();
        assert!(!fid.is_empty());
    }

    #[test]
    fn niche_name_matches_convention() {
        assert_eq!(NICHE_NAME, "hotspring");
        assert!(NICHE_NAME.chars().all(|c| c.is_ascii_lowercase()));
    }

    #[test]
    fn registration_target_is_biomeos() {
        assert_eq!(REGISTRATION_TARGET, "biomeos");
    }

    #[test]
    fn register_with_target_graceful_when_biomeos_unreachable() {
        let sock = Path::new("/tmp/hotspring-niche-test-nonexistent.sock");
        register_with_target(sock);
    }

    #[test]
    fn capabilities_match_registry_toml() {
        let toml_str = include_str!("../../config/capability_registry.toml");
        let parsed: toml::Value = toml::from_str(toml_str).expect("parse registry");
        let caps_in_toml: Vec<&str> = parsed["capabilities"]
            .as_array()
            .expect("capabilities array")
            .iter()
            .filter_map(|c| c.get("method")?.as_str())
            .collect();

        let all = all_capabilities();
        for code_cap in &all {
            assert!(
                caps_in_toml.contains(code_cap),
                "capability '{code_cap}' is in niche but missing from \
                 config/capability_registry.toml"
            );
        }
        for toml_cap in &caps_in_toml {
            assert!(
                all.contains(toml_cap),
                "capability '{toml_cap}' is in capability_registry.toml but missing from \
                 niche capabilities"
            );
        }
    }

    #[test]
    fn backward_compat_capabilities_alias() {
        assert!(std::ptr::eq(CAPABILITIES, LOCAL_CAPABILITIES));
    }
}
