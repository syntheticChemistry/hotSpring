// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: hotQCD Dispatch Pipeline.
//!
//! Validates the full lattice QCD compute pipeline through the compute trio:
//!
//! 1. **barraCuda** provides QCD WGSL shaders (wilson_plaquette, su3_gauge_force, etc.)
//! 2. **coralReef** compiles WGSL → GPU binary via `shader.compile.wgsl`
//! 3. **toadStool** dispatches the workload via `compute.dispatch.submit`
//! 4. **barraCuda** precision routing advises f64 strategy via `precision.route`
//!
//! The QCD shader corpus (51 lattice shaders + 7 silicon routing shaders) is
//! the densest f64 workload in the ecosystem. This scenario validates that the
//! key shaders compile through the IPC pipeline and that dispatch is routable.
//!
//! In standalone mode, all IPC probes gracefully degrade.

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "hotqcd-dispatch",
        track: Track::GpuCompute,
        tier: Tier::Live,
        provenance_crate: "validate_compute_trio_pipeline",
        provenance_date: "2026-05-12",
        description: "hotQCD: barraCuda lattice shaders → coralReef compile → toadStool dispatch",
    },
    run,
};

/// Core QCD shaders that must compile for sovereign lattice compute.
/// Relative to the crate root (CARGO_MANIFEST_DIR at build time).
const QCD_PROBE_SHADER_NAMES: &[&str] = &[
    "wilson_plaquette_f64.wgsl",
    "su3_gauge_force_f64.wgsl",
    "su3_link_update_f64.wgsl",
    "dirac_staggered_f64.wgsl",
    "cg_kernels_f64.wgsl",
    "hmc_leapfrog_f64.wgsl",
];

fn qcd_shader_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("HOTSPRING_QCD_SHADER_DIR") {
        return std::path::PathBuf::from(dir);
    }
    let base = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(base).join("src/lattice/shaders")
}

fn routing_shader_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("HOTSPRING_ROUTING_SHADER_DIR") {
        return std::path::PathBuf::from(dir);
    }
    let base = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(base).join("src/bin/shaders/qcd_silicon_routing")
}

pub fn run(v: &mut ValidationHarness) {
    use crate::ipc::tier2;
    use crate::primal_bridge::NucleusContext;

    let nucleus = NucleusContext::detect();

    // --- Phase 1: Precision advisory for lattice QCD ---
    let advisory = tier2::precision_advisory(&nucleus, "lattice_qcd");
    let math_alive = nucleus.by_domain("math").is_some_and(|ep| ep.alive);
    v.check_bool(
        "hotqcd:precision_advisory",
        advisory.is_some() || !math_alive,
    );
    if let Some(ref adv) = advisory {
        let tier_ok = adv
            .tier
            .as_deref()
            .is_some_and(|t| t.contains("f64") || t.contains("mixed") || t.contains("df64"));
        v.check_bool("hotqcd:precision_f64_or_mixed", tier_ok || adv.fma_safe);
    }

    // --- Phase 2: QCD shader compilation through coralReef ---
    let shader_alive = nucleus.by_domain("shader").is_some_and(|ep| ep.alive);
    let mut compiled_count = 0usize;
    let total = QCD_PROBE_SHADER_NAMES.len();

    let shader_dir = qcd_shader_dir();
    for shader_name in QCD_PROBE_SHADER_NAMES {
        let shader_path = shader_dir.join(shader_name);
        if !shader_alive {
            v.check_bool(
                &format!("hotqcd:compile:{}", short_name(shader_name)),
                true, // standalone: skip is acceptable
            );
            continue;
        }

        let source = match std::fs::read_to_string(&shader_path) {
            Ok(s) => s,
            Err(_) => {
                v.check_bool(
                    &format!("hotqcd:compile:{}", short_name(shader_name)),
                    false,
                );
                continue;
            }
        };

        let params = serde_json::json!({
            "source": source,
            "source_type": "wgsl",
        });

        match nucleus.call_by_capability("shader", "shader.compile.wgsl", params) {
            Ok(resp) => {
                let ok = resp.get("error").is_none();
                if ok {
                    compiled_count += 1;
                }
                v.check_bool(
                    &format!("hotqcd:compile:{}", short_name(shader_name)),
                    ok,
                );
            }
            Err(_) => {
                v.check_bool(
                    &format!("hotqcd:compile:{}", short_name(shader_name)),
                    false,
                );
            }
        }
    }

    if shader_alive {
        v.check_bool(
            "hotqcd:all_core_shaders_compiled",
            compiled_count == total,
        );
    }

    // --- Phase 3: toadStool workload preflight for QCD ---
    let preflight = tier2::workload_preflight(&nucleus, "hotqcd-lattice");
    let compute_alive = nucleus.by_domain("compute").is_some_and(|ep| ep.alive);
    v.check_bool(
        "hotqcd:workload_preflight",
        preflight.is_some() || !compute_alive,
    );

    // --- Phase 4: QCD dispatch probe (dry-run) ---
    if compute_alive {
        let dispatch_params = serde_json::json!({
            "workload": "hotqcd-wilson-plaquette",
            "kind": "lattice_qcd",
            "dry_run": true,
            "spring": "hotSpring",
            "domain": "lattice_qcd",
            "precision": advisory.as_ref().and_then(|a| a.tier.as_deref()).unwrap_or("f64"),
        });

        match nucleus.call_by_capability("compute", "compute.dispatch.submit", dispatch_params) {
            Ok(_) => v.check_bool("hotqcd:dispatch_submit", true),
            Err(_) => v.check_bool("hotqcd:dispatch_submit", false),
        }
    } else {
        v.check_bool("hotqcd:compute_domain_available", compute_alive);
    }

    // --- Phase 5: QCD silicon routing shaders (barrier-heavy) ---
    let routing_dir = routing_shader_dir();
    let routing_names = [
        "reduce_shared.wgsl",
        "force_alu.wgsl",
        "stencil_storage.wgsl",
    ];

    for shader_name in &routing_names {
        let shader_path = routing_dir.join(shader_name);
        if !shader_alive {
            v.check_bool(
                &format!("hotqcd:routing:{}", short_name(shader_name)),
                true,
            );
            continue;
        }

        let ok = match std::fs::read_to_string(&shader_path) {
            Ok(source) => {
                let params = serde_json::json!({
                    "source": source,
                    "source_type": "wgsl",
                });
                nucleus
                    .call_by_capability("shader", "shader.compile.wgsl", params)
                    .is_ok()
            }
            Err(_) => false,
        };
        v.check_bool(
            &format!("hotqcd:routing:{}", short_name(shader_name)),
            ok,
        );
    }
}

fn short_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path).trim_end_matches(".wgsl")
}
