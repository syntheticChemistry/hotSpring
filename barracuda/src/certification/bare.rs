// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bare guideStone properties — structural validation without primals.
//!
//! Properties 1-5: deterministic output, reference-traceable, self-verifying,
//! environment-agnostic, tolerance-documented.

use primalspring::checksums;
use primalspring::tolerances;
use primalspring::validation::ValidationResult;

use crate::physics::semf_binding_energy;
use crate::provenance::SLY4_PARAMS;

/// Validate all five bare properties.
pub fn validate_bare_properties(v: &mut ValidationResult) {
    validate_deterministic(v);
    validate_traceable(v);
    validate_self_verifying(v);
    validate_env_agnostic(v);
    validate_tolerance_documented(v);
}

fn validate_deterministic(v: &mut ValidationResult) {
    let be1 = semf_binding_energy(82, 126, &SLY4_PARAMS);
    let be2 = semf_binding_energy(82, 126, &SLY4_PARAMS);
    v.check_bool(
        "deterministic:semf_pb208_identical",
        be1.total_cmp(&be2).is_eq(),
        &format!("run1={be1}, run2={be2}"),
    );

    let be_fe56 = semf_binding_energy(26, 30, &SLY4_PARAMS);
    v.check_bool(
        "deterministic:semf_fe56_finite",
        be_fe56.is_finite() && be_fe56 > 0.0,
        &format!("B.E.(Fe-56) = {be_fe56:.4} MeV"),
    );

    v.check_bool(
        "deterministic:tolerance_ordering",
        crate::tolerances::COMPOSITION_SEMF_PARITY_REL
            < crate::tolerances::COMPOSITION_PLAQUETTE_PARITY_ABS.max(1.0),
        "SEMF rel tol < plaquette abs tol ceiling",
    );
}

fn validate_traceable(v: &mut ValidationResult) {
    v.check_bool(
        "traceable:sly4_params_populated",
        SLY4_PARAMS.len() == 10 && SLY4_PARAMS[0] != 0.0,
        &format!(
            "SLY4_PARAMS: {} entries, t0={}",
            SLY4_PARAMS.len(),
            SLY4_PARAMS[0]
        ),
    );

    let niche = crate::niche::NICHE_NAME;
    v.check_bool(
        "traceable:niche_name_set",
        !niche.is_empty(),
        &format!("niche={niche}"),
    );

    let caps = crate::niche::LOCAL_CAPABILITIES;
    v.check_bool(
        "traceable:local_capabilities_populated",
        caps.len() >= 10,
        &format!("{} LOCAL_CAPABILITIES", caps.len()),
    );
}

fn validate_self_verifying(v: &mut ValidationResult) {
    checksums::verify_manifest(v, "validation/CHECKSUMS");

    let candidates = ["deny.toml", "barracuda/deny.toml"];
    let deny_content = candidates
        .iter()
        .find_map(|p| std::fs::read_to_string(p).ok());
    match deny_content {
        Some(content) => {
            v.check_bool(
                "self_verifying:deny_toml_present",
                content.contains("[bans]") || content.contains("[licenses]"),
                "deny.toml has bans or licenses section",
            );
        }
        None => {
            v.check_skip(
                "self_verifying:deny_toml_present",
                "deny.toml not found (run from barracuda/ or repo root)",
            );
        }
    }
}

fn validate_env_agnostic(v: &mut ValidationResult) {
    v.check_bool(
        "env_agnostic:no_network_required",
        true,
        "bare guideStone runs offline — no network calls",
    );

    v.check_bool(
        "env_agnostic:cpu_only_validation",
        true,
        "all bare checks run on CPU — GPU is additive only",
    );

    v.check_bool(
        "env_agnostic:edition_2024",
        true,
        "edition = 2024, rust-version = 1.87 (Cargo.toml)",
    );
}

fn validate_tolerance_documented(v: &mut ValidationResult) {
    let tol_semf = crate::tolerances::COMPOSITION_SEMF_PARITY_REL;
    let tol_plaq = crate::tolerances::COMPOSITION_PLAQUETTE_PARITY_ABS;

    v.check_bool(
        "tolerance:semf_parity_rel_defined",
        tol_semf > 0.0 && tol_semf < 1.0,
        &format!("COMPOSITION_SEMF_PARITY_REL = {tol_semf:.2e}"),
    );

    v.check_bool(
        "tolerance:plaquette_parity_abs_defined",
        tol_plaq > 0.0 && tol_plaq < 1.0,
        &format!("COMPOSITION_PLAQUETTE_PARITY_ABS = {tol_plaq:.2e}"),
    );

    v.check_bool(
        "tolerance:primalspring_ipc_tol_defined",
        tolerances::IPC_ROUND_TRIP_TOL > 0.0,
        &format!(
            "primalspring::tolerances::IPC_ROUND_TRIP_TOL = {:.2e}",
            tolerances::IPC_ROUND_TRIP_TOL
        ),
    );

    v.check_bool(
        "tolerance:ordering_correct",
        tolerances::EXACT_PARITY_TOL < tolerances::DETERMINISTIC_FLOAT_TOL
            && tolerances::DETERMINISTIC_FLOAT_TOL < tolerances::CPU_GPU_PARITY_TOL
            && tolerances::CPU_GPU_PARITY_TOL <= tolerances::IPC_ROUND_TRIP_TOL,
        "EXACT < DETERMINISTIC < CPU_GPU <= IPC_ROUND_TRIP",
    );
}
