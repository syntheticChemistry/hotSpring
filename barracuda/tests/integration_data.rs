// SPDX-License-Identifier: AGPL-3.0-only

//! Integration tests: data loading, provenance, and discovery.
//!
//! Validates that the data loading pipeline (JSON→structs), provenance records,
//! and capability-based discovery work end-to-end.

use hotspring_barracuda::provenance;
use hotspring_barracuda::tolerances;

#[test]
fn sly4_params_has_10_elements() {
    assert_eq!(
        provenance::SLY4_PARAMS.len(),
        10,
        "SLy4 parametrization has 10 Skyrme parameters"
    );
}

#[test]
fn sly4_params_all_finite() {
    for (i, &p) in provenance::SLY4_PARAMS.iter().enumerate() {
        assert!(p.is_finite(), "SLy4 param [{i}] must be finite, got {p}");
    }
}

#[test]
fn nmp_targets_are_physical() {
    let nmp = &provenance::NMP_TARGETS;
    assert!(
        nmp.rho0.value > 0.14 && nmp.rho0.value < 0.18,
        "saturation density ~ 0.16 fm^-3"
    );
    assert!(
        nmp.e_a.value > -17.0 && nmp.e_a.value < -15.0,
        "saturation energy ~ -16 MeV"
    );
    assert!(
        nmp.k_inf.value > 200.0 && nmp.k_inf.value < 300.0,
        "incompressibility ~ 230 MeV"
    );
    assert!(
        nmp.m_eff.value > 0.0 && nmp.m_eff.value < 1.0,
        "effective mass ratio in (0,1)"
    );
    assert!(
        nmp.j.value > 25.0 && nmp.j.value < 40.0,
        "symmetry energy ~ 32 MeV"
    );
}

#[test]
fn tolerance_hierarchy_consistent() {
    let tols = [
        ("EXACT_F64", tolerances::EXACT_F64),
        ("ITERATIVE_F64", tolerances::ITERATIVE_F64),
        ("GPU_VS_CPU_F64", tolerances::GPU_VS_CPU_F64),
        ("MD_FORCE_TOLERANCE", tolerances::MD_FORCE_TOLERANCE),
    ];

    for window in tols.windows(2) {
        assert!(
            window[0].1 < window[1].1,
            "{} ({}) should be < {} ({})",
            window[0].0,
            window[0].1,
            window[1].0,
            window[1].1
        );
    }
}

#[test]
fn esn_regularization_matches_config_default() {
    assert!(
        (tolerances::ESN_REGULARIZATION - 1e-2).abs() < f64::EPSILON,
        "ESN regularization should be 1e-2"
    );
}

#[test]
fn provenance_records_have_content() {
    let records = [
        &provenance::L1_PYTHON_CHI2,
        &provenance::L2_PYTHON_CHI2,
        &provenance::HOTQCD_EOS_PROVENANCE,
        &provenance::SCREENED_COULOMB_PROVENANCE,
        &provenance::DALIGAULT_CALIBRATION_PROVENANCE,
    ];

    for p in &records {
        assert!(!p.label.is_empty(), "label must not be empty");
        assert!(!p.script.is_empty(), "script must not be empty");
        assert!(!p.commit.is_empty(), "commit must not be empty");
        assert!(!p.date.is_empty(), "date must not be empty");
    }
}

#[test]
fn screened_coulomb_script_path_has_no_control_prefix() {
    let path = provenance::SCREENED_COULOMB_PROVENANCE.script;
    assert!(
        !path.starts_with("control/"),
        "script path should not have control/ prefix: {path}"
    );
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn transport_tolerances_ordered() {
    assert!(
        tolerances::TRANSPORT_D_STAR_VS_SARKAS < tolerances::TRANSPORT_D_STAR_VS_FIT_LITE,
        "Sarkas parity should be tighter than lite fit"
    );
}

#[test]
fn provenance_baseline_values_are_finite() {
    let baselines = [
        &provenance::L1_PYTHON_CHI2,
        &provenance::L1_PYTHON_CANDIDATES,
        &provenance::L2_PYTHON_CHI2,
        &provenance::L2_PYTHON_CANDIDATES,
    ];
    for bp in &baselines {
        assert!(bp.value.is_finite(), "{}: value must be finite", bp.label);
        assert!(bp.value > 0.0, "{}: value must be positive", bp.label);
    }
}

#[test]
fn provenance_dates_are_iso8601() {
    let baselines = [
        &provenance::L1_PYTHON_CHI2,
        &provenance::L2_PYTHON_CHI2,
        &provenance::HOTQCD_EOS_PROVENANCE,
    ];
    for bp in &baselines {
        assert!(
            bp.date.len() == 10 && bp.date.chars().nth(4) == Some('-'),
            "{}: date should be ISO 8601: {}",
            bp.label,
            bp.date
        );
    }
}

#[test]
fn validation_harness_print_provenance_runs() {
    use hotspring_barracuda::validation::ValidationHarness;
    let h = ValidationHarness::new("provenance_test");
    h.print_provenance(&[&provenance::L1_PYTHON_CHI2]);
}

#[test]
fn all_doi_constants_are_valid() {
    let dois = [
        provenance::AME2020_DOI,
        provenance::HOTQCD_DOI,
        provenance::STANTON_MURILLO_DOI,
        provenance::DALIGAULT_DOI,
    ];
    for doi in &dois {
        assert!(doi.starts_with("10."), "DOI must start with 10.: {doi}");
        assert!(doi.contains('/'), "DOI must contain /: {doi}");
    }
}
