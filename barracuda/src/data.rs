// SPDX-License-Identifier: AGPL-3.0-only

//! Experimental data and parameter bounds loading
//!
//! Supports two AME2020 datasets:
//!   - `ame2020_selected.json` — 52 curated nuclei (fast validation)
//!   - `ame2020_full.json` — 2,042 experimentally measured nuclei (full AME2020)
//!
//! Use `--nuclei=full` CLI flag to switch to the full dataset.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A single nucleus from AME2020
#[derive(Debug, Clone, Deserialize)]
pub struct Nucleus {
    #[serde(rename = "Z")]
    pub z: usize,
    #[serde(rename = "N")]
    pub n: usize,
    #[serde(rename = "A")]
    pub a: usize,
    pub element: String,
    #[serde(rename = "binding_energy_MeV")]
    pub binding_energy_mev: f64,
    #[serde(rename = "uncertainty_MeV")]
    pub uncertainty_mev: f64,
}

#[derive(Debug, Deserialize)]
struct NucleiFile {
    nuclei: Vec<Nucleus>,
}

/// Which AME2020 dataset to use
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NucleiSet {
    /// 52 curated nuclei (doubly-magic, Sn chain, etc.) — fast validation
    Selected,
    /// All 2,042 experimentally measured nuclei from AME2020
    Full,
}

impl NucleiSet {
    /// Parse from CLI argument string
    pub fn from_arg(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "full" | "all" | "2042" => NucleiSet::Full,
            "selected" | "52" | "default" => NucleiSet::Selected,
            _ => {
                eprintln!("  WARNING: Unknown nuclei set '{s}', using selected (52)");
                NucleiSet::Selected
            }
        }
    }

    /// JSON filename for this dataset
    pub fn filename(&self) -> &'static str {
        match self {
            NucleiSet::Selected => "ame2020_selected.json",
            NucleiSet::Full => "ame2020_full.json",
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            NucleiSet::Selected => "AME2020 selected (52 nuclei)",
            NucleiSet::Full => "AME2020 full (2,042 experimentally measured nuclei)",
        }
    }
}

/// Parse --nuclei=... from CLI args, defaulting to Selected
pub fn parse_nuclei_set_from_args() -> NucleiSet {
    std::env::args()
        .find(|a| a.starts_with("--nuclei="))
        .map_or(NucleiSet::Selected, |a| NucleiSet::from_arg(&a[9..]))
}

/// Resolve the path to the nuclei JSON file for a given dataset
pub fn nuclei_data_path(base_dir: &Path, set: NucleiSet) -> PathBuf {
    base_dir.join("exp_data").join(set.filename())
}

/// Load AME2020 experimental data → HashMap<(Z, N), (B_exp, σ)>
pub fn load_experimental_data(
    path: &Path,
) -> Result<HashMap<(usize, usize), (f64, f64)>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let file: NucleiFile = serde_json::from_str(&text)?;
    let mut map = HashMap::new();
    for nuc in file.nuclei {
        map.insert(
            (nuc.z, nuc.n),
            (nuc.binding_energy_mev, nuc.uncertainty_mev),
        );
    }
    Ok(map)
}

/// Load experimental data with dataset selection (convenience wrapper)
pub fn load_nuclei(
    base_dir: &Path,
    set: NucleiSet,
) -> Result<HashMap<(usize, usize), (f64, f64)>, Box<dyn std::error::Error>> {
    let path = nuclei_data_path(base_dir, set);
    println!("  Dataset: {} ({})", set.description(), path.display());
    load_experimental_data(&path)
}

/// Parameter range from bounds JSON
#[derive(Debug, Deserialize)]
struct ParamInfo {
    typical_range: [f64; 2],
}

#[derive(Debug, Deserialize)]
struct BoundsFile {
    parameters: HashMap<String, ParamInfo>,
}

/// Parameter ordering (must match Python)
pub const PARAM_NAMES: [&str; 10] = [
    "t0", "t1", "t2", "t3", "x0", "x1", "x2", "x3", "alpha", "W0",
];

/// Load parameter bounds → Vec<(min, max)>
pub fn load_bounds(path: &Path) -> Result<Vec<(f64, f64)>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let file: BoundsFile = serde_json::from_str(&text)?;
    let mut bounds = Vec::with_capacity(10);
    for name in &PARAM_NAMES {
        let info = file
            .parameters
            .get(*name)
            .ok_or_else(|| format!("Missing parameter: {name}"))?;
        bounds.push((info.typical_range[0], info.typical_range[1]));
    }
    Ok(bounds)
}

// ═══════════════════════════════════════════════════════════════════
// EOS context: shared setup for nuclear EOS binaries
// ═══════════════════════════════════════════════════════════════════

/// Loaded EOS context: base path, experimental data, parameter bounds.
///
/// Shared across all nuclear EOS binaries (L1, L2, L3, GPU variants).
/// Resolves the `control/surrogate/nuclear-eos` directory relative to
/// `CARGO_MANIFEST_DIR/../` and loads experimental data and parameter bounds.
pub struct EosContext {
    /// Base directory for nuclear EOS data files.
    pub base: PathBuf,
    /// Experimental binding energies: (Z, N) → (B_exp, σ).
    pub exp_data: std::sync::Arc<HashMap<(usize, usize), (f64, f64)>>,
    /// Skyrme parameter bounds: Vec<(min, max)>.
    pub bounds: Vec<(f64, f64)>,
}

/// Load the standard nuclear EOS context.
///
/// Resolves `control/surrogate/nuclear-eos` relative to the crate's parent
/// directory (hotSpring root), loads experimental data for the selected
/// nuclei set, and reads parameter bounds from `skyrme_bounds.json`.
///
/// # Panics
///
/// Panics if the data directory, experimental data, or bounds file cannot be read.
pub fn load_eos_context() -> EosContext {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR has no parent")
        .join("control/surrogate/nuclear-eos");

    let exp_data = std::sync::Arc::new(
        load_nuclei(&base, parse_nuclei_set_from_args()).expect("Failed to load experimental data"),
    );
    let bounds = load_bounds(&base.join("wrapper/skyrme_bounds.json"))
        .expect("Failed to load parameter bounds");

    EosContext {
        base,
        exp_data,
        bounds,
    }
}

/// Compute per-datum χ² for a binding energy function.
///
/// For each nucleus in `exp_data`, evaluates `binding_energy_fn(z, n, params)`
/// and computes `((B_calc - B_exp) / sigma_theo)²`. Returns the mean χ²/datum.
///
/// Uses [`crate::tolerances::sigma_theo`] for the theoretical uncertainty.
pub fn chi2_per_datum(
    params: &[f64],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    binding_energy_fn: impl Fn(usize, usize, &[f64]) -> f64,
) -> f64 {
    let mut chi2 = 0.0;
    let n = exp_data.len();
    if n == 0 {
        return 0.0;
    }
    for (&(z, nn), &(b_exp, _sigma_exp)) in exp_data {
        let b_calc = binding_energy_fn(z, nn, params);
        let sigma = crate::tolerances::sigma_theo(b_exp);
        chi2 += ((b_calc - b_exp) / sigma).powi(2);
    }
    chi2 / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::SLY4_PARAMS;

    #[test]
    fn parse_nuclei_json_without_file() {
        let json = r#"{"nuclei": [{"Z": 28, "N": 28, "A": 56, "element": "Ni", "binding_energy_MeV": 483.99, "uncertainty_MeV": 0.5}]}"#;
        #[derive(serde::Deserialize)]
        struct NucleiFile {
            nuclei: Vec<Nucleus>,
        }
        let file: NucleiFile = serde_json::from_str(json).expect("parse");
        assert_eq!(file.nuclei.len(), 1);
        let n = &file.nuclei[0];
        assert_eq!(n.z, 28);
        assert_eq!(n.n, 28);
        assert_eq!(n.a, 56);
        assert_eq!(n.element, "Ni");
        assert!((n.binding_energy_mev - 483.99).abs() < 1e-10);
        assert!((n.uncertainty_mev - 0.5).abs() < 1e-10);
    }

    #[test]
    fn nuclei_struct_fields() {
        let n = Nucleus {
            z: 82,
            n: 126,
            a: 208,
            element: "Pb".to_string(),
            binding_energy_mev: 1636.43,
            uncertainty_mev: 0.3,
        };
        assert_eq!(n.z, 82);
        assert_eq!(n.a, 208);
        assert!(n.binding_energy_mev > 1600.0);
    }

    #[test]
    fn param_names_correct_count() {
        assert_eq!(PARAM_NAMES.len(), 10);
    }

    #[test]
    fn param_names_standard_order() {
        assert_eq!(PARAM_NAMES[0], "t0");
        assert_eq!(PARAM_NAMES[8], "alpha");
        assert_eq!(PARAM_NAMES[9], "W0");
    }

    #[test]
    fn nuclei_set_parsing() {
        assert_eq!(NucleiSet::from_arg("full"), NucleiSet::Full);
        assert_eq!(NucleiSet::from_arg("all"), NucleiSet::Full);
        assert_eq!(NucleiSet::from_arg("2042"), NucleiSet::Full);
        assert_eq!(NucleiSet::from_arg("selected"), NucleiSet::Selected);
        assert_eq!(NucleiSet::from_arg("52"), NucleiSet::Selected);
        assert_eq!(NucleiSet::from_arg("default"), NucleiSet::Selected);
        assert_eq!(NucleiSet::from_arg("garbage"), NucleiSet::Selected);
    }

    #[test]
    #[allow(clippy::case_sensitive_file_extension_comparisons)]
    fn nuclei_set_filenames() {
        assert!(NucleiSet::Selected.filename().ends_with(".json"));
        assert!(NucleiSet::Full.filename().ends_with(".json"));
        assert!(NucleiSet::Selected.filename().contains("selected"));
        assert!(NucleiSet::Full.filename().contains("full"));
    }

    #[test]
    fn nuclei_data_path_construction() {
        let base = Path::new("/test/base");
        let path = nuclei_data_path(base, NucleiSet::Selected);
        assert!(path.to_str().unwrap().contains("exp_data"));
        assert!(path.to_str().unwrap().contains("ame2020_selected"));
    }

    #[test]
    fn load_experimental_data_from_disk() {
        // Integration test: load actual data if available
        let base = crate::discovery::nuclear_eos_dir();
        let path = nuclei_data_path(&base, NucleiSet::Selected);
        if path.exists() {
            let data = load_experimental_data(&path).expect("should parse JSON");
            assert!(!data.is_empty(), "should have nuclei");
            // Check Pb-208 is present
            assert!(data.contains_key(&(82, 126)), "should contain Pb-208");
            let (b_exp, sigma) = data[&(82, 126)];
            assert!(b_exp > 1600.0 && b_exp < 1700.0, "Pb-208 B ≈ 1636 MeV");
            assert!(sigma > 0.0, "uncertainty should be positive");
        }
    }

    #[test]
    fn load_bounds_from_disk() {
        let path = crate::discovery::skyrme_bounds_path();
        if path.exists() {
            let bounds = load_bounds(&path).expect("should parse bounds");
            assert_eq!(bounds.len(), 10, "should have 10 parameter bounds");
            for (min, max) in &bounds {
                assert!(min < max, "min should be less than max: {min} < {max}");
            }
        }
    }

    #[test]
    fn json_load_round_trip_consistency() {
        // Integration: loading the same file twice produces identical data.
        let base = crate::discovery::nuclear_eos_dir();
        let path = nuclei_data_path(&base, NucleiSet::Selected);
        if path.exists() {
            let a = load_experimental_data(&path).expect("first load");
            let b = load_experimental_data(&path).expect("second load");
            assert_eq!(a.len(), b.len(), "reload produced different count");
            for ((z, n), (be_a, sig_a)) in &a {
                let (be_b, sig_b) = b
                    .get(&(*z, *n))
                    .unwrap_or_else(|| panic!("missing ({z},{n}) in second load"));
                assert_eq!(
                    be_a.to_bits(),
                    be_b.to_bits(),
                    "B({z},{n}) differs on reload"
                );
                assert_eq!(
                    sig_a.to_bits(),
                    sig_b.to_bits(),
                    "σ({z},{n}) differs on reload"
                );
            }
        }
    }

    #[test]
    fn pipeline_discovery_to_semf() {
        // Integration: discovery → data load → SEMF computation.
        // Verifies the full path from disk files to physics.
        let base = crate::discovery::nuclear_eos_dir();
        let path = nuclei_data_path(&base, NucleiSet::Selected);
        if path.exists() {
            let data = load_experimental_data(&path).expect("load failed");
            let bounds_path = crate::discovery::skyrme_bounds_path();
            if bounds_path.exists() {
                let bounds = load_bounds(&bounds_path).expect("bounds load failed");
                assert_eq!(bounds.len(), 10, "wrong param count");
                // Use SLy4 parameters (known to be within bounds)
                // Verify SLy4 produces reasonable binding energies for loaded nuclei
                let mut checked = 0;
                for (&(z, n), &(b_exp, _sigma)) in &data {
                    let b_calc = crate::physics::semf::semf_binding_energy(z, n, &SLY4_PARAMS);
                    if b_exp > 0.0 && b_calc > 0.0 {
                        let ratio = b_calc / b_exp;
                        assert!(
                            ratio > 0.5 && ratio < 2.0,
                            "SEMF/exp ratio out of range for ({z},{n}): {b_calc}/{b_exp}={ratio}"
                        );
                        checked += 1;
                    }
                }
                assert!(
                    checked > 20,
                    "should verify at least 20 nuclei, got {checked}"
                );
            }
        }
    }

    #[test]
    fn chi2_per_datum_perfect_fit() {
        let mut exp = HashMap::new();
        exp.insert((28, 28), (483.99, 0.5));
        exp.insert((50, 82), (1102.85, 0.5));

        // If binding_energy_fn returns exact experimental values, chi2 = 0
        let chi2 = chi2_per_datum(&[], &exp, |z, n, _params| {
            *exp.get(&(z, n)).map(|(b, _)| b).unwrap()
        });
        assert!(chi2 < 1e-10, "perfect fit should give chi2 ≈ 0, got {chi2}");
    }

    #[test]
    fn chi2_per_datum_known_deviation() {
        let mut exp = HashMap::new();
        exp.insert((28, 28), (484.0, 0.5));

        // B_calc = 484 + sigma_theo(484) = 484 + 4.84 → chi2 = 1.0
        let sigma = crate::tolerances::sigma_theo(484.0);
        let chi2 = chi2_per_datum(&[], &exp, |_z, _n, _| 484.0 + sigma);
        assert!(
            (chi2 - 1.0).abs() < 1e-6,
            "one-sigma deviation should give chi2 = 1.0, got {chi2}"
        );
    }

    #[test]
    fn chi2_per_datum_determinism() {
        // Same inputs → identical chi2 on rerun
        let params = crate::provenance::SLY4_PARAMS;
        let mut exp = std::collections::HashMap::new();
        exp.insert((28, 28), (483.99, 1.0));
        exp.insert((82, 126), (1636.43, 1.0));
        let run = || chi2_per_datum(&params, &exp, crate::physics::semf_binding_energy);
        let a = run();
        let b = run();
        assert_eq!(a.to_bits(), b.to_bits(), "chi2_per_datum not deterministic");
    }
}
