// SPDX-License-Identifier: AGPL-3.0-only

//! Experimental data and parameter bounds loading
//!
//! Supports two AME2020 datasets:
//!   - `ame2020_selected.json` — 52 curated nuclei (fast validation)
//!   - `ame2020_full.json` — 2,042 experimentally measured nuclei (full AME2020)
//!
//! Use `--nuclei=full` CLI flag to switch to the full dataset.

use crate::error::HotSpringError;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A single nucleus from AME2020
#[derive(Debug, Clone, Deserialize)]
#[allow(missing_docs)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NucleiSet {
    /// 52 curated nuclei (doubly-magic, Sn chain, etc.) — fast validation
    Selected,
    /// All 2,042 experimentally measured nuclei from AME2020
    Full,
}

impl NucleiSet {
    /// Parse from CLI argument string
    #[must_use]
    pub fn from_arg(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "full" | "all" | "2042" => Self::Full,
            "selected" | "52" | "default" => Self::Selected,
            _ => {
                eprintln!("  WARNING: Unknown nuclei set '{s}', using selected (52)");
                Self::Selected
            }
        }
    }

    /// JSON filename for this dataset
    #[must_use]
    pub const fn filename(&self) -> &'static str {
        match self {
            Self::Selected => "ame2020_selected.json",
            Self::Full => "ame2020_full.json",
        }
    }

    /// Human-readable description
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Selected => "AME2020 selected (52 nuclei)",
            Self::Full => "AME2020 full (2,042 experimentally measured nuclei)",
        }
    }
}

/// Parse --nuclei=... from CLI args, defaulting to Selected
#[must_use]
pub fn parse_nuclei_set_from_args() -> NucleiSet {
    std::env::args()
        .find(|a| a.starts_with("--nuclei="))
        .map_or(NucleiSet::Selected, |a| NucleiSet::from_arg(&a[9..]))
}

/// Resolve the path to the nuclei JSON file for a given dataset
#[must_use]
pub fn nuclei_data_path(base_dir: &Path, set: NucleiSet) -> PathBuf {
    base_dir.join("exp_data").join(set.filename())
}

/// Load AME2020 experimental data → `HashMap`<(Z, N), (`B_exp`, σ)>
///
/// Uses streaming `from_reader` to avoid buffering the entire JSON file
/// in memory as an intermediate string.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or JSON deserialization fails.
pub fn load_experimental_data(
    path: &Path,
) -> Result<HashMap<(usize, usize), (f64, f64)>, Box<dyn std::error::Error>> {
    let reader = std::io::BufReader::new(std::fs::File::open(path)?);
    let file: NucleiFile = serde_json::from_reader(reader)?;
    let mut map = HashMap::with_capacity(file.nuclei.len());
    for nuc in file.nuclei {
        map.insert(
            (nuc.z, nuc.n),
            (nuc.binding_energy_mev, nuc.uncertainty_mev),
        );
    }
    Ok(map)
}

/// Load experimental data with dataset selection (convenience wrapper)
///
/// # Errors
///
/// Returns an error if the file cannot be opened or JSON deserialization fails.
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

/// Parameter ordering (must match Python).
///
/// Re-exported from [`crate::provenance::PARAM_NAMES`] — single source of truth.
pub use crate::provenance::PARAM_NAMES;

/// Load parameter bounds → Vec<(min, max)>
///
/// Uses streaming `from_reader` to avoid buffering the entire JSON file.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, JSON deserialization fails,
/// or a required parameter is missing from the bounds file.
pub fn load_bounds(path: &Path) -> Result<Vec<(f64, f64)>, Box<dyn std::error::Error>> {
    let reader = std::io::BufReader::new(std::fs::File::open(path)?);
    let file: BoundsFile = serde_json::from_reader(reader)?;
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
#[must_use]
pub struct EosContext {
    /// Base directory for nuclear EOS data files.
    pub base: PathBuf,
    /// Experimental binding energies: (Z, N) → (`B_exp`, σ).
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
/// # Errors
///
/// Returns [`HotSpringError::DataLoad`] if the data directory, experimental
/// data, or bounds file cannot be read.
pub fn load_eos_context() -> Result<EosContext, HotSpringError> {
    let base = crate::discovery::nuclear_eos_dir();

    let exp_data = std::sync::Arc::new(
        load_nuclei(&base, parse_nuclei_set_from_args())
            .map_err(|e| HotSpringError::DataLoad(format!("experimental data: {e}")))?,
    );
    let bounds = load_bounds(&base.join("wrapper/skyrme_bounds.json"))
        .map_err(|e| HotSpringError::DataLoad(format!("parameter bounds: {e}")))?;

    Ok(EosContext {
        base,
        exp_data,
        bounds,
    })
}

/// Compute per-datum χ² for a binding energy function.
///
/// For each nucleus in `exp_data`, evaluates `binding_energy_fn(z, n, params)`
/// and computes `((B_calc - B_exp) / sigma_theo)²`. Returns the mean χ²/datum.
///
/// Uses [`crate::tolerances::sigma_theo`] for the theoretical uncertainty.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use hotspring_barracuda::data::chi2_per_datum;
///
/// let mut exp = HashMap::new();
/// exp.insert((28, 28), (484.0, 0.1));  // Ni-56: (B_exp, sigma_exp)
/// let params: &[f64] = &[];
/// let chi2 = chi2_per_datum(params, &exp, |_z, _n, _| 484.0);
/// assert!(chi2 < 0.01);  // perfect fit
/// ```
pub fn chi2_per_datum<S: std::hash::BuildHasher>(
    params: &[f64],
    exp_data: &HashMap<(usize, usize), (f64, f64), S>,
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

/// Save JSON to a results subdirectory under the given base path.
///
/// Creates `{base}/results/` if needed, writes to `{base}/results/{filename}`.
/// Used by nuclear EOS binaries (L1, L2, L3) for engine-specific result files.
pub fn save_json_to_results(base: &Path, filename: &str, json: &serde_json::Value) {
    let results_dir = base.join("results");
    let _ = std::fs::create_dir_all(&results_dir);
    let path = results_dir.join(filename);
    if let Ok(s) = serde_json::to_string_pretty(json) {
        let _ = std::fs::write(&path, s);
        println!("\n  Results saved to: {}", path.display());
    }
}

/// Print an L2 result summary box (χ², evals, time, throughput).
///
/// Shared format across L2 heterogeneous, screen, and direct modes.
pub fn print_l2_result_box(
    title: &str,
    chi2: f64,
    log_chi2: f64,
    evals: usize,
    time_s: f64,
    throughput: f64,
) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  {title:<58} ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  χ²/datum:       {chi2:12.4}                              ║");
    println!("║  log(1+χ²):      {log_chi2:12.4}                              ║");
    println!("║  HFB evals:      {evals:6}                                    ║");
    println!("║  Total time:     {time_s:6.1}s                                   ║");
    println!("║  HFB throughput: {throughput:6.1} evals/s                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

/// Parse `--key=value` from CLI args as `usize`, returning `default` if missing or invalid.
#[must_use]
pub fn parse_cli_usize(args: &[String], key: &str, default: usize) -> usize {
    let prefix = format!("{key}=");
    args.iter()
        .find(|a| a.starts_with(&prefix))
        .and_then(|a| a.strip_prefix(&prefix)?.parse().ok())
        .unwrap_or(default)
}

/// Save a JSON result to the benchmark results directory.
///
/// Creates the output directory if needed, serializes `value` to pretty JSON,
/// and writes it to `{results_dir}/{filename}`. Returns the written path.
///
/// # Errors
///
/// Returns `HotSpringError::DataLoad` if directory creation or file writing fails.
pub fn save_results_json(
    filename: &str,
    value: &serde_json::Value,
) -> Result<std::path::PathBuf, HotSpringError> {
    let dir = crate::discovery::benchmark_results_dir()
        .map_err(|e| HotSpringError::DataLoad(format!("results dir: {e}")))?;
    let path = dir.join(filename);
    let json = serde_json::to_string_pretty(value)
        .map_err(|e| HotSpringError::DataLoad(format!("JSON serialize: {e}")))?;
    std::fs::write(&path, json)
        .map_err(|e| HotSpringError::DataLoad(format!("write {}: {e}", path.display())))?;
    Ok(path)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::provenance::SLY4_PARAMS;

    #[test]
    fn parse_nuclei_json_without_file() {
        #[derive(serde::Deserialize)]
        struct NucleiFile {
            nuclei: Vec<Nucleus>,
        }
        let json = r#"{"nuclei": [{"Z": 28, "N": 28, "A": 56, "element": "Ni", "binding_energy_MeV": 483.99, "uncertainty_MeV": 0.5}]}"#;
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
    fn nuclei_set_filenames() {
        assert!(NucleiSet::Selected
            .filename()
            .to_lowercase()
            .ends_with(".json"));
        assert!(NucleiSet::Full.filename().to_lowercase().ends_with(".json"));
        assert!(NucleiSet::Selected.filename().contains("selected"));
        assert!(NucleiSet::Full.filename().contains("full"));
    }

    #[test]
    fn nuclei_set_description_selected() {
        assert_eq!(
            NucleiSet::Selected.description(),
            "AME2020 selected (52 nuclei)"
        );
    }

    #[test]
    fn nuclei_set_description_full() {
        assert_eq!(
            NucleiSet::Full.description(),
            "AME2020 full (2,042 experimentally measured nuclei)"
        );
    }

    #[test]
    fn nuclei_data_path_construction() {
        let base = Path::new("/test/base");
        let path = nuclei_data_path(base, NucleiSet::Selected);
        assert!(path
            .to_str()
            .expect("path is valid UTF-8")
            .contains("exp_data"));
        assert!(path
            .to_str()
            .expect("path is valid UTF-8")
            .contains("ame2020_selected"));
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
            *exp.get(&(z, n)).map(|(b, _)| b).expect("exp has (z,n)")
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

    #[test]
    #[allow(clippy::float_cmp)] // exact known value (0.0)
    fn chi2_per_datum_empty_exp_returns_zero() {
        let exp: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
        let chi2 = chi2_per_datum(&crate::provenance::SLY4_PARAMS, &exp, |_z, _n, _| 100.0);
        assert_eq!(chi2, 0.0, "empty exp_data should return 0");
    }

    #[test]
    fn chi2_per_datum_single_point() {
        let mut exp = HashMap::new();
        exp.insert((28, 28), (484.0, 0.5));
        let chi2 = chi2_per_datum(&[], &exp, |z, n, _| {
            *exp.get(&(z, n)).map(|(b, _)| b).expect("exp has (z,n)")
        });
        assert!(chi2 < 1e-10, "single point perfect fit: chi2={chi2}");
    }

    #[test]
    fn load_experimental_data_missing_file_errors() {
        let path = std::path::Path::new("/nonexistent/ame2020_nonexistent.json");
        let result = load_experimental_data(path);
        assert!(result.is_err(), "missing file should error");
    }

    #[test]
    fn load_experimental_data_malformed_json_errors() {
        let temp = std::env::temp_dir().join("barracuda_test_malformed.json");
        std::fs::write(&temp, "{invalid json").expect("write temp file");
        let result = load_experimental_data(&temp);
        std::fs::remove_file(&temp).ok();
        assert!(result.is_err(), "malformed JSON should error");
    }

    #[test]
    fn load_bounds_missing_file_errors() {
        let path = std::path::Path::new("/nonexistent/skyrme_bounds.json");
        let result = load_bounds(path);
        assert!(result.is_err(), "missing bounds file should error");
    }

    #[test]
    fn load_bounds_malformed_json_errors() {
        let temp = std::env::temp_dir().join("barracuda_test_bounds_malformed.json");
        std::fs::write(&temp, "{broken}").expect("write temp file");
        let result = load_bounds(&temp);
        std::fs::remove_file(&temp).ok();
        assert!(result.is_err(), "malformed bounds JSON should error");
    }

    #[test]
    fn load_bounds_missing_parameter_errors() {
        let temp = std::env::temp_dir().join("barracuda_test_bounds_incomplete.json");
        let json = r#"{"parameters": {"t0": {"typical_range": [0, 1]}}}"#;
        std::fs::write(&temp, json).expect("write temp file");
        let result = load_bounds(&temp);
        std::fs::remove_file(&temp).ok();
        assert!(result.is_err(), "missing parameter should error");
    }

    #[test]
    fn load_experimental_data_empty_nuclei() {
        let temp = std::env::temp_dir().join("barracuda_test_empty_nuclei.json");
        std::fs::write(&temp, r#"{"nuclei": []}"#).expect("write temp file");
        let data = load_experimental_data(&temp).expect("empty nuclei should parse");
        std::fs::remove_file(&temp).ok();
        assert!(data.is_empty());
    }

    #[test]
    fn load_experimental_data_invalid_nucleus_missing_field() {
        let temp = std::env::temp_dir().join("barracuda_test_invalid_nucleus.json");
        let json = r#"{"nuclei": [{"Z": 1, "N": 0, "element": "H"}]}"#;
        std::fs::write(&temp, json).expect("write temp file");
        let result = load_experimental_data(&temp);
        std::fs::remove_file(&temp).ok();
        assert!(
            result.is_err(),
            "nucleus missing binding_energy_MeV should error"
        );
    }

    #[test]
    fn load_experimental_data_invalid_nucleus_wrong_type() {
        let temp = std::env::temp_dir().join("barracuda_test_nucleus_wrong_type.json");
        let json = r#"{"nuclei": [{"Z": "not_a_number", "N": 28, "A": 56, "element": "Ni", "binding_energy_MeV": 483.99, "uncertainty_MeV": 0.5}]}"#;
        std::fs::write(&temp, json).expect("write temp file");
        let result = load_experimental_data(&temp);
        std::fs::remove_file(&temp).ok();
        assert!(result.is_err());
    }

    #[test]
    fn load_bounds_empty_parameters_errors() {
        let temp = std::env::temp_dir().join("barracuda_test_bounds_empty.json");
        std::fs::write(&temp, r#"{"parameters": {}}"#).expect("write temp file");
        let result = load_bounds(&temp);
        std::fs::remove_file(&temp).ok();
        assert!(result.is_err());
    }

    #[test]
    fn nuclei_set_from_arg_case_insensitive() {
        assert_eq!(NucleiSet::from_arg("FULL"), NucleiSet::Full);
        assert_eq!(NucleiSet::from_arg("SELECTED"), NucleiSet::Selected);
    }

    #[test]
    fn nuclei_set_from_arg_empty_falls_back_to_selected() {
        assert_eq!(NucleiSet::from_arg(""), NucleiSet::Selected);
    }

    #[test]
    fn nuclei_data_path_full_set() {
        let base = Path::new("/data");
        let path = nuclei_data_path(base, NucleiSet::Full);
        assert!(path.to_str().unwrap().contains("ame2020_full"));
    }

    #[test]
    fn chi2_per_datum_uses_sigma_theo() {
        // sigma_theo scales with B_exp; verify chi2 scales correctly
        let mut exp = HashMap::new();
        exp.insert((28, 28), (484.0, 0.5));
        let sigma = crate::tolerances::sigma_theo(484.0);
        let chi2 = chi2_per_datum(&[], &exp, |_z, _n, _| 484.0 + sigma);
        assert!((chi2 - 1.0).abs() < 0.1, "one-sigma deviation → chi2≈1");
    }
}
