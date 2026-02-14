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
    pub binding_energy_MeV: f64,
    pub uncertainty_MeV: f64,
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
                eprintln!("  WARNING: Unknown nuclei set '{}', using selected (52)", s);
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
        .map(|a| NucleiSet::from_arg(&a[9..]))
        .unwrap_or(NucleiSet::Selected)
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
        map.insert((nuc.z, nuc.n), (nuc.binding_energy_MeV, nuc.uncertainty_MeV));
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
pub fn load_bounds(
    path: &Path,
) -> Result<Vec<(f64, f64)>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let file: BoundsFile = serde_json::from_str(&text)?;
    let mut bounds = Vec::with_capacity(10);
    for name in &PARAM_NAMES {
        let info = file
            .parameters
            .get(*name)
            .ok_or_else(|| format!("Missing parameter: {}", name))?;
        bounds.push((info.typical_range[0], info.typical_range[1]));
    }
    Ok(bounds)
}

