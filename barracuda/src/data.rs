//! Experimental data and parameter bounds loading

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

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

