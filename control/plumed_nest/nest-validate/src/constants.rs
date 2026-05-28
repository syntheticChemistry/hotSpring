//! Runtime constants loaded from the pseudoSpore derivation file.
//!
//! This module implements the "Planck's constant" principle: every numeric
//! threshold used in pass/fail decisions lives in ONE canonical location
//! (derivations/threshold_calibration.toml) and is read at runtime.
//!
//! The code NEVER defines its own values for these constants. If the file
//! is missing, validation refuses to run (fail-closed, not fail-open).

use std::path::Path;

#[derive(Debug, Clone)]
pub struct GuideStoneConstants {
    // Cross-landscape detection
    pub cross_landscape_1d_kjmol: f64,
    pub cross_landscape_2d_kjmol: f64,

    // Binding distance
    pub binding_distance_max_nm: f64,

    // KS test
    pub ks_critical_coeff: f64,

    // Enzyme barrier
    pub enzyme_barrier_reduction_kjmol: f64,

    // Pipeline thresholds
    pub nucleus_readiness_fraction: f64,
    pub parity_rmsd_max_kjmol: f64,
    pub refresh_shift_max_kjmol: f64,
    pub fes_grid_margin_sigma: f64,
    pub alanine_minima_depth_kjmol: f64,
    pub alanine_top_n_minima: usize,
    pub opes_fold_above: f64,
    pub opes_unfold_below: f64,

    // Metadata validation
    pub pdb_resolution_tol: f64,
    pub metadata_temp_tol_k: f64,
    pub min_time_tol_ns: f64,
    pub time_relative_tol: f64,

    // Library constants
    pub basin_4c1_max_deg: f64,
    pub basin_1c4_min_deg: f64,
    pub wall_active_threshold: f64,
    pub verdict_midpoint: f64,
}

impl GuideStoneConstants {
    /// Load constants from the canonical derivation file inside a pseudoSpore directory.
    /// Returns Err if the file is missing or any required value is absent.
    pub fn load(guidestone_dir: &Path) -> Result<Self, String> {
        let derivation_path = guidestone_dir.join("derivations/threshold_calibration.toml");

        if !derivation_path.exists() {
            return Err(format!(
                "FATAL: derivations/threshold_calibration.toml not found in {}. \
                 Cannot validate without canonical constant definitions. \
                 This file is the single source of truth for all numeric thresholds.",
                guidestone_dir.display()
            ));
        }

        let content = std::fs::read_to_string(&derivation_path)
            .map_err(|e| format!("Failed to read {}: {}", derivation_path.display(), e))?;

        let doc: toml::Value = content.parse()
            .map_err(|e| format!("Failed to parse {}: {}", derivation_path.display(), e))?;

        let get_f64 = |path: &[&str]| -> Result<f64, String> {
            let mut v = &doc;
            for &key in path {
                v = v.get(key).ok_or_else(|| {
                    format!("Missing key '{}' in derivation path {:?}", key, path)
                })?;
            }
            v.as_float()
                .or_else(|| v.as_integer().map(|i| i as f64))
                .ok_or_else(|| format!("Key {:?} is not a number (got {:?})", path, v))
        };

        let get_usize = |path: &[&str]| -> Result<usize, String> {
            let mut v = &doc;
            for &key in path {
                v = v.get(key).ok_or_else(|| {
                    format!("Missing key '{}' in derivation path {:?}", key, path)
                })?;
            }
            v.as_integer()
                .map(|i| i as usize)
                .ok_or_else(|| format!("Key {:?} is not an integer", path))
        };

        Ok(Self {
            cross_landscape_1d_kjmol: get_f64(&["thresholds", "cross_landscape_1d", "value"])?,
            cross_landscape_2d_kjmol: get_f64(&["thresholds", "cross_landscape_2d", "value"])?,
            binding_distance_max_nm: get_f64(&["thresholds", "binding_distance_max", "value"])?,
            ks_critical_coeff: get_f64(&["library", "ks_critical_coefficient", "value"])?,
            enzyme_barrier_reduction_kjmol: get_f64(&["thresholds", "enzyme_barrier_reduction", "value"])?,
            nucleus_readiness_fraction: get_f64(&["pipeline", "nucleus_readiness", "value"])?,
            parity_rmsd_max_kjmol: get_f64(&["pipeline", "parity_rmsd_1d", "value"])?,
            refresh_shift_max_kjmol: get_f64(&["pipeline", "refresh_shift_boundary", "value"])?,
            fes_grid_margin_sigma: get_f64(&["pipeline", "fes_grid_margin", "value"])?,
            alanine_minima_depth_kjmol: get_f64(&["pipeline", "alanine_minima_depth", "value"])?,
            alanine_top_n_minima: get_usize(&["pipeline", "alanine_minima_count", "value"])?,
            opes_fold_above: get_f64(&["pipeline", "opes_fold_threshold", "value"])?,
            opes_unfold_below: get_f64(&["pipeline", "opes_unfold_threshold", "value"])?,
            pdb_resolution_tol: get_f64(&["pipeline", "pdb_resolution_tolerance", "value"])?,
            metadata_temp_tol_k: get_f64(&["pipeline", "metadata_temperature_tolerance", "value"])?,
            min_time_tol_ns: get_f64(&["pipeline", "simulation_time_tolerance_absolute", "value"])?,
            time_relative_tol: get_f64(&["pipeline", "simulation_time_tolerance_relative", "value"])?,
            basin_4c1_max_deg: get_f64(&["library", "basin_boundary_4c1", "value"])?,
            basin_1c4_min_deg: get_f64(&["library", "basin_boundary_1c4", "value"])?,
            wall_active_threshold: get_f64(&["library", "wall_active_threshold", "value"])?,
            verdict_midpoint: get_f64(&["library", "cross_landscape_verdict_midpoint", "expression_value"])
                .unwrap_or(0.5), // fallback: 0.5 is the only constant with a mathematical derivation (not empirical)
        })
    }

    /// Fallback defaults (ONLY for finalize, which is preparation not validation).
    /// guidestone_validate REFUSES to use these — it requires the derivation file.
    pub fn defaults() -> Self {
        Self {
            cross_landscape_1d_kjmol: 1.5,
            cross_landscape_2d_kjmol: 0.3,
            binding_distance_max_nm: 2.0,
            ks_critical_coeff: 1.36,
            enzyme_barrier_reduction_kjmol: 2.0,
            nucleus_readiness_fraction: 0.80,
            parity_rmsd_max_kjmol: 2.0,
            refresh_shift_max_kjmol: 3.0,
            fes_grid_margin_sigma: 3.0,
            alanine_minima_depth_kjmol: 5.0,
            alanine_top_n_minima: 3,
            opes_fold_above: 1.2,
            opes_unfold_below: 0.3,
            pdb_resolution_tol: 0.01,
            metadata_temp_tol_k: 0.1,
            min_time_tol_ns: 0.5,
            time_relative_tol: 0.05,
            basin_4c1_max_deg: 40.0,
            basin_1c4_min_deg: 140.0,
            wall_active_threshold: 0.001,
            verdict_midpoint: 0.5,
        }
    }

    /// Verify that compiled-in constants (used by helper functions) match
    /// the derivation file values. Any divergence is a FAIL — the single source
    /// of truth has been updated but the code hasn't been recompiled.
    pub fn verify_compiled_constants(&self) -> Vec<String> {
        let mut mismatches = Vec::new();

        macro_rules! check {
            ($field:ident, $compiled:expr, $name:expr) => {
                let delta = (self.$field - $compiled).abs();
                if delta > 1e-10 {
                    mismatches.push(format!(
                        "{}: derivation={} vs compiled={} (Δ={})",
                        $name, self.$field, $compiled, delta
                    ));
                }
            };
        }

        check!(pdb_resolution_tol, crate::PDB_RESOLUTION_TOL, "PDB_RESOLUTION_TOL");
        check!(metadata_temp_tol_k, crate::METADATA_TEMP_TOL, "METADATA_TEMP_TOL");
        check!(min_time_tol_ns, crate::MIN_TIME_TOL_NS, "MIN_TIME_TOL_NS");
        check!(time_relative_tol, crate::TIME_RELATIVE_TOL, "TIME_RELATIVE_TOL");

        mismatches
    }

    /// Print loaded constants for transparency
    pub fn print_summary(&self) {
        println!("  \x1b[36m┌─ Constants (from derivations/threshold_calibration.toml) ─┐\x1b[0m");
        println!("    cross_landscape_1d: {:.1} kJ/mol", self.cross_landscape_1d_kjmol);
        println!("    cross_landscape_2d: {:.1} kJ/mol", self.cross_landscape_2d_kjmol);
        println!("    binding_distance_max: {:.1} nm", self.binding_distance_max_nm);
        println!("    parity_rmsd_max: {:.1} kJ/mol", self.parity_rmsd_max_kjmol);
        println!("    basin_4c1_max: {:.0}°, basin_1c4_min: {:.0}°",
            self.basin_4c1_max_deg, self.basin_1c4_min_deg);
        println!("    ks_coeff: {:.2}, opes_fold: {:.1}/{:.1}",
            self.ks_critical_coeff, self.opes_fold_above, self.opes_unfold_below);
        println!("  \x1b[36m└────────────────────────────────────────────────────────────┘\x1b[0m");
    }
}
