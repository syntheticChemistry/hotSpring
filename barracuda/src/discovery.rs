// SPDX-License-Identifier: AGPL-3.0-only

//! Capability-based data discovery for validation resources.
//!
//! Follows the ecoPrimals primal pattern: code has self-knowledge only and
//! discovers resources at runtime. No hardcoded absolute paths.
//!
//! # Discovery order
//!
//! 1. Environment variable (`HOTSPRING_DATA_ROOT`)
//! 2. `CARGO_MANIFEST_DIR` parent (development layout)
//! 3. Current working directory
//!
//! This replaces scattered `PathBuf::from(env!("CARGO_MANIFEST_DIR"))` calls
//! throughout validation binaries with a single, overridable discovery.

use std::path::{Path, PathBuf};

/// Well-known subdirectories within the hotSpring data root.
pub mod paths {
    /// Nuclear EOS control data (AME2020, Skyrme bounds)
    pub const NUCLEAR_EOS: &str = "control/surrogate/nuclear-eos";
    /// Experimental data subdirectory
    pub const EXP_DATA: &str = "exp_data";
    /// Skyrme parameter bounds
    pub const SKYRME_BOUNDS: &str = "wrapper/skyrme_bounds.json";
    /// Benchmark results output
    pub const BENCHMARK_RESULTS: &str = "benchmarks/nuclear-eos/results";
    /// Sarkas control data
    pub const SARKAS_CONTROL: &str = "control/sarkas";
    /// Surrogate control data
    pub const SURROGATE_CONTROL: &str = "control/surrogate";
}

/// Discover the hotSpring data root directory.
///
/// Checks, in order:
/// 1. `HOTSPRING_DATA_ROOT` environment variable
/// 2. Parent of `CARGO_MANIFEST_DIR` (standard development layout)
/// 3. Current working directory
///
/// Returns the first path that exists and contains a `control/` subdirectory.
#[must_use]
pub fn discover_data_root() -> PathBuf {
    // 1. Explicit environment override
    if let Ok(root) = std::env::var("HOTSPRING_DATA_ROOT") {
        let p = PathBuf::from(&root);
        if is_valid_root(&p) {
            return p;
        }
        eprintln!("  WARNING: HOTSPRING_DATA_ROOT={root} does not contain control/ — falling back");
    }

    // 2. CARGO_MANIFEST_DIR parent (development: barracuda/ → hotSpring/)
    let manifest_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest_parent = manifest_root.parent().map(Path::to_path_buf);
    if let Some(ref parent) = manifest_parent {
        if is_valid_root(parent) {
            return parent.clone();
        }
    }

    // 3. Current working directory
    if let Ok(cwd) = std::env::current_dir() {
        if is_valid_root(&cwd) {
            return cwd;
        }
    }

    // Last resort: use manifest parent anyway (will fail gracefully downstream)
    manifest_parent.unwrap_or(manifest_root)
}

/// Check if a directory looks like a valid hotSpring root.
fn is_valid_root(path: &Path) -> bool {
    path.join("control").is_dir()
}

/// Resolve the nuclear EOS data directory.
#[must_use]
pub fn nuclear_eos_dir() -> PathBuf {
    discover_data_root().join(paths::NUCLEAR_EOS)
}

/// Resolve the Skyrme bounds file path.
#[must_use]
pub fn skyrme_bounds_path() -> PathBuf {
    nuclear_eos_dir().join(paths::SKYRME_BOUNDS)
}

/// Resolve the benchmark results output directory, creating it if needed.
///
/// # Errors
///
/// Returns an error if the directory cannot be created.
pub fn benchmark_results_dir() -> std::io::Result<PathBuf> {
    let dir = discover_data_root().join(paths::BENCHMARK_RESULTS);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_finds_root() {
        let root = discover_data_root();
        // In development, this should find the hotSpring root
        assert!(root.exists(), "discovered root {root:?} should exist");
    }

    #[test]
    fn nuclear_eos_dir_is_reasonable() {
        let dir = nuclear_eos_dir();
        assert!(
            dir.to_str()
                .expect("path is valid UTF-8")
                .contains("nuclear-eos"),
            "nuclear EOS dir should contain 'nuclear-eos': {dir:?}"
        );
    }

    #[test]
    fn paths_are_not_empty() {
        assert!(!paths::NUCLEAR_EOS.is_empty());
        assert!(!paths::EXP_DATA.is_empty());
        assert!(!paths::SKYRME_BOUNDS.is_empty());
    }

    #[test]
    fn skyrme_bounds_path_contains_filename() {
        let p = skyrme_bounds_path();
        assert!(
            p.to_str()
                .expect("path is valid UTF-8")
                .contains("skyrme_bounds"),
            "path should contain skyrme_bounds: {p:?}"
        );
    }

    #[test]
    fn benchmark_results_dir_creates_or_exists() {
        let result = benchmark_results_dir();
        assert!(result.is_ok(), "benchmark_results_dir should succeed");
        let dir = result.expect("benchmark_results_dir returned Ok");
        assert!(dir.is_dir() || dir.parent().is_some_and(std::path::Path::is_dir));
        assert!(
            dir.to_str()
                .expect("path is valid UTF-8")
                .contains("benchmarks"),
            "path should contain benchmarks: {dir:?}"
        );
    }

    #[test]
    fn discover_data_root_with_invalid_env_falls_back() {
        // When HOTSPRING_DATA_ROOT points to dir without control/, we fall back.
        let bad_root = std::env::temp_dir().join("barracuda_no_control");
        std::fs::create_dir_all(&bad_root).ok();
        let prev = std::env::var("HOTSPRING_DATA_ROOT").ok();
        std::env::set_var("HOTSPRING_DATA_ROOT", bad_root.as_os_str());
        let root = discover_data_root();
        if let Some(p) = prev {
            std::env::set_var("HOTSPRING_DATA_ROOT", p);
        } else {
            std::env::remove_var("HOTSPRING_DATA_ROOT");
        }
        std::fs::remove_dir_all(&bad_root).ok();
        // Should have fallen back to manifest parent or cwd, not the invalid path
        assert!(
            root.join("control").is_dir(),
            "fallback root must have control/: {root:?}"
        );
    }

    #[test]
    fn paths_constants_sensible() {
        assert!(paths::NUCLEAR_EOS.contains("nuclear-eos"));
        assert!(paths::EXP_DATA.contains("exp"));
        assert!(paths::SKYRME_BOUNDS.to_ascii_lowercase().ends_with(".json"));
        assert!(paths::BENCHMARK_RESULTS.contains("benchmark"));
    }
}
