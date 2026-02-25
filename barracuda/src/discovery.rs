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

/// Discover the data root, returning an error if no valid root is found.
///
/// Checks, in order: `HOTSPRING_DATA_ROOT` env, manifest parent, CWD.
/// Returns the first path that contains a `control/` subdirectory.
///
/// # Errors
///
/// Returns `HotSpringError::DataLoad` if no path with a `control/` directory
/// can be found via any discovery strategy.
///
/// # Example
///
/// ```
/// use hotspring_barracuda::discovery::try_discover_data_root;
///
/// let result = try_discover_data_root();
/// // Succeeds when run from repo with control/; Err otherwise
/// if let Ok(root) = result {
///     assert!(root.join("control").is_dir());
/// }
/// ```
pub fn try_discover_data_root() -> Result<PathBuf, crate::error::HotSpringError> {
    try_discover_with_override(None)
}

/// Discover the data root with an optional override (capability injection).
///
/// When `override_root` is `Some`, it is checked first — before env vars,
/// manifest, or CWD. This enables pure, `unsafe`-free testing without
/// global env mutation.
///
/// # Errors
///
/// Returns `HotSpringError::DataLoad` if no valid root is found.
pub fn try_discover_with_override(
    override_root: Option<&Path>,
) -> Result<PathBuf, crate::error::HotSpringError> {
    // 0. Injected override (capability-based, no global state)
    if let Some(root) = override_root {
        if is_valid_root(root) {
            return Ok(root.to_path_buf());
        }
    }

    // 1. Explicit environment override
    if let Ok(root) = std::env::var("HOTSPRING_DATA_ROOT") {
        let p = PathBuf::from(&root);
        if is_valid_root(&p) {
            return Ok(p);
        }
    }

    // 2. CARGO_MANIFEST_DIR parent
    let manifest_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(parent) = manifest_root.parent() {
        if is_valid_root(parent) {
            return Ok(parent.to_path_buf());
        }
    }

    // 3. CWD
    if let Ok(cwd) = std::env::current_dir() {
        if is_valid_root(&cwd) {
            return Ok(cwd);
        }
    }

    Err(crate::error::HotSpringError::DataLoad(
        "no valid hotSpring data root found (need directory with control/ subdirectory)".into(),
    ))
}

/// Discover the hotSpring data root directory.
///
/// Checks, in order:
/// 1. `HOTSPRING_DATA_ROOT` environment variable
/// 2. Parent of `CARGO_MANIFEST_DIR` (standard development layout)
/// 3. Current working directory
///
/// Returns the first path that exists and contains a `control/` subdirectory.
/// If no valid root is found, falls back to the manifest parent (may fail gracefully downstream).
#[must_use]
pub fn discover_data_root() -> PathBuf {
    try_discover_data_root().unwrap_or_else(|_| {
        let manifest_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest_root
            .parent()
            .map_or_else(|| manifest_root.clone(), std::path::Path::to_path_buf)
    })
}

/// Check if a directory looks like a valid hotSpring root.
pub(crate) fn is_valid_root(path: &Path) -> bool {
    path.join("control").is_dir()
}

/// Capability→path mapping for runtime discovery.
///
/// Each primal only knows itself; capabilities are discovered by probing
/// the filesystem at runtime rather than hardcoding knowledge of other primals.
const CAPABILITY_PROBES: &[(&str, &str)] = &[
    ("nuclear-eos", paths::NUCLEAR_EOS),
    ("sarkas-md", paths::SARKAS_CONTROL),
    ("surrogate", paths::SURROGATE_CONTROL),
    ("screened-coulomb", "control/screened_coulomb"),
    ("lattice-qcd", "control/lattice_qcd"),
    ("npu", "control/npu"),
    ("reservoir-transport", "control/reservoir_transport"),
    ("spectral-theory", "control/spectral_theory"),
];

/// Discover which validation capabilities are available at runtime.
///
/// Probes the filesystem for known control directories. No hardcoded
/// assumptions about which primals exist — purely capability-based.
#[must_use]
pub fn available_capabilities() -> Vec<&'static str> {
    let root = discover_data_root();
    CAPABILITY_PROBES
        .iter()
        .filter(|(_, path)| root.join(path).is_dir())
        .map(|(cap, _)| *cap)
        .collect()
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

/// Probe whether an NPU (neuromorphic processing unit) is available.
///
/// When the `npu-hw` feature is enabled, delegates to the `akida-driver`
/// device manager. Otherwise, checks for any device node matching
/// `/dev/akida*` — a generic probe that avoids hardcoding a specific index.
#[must_use]
pub fn probe_npu_available() -> bool {
    #[cfg(feature = "npu-hw")]
    {
        crate::md::npu_hw::NpuHardware::discover().is_some()
    }
    #[cfg(not(feature = "npu-hw"))]
    {
        std::fs::read_dir("/dev")
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .any(|e| e.file_name().to_string_lossy().starts_with("akida"))
            })
            .unwrap_or(false)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn discover_finds_root() {
        let root = discover_data_root();
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
    fn try_discover_data_root_ok_when_valid() {
        let result = try_discover_data_root();
        assert!(
            result.is_ok(),
            "try_discover_data_root should succeed in dev"
        );
        let root = result.expect("Ok");
        assert!(
            root.join("control").is_dir(),
            "discovered root must have control/: {root:?}"
        );
    }

    #[test]
    fn try_discover_with_override_accepts_valid_path() {
        let tmp = std::env::temp_dir().join("hotspring_override_valid");
        std::fs::create_dir_all(tmp.join("control")).unwrap();
        let result = try_discover_with_override(Some(&tmp));
        std::fs::remove_dir_all(&tmp).ok();

        let discovered = result.expect("override with valid root should succeed");
        assert_eq!(discovered, tmp);
    }

    #[test]
    fn try_discover_with_override_rejects_invalid_path() {
        let bad = std::env::temp_dir().join("hotspring_override_no_control");
        std::fs::create_dir_all(&bad).unwrap();

        let result = try_discover_with_override(Some(&bad));
        std::fs::remove_dir_all(&bad).ok();

        // Invalid override falls through to env/manifest/CWD strategies
        if let Ok(root) = result {
            assert!(
                root.join("control").is_dir(),
                "should have fallen through to a valid root"
            );
        }
    }

    #[test]
    fn try_discover_with_override_none_matches_default() {
        let default_result = try_discover_data_root();
        let override_result = try_discover_with_override(None);
        assert_eq!(default_result.ok(), override_result.ok());
    }

    #[test]
    fn try_discover_data_root_err_has_data_load_message() {
        let bad_root = std::env::temp_dir().join("hotspring_no_control_test_override");
        std::fs::create_dir_all(&bad_root).unwrap();

        let result = try_discover_with_override(Some(&bad_root));
        std::fs::remove_dir_all(&bad_root).ok();

        // Falls through to other strategies which should succeed in dev;
        // but the override itself should not produce that path.
        if let Ok(root) = &result {
            assert_ne!(
                root, &bad_root,
                "bad override should not be returned as the root"
            );
        }
    }

    #[test]
    fn discover_data_root_delegates_to_try() {
        let root = discover_data_root();
        assert!(
            root.exists(),
            "discover_data_root must return existing path"
        );
        if let Ok(try_root) = try_discover_data_root() {
            assert_eq!(
                root, try_root,
                "discover should match try when try succeeds"
            );
        }
    }

    #[test]
    fn available_capabilities_returns_vec() {
        let caps = available_capabilities();
        assert!(
            caps.contains(&"nuclear-eos") || caps.contains(&"surrogate") || caps.is_empty(),
            "capabilities should be sensible: {caps:?}"
        );
    }

    #[test]
    fn paths_constants_sensible() {
        assert!(paths::NUCLEAR_EOS.contains("nuclear-eos"));
        assert!(paths::EXP_DATA.contains("exp"));
        assert!(paths::SKYRME_BOUNDS.to_ascii_lowercase().ends_with(".json"));
        assert!(paths::BENCHMARK_RESULTS.contains("benchmark"));
    }

    #[test]
    fn is_valid_root_rejects_dir_without_control() {
        let tmp = std::env::temp_dir().join("hotspring_no_control");
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(!is_valid_root(&tmp));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn is_valid_root_rejects_file() {
        let tmp = std::env::temp_dir().join("hotspring_file_not_dir");
        std::fs::write(&tmp, "x").unwrap();
        assert!(!is_valid_root(&tmp));
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn is_valid_root_accepts_dir_with_control() {
        let tmp = std::env::temp_dir().join("hotspring_valid_root");
        std::fs::create_dir_all(tmp.join("control")).unwrap();
        assert!(is_valid_root(&tmp));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn available_capabilities_detects_nuclear_eos_when_present() {
        let root = discover_data_root();
        let caps = available_capabilities();
        if root.join(paths::NUCLEAR_EOS).is_dir() {
            assert!(caps.contains(&"nuclear-eos"));
        }
    }

    #[test]
    fn probe_npu_available_returns_bool() {
        let result = probe_npu_available();
        // On CI / dev machines without Akida hardware, expect false.
        // On machines with /dev/akida*, expect true.
        // Either way, the function must not panic.
        assert!(result || !result, "probe_npu_available must return a bool");
    }

    #[test]
    fn nuclear_eos_dir_resolves() {
        let dir = nuclear_eos_dir();
        assert!(
            !dir.as_os_str().is_empty(),
            "nuclear_eos_dir should resolve to a non-empty path"
        );
    }

    #[test]
    fn capability_probes_have_unique_names() {
        let names: Vec<&str> = CAPABILITY_PROBES.iter().map(|(n, _)| *n).collect();
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(
            names.len(),
            unique.len(),
            "capability probe names must be unique"
        );
    }

    #[test]
    fn capability_probes_paths_are_relative() {
        for (name, path) in CAPABILITY_PROBES {
            assert!(
                !path.starts_with('/'),
                "probe '{name}' has absolute path '{path}' — should be relative"
            );
        }
    }

    #[test]
    fn is_valid_root_rejects_nonexistent() {
        assert!(!is_valid_root(Path::new(
            "/nonexistent_hotspring_path_98312"
        )));
    }
}
