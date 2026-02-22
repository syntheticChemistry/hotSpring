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
/// # Errors
///
/// Returns `HotSpringError::DataLoad` if no path with a `control/` directory
/// can be found via any discovery strategy.
pub fn try_discover_data_root() -> Result<PathBuf, crate::error::HotSpringError> {
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
    ("npu", "control/metalforge_npu"),
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

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    /// RAII guard for temporarily overriding an environment variable.
    ///
    /// Restores the previous value (or removes the variable) on drop.
    /// Uses `unsafe` env mutation — safe here because cargo test runs
    /// unit tests in a single process, and these tests are non-concurrent
    /// (no `#[test]` spawns threads that read `HOTSPRING_DATA_ROOT`).
    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &std::ffi::OsStr) -> Self {
            let prev = std::env::var(key).ok();
            // SAFETY: No concurrent threads read this env var during these tests.
            unsafe { std::env::set_var(key, value) };
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: Restoring the env var we changed — same thread, no concurrent readers.
            unsafe {
                if let Some(ref p) = self.prev {
                    std::env::set_var(self.key, p);
                } else {
                    std::env::remove_var(self.key);
                }
            }
        }
    }

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
    fn try_discover_data_root_err_has_data_load_message() {
        let bad_root = std::env::temp_dir().join("hotspring_no_control_test");
        std::fs::create_dir_all(&bad_root).ok();
        let _guard = EnvGuard::set("HOTSPRING_DATA_ROOT", bad_root.as_os_str());

        let result = try_discover_data_root();
        std::fs::remove_dir_all(&bad_root).ok();

        if let Err(e) = result {
            assert!(
                e.to_string().contains("Data loading failed"),
                "DataLoad error should mention data loading: {e}"
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
    fn discover_data_root_with_invalid_env_falls_back() {
        let bad_root = std::env::temp_dir().join("barracuda_no_control");
        std::fs::create_dir_all(&bad_root).ok();
        let guard = EnvGuard::set("HOTSPRING_DATA_ROOT", bad_root.as_os_str());
        let root = discover_data_root();
        drop(guard);
        std::fs::remove_dir_all(&bad_root).ok();
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

    #[test]
    #[allow(clippy::unwrap_used)]
    fn is_valid_root_rejects_dir_without_control() {
        let tmp = std::env::temp_dir().join("hotspring_no_control");
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(!is_valid_root(&tmp));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn is_valid_root_rejects_file() {
        let tmp = std::env::temp_dir().join("hotspring_file_not_dir");
        std::fs::write(&tmp, "x").unwrap();
        assert!(!is_valid_root(&tmp));
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn is_valid_root_accepts_dir_with_control() {
        let tmp = std::env::temp_dir().join("hotspring_valid_root");
        std::fs::create_dir_all(tmp.join("control")).unwrap();
        assert!(is_valid_root(&tmp));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn try_discover_with_valid_env_root() {
        let tmp = std::env::temp_dir().join("hotspring_valid_env_root");
        std::fs::create_dir_all(tmp.join("control")).unwrap();
        let guard = EnvGuard::set("HOTSPRING_DATA_ROOT", tmp.as_os_str());

        let result = try_discover_data_root();
        drop(guard);
        std::fs::remove_dir_all(&tmp).ok();

        let discovered = result.expect("valid env root should succeed");
        assert_eq!(discovered, tmp);
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn available_capabilities_detects_nuclear_eos_when_present() {
        let root = discover_data_root();
        let caps = available_capabilities();
        if root.join(paths::NUCLEAR_EOS).is_dir() {
            assert!(caps.contains(&"nuclear-eos"));
        }
    }
}
