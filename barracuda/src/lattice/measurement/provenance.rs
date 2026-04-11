// SPDX-License-Identifier: AGPL-3.0-or-later

//! Run manifests, NUCLEUS metadata, and implementation auto-detection.

use serde::{Deserialize, Serialize};

use super::time_host::{hostname_best_effort, iso8601_now};

/// Implementation provenance for a measurement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImplementationInfo {
    /// Code name.
    pub code_name: String,
    /// Code version.
    pub code_version: String,
    /// Machine name.
    pub machine: String,
    /// Machine institution.
    pub institution: String,
    /// Machine type (e.g. "CPU workstation", "GPU workstation", "HPC cluster").
    pub machine_type: String,
    /// Git commit hash (if available at build or run time).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
    /// GPU adapter names discovered on this machine.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpus: Option<Vec<String>>,
}

impl ImplementationInfo {
    /// Auto-detect implementation info from the current environment.
    ///
    /// Enumerates GPU adapters via wgpu if available, captures git commit
    /// from `GIT_COMMIT` env var or build-time embedding.
    pub fn auto_detect() -> Self {
        let gpu_adapters = crate::gpu::GpuF64::enumerate_adapters();
        let gpu_names: Vec<String> = gpu_adapters
            .iter()
            .filter(|a| a.has_f64)
            .map(|a| a.name.clone())
            .collect();
        let has_gpu = !gpu_names.is_empty();
        Self {
            code_name: "hotSpring-barracuda".to_string(),
            code_version: format!("hotSpring-barracuda {}", env!("CARGO_PKG_VERSION")),
            machine: hostname_best_effort(),
            institution: "ecoPrimals".to_string(),
            machine_type: if has_gpu {
                "GPU workstation".to_string()
            } else {
                "CPU workstation".to_string()
            },
            git_commit: std::env::var("GIT_COMMIT")
                .ok()
                .or_else(|| option_env!("GIT_COMMIT").map(String::from)),
            gpus: if gpu_names.is_empty() {
                None
            } else {
                Some(gpu_names)
            },
        }
    }

    /// Lightweight detection without GPU enumeration (for binaries that
    /// don't need wgpu startup overhead).
    pub fn auto_detect_cpu_only() -> Self {
        Self {
            code_name: "hotSpring-barracuda".to_string(),
            code_version: format!("hotSpring-barracuda {}", env!("CARGO_PKG_VERSION")),
            machine: hostname_best_effort(),
            institution: "ecoPrimals".to_string(),
            machine_type: "CPU workstation".to_string(),
            git_commit: std::env::var("GIT_COMMIT")
                .ok()
                .or_else(|| option_env!("GIT_COMMIT").map(String::from)),
            gpus: None,
        }
    }
}

/// NUCLEUS layer metadata embedded in the run manifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NucleusManifest {
    /// Which primals were detected at runtime.
    pub primals_detected: Vec<String>,
    /// rhizoCrypt DAG session ID (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dag_session: Option<String>,
    /// Merkle root of the computation DAG (if dehydrated).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merkle_root: Option<String>,
    /// bearDog Ed25519 signature hex (if receipt was signed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// NUCLEUS family ID used for socket discovery.
    pub family_id: String,
}

/// Run manifest — captures everything needed to reproduce or compare a run.
///
/// Every chuna binary populates this at startup via `RunManifest::capture()`
/// and embeds it in its JSON output as a `"run"` key. This is the receipt
/// header that makes every output self-documenting.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunManifest {
    /// Schema version for the manifest format.
    pub schema_version: String,
    /// Binary name that produced this output.
    pub binary: String,
    /// Engine version (CARGO_PKG_VERSION).
    pub engine_version: String,
    /// ISO 8601 UTC timestamp of when the run started.
    pub timestamp: String,
    /// Hostname of the machine.
    pub hostname: String,
    /// CPU architecture.
    pub arch: String,
    /// Operating system.
    pub os: String,
    /// Full CLI invocation (argv) for reproducibility.
    pub argv: Vec<String>,
    /// Git commit hash (from GIT_COMMIT env var or build-time embedding).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
    /// GPU adapter name (if GPU was used or discovered).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    /// NUCLEUS layer metadata (present when primals are detected).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nucleus: Option<NucleusManifest>,
}

impl RunManifest {
    /// Capture run metadata from the current environment.
    ///
    /// Call once at the start of main() with the binary name. All fields
    /// are populated automatically from the environment — no arguments needed.
    pub fn capture(binary: &str) -> Self {
        Self {
            schema_version: "1.0".to_string(),
            binary: binary.to_string(),
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: iso8601_now(),
            hostname: hostname_best_effort(),
            arch: std::env::consts::ARCH.to_string(),
            os: std::env::consts::OS.to_string(),
            argv: std::env::args().collect(),
            git_commit: std::env::var("GIT_COMMIT")
                .ok()
                .or_else(|| option_env!("GIT_COMMIT").map(String::from)),
            gpu: None,
            nucleus: None,
        }
    }

    /// Set the GPU adapter name after discovery.
    pub fn with_gpu(mut self, name: &str) -> Self {
        self.gpu = Some(name.to_string());
        self
    }

    /// Attach NUCLEUS metadata from a detected context.
    pub fn with_nucleus(mut self, ctx: &crate::primal_bridge::NucleusContext) -> Self {
        let names = ctx
            .alive_names()
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        self.nucleus = Some(NucleusManifest {
            primals_detected: names,
            dag_session: None,
            merkle_root: None,
            signature: None,
            family_id: ctx.family_id.clone(),
        });
        self
    }

    /// Update NUCLEUS manifest with DAG provenance after dehydration.
    pub fn set_dag_provenance(&mut self, prov: &crate::dag_provenance::DagProvenance) {
        if let Some(ref mut n) = self.nucleus {
            n.dag_session = Some(prov.dag_session_id.clone());
            n.merkle_root = Some(prov.merkle_root.clone());
        }
    }

    /// Update NUCLEUS manifest with signature after signing.
    pub fn set_signature(&mut self, sig_hex: &str) {
        if let Some(ref mut n) = self.nucleus {
            n.signature = Some(sig_hex.to_string());
        }
    }

    /// Serialize to a JSON string for embedding in hand-built JSON.
    pub fn to_json_value(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
    }
}
