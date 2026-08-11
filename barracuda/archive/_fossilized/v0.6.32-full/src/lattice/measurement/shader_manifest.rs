// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shader provenance manifests and cross-path validation receipts.

use serde::{Deserialize, Serialize};

use super::time_host::iso8601_now;

/// Shader provenance manifest — guideStone-grade metadata for a validated shader.
///
/// Every validated WGSL shader carries a manifest that documents its origin,
/// paper reference, precision tier, and cross-path validation status. This is
/// the shader equivalent of the guideStone receipt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShaderManifest {
    /// Schema version.
    pub schema_version: String,
    /// Shader name (e.g. `"rk4_integrator"`, `"wilson_dslash"`).
    pub name: String,
    /// Semantic version of this shader.
    pub version: String,
    /// Author (person or spring that wrote it).
    pub author: String,
    /// Paper reference in the same format as guideStone check citations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paper_ref: Option<PaperReference>,
    /// Precision tier: `"f32"`, `"df64"`, or `"f64"`.
    pub precision_tier: String,
    /// WGSL workgroup size (e.g. `[256, 1, 1]`).
    pub workgroup_size: [u32; 3],
    /// Input buffer layouts (names and element types).
    pub inputs: Vec<ShaderBufferLayout>,
    /// Output buffer layouts.
    pub outputs: Vec<ShaderBufferLayout>,
    /// Reference values for small-lattice validation (known results).
    pub reference_values: Vec<ShaderReferenceValue>,
    /// Cross-path validation results (CPU, NagaExec, GPU, JIT agreement).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation: Option<ShaderValidationResult>,
    /// ISO 8601 timestamp of manifest creation.
    pub created: String,
    /// Origin spring (e.g. `"hotSpring"`, `"wetSpring"`).
    pub origin_spring: String,
}

/// Paper reference for a shader's mathematical basis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaperReference {
    /// arXiv ID or DOI.
    pub citation: String,
    /// Specific equation or section referenced.
    pub equation: String,
    /// Short description of what is implemented.
    pub description: String,
}

/// Buffer layout for a shader input or output.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShaderBufferLayout {
    /// Binding group index.
    pub group: u32,
    /// Binding index within the group.
    pub binding: u32,
    /// Human-readable name (e.g. `"gauge_links"`, `"result"`).
    pub name: String,
    /// Element type (e.g. `"f32"`, `"vec4<f32>"`, `"array<f64, N>`).
    pub element_type: String,
}

/// Known reference value for shader validation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShaderReferenceValue {
    /// Test case name (e.g. `"unit_gauge_plaquette"`).
    pub name: String,
    /// Expected output value.
    pub expected: f64,
    /// Tolerance for this test case.
    pub tolerance: f64,
    /// Tolerance justification.
    pub justification: String,
}

/// Cross-path validation result for a shader.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShaderValidationResult {
    /// CPU reference path (Rust f64) — always available.
    pub cpu_reference: PathValidation,
    /// NagaExecutor (CPU shader interpreter) — available via barraCuda.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub naga_executor: Option<PathValidation>,
    /// GPU dispatch via wgpu — available when f64 GPU present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_wgpu: Option<PathValidation>,
    /// coralReef JIT (Cranelift) — available when coralReef present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coral_jit: Option<PathValidation>,
    /// Maximum delta across all path pairs.
    pub max_cross_path_delta: f64,
    /// Whether all paths agree within manifest tolerances.
    pub all_paths_agree: bool,
}

/// Validation result for a single execution path.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PathValidation {
    /// Path name.
    pub path: String,
    /// Whether this path was executed.
    pub executed: bool,
    /// All reference values matched within tolerance.
    pub passed: bool,
    /// Maximum absolute delta from reference.
    pub max_delta: f64,
    /// Wall time for the validation run (seconds).
    pub wall_seconds: f64,
}

impl ShaderManifest {
    /// Create a new manifest with minimal required fields.
    pub fn new(name: &str, version: &str, author: &str, precision_tier: &str) -> Self {
        Self {
            schema_version: "1.0".to_string(),
            name: name.to_string(),
            version: version.to_string(),
            author: author.to_string(),
            paper_ref: None,
            precision_tier: precision_tier.to_string(),
            workgroup_size: [256, 1, 1],
            inputs: Vec::new(),
            outputs: Vec::new(),
            reference_values: Vec::new(),
            validation: None,
            created: iso8601_now(),
            origin_spring: "hotSpring".to_string(),
        }
    }
}
