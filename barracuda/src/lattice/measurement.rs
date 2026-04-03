// SPDX-License-Identifier: AGPL-3.0-only

//! Formalized measurement output schema for MILC/Bazavov ecosystem.
//!
//! Defines structured JSON schemas for gauge ensemble manifests and
//! per-configuration measurements. These "receipts" pair with ILDG binary
//! configs to produce complete, documented datasets.
//!
//! # Schema overview
//!
//! ```text
//! EnsembleManifest (one per ensemble)
//!   ├── action parameters (beta, mass, Nf, action type)
//!   ├── hardware info (GPU, CPU, engine version)
//!   ├── algorithm parameters (integrator, dt, n_md, CG tol)
//!   └── configs[] → list of ConfigEntry (trajectory, filename, checksum)
//!
//! ConfigMeasurement (one per measured configuration)
//!   ├── config identity (trajectory, ensemble_id, ildg_lfn)
//!   ├── gauge observables (plaquette, Polyakov, action density)
//!   ├── flow results (t0, w0, flow_curves[])
//!   ├── topology (Q, Q_density_rms)
//!   ├── wilson_loops (W(R,T) grid)
//!   ├── fermion observables (chiral condensate, correlators)
//!   └── diagnostics (acceptance, delta_h, CG iters, wall time)
//! ```

use serde::{Deserialize, Serialize};

/// Ensemble-level manifest — one per production run.
///
/// Written alongside the ILDG config files as `ensemble.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnsembleManifest {
    /// Schema version for forward compatibility.
    pub schema_version: String,
    /// Unique ensemble identifier (matches ILDG metadata).
    pub ensemble_id: String,
    /// ISO 8601 timestamp of ensemble creation.
    pub created: String,

    // Action parameters
    /// Gauge action type ("Wilson", "Symanzik", etc.).
    pub gauge_action: String,
    /// Fermion action ("none", "staggered", "HISQ").
    pub fermion_action: String,
    /// Inverse bare coupling.
    pub beta: f64,
    /// Quark mass (0.0 for quenched).
    pub mass: f64,
    /// Number of dynamical flavors.
    pub nf: usize,
    /// Lattice dimensions `[Nx, Ny, Nz, Nt]`.
    pub dims: [usize; 4],

    /// Hardware and software provenance.
    pub provenance: Provenance,
    /// Algorithm parameters used for generation.
    pub algorithm: AlgorithmParams,
    /// Run manifest with full invocation metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<RunManifest>,
    /// List of configurations in this ensemble.
    pub configs: Vec<ConfigEntry>,
}

/// Hardware + software provenance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Provenance {
    /// Engine name and version.
    pub engine: String,
    /// GPU adapter name (if GPU was used).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    /// CPU architecture.
    pub arch: String,
    /// Operating system.
    pub os: String,
    /// Hostname.
    pub hostname: String,
}

/// HMC/RHMC algorithm parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlgorithmParams {
    /// Algorithm type ("HMC", "RHMC").
    pub algorithm: String,
    /// Integrator type ("Leapfrog", "Omelyan").
    pub integrator: String,
    /// MD step size.
    pub dt: f64,
    /// Number of MD steps per trajectory.
    pub n_md_steps: usize,
    /// Number of thermalization trajectories.
    pub n_therm: usize,
    /// Measurement interval (trajectories between saved configs).
    pub meas_interval: usize,
    /// CG solver tolerance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cg_tol: Option<f64>,
    /// Random seed.
    pub seed: u64,
}

/// Entry for a single configuration within an ensemble.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfigEntry {
    /// HMC trajectory index.
    pub trajectory: usize,
    /// Filename of the ILDG binary file.
    pub filename: String,
    /// ILDG logical file name.
    pub ildg_lfn: String,
    /// CRC32 checksum of the ILDG file (Ethernet CRC32, legacy).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum_crc32: Option<String>,
    /// ILDG-standard CRC (POSIX cksum / GNU cksum algorithm).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum_ildg_crc: Option<u32>,
    /// Average plaquette at save time.
    pub plaquette: f64,
}

/// Per-configuration measurement results.
///
/// Written as `measurements/conf_NNNNNN.json` alongside the ILDG files.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfigMeasurement {
    /// Schema version.
    pub schema_version: String,
    /// Ensemble this config belongs to.
    pub ensemble_id: String,
    /// ILDG logical file name.
    pub ildg_lfn: String,
    /// HMC trajectory index.
    pub trajectory: usize,

    /// Gauge observables.
    pub gauge: GaugeObservables,
    /// Gradient flow results (if measured).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flow: Option<FlowResults>,
    /// Topological charge (if measured).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topology: Option<TopologyResults>,
    /// Wilson loop measurements (if measured).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wilson_loops: Option<Vec<WilsonLoopEntry>>,
    /// Fermion observables (if measured).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fermion: Option<FermionObservables>,
    /// HMC diagnostics for this trajectory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<HmcDiagnostics>,
    /// Implementation provenance for this measurement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub implementation: Option<ImplementationInfo>,
    /// Run manifest with full invocation metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<RunManifest>,
    /// Scale-setting context (lattice spacing in physical units).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale_setting: Option<ScaleSetting>,
    /// HVP (hadronic vacuum polarization) correlator and integral.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hvp: Option<HvpResults>,

    /// Wall-clock time for all measurements on this config (seconds).
    pub wall_seconds: f64,
}

/// Basic gauge observables measured on every configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaugeObservables {
    /// Average plaquette.
    pub plaquette: f64,
    /// Polyakov loop magnitude (spatial average).
    pub polyakov_abs: f64,
    /// Polyakov loop (Re, Im) spatial average.
    pub polyakov_re: f64,
    /// Polyakov loop imaginary part.
    pub polyakov_im: f64,
    /// Wilson action density = 6(1 - P).
    pub action_density: f64,
}

/// Gradient flow measurement results.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlowResults {
    /// Flow integrator used ("Rk3Luscher", "Lscfrk3w7", "Lscfrk4ck").
    pub integrator: String,
    /// Flow step size epsilon.
    pub epsilon: f64,
    /// Maximum flow time.
    pub t_max: f64,
    /// Scale t0 (from t^2 E(t) = 0.3), or null if not found.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub t0: Option<f64>,
    /// Scale w0, or null if not found.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub w0: Option<f64>,
    /// Flow curve: list of (t, E(t), t^2 E(t)) measurements.
    pub flow_curve: Vec<FlowPoint>,
}

/// Single point on the gradient flow curve.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlowPoint {
    /// Flow time.
    pub t: f64,
    /// Energy density E(t).
    pub energy_density: f64,
    /// t^2 E(t).
    pub t2_e: f64,
}

/// Topological charge measurement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopologyResults {
    /// Topological charge Q (measured on flowed config).
    pub charge: f64,
    /// Flow time at which Q was measured.
    pub flow_time: f64,
}

/// Single Wilson loop measurement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WilsonLoopEntry {
    /// Spatial extent R.
    pub r: usize,
    /// Temporal extent T.
    pub t: usize,
    /// Average (1/3) Re Tr W(R,T).
    pub value: f64,
}

/// Fermion observables.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FermionObservables {
    /// Chiral condensate <psi-bar psi>.
    pub chiral_condensate: f64,
    /// Statistical error on condensate.
    pub chiral_condensate_error: f64,
    /// Number of stochastic sources.
    pub n_sources: usize,
    /// Mass used for the measurement.
    pub mass: f64,
}

/// HVP (hadronic vacuum polarization) correlator and integral.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HvpResults {
    /// Quark mass used for the propagator inversion.
    pub mass: f64,
    /// CG solver residual achieved.
    pub cg_residual: f64,
    /// Number of CG iterations.
    pub cg_iterations: usize,
    /// Time-slice correlator C(t) = Σ_x |G(x,t)|².
    pub correlator: Vec<f64>,
    /// Lattice-units HVP integral: a_μ^HVP ∝ Σ_t K(t) C(t).
    pub hvp_integral: f64,
}

/// HMC trajectory diagnostics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HmcDiagnostics {
    /// Whether this trajectory was accepted.
    pub accepted: bool,
    /// Hamiltonian violation delta_H.
    pub delta_h: f64,
    /// Total CG iterations (for dynamical fermions).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cg_iterations: Option<usize>,
    /// Wall time for this trajectory in seconds.
    pub trajectory_seconds: f64,
}

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
        let gpu_names: Vec<String> = gpu_adapters.iter()
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
            gpus: if gpu_names.is_empty() { None } else { Some(gpu_names) },
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
        let names = ctx.alive_names().iter().map(|s| s.to_string()).collect();
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

/// Scale-setting context: lattice spacing in physical units.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScaleSetting {
    /// Lattice spacing `a` in fm (from t0 or w0 scale).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub a_fm: Option<f64>,
    /// Method used for scale setting ("t0", "w0", "r0", "string_tension").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// Physical value of the reference scale used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_value: Option<f64>,
    /// Reference: paper or value used (e.g. "BMW 2012: sqrt(t0) = 0.1465 fm").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
}

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

/// Statistical analysis metadata for ensemble-level observables.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    /// Number of configurations in the analysis.
    pub n_configs: usize,
    /// Integrated autocorrelation time (in units of measurement interval).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tau_int: Option<f64>,
    /// Error on tau_int.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tau_int_error: Option<f64>,
    /// Error estimation method ("jackknife", "bootstrap", "binning", "autocorr").
    pub error_method: String,
    /// Number of jackknife/bootstrap samples.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_samples: Option<usize>,
    /// Bin size used (for binning analysis).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bin_size: Option<usize>,
}

/// Ensemble-level summary of an observable with statistics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservableSummary {
    /// Observable name (e.g. "plaquette", "t0", "w0", "Q", "pbp").
    pub name: String,
    /// Central value (mean).
    pub value: f64,
    /// Statistical error.
    pub error: f64,
    /// Statistical analysis metadata.
    pub analysis: StatisticalAnalysis,
}

/// Compute jackknife error estimate for a set of measurements.
///
/// Returns (mean, jackknife_error).
pub fn jackknife_error(data: &[f64]) -> (f64, f64) {
    let n = data.len();
    if n < 2 {
        return (data.first().copied().unwrap_or(0.0), 0.0);
    }

    let total: f64 = data.iter().sum();
    let mean = total / n as f64;

    let mut jk_means = Vec::with_capacity(n);
    for i in 0..n {
        let jk_sum = total - data[i];
        jk_means.push(jk_sum / (n - 1) as f64);
    }

    let jk_mean: f64 = jk_means.iter().sum::<f64>() / n as f64;
    let jk_var: f64 = jk_means
        .iter()
        .map(|&jk| (jk - jk_mean).powi(2))
        .sum::<f64>()
        * (n - 1) as f64
        / n as f64;

    (mean, jk_var.sqrt())
}

/// Estimate integrated autocorrelation time using binning analysis.
///
/// Returns (tau_int, tau_int_error) where tau_int is in units of
/// the measurement separation.
pub fn estimate_tau_int(data: &[f64]) -> (f64, f64) {
    let n = data.len();
    if n < 10 {
        return (1.0, 0.0);
    }

    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let naive_var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    if naive_var < 1e-30 {
        return (1.0, 0.0);
    }

    let mut best_tau = 1.0;
    let mut best_tau_err = 0.0;

    // Try increasing bin sizes
    let max_bin = (n / 4).max(2);
    for bin_size in 1..=max_bin {
        let n_bins = n / bin_size;
        if n_bins < 4 {
            break;
        }

        let mut bin_means = Vec::with_capacity(n_bins);
        for b in 0..n_bins {
            let start = b * bin_size;
            let end = start + bin_size;
            let bin_mean: f64 = data[start..end].iter().sum::<f64>() / bin_size as f64;
            bin_means.push(bin_mean);
        }

        let bin_var: f64 = bin_means
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (n_bins - 1) as f64;

        let tau = 0.5 * bin_size as f64 * bin_var / naive_var;
        let tau_err = tau * (2.0 / n_bins as f64).sqrt();

        if tau > best_tau {
            best_tau = tau;
            best_tau_err = tau_err;
        }
    }

    (best_tau, best_tau_err)
}

impl EnsembleManifest {
    /// Create a new manifest with defaults filled in.
    pub fn new(ensemble_id: &str, dims: [usize; 4], beta: f64) -> Self {
        Self {
            schema_version: "1.0".to_string(),
            ensemble_id: ensemble_id.to_string(),
            created: iso8601_now(),
            gauge_action: "Wilson".to_string(),
            fermion_action: "none".to_string(),
            beta,
            mass: 0.0,
            nf: 0,
            dims,
            provenance: Provenance {
                engine: format!("hotSpring-barracuda {}", env!("CARGO_PKG_VERSION")),
                gpu: None,
                arch: std::env::consts::ARCH.to_string(),
                os: std::env::consts::OS.to_string(),
                hostname: hostname_best_effort(),
            },
            algorithm: AlgorithmParams {
                algorithm: "HMC".to_string(),
                integrator: "Omelyan".to_string(),
                dt: 0.01,
                n_md_steps: 20,
                n_therm: 200,
                meas_interval: 10,
                cg_tol: None,
                seed: 42,
            },
            run: None,
            configs: Vec::new(),
        }
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
    }
}

impl ConfigMeasurement {
    /// Create a minimal measurement record.
    pub fn new(ensemble_id: &str, trajectory: usize, lfn: &str) -> Self {
        Self {
            schema_version: "1.0".to_string(),
            ensemble_id: ensemble_id.to_string(),
            ildg_lfn: lfn.to_string(),
            trajectory,
            gauge: GaugeObservables {
                plaquette: 0.0,
                polyakov_abs: 0.0,
                polyakov_re: 0.0,
                polyakov_im: 0.0,
                action_density: 0.0,
            },
            flow: None,
            topology: None,
            wilson_loops: None,
            fermion: None,
            diagnostics: None,
            implementation: None,
            run: None,
            scale_setting: None,
            hvp: None,
            wall_seconds: 0.0,
        }
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
    }
}

pub fn iso8601_now() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86400) as i64;
    let time_of_day = secs % 86400;
    let h = time_of_day / 3600;
    let m = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let yr = if mo <= 2 { y + 1 } else { y };
    format!("{yr:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn hostname_best_effort() -> String {
    std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Parse lattice dimensions from CLI arguments.
///
/// Supports three formats (checked in priority order):
///   - `--dims=Nx,Ny,Nz,Nt` — fully specified
///   - `--ns=Ns` + optional `--nt=Nt` — cubic spatial with optional temporal override
///   - `--lattice=N` — isotropic N^4
///
/// Returns `None` if no geometry flag is found (caller should use its default).
pub fn parse_dims_from_args(args: &[String]) -> Option<[usize; 4]> {
    let mut dims_val: Option<String> = None;
    let mut ns_val: Option<usize> = None;
    let mut nt_val: Option<usize> = None;
    let mut lattice_val: Option<usize> = None;

    for arg in args {
        if let Some(v) = arg.strip_prefix("--dims=") {
            dims_val = Some(v.to_string());
        } else if let Some(v) = arg.strip_prefix("--ns=") {
            ns_val = v.parse().ok();
        } else if let Some(v) = arg.strip_prefix("--nt=") {
            nt_val = v.parse().ok();
        } else if let Some(v) = arg.strip_prefix("--lattice=") {
            lattice_val = v.parse().ok();
        }
    }

    if let Some(d) = dims_val {
        let parts: Vec<usize> = d.split(',').filter_map(|s| s.parse().ok()).collect();
        if parts.len() == 4 {
            return Some([parts[0], parts[1], parts[2], parts[3]]);
        }
        panic!("--dims requires exactly 4 comma-separated values (Nx,Ny,Nz,Nt)");
    }

    if let Some(ns) = ns_val {
        let nt = nt_val.unwrap_or(ns);
        return Some([ns, ns, ns, nt]);
    }

    if let Some(nt) = nt_val {
        if let Some(l) = lattice_val {
            return Some([l, l, l, nt]);
        }
    }

    lattice_val.map(|l| [l, l, l, l])
}

/// Format lattice dimensions as a human-readable string.
///
/// Returns "N^4" for isotropic, "Ns^3 x Nt" for cubic spatial with
/// different temporal, or "Nx x Ny x Nz x Nt" for fully anisotropic.
pub fn format_dims(dims: [usize; 4]) -> String {
    let [nx, ny, nz, nt] = dims;
    if nx == ny && ny == nz && nz == nt {
        format!("{}⁴", nx)
    } else if nx == ny && ny == nz {
        format!("{}³×{}", nx, nt)
    } else {
        format!("{}×{}×{}×{}", nx, ny, nz, nt)
    }
}

/// Format lattice dimensions for use in ensemble/file identifiers.
///
/// Returns "L8" for isotropic 8^4, "L8_Nt16" for 8^3 x 16,
/// or "8x8x8x16" for fully anisotropic.
pub fn format_dims_id(dims: [usize; 4]) -> String {
    let [nx, ny, nz, nt] = dims;
    if nx == ny && ny == nz && nz == nt {
        format!("L{nx}")
    } else if nx == ny && ny == nz {
        format!("L{nx}_Nt{nt}")
    } else {
        format!("{nx}x{ny}x{nz}x{nt}")
    }
}

/// Minimum spatial dimension (for bounding Wilson loop extents).
pub fn min_spatial_dim(dims: [usize; 4]) -> usize {
    dims[0].min(dims[1]).min(dims[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_serializes() {
        let manifest = EnsembleManifest::new("test_ensemble", [8, 8, 8, 8], 6.0);
        let json = manifest.to_json();
        assert!(json.contains("\"ensemble_id\": \"test_ensemble\""));
        assert!(json.contains("\"beta\": 6.0"));
        assert!(json.contains("\"schema_version\": \"1.0\""));
    }

    #[test]
    fn measurement_serializes() {
        let mut meas = ConfigMeasurement::new("test", 42, "/test/lfn");
        meas.gauge.plaquette = 0.598;
        meas.flow = Some(FlowResults {
            integrator: "Lscfrk3w7".to_string(),
            epsilon: 0.01,
            t_max: 4.0,
            t0: Some(1.234),
            w0: Some(0.987),
            flow_curve: vec![FlowPoint {
                t: 0.1,
                energy_density: 0.5,
                t2_e: 0.005,
            }],
        });
        let json = meas.to_json();
        assert!(json.contains("\"plaquette\": 0.598"));
        assert!(json.contains("\"t0\": 1.234"));
        assert!(json.contains("\"integrator\": \"Lscfrk3w7\""));
    }

    #[test]
    fn manifest_roundtrip() {
        let manifest = EnsembleManifest::new("rt", [16, 16, 16, 32], 5.8);
        let json = manifest.to_json();
        let parsed: EnsembleManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.ensemble_id, "rt");
        assert_eq!(parsed.dims, [16, 16, 16, 32]);
        assert!((parsed.beta - 5.8).abs() < 1e-10);
    }

    #[test]
    fn jackknife_constant_data() {
        let data = vec![1.0; 10];
        let (mean, err) = jackknife_error(&data);
        assert!((mean - 1.0).abs() < 1e-12);
        assert!(err < 1e-12);
    }

    #[test]
    fn jackknife_known_values() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, err) = jackknife_error(&data);
        assert!((mean - 3.0).abs() < 1e-12);
        assert!(err > 0.0);
    }

    #[test]
    fn tau_int_uncorrelated() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let (tau, _) = estimate_tau_int(&data);
        assert!(tau >= 1.0, "tau_int should be >= 1: got {tau}");
    }

    #[test]
    fn implementation_auto_detect() {
        let info = ImplementationInfo::auto_detect();
        assert_eq!(info.code_name, "hotSpring-barracuda");
        assert!(!info.code_version.is_empty());
    }

    #[test]
    fn measurement_with_new_fields() {
        let mut meas = ConfigMeasurement::new("test", 1, "/test");
        meas.implementation = Some(ImplementationInfo::auto_detect());
        meas.scale_setting = Some(ScaleSetting {
            a_fm: Some(0.1),
            method: Some("t0".to_string()),
            reference_value: Some(0.1465),
            reference: Some("BMW 2012".to_string()),
        });
        let json = meas.to_json();
        assert!(json.contains("\"code_name\": \"hotSpring-barracuda\""));
        assert!(json.contains("\"a_fm\": 0.1"));
        assert!(json.contains("\"method\": \"t0\""));
    }
}
