// SPDX-License-Identifier: AGPL-3.0-only

//! `MdBenchmarkBackend` trait — swappable backend interface for Yukawa MD
//! performance comparison across dispatch tiers.
//!
//! Allows the same MD benchmark specification to be run on multiple backends
//! (barraCuda GPU via wgpu/Vulkan, Kokkos/LAMMPS via external process,
//! coralReef sovereign once dispatch is unblocked) with uniform result
//! collection.
//!
//! Designed for Kokkos-parity benchmarking: the 9 PP Yukawa DSF cases
//! from hotSpring's Sarkas study provide a physics-validated comparison
//! matrix across all three dispatch tiers.
//!
//! See `specs/MULTI_BACKEND_DISPATCH.md` for the three-tier architecture.
//!
//! Provenance: hotSpring v0.6.28 — multi-backend Kokkos parity initiative.

use super::BackendKind;
use crate::md::config::MdConfig;
use std::fmt;
use std::time::Duration;

/// Force calculation method for MD benchmarks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForceMethod {
    /// O(N^2) direct summation.
    AllPairs,
    /// Spatial binning with linked-cell list.
    CellList,
    /// Verlet neighbor list with skin distance.
    VerletList,
}

impl fmt::Display for ForceMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllPairs => write!(f, "AllPairs"),
            Self::CellList => write!(f, "CellList"),
            Self::VerletList => write!(f, "Verlet"),
        }
    }
}

/// Specification for a Yukawa MD benchmark run.
#[derive(Clone, Debug)]
pub struct MdBenchmarkSpec {
    /// Label for this case (e.g. "k2_G158").
    pub label: String,
    /// Number of particles.
    pub n_particles: usize,
    /// Screening parameter kappa.
    pub kappa: f64,
    /// Coupling parameter Gamma.
    pub gamma: f64,
    /// Cutoff radius in a_ws.
    pub rc: f64,
    /// Reduced timestep.
    pub dt: f64,
    /// Equilibration steps.
    pub equil_steps: usize,
    /// Production steps.
    pub prod_steps: usize,
    /// Force calculation method.
    pub force_method: ForceMethod,
}

impl MdBenchmarkSpec {
    /// Build from an existing `MdConfig`, auto-selecting force method.
    #[must_use]
    pub fn from_config(config: &MdConfig) -> Self {
        use crate::md::neighbor::{AlgorithmSelector, ForceAlgorithm};
        let selector = AlgorithmSelector::from_config(config);
        let method = match selector.select() {
            ForceAlgorithm::AllPairs => ForceMethod::AllPairs,
            ForceAlgorithm::CellList => ForceMethod::CellList,
            ForceAlgorithm::VerletList { .. } => ForceMethod::VerletList,
        };
        Self {
            label: config.label.clone(),
            n_particles: config.n_particles,
            kappa: config.kappa,
            gamma: config.gamma,
            rc: config.rc,
            dt: config.dt,
            equil_steps: config.equil_steps,
            prod_steps: config.prod_steps,
            force_method: method,
        }
    }

    /// Convert back to `MdConfig` for execution.
    pub fn to_md_config(&self) -> MdConfig {
        MdConfig {
            label: self.label.clone(),
            n_particles: self.n_particles,
            kappa: self.kappa,
            gamma: self.gamma,
            dt: self.dt,
            rc: self.rc,
            equil_steps: self.equil_steps,
            prod_steps: self.prod_steps,
            dump_step: 10,
            berendsen_tau: 5.0,
            rdf_bins: 500,
            vel_snapshot_interval: 100,
        }
    }

    /// The 9 PP Yukawa DSF cases used for Kokkos parity benchmarking.
    #[must_use]
    pub fn kokkos_parity_cases(n_particles: usize) -> Vec<Self> {
        crate::md::config::dsf_pp_cases(n_particles, true)
            .iter()
            .map(Self::from_config)
            .collect()
    }
}

/// Result of an MD benchmark run on a single backend.
#[derive(Clone, Debug)]
pub struct MdBenchmarkResult {
    /// Backend identifier.
    pub backend_name: String,
    /// Which dispatch tier ran this.
    pub backend_kind: BackendKind,
    /// The spec that produced this result.
    pub label: String,
    /// Number of particles.
    pub n_particles: usize,
    /// Kappa value.
    pub kappa: f64,
    /// Gamma value.
    pub gamma: f64,
    /// Steps per second (production phase).
    pub steps_per_sec: f64,
    /// Energy drift percentage.
    pub energy_drift_pct: f64,
    /// Total wall-clock time for the full run (equil + production).
    pub wall_time: Duration,
    /// Force method used.
    pub force_method: ForceMethod,
    /// GPU adapter name (if applicable).
    pub adapter_name: String,
    /// Driver info (e.g. "NVK/Mesa 25.1.5", "NVIDIA proprietary 580.119.02").
    pub driver_info: String,
}

impl MdBenchmarkResult {
    /// Milliseconds per production step.
    #[must_use]
    pub fn ms_per_step(&self) -> f64 {
        if self.steps_per_sec > 0.0 {
            1000.0 / self.steps_per_sec
        } else {
            f64::NAN
        }
    }
}

impl fmt::Display for MdBenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}): {:.1} steps/s, drift={:.3}%, method={}",
            self.backend_name,
            self.label,
            self.steps_per_sec,
            self.energy_drift_pct,
            self.force_method,
        )
    }
}

/// Trait for swappable MD compute backends in Kokkos parity benchmarking.
pub trait MdBenchmarkBackend {
    /// Human-readable name.
    fn name(&self) -> &'static str;
    /// Backend classification.
    fn kind(&self) -> BackendKind;
    /// Whether this backend can run right now.
    fn available(&self) -> bool;
    /// Execute Yukawa MD according to `spec` and return measurements.
    fn run_yukawa_md(&self, spec: &MdBenchmarkSpec) -> Result<MdBenchmarkResult, String>;
}

/// Run the same spec on all available MD backends and print comparison.
pub fn compare_md_backends(
    backends: &[&dyn MdBenchmarkBackend],
    spec: &MdBenchmarkSpec,
) -> Vec<Result<MdBenchmarkResult, String>> {
    let mut results = Vec::with_capacity(backends.len());

    println!(
        "  {} (κ={}, Γ={}, N={}, {}):",
        spec.label, spec.kappa, spec.gamma, spec.n_particles, spec.force_method
    );

    for backend in backends {
        if !backend.available() {
            println!("    {} — SKIPPED (not available)", backend.name());
            results.push(Err(format!("{} not available", backend.name())));
            continue;
        }

        print!("    {} ... ", backend.name());
        match backend.run_yukawa_md(spec) {
            Ok(r) => {
                println!("{:.1} steps/s, drift={:.3}%", r.steps_per_sec, r.energy_drift_pct);
                results.push(Ok(r));
            }
            Err(e) => {
                println!("FAILED: {e}");
                results.push(Err(e));
            }
        }
    }

    results
}

/// Print a gap analysis table comparing MD results across backends.
pub fn print_gap_analysis(all_results: &[(String, Vec<Result<MdBenchmarkResult, String>>)]) {
    println!();
    println!(
        "  {:>12} {:>12} {:>12} {:>8} {:>8}",
        "Case", "barraCuda", "Kokkos", "Gap", "Method"
    );
    println!(
        "  {:>12} {:>12} {:>12} {:>8} {:>8}",
        "────", "steps/s", "steps/s", "────", "──────"
    );

    for (label, results) in all_results {
        let bc = results.iter().find_map(|r| {
            r.as_ref()
                .ok()
                .filter(|r| r.backend_kind == BackendKind::BarraCudaGpu)
        });
        let kk = results.iter().find_map(|r| {
            r.as_ref()
                .ok()
                .filter(|r| r.backend_kind == BackendKind::KokkosCuda)
        });

        let bc_sps = bc.map_or(f64::NAN, |r| r.steps_per_sec);
        let kk_sps = kk.map_or(f64::NAN, |r| r.steps_per_sec);
        let gap = if kk_sps.is_finite() && bc_sps > 0.0 {
            format!("{:.1}×", kk_sps / bc_sps)
        } else {
            "—".to_string()
        };
        let method = bc.map_or_else(|| "—".to_string(), |r| r.force_method.to_string());

        println!(
            "  {:>12} {:>12} {:>12} {:>8} {:>8}",
            label,
            if bc_sps.is_finite() {
                format!("{bc_sps:.1}")
            } else {
                "—".to_string()
            },
            if kk_sps.is_finite() {
                format!("{kk_sps:.1}")
            } else {
                "—".to_string()
            },
            gap,
            method,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Backend Implementations
// ═══════════════════════════════════════════════════════════════════════

/// barraCuda GPU backend — wgpu/Vulkan (Tier 1 dispatch).
pub struct BarraCudaMdBackend {
    adapter_name: String,
    driver_info: String,
}

impl BarraCudaMdBackend {
    /// Create a new barraCuda MD backend, probing the default GPU.
    pub fn new() -> Result<Self, String> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| format!("runtime: {e}"))?;
        let gpu = rt
            .block_on(crate::gpu::GpuF64::new())
            .map_err(|e| format!("GPU: {e}"))?;
        let profile = gpu.driver_profile();
        let driver = format!("{:?}/{:?}", profile.driver, profile.arch);
        Ok(Self {
            adapter_name: gpu.adapter_name.clone(),
            driver_info: driver,
        })
    }
}

impl MdBenchmarkBackend for BarraCudaMdBackend {
    fn name(&self) -> &'static str {
        "barraCuda-GPU"
    }

    fn kind(&self) -> BackendKind {
        BackendKind::BarraCudaGpu
    }

    fn available(&self) -> bool {
        true
    }

    fn run_yukawa_md(&self, spec: &MdBenchmarkSpec) -> Result<MdBenchmarkResult, String> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| format!("runtime: {e}"))?;
        let config = spec.to_md_config();
        let adapter_name = self.adapter_name.clone();
        let driver_info = self.driver_info.clone();

        rt.block_on(async {
            let t0 = std::time::Instant::now();

            use crate::md::neighbor::{AlgorithmSelector, ForceAlgorithm};
            use crate::md::simulation;

            let selector = AlgorithmSelector::from_config(&config);
            let result = match selector.select() {
                ForceAlgorithm::AllPairs => simulation::run_simulation(&config).await,
                ForceAlgorithm::CellList => simulation::run_simulation_celllist(&config).await,
                ForceAlgorithm::VerletList { skin } => {
                    simulation::run_simulation_verlet_with_brain(&config, skin, None).await
                }
            };

            let wall_time = t0.elapsed();

            match result {
                Ok(sim) => {
                    let energy_val =
                        crate::md::observables::validate_energy(&sim.energy_history, &config);
                    Ok(MdBenchmarkResult {
                        backend_name: format!("barraCuda-GPU ({adapter_name})"),
                        backend_kind: BackendKind::BarraCudaGpu,
                        label: spec.label.clone(),
                        n_particles: spec.n_particles,
                        kappa: spec.kappa,
                        gamma: spec.gamma,
                        steps_per_sec: sim.steps_per_sec,
                        energy_drift_pct: energy_val.drift_pct,
                        wall_time,
                        force_method: spec.force_method,
                        adapter_name,
                        driver_info,
                    })
                }
                Err(e) => Err(format!("MD simulation failed: {e}")),
            }
        })
    }
}

/// Backend-agnostic MD backend — sovereign engine with smart device selection.
///
/// Tries sovereign (coralReef → DRM) first, falls back to wgpu/Vulkan.
/// Uses `run_simulation_generic<B>` from `md::sovereign_engine`.
pub struct GenericMdBackend {
    adapter_name: String,
    driver_info: String,
    dispatch_tier: &'static str,
}

impl GenericMdBackend {
    /// Probe hardware and select the best available backend.
    ///
    /// Priority: sovereign (coralReef) → wgpu (Vulkan/Metal).
    pub fn new() -> Result<Self, String> {
        #[cfg(feature = "sovereign-dispatch")]
        {
            use barracuda::device::backend::GpuBackend;
            use barracuda::device::CoralReefDevice;

            let strategies: &[(&str, &str, Box<dyn Fn() -> barracuda::error::Result<CoralReefDevice>>)] = &[
                ("nouveau", "SM70/Volta", Box::new(|| {
                    CoralReefDevice::from_descriptor("nvidia", Some("sm70"), Some("nouveau"))
                })),
                ("amdgpu", "RDNA2", Box::new(|| {
                    CoralReefDevice::from_descriptor("amd", None, None)
                })),
                ("auto", "sovereign", Box::new(CoralReefDevice::with_auto_device)),
            ];

            for (driver, desc, init_fn) in strategies {
                if let Ok(dev) = init_fn() {
                    if dev.has_dispatch() {
                        let name = GpuBackend::name(&dev);
                        return Ok(Self {
                            adapter_name: name.to_string(),
                            driver_info: format!("sovereign:{driver} ({desc})"),
                            dispatch_tier: "Tier 2: Sovereign/DRM",
                        });
                    }
                }
            }
        }

        let rt = tokio::runtime::Runtime::new().map_err(|e| format!("runtime: {e}"))?;
        let dev = rt.block_on(barracuda::device::WgpuDevice::new())
            .map_err(|e| format!("no GPU available: {e}"))?;
        let name = barracuda::device::backend::GpuBackend::name(&dev);
        Ok(Self {
            adapter_name: name.to_string(),
            driver_info: "wgpu/Vulkan".to_string(),
            dispatch_tier: "Tier 1: wgpu/Vulkan",
        })
    }

    /// Which dispatch tier was selected.
    #[must_use]
    pub fn dispatch_tier(&self) -> &str {
        self.dispatch_tier
    }
}

impl MdBenchmarkBackend for GenericMdBackend {
    fn name(&self) -> &'static str {
        "generic-GPU"
    }

    fn kind(&self) -> BackendKind {
        BackendKind::BarraCudaGpu
    }

    fn available(&self) -> bool {
        true
    }

    fn run_yukawa_md(&self, spec: &MdBenchmarkSpec) -> Result<MdBenchmarkResult, String> {
        let config = spec.to_md_config();
        let adapter_name = self.adapter_name.clone();
        let driver_info = self.driver_info.clone();

        let t0 = std::time::Instant::now();

        let sim = if self.dispatch_tier.contains("Sovereign") {
            #[cfg(feature = "sovereign-dispatch")]
            {
                use barracuda::device::CoralReefDevice;
                let dev = CoralReefDevice::with_auto_device()
                    .map_err(|e| format!("sovereign device: {e}"))?;
                crate::md::sovereign_engine::run_simulation_generic(&dev, &config)?
            }
            #[cfg(not(feature = "sovereign-dispatch"))]
            {
                return Err("sovereign-dispatch feature not enabled".to_string());
            }
        } else {
            let rt = tokio::runtime::Runtime::new().map_err(|e| format!("runtime: {e}"))?;
            let dev = rt.block_on(barracuda::device::WgpuDevice::new())
                .map_err(|e| format!("wgpu device: {e}"))?;
            crate::md::sovereign_engine::run_simulation_generic(&dev, &config)?
        };

        let wall_time = t0.elapsed();
        let energy_val =
            crate::md::observables::validate_energy(&sim.energy_history, &config);

        Ok(MdBenchmarkResult {
            backend_name: format!("generic-GPU ({adapter_name}, {})", self.dispatch_tier),
            backend_kind: BackendKind::BarraCudaGpu,
            label: spec.label.clone(),
            n_particles: spec.n_particles,
            kappa: spec.kappa,
            gamma: spec.gamma,
            steps_per_sec: sim.steps_per_sec,
            energy_drift_pct: energy_val.drift_pct,
            wall_time,
            force_method: spec.force_method,
            adapter_name,
            driver_info,
        })
    }
}

/// Kokkos/LAMMPS backend — external process (Tier 3 reference).
///
/// Spawns LAMMPS with Kokkos-CUDA backend, writes a temporary input file
/// from the benchmark spec, parses thermo output for steps/s and energy.
/// Gracefully unavailable if `lmp` is not installed.
pub struct KokkosLammpsBackend {
    lmp_path: Option<String>,
}

impl Default for KokkosLammpsBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl KokkosLammpsBackend {
    /// Probe for LAMMPS installation.
    #[must_use]
    pub fn new() -> Self {
        let lmp_path = which_lmp();
        if let Some(ref p) = lmp_path {
            println!("    LAMMPS found: {p}");
        }
        Self { lmp_path }
    }
}

fn which_lmp() -> Option<String> {
    std::process::Command::new("which")
        .arg("lmp")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

impl KokkosLammpsBackend {
    /// Generate a LAMMPS input file for a Yukawa MD benchmark.
    ///
    /// FCC lattice in LAMMPS: `region` dimensions are in lattice units.
    /// For N particles in FCC (4 atoms/cell): L = ceil((N/4)^{1/3}).
    /// Actual atom count = 4 * L^3 (may differ slightly from spec.n_particles).
    fn write_lammps_input(
        spec: &MdBenchmarkSpec,
        path: &std::path::Path,
    ) -> Result<(), String> {
        let density = 3.0 / (4.0 * std::f64::consts::PI);
        let n_cells = ((spec.n_particles as f64 / 4.0).cbrt()).round() as usize;
        let n_actual = 4 * n_cells * n_cells * n_cells;
        let temperature = 1.0 / spec.gamma;
        let total_steps = spec.equil_steps + spec.prod_steps;

        let kappa = spec.kappa;
        let rc = spec.rc;
        let temp = temperature;
        let tau = 5.0 * spec.dt;
        let dt = spec.dt;
        let thermo_interval = 1000;

        let input = format!(
            r"# Auto-generated by hotSpring bench_md_parity
# Case: {label} (kappa={kappa}, Gamma={gamma}, N_target={n}, N_actual={n_actual})
package kokkos neigh full comm device
units lj
atom_style atomic
boundary p p p
lattice fcc {density}
region box block 0 {n_cells} 0 {n_cells} 0 {n_cells}
create_box 1 box
create_atoms 1 box
mass 1 3.0

pair_style yukawa {kappa} {rc}
pair_coeff 1 1 1.0

velocity all create {temp} 42 dist gaussian

fix 1 all nvt temp {temp} {temp} {tau}
timestep {dt}

thermo_style custom step temp pe ke etotal press
thermo {thermo_interval}

run {total_steps}
",
            label = spec.label,
            gamma = spec.gamma,
            n = spec.n_particles,
        );

        std::fs::write(path, input).map_err(|e| format!("Failed to write LAMMPS input: {e}"))
    }

    /// Parse LAMMPS log output for performance metrics.
    fn parse_lammps_output(
        output: &str,
        spec: &MdBenchmarkSpec,
        wall_time: Duration,
    ) -> MdBenchmarkResult {
        let steps_per_sec = output
            .lines()
            .find(|l| l.contains("Loop time of"))
            .and_then(|l| {
                let total_steps = (spec.equil_steps + spec.prod_steps) as f64;
                l.split_whitespace()
                    .nth(3)
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(|loop_time| total_steps / loop_time)
            })
            .unwrap_or(0.0);

        let energy_lines: Vec<&str> = output
            .lines()
            .filter(|l| {
                l.split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                    .is_some()
            })
            .collect();

        let drift = if energy_lines.len() >= 2 {
            let parse_etotal = |line: &str| -> Option<f64> {
                line.split_whitespace().nth(4).and_then(|s| s.parse().ok())
            };
            let e_first = parse_etotal(energy_lines[0]).unwrap_or(0.0);
            let e_last = parse_etotal(energy_lines[energy_lines.len() - 1]).unwrap_or(0.0);
            if e_first.abs() > 1e-15 {
                ((e_last - e_first) / e_first).abs() * 100.0
            } else {
                0.0
            }
        } else {
            f64::NAN
        };

        MdBenchmarkResult {
            backend_name: "Kokkos-CUDA (LAMMPS)".to_string(),
            backend_kind: BackendKind::KokkosCuda,
            label: spec.label.clone(),
            n_particles: spec.n_particles,
            kappa: spec.kappa,
            gamma: spec.gamma,
            steps_per_sec,
            energy_drift_pct: drift,
            wall_time,
            force_method: spec.force_method,
            adapter_name: "CUDA".to_string(),
            driver_info: "Kokkos".to_string(),
        }
    }
}

impl MdBenchmarkBackend for KokkosLammpsBackend {
    fn name(&self) -> &'static str {
        "Kokkos-CUDA"
    }

    fn kind(&self) -> BackendKind {
        BackendKind::KokkosCuda
    }

    fn available(&self) -> bool {
        self.lmp_path.is_some()
    }

    fn run_yukawa_md(&self, spec: &MdBenchmarkSpec) -> Result<MdBenchmarkResult, String> {
        let lmp = self
            .lmp_path
            .as_ref()
            .ok_or_else(|| "LAMMPS not installed".to_string())?;

        let tmp_dir =
            std::env::temp_dir().join(format!("hotspring_kokkos_{}", spec.label));
        std::fs::create_dir_all(&tmp_dir)
            .map_err(|e| format!("Failed to create temp dir: {e}"))?;

        let input_path = tmp_dir.join("input.lammps");
        Self::write_lammps_input(spec, &input_path)?;

        let t0 = std::time::Instant::now();
        let output = std::process::Command::new(lmp)
            .args(["-k", "on", "g", "1", "-sf", "kk", "-in"])
            .arg(&input_path)
            .current_dir(&tmp_dir)
            .output()
            .map_err(|e| format!("Failed to run LAMMPS: {e}"))?;

        let wall_time = t0.elapsed();

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("LAMMPS exited with {}: {stderr}", output.status));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result = Self::parse_lammps_output(&stdout, spec, wall_time);

        let _ = std::fs::remove_dir_all(&tmp_dir);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn md_spec_from_config_roundtrip() {
        let config = crate::md::config::quick_test_case(500);
        let spec = MdBenchmarkSpec::from_config(&config);
        assert_eq!(spec.n_particles, 500);
        assert!((spec.kappa - 2.0).abs() < 1e-10);
        assert!((spec.gamma - 158.0).abs() < 1e-10);
        let config2 = spec.to_md_config();
        assert_eq!(config2.n_particles, 500);
        assert!((config2.kappa - 2.0).abs() < 1e-10);
    }

    #[test]
    fn kokkos_parity_cases_nine() {
        let cases = MdBenchmarkSpec::kokkos_parity_cases(2000);
        assert_eq!(cases.len(), 9, "9 PP Yukawa DSF cases");
    }

    #[test]
    fn force_method_display() {
        assert_eq!(format!("{}", ForceMethod::AllPairs), "AllPairs");
        assert_eq!(format!("{}", ForceMethod::VerletList), "Verlet");
        assert_eq!(format!("{}", ForceMethod::CellList), "CellList");
    }

    #[test]
    fn md_result_ms_per_step() {
        let result = MdBenchmarkResult {
            backend_name: "test".to_string(),
            backend_kind: BackendKind::BarraCudaGpu,
            label: "test".to_string(),
            n_particles: 500,
            kappa: 2.0,
            gamma: 158.0,
            steps_per_sec: 1000.0,
            energy_drift_pct: 0.0,
            wall_time: Duration::from_secs(1),
            force_method: ForceMethod::VerletList,
            adapter_name: "test".to_string(),
            driver_info: "test".to_string(),
        };
        assert!((result.ms_per_step() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn kokkos_backend_unavailable_without_lammps() {
        let backend = KokkosLammpsBackend { lmp_path: None };
        assert!(!backend.available());
    }
}
