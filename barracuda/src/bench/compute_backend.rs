// SPDX-License-Identifier: AGPL-3.0-or-later

//! `ComputeBackend` trait — swappable backend interface for benchmark comparison.
//!
//! Allows the same benchmark specification to be run on multiple backends
//! (barraCuda GPU, barraCuda CPU, Kokkos/LAMMPS, Python/Sarkas) with
//! uniform result collection.
//!
//! Provenance: hotSpring v0.6.17 — proposed in asymmetric lattice handoff.

use std::fmt;
use std::time::Duration;

/// Specification for a quenched SU(3) HMC benchmark run.
#[derive(Clone, Debug)]
pub struct BenchmarkSpec {
    /// Lattice dimensions `[Nx, Ny, Nz, Nt]`.
    pub dims: [usize; 4],
    /// Inverse coupling.
    pub beta: f64,
    /// Thermalization trajectories (discarded).
    pub n_therm: usize,
    /// Measurement trajectories.
    pub n_meas: usize,
    /// Leapfrog / Omelyan MD steps per trajectory.
    pub n_md_steps: usize,
    /// MD step size.
    pub dt: f64,
    /// RNG seed.
    pub seed: u64,
}

impl BenchmarkSpec {
    /// Total lattice volume (product of dims).
    #[must_use]
    pub fn volume(&self) -> usize {
        self.dims.iter().product()
    }

    /// True if `N_t` differs from `N_s` (finite-temperature geometry).
    #[must_use]
    pub const fn is_asymmetric(&self) -> bool {
        let [nx, ny, nz, nt] = self.dims;
        nt != nx || ny != nx || nz != nx
    }

    /// Human-readable lattice label (e.g. "32³×8" or "16⁴").
    #[must_use]
    pub fn label(&self) -> String {
        let [nx, _, _, nt] = self.dims;
        if self.is_asymmetric() {
            format!("{nx}³×{nt}")
        } else {
            format!("{nx}⁴")
        }
    }

    /// Construct a spec with auto-tuned HMC parameters for quenched runs.
    #[must_use]
    pub fn quenched_default(dims: [usize; 4], beta: f64) -> Self {
        let vol: usize = dims.iter().product();
        let ref_vol = 4096.0_f64;
        let scale = (ref_vol / vol as f64).powf(0.25);
        let dt = (0.05 * scale).max(0.002);
        let n_md = ((0.5 / dt).round() as usize).max(10);
        Self {
            dims,
            beta,
            n_therm: 200,
            n_meas: 500,
            n_md_steps: n_md,
            dt,
            seed: 42,
        }
    }
}

/// Result of a benchmark run on a single backend.
#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    /// Backend identifier string.
    pub backend_name: String,
    /// Which compute substrate ran this.
    pub backend_kind: BackendKind,
    /// The spec that produced this result.
    pub spec: BenchmarkSpec,
    /// Mean plaquette over measurement trajectories.
    pub mean_plaquette: f64,
    /// Standard deviation of plaquette.
    pub std_plaquette: f64,
    /// Polyakov loop magnitude (deconfinement order parameter).
    pub polyakov_mag: f64,
    /// Fraction of trajectories accepted by Metropolis test.
    pub acceptance_rate: f64,
    /// Total wall-clock time for measurement phase.
    pub wall_time: Duration,
    /// Milliseconds per trajectory.
    pub ms_per_trajectory: f64,
    /// Precision strategy used.
    pub precision_mode: PrecisionMode,
}

impl BenchmarkResult {
    /// Throughput in trajectories per second.
    #[must_use]
    pub fn trajectories_per_second(&self) -> f64 {
        if self.ms_per_trajectory > 0.0 {
            1000.0 / self.ms_per_trajectory
        } else {
            0.0
        }
    }
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({:?}/{:?}): ⟨P⟩={:.6}±{:.6} |L|={:.4} acc={:.0}% {:.1}ms/traj",
            self.backend_name,
            self.backend_kind,
            self.precision_mode,
            self.mean_plaquette,
            self.std_plaquette,
            self.polyakov_mag,
            self.acceptance_rate * 100.0,
            self.ms_per_trajectory,
        )
    }
}

/// Which compute substrate produced a benchmark result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// barraCuda Rust CPU path.
    BarraCudaCpu,
    /// barraCuda GPU via wgpu/Vulkan (Tier 1 dispatch).
    BarraCudaGpu,
    /// coralReef sovereign dispatch via direct DRM ioctls (Tier 2 dispatch).
    CoralReefSovereign,
    /// Kokkos-CUDA via LAMMPS (Tier 3 reference target).
    KokkosCuda,
    /// Python/Sarkas reference (external process).
    PythonSarkas,
    /// Other external backend.
    External,
}

/// Floating-point precision strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionMode {
    /// Native 64-bit float.
    F64,
    /// Double-float emulation on f32 cores.
    DF64,
    /// 32-bit float only.
    F32,
    /// Mixed precision.
    Mixed,
}

/// Trait for swappable compute backends in benchmark comparison.
pub trait ComputeBackend {
    /// Human-readable name.
    fn name(&self) -> &str;
    /// Backend classification.
    fn kind(&self) -> BackendKind;
    /// Precision strategy.
    fn precision(&self) -> PrecisionMode;
    /// Whether this backend can run right now.
    fn available(&self) -> bool;
    /// Execute quenched HMC according to `spec` and return measurements.
    fn run_quenched_hmc(&self, spec: &BenchmarkSpec) -> Result<BenchmarkResult, String>;
}

/// Run the same spec on all available backends and print comparison.
pub fn compare_backends(
    backends: &[&dyn ComputeBackend],
    spec: &BenchmarkSpec,
) -> Vec<Result<BenchmarkResult, String>> {
    let mut results = Vec::with_capacity(backends.len());

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║  ComputeBackend Comparison — {} β={:.4}               ║",
        spec.label(),
        spec.beta
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    for backend in backends {
        if !backend.available() {
            println!("  {} — SKIPPED (not available)", backend.name());
            results.push(Err(format!("{} not available", backend.name())));
            continue;
        }

        println!("  Running {}...", backend.name());
        match backend.run_quenched_hmc(spec) {
            Ok(r) => {
                println!("    {r}");
                results.push(Ok(r));
            }
            Err(e) => {
                println!("    FAILED: {e}");
                results.push(Err(e));
            }
        }
    }

    if results.len() >= 2 {
        println!();
        println!("  ── Speedup Matrix ──");
        let ok_results: Vec<&BenchmarkResult> =
            results.iter().filter_map(|r| r.as_ref().ok()).collect();
        if ok_results.len() >= 2 {
            let baseline = &ok_results[0];
            for r in &ok_results[1..] {
                let speedup = baseline.ms_per_trajectory / r.ms_per_trajectory;
                let plaq_diff = (baseline.mean_plaquette - r.mean_plaquette).abs();
                println!(
                    "    {} vs {}: {:.1}× speedup, ΔP={:.6}",
                    r.backend_name, baseline.backend_name, speedup, plaq_diff
                );
            }
        }
    }

    results
}

/// barraCuda CPU backend — pure Rust, no GPU required.
pub struct BarraCudaCpuBackend;

impl ComputeBackend for BarraCudaCpuBackend {
    fn name(&self) -> &'static str {
        "barraCuda-CPU"
    }
    fn kind(&self) -> BackendKind {
        BackendKind::BarraCudaCpu
    }
    fn precision(&self) -> PrecisionMode {
        PrecisionMode::F64
    }
    fn available(&self) -> bool {
        true
    }

    fn run_quenched_hmc(&self, spec: &BenchmarkSpec) -> Result<BenchmarkResult, String> {
        use crate::lattice::hmc::{self, HmcConfig, IntegratorType};
        use crate::lattice::wilson::Lattice;
        use std::time::Instant;

        let mut lat = Lattice::hot_start(spec.dims, spec.beta, spec.seed);
        let mut cfg = HmcConfig {
            n_md_steps: spec.n_md_steps,
            dt: spec.dt,
            seed: spec.seed,
            integrator: IntegratorType::Omelyan,
        };

        for _ in 0..spec.n_therm {
            hmc::hmc_trajectory(&mut lat, &mut cfg);
        }

        let start = Instant::now();
        let mut plaq_vals = Vec::with_capacity(spec.n_meas);
        let mut accepted = 0usize;
        for _ in 0..spec.n_meas {
            let r = hmc::hmc_trajectory(&mut lat, &mut cfg);
            plaq_vals.push(r.plaquette);
            if r.accepted {
                accepted += 1;
            }
        }
        let wall = start.elapsed();
        let ms_per = wall.as_secs_f64() * 1000.0 / spec.n_meas as f64;

        let mean_plaq = plaq_vals.iter().sum::<f64>() / plaq_vals.len() as f64;
        let var_plaq = plaq_vals
            .iter()
            .map(|p| (p - mean_plaq).powi(2))
            .sum::<f64>()
            / (plaq_vals.len() - 1).max(1) as f64;
        let poly_mag = lat.average_polyakov_loop().abs();

        Ok(BenchmarkResult {
            backend_name: self.name().to_string(),
            backend_kind: self.kind(),
            spec: spec.clone(),
            mean_plaquette: mean_plaq,
            std_plaquette: var_plaq.sqrt(),
            polyakov_mag: poly_mag,
            acceptance_rate: accepted as f64 / spec.n_meas as f64,
            wall_time: wall,
            ms_per_trajectory: ms_per,
            precision_mode: self.precision(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_spec_label() {
        let sym = BenchmarkSpec::quenched_default([8, 8, 8, 8], 6.0);
        assert_eq!(sym.label(), "8⁴");
        assert!(!sym.is_asymmetric());

        let asym = BenchmarkSpec::quenched_default([32, 32, 32, 4], 5.69);
        assert_eq!(asym.label(), "32³×4");
        assert!(asym.is_asymmetric());
    }

    #[test]
    fn cpu_backend_small_lattice() {
        let spec = BenchmarkSpec {
            dims: [4, 4, 4, 4],
            beta: 6.0,
            n_therm: 10,
            n_meas: 20,
            n_md_steps: 10,
            dt: 0.05,
            seed: 42,
        };
        let backend = BarraCudaCpuBackend;
        assert!(backend.available());
        let result = match backend.run_quenched_hmc(&spec) {
            Ok(r) => r,
            Err(e) => panic!("CPU backend failed on small lattice: {e}"),
        };
        assert!(result.mean_plaquette > 0.3);
        assert!(result.acceptance_rate > 0.3);
        assert!(result.ms_per_trajectory > 0.0);
    }
}
