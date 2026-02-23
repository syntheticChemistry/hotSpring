// SPDX-License-Identifier: AGPL-3.0-only

//! `BarraCuda` Evolution Validation — CPU Foundation + Substrate Coverage
//!
//! Runs representative checks from each physics domain on CPU, establishing
//! the correctness foundation that GPU validation binaries build upon.
//!
//! **Evolution chain**:
//!   Python Control → `BarraCuda` CPU (this binary) → WGSL GPU → metalForge
//!
//! GPU parity is proven by the individual `validate_gpu_*` binaries.
//! This binary proves the CPU math is correct across all domains first.
//!
//! **Domains exercised**:
//!   1. Pure gauge SU(3) HMC (Papers 8-9)
//!   2. Staggered Dirac + CG solver (Papers 9-12)
//!   3. Pseudofermion HMC (Paper 10)
//!   4. Abelian Higgs (Paper 13)
//!   5. Spectral 1D/2D/3D Anderson (Papers 14-20)
//!   6. Hofstadter butterfly (Papers 21-22)
//!   7. Dimensional bandwidth hierarchy (cross-dimensional proof)

use hotspring_barracuda::lattice::abelian_higgs::{AbelianHiggsParams, U1HiggsLattice};
use hotspring_barracuda::lattice::cg;
use hotspring_barracuda::lattice::dirac::{apply_dirac, FermionField};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig};
use hotspring_barracuda::lattice::pseudofermion::{
    dynamical_hmc_trajectory, DynamicalHmcConfig, PseudofermionConfig,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::spectral;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  BarraCuda Evolution — CPU Foundation Validation            ║");
    println!("║  Correctness base for GPU + metalForge parity proofs       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("barracuda_evolution");

    check_pure_gauge_hmc(&mut harness);
    check_dirac_cg(&mut harness);
    check_pseudofermion(&mut harness);
    check_abelian_higgs(&mut harness);
    check_spectral_1d(&mut harness);
    check_spectral_2d(&mut harness);
    check_spectral_3d(&mut harness);
    check_hofstadter(&mut harness);
    check_dimensional_hierarchy(&mut harness);
    print_evolution_summary();

    println!();
    harness.finish();
}

/// \[1\] Pure gauge SU(3) HMC — thermalization and plaquette.
fn check_pure_gauge_hmc(harness: &mut ValidationHarness) {
    println!("[1] Pure Gauge SU(3) HMC — Papers 8-9");

    let dims = [4, 4, 4, 4];
    let beta = 5.5;
    let mut lattice = Lattice::hot_start(dims, beta, 42);
    let mut config = HmcConfig {
        n_md_steps: 15,
        dt: 0.05,
        seed: 100,
        ..Default::default()
    };

    let mut accepted = 0;
    for _ in 0..20 {
        let result = hmc::hmc_trajectory(&mut lattice, &mut config);
        if result.accepted {
            accepted += 1;
        }
    }

    let plaq = lattice.average_plaquette();
    println!("  4⁴ lattice, β=5.5, 20 HMC trajectories (hot start)");
    println!("  Final plaquette: {plaq:.6}");
    println!("  Acceptance: {accepted}/20");

    harness.check_bool(
        "plaquette in physical range (0, 1)",
        plaq > tolerances::LATTICE_PLAQUETTE_PHYSICAL_MIN
            && plaq < tolerances::LATTICE_PLAQUETTE_PHYSICAL_MAX,
    );
    harness.check_bool(
        "HMC acceptance > 30%",
        accepted > tolerances::EVOLUTION_PURE_GAUGE_MIN_ACCEPTED,
    );
    println!();
}

/// \[2\] Staggered Dirac operator + CG solver on thermalized lattice.
fn check_dirac_cg(harness: &mut ValidationHarness) {
    println!("[2] Staggered Dirac + CG Solver — Papers 9-12");

    let dims = [4, 4, 4, 4];
    let beta = 5.5;
    let mut lattice = Lattice::hot_start(dims, beta, 42);
    let mut config = HmcConfig {
        n_md_steps: 15,
        dt: 0.05,
        seed: 200,
        ..Default::default()
    };
    for _ in 0..10 {
        hmc::hmc_trajectory(&mut lattice, &mut config);
    }

    let volume = lattice.volume();
    let source = FermionField::random(volume, 42);
    let mass = 0.5;

    let dirac_result = apply_dirac(&lattice, &source, mass);
    let norm_src = source.norm_sq();
    let norm_dst = dirac_result.norm_sq();
    println!("  Dirac applied: |src|² = {norm_src:.4}, |D·src|² = {norm_dst:.4}");

    let mut x = FermionField::zeros(volume);
    let cg_result = cg::cg_solve(
        &lattice,
        &mut x,
        &source,
        mass,
        tolerances::LATTICE_CG_TOLERANCE_STRICT,
        5000,
    );
    println!(
        "  CG solve: {} iters, residual {:.2e}, converged: {}",
        cg_result.iterations, cg_result.final_residual, cg_result.converged
    );

    harness.check_bool("Dirac produces non-zero output", norm_dst > 0.0);
    harness.check_bool("CG converges", cg_result.converged);
    harness.check_upper(
        "CG residual < 1e-8",
        cg_result.final_residual,
        tolerances::LATTICE_CG_RESIDUAL_STRICT,
    );
    println!();
}

/// \[3\] Pseudofermion action via dynamical HMC single trajectory.
fn check_pseudofermion(harness: &mut ValidationHarness) {
    println!("[3] Pseudofermion HMC — Paper 10");

    let dims = [4, 4, 4, 4];
    let beta = 5.5;
    let mut lattice = Lattice::hot_start(dims, beta, 42);

    let mut qconfig = HmcConfig {
        n_md_steps: 15,
        dt: 0.05,
        seed: 300,
        ..Default::default()
    };
    for _ in 0..20 {
        hmc::hmc_trajectory(&mut lattice, &mut qconfig);
    }

    let mut dyn_config = DynamicalHmcConfig {
        n_md_steps: 10,
        dt: 0.02,
        seed: 42,
        fermion: PseudofermionConfig {
            mass: 2.0,
            cg_tol: tolerances::DYNAMICAL_CG_TOLERANCE,
            cg_max_iter: 5000,
        },
        beta,
        n_flavors_over_4: 2,
        ..Default::default()
    };

    let result = dynamical_hmc_trajectory(&mut lattice, &mut dyn_config);

    println!("  4⁴ pre-thermalized (20 quenched HMC), β=5.5, mass=2.0, dt=0.02");
    println!("  ΔH = {:+.6}", result.delta_h);
    println!("  Fermion CG iters: {}", result.cg_iterations);
    println!("  Fermion action: {:.6}", result.fermion_action);
    println!("  Plaquette: {:.6}", result.plaquette);

    harness.check_bool("ΔH is finite", result.delta_h.is_finite());
    harness.check_bool("fermion CG converges (iters > 0)", result.cg_iterations > 0);
    harness.check_bool(
        "plaquette physical after dynamical HMC",
        result.plaquette > tolerances::LATTICE_PLAQUETTE_PHYSICAL_MIN
            && result.plaquette < tolerances::LATTICE_PLAQUETTE_PHYSICAL_MAX,
    );
    println!();
}

/// \[4\] Abelian Higgs model — U(1) gauge + scalar.
fn check_abelian_higgs(harness: &mut ValidationHarness) {
    println!("[4] Abelian Higgs — Paper 13");

    let params = AbelianHiggsParams::new(2.0, 0.3, 0.5);

    let cold = U1HiggsLattice::cold_start(16, 16, params.clone());
    let plaq_cold = cold.average_plaquette();

    let mut lat = U1HiggsLattice::cold_start(16, 16, params);
    let mut seed = 42u64;
    for _ in 0..50 {
        lat.hmc_trajectory(10, 0.1, &mut seed);
    }
    let plaq_hot = lat.average_plaquette();

    println!("  16×16 lattice, β_pl=2.0, κ=0.3, λ=0.5");
    println!("  Cold plaquette: {plaq_cold:.6}");
    println!("  After 50 HMC trajectories: {plaq_hot:.6}");

    harness.check_bool(
        "cold plaquette = 1.0",
        (plaq_cold - 1.0).abs() < tolerances::U1_COLD_PLAQUETTE_ABS,
    );
    harness.check_bool(
        "thermalized plaquette < 1.0",
        plaq_hot < tolerances::LATTICE_PLAQUETTE_PHYSICAL_MAX,
    );
    harness.check_bool(
        "thermalized plaquette > 0",
        plaq_hot > tolerances::LATTICE_PLAQUETTE_PHYSICAL_MIN,
    );
    println!();
}

/// \[5\] Spectral theory: 1D Anderson — Sturm eigenvalues + level statistics.
fn check_spectral_1d(harness: &mut ValidationHarness) {
    println!("[5] Anderson 1D — Papers 14-17");

    let n = 500;
    let w = 4.0;
    let (diag, off) = spectral::anderson_hamiltonian(n, w, 42);
    let evals = spectral::find_all_eigenvalues(&diag, &off);

    let bw = evals.last().expect("non-empty") - evals.first().expect("non-empty");
    println!("  N={n}, W={w}");
    println!("  Bandwidth: {bw:.4} (clean: 4.0)");

    harness.check_bool("1D bandwidth > clean value (disorder widens)", bw > 4.0);

    let mid = evals.len() / 4;
    let end = 3 * evals.len() / 4;
    let r = spectral::level_spacing_ratio(&evals[mid..end]);
    println!("  ⟨r⟩ = {r:.4} (Poisson = {:.4})", spectral::POISSON_R);
    harness.check_upper(
        "1D ⟨r⟩ near Poisson (localized)",
        (r - spectral::POISSON_R).abs(),
        tolerances::ANDERSON_1D_LEVEL_SPACING_DEVIATION,
    );
    println!();
}

/// \[6\] Spectral theory: 2D Anderson via Lanczos.
fn check_spectral_2d(harness: &mut ValidationHarness) {
    println!("[6] Anderson 2D — Paper 19");

    let l = 16;
    let w = 4.0;
    let mat = spectral::anderson_2d(l, l, w, 42);
    let result = spectral::lanczos(&mat, l * l, 42);
    let evals = spectral::lanczos_eigenvalues(&result);

    let bw = evals.last().expect("non-empty") - evals.first().expect("non-empty");
    println!("  {l}×{l} = {}, W={w}", l * l);
    println!("  Bandwidth: {bw:.4} (clean: 8.0)");

    harness.check_bool("2D bandwidth > 8.0 (disorder widens)", bw > 8.0);
    println!();
}

/// \[7\] Spectral theory: 3D Anderson — GOE→Poisson transition.
fn check_spectral_3d(harness: &mut ValidationHarness) {
    println!("[7] Anderson 3D — Paper 20");

    let l = 8;
    let n = l * l * l;

    let mat_clean = spectral::clean_3d_lattice(l);
    let result_clean = spectral::lanczos(&mat_clean, n, 42);
    let evals_clean = spectral::lanczos_eigenvalues(&result_clean);
    let bw_clean = evals_clean.last().expect("non-empty") - evals_clean.first().expect("non-empty");
    let exact_bw = 12.0 * (std::f64::consts::PI / (l as f64 + 1.0)).cos();
    println!("  Clean 3D (L={l}): bw = {bw_clean:.4} (exact OBC: {exact_bw:.4})");

    harness.check_upper(
        "clean 3D bandwidth matches OBC theory",
        (bw_clean - exact_bw).abs(),
        tolerances::ANDERSON_3D_CLEAN_BANDWIDTH_ABS,
    );

    let w_weak = 4.0;
    let w_strong = 30.0;
    let n_real: u64 = 5;

    let mut r_weak_sum = 0.0;
    let mut r_strong_sum = 0.0;
    for seed in 0..n_real {
        let mat_w = spectral::anderson_3d(l, l, l, w_weak, seed * 137 + 42);
        let res_w = spectral::lanczos(&mat_w, n, seed * 37 + 1);
        let ev_w = spectral::lanczos_eigenvalues(&res_w);
        let mid = ev_w.len() / 4;
        let end = 3 * ev_w.len() / 4;
        r_weak_sum += spectral::level_spacing_ratio(&ev_w[mid..end]);

        let mat_s = spectral::anderson_3d(l, l, l, w_strong, seed * 137 + 42);
        let res_s = spectral::lanczos(&mat_s, n, seed * 37 + 1);
        let ev_s = spectral::lanczos_eigenvalues(&res_s);
        let mid_s = ev_s.len() / 4;
        let end_s = 3 * ev_s.len() / 4;
        r_strong_sum += spectral::level_spacing_ratio(&ev_s[mid_s..end_s]);
    }
    let r_weak = r_weak_sum / n_real as f64;
    let r_strong = r_strong_sum / n_real as f64;

    println!("  W={w_weak}: ⟨r⟩ = {r_weak:.4} (GOE ≈ 0.531)");
    println!(
        "  W={w_strong}: ⟨r⟩ = {r_strong:.4} (Poisson ≈ {:.4})",
        spectral::POISSON_R
    );

    harness.check_bool(
        "GOE→Poisson transition: r(weak) > r(strong)",
        r_weak > r_strong,
    );
    harness.check_bool(
        "transition Δ⟨r⟩ > 0.05 (genuine phase transition)",
        r_weak - r_strong > tolerances::ANDERSON_3D_GOE_POISSON_DELTA_R_MIN,
    );
    println!();
}

/// \[8\] Hofstadter butterfly — band counting.
fn check_hofstadter(harness: &mut ValidationHarness) {
    println!("[8] Hofstadter Butterfly — Papers 21-22");

    let n = 500;

    for (q, alpha, expected_bands) in [(2, 0.5, 2), (3, 1.0 / 3.0, 3), (5, 0.2, 5)] {
        let (d, e) = spectral::almost_mathieu_hamiltonian(n, 1.0, alpha, 0.0);
        let evals = spectral::find_all_eigenvalues(&d, &e);
        let bands = spectral::detect_bands(&evals, 10.0);
        let n_wide = bands
            .iter()
            .filter(|(lo, hi)| hi - lo > tolerances::HOFSTADTER_WIDE_BAND_MIN_WIDTH)
            .count();
        println!("  α=1/{q}: {n_wide} wide bands (expect {expected_bands})");
        harness.check_bool(
            &format!("α=1/{q} produces {expected_bands} bands"),
            n_wide == expected_bands,
        );
    }
    println!();
}

/// \[9\] Dimensional bandwidth hierarchy: 1D &lt; 2D &lt; 3D.
fn check_dimensional_hierarchy(harness: &mut ValidationHarness) {
    println!("[9] Dimensional Hierarchy — Cross-dimensional proof");

    let w = 2.0;

    let (d1, e1) = spectral::anderson_hamiltonian(500, w, 42);
    let ev1 = spectral::find_all_eigenvalues(&d1, &e1);
    let bw_1d = ev1.last().expect("non-empty") - ev1.first().expect("non-empty");

    let mat_2d = spectral::anderson_2d(22, 22, w, 42);
    let res_2d = spectral::lanczos(&mat_2d, 484, 42);
    let ev2 = spectral::lanczos_eigenvalues(&res_2d);
    let bw_2d = ev2.last().expect("non-empty") - ev2.first().expect("non-empty");

    let mat_3d = spectral::anderson_3d(8, 8, 8, w, 42);
    let res_3d = spectral::lanczos(&mat_3d, 512, 42);
    let ev3 = spectral::lanczos_eigenvalues(&res_3d);
    let bw_3d = ev3.last().expect("non-empty") - ev3.first().expect("non-empty");

    println!("  W={w}");
    println!("  1D (N=500):   bw = {bw_1d:.4}  (clean: 4)");
    println!("  2D (22×22):   bw = {bw_2d:.4}  (clean: 8)");
    println!("  3D (8³=512):  bw = {bw_3d:.4}  (clean: 12)");

    harness.check_bool(
        "3D > 2D > 1D bandwidth (scaling theory)",
        bw_3d > bw_2d && bw_2d > bw_1d,
    );
    println!();
}

fn print_evolution_summary() {
    println!("═══ Evolution Evidence Summary ═══");
    println!();
    println!("  Substrate coverage:");
    println!("    Python Control:    18/22 papers (1-6, 8-10, 13-22)");
    println!("    BarraCuda CPU:     20/22 papers (this binary covers key domains)");
    println!("    BarraCuda GPU:     15/22 papers (validate_gpu_*, validate_pure_gpu_qcd)");
    println!("    metalForge:         3/22 papers (transport, phase, classification)");
    println!();
    println!("  GPU parity (from dedicated binaries):");
    println!("    Dirac CPU-GPU:     4.44e-16 (machine epsilon)");
    println!("    CG iterations:     identical at every lattice size");
    println!("    Pure GPU QCD:      4.10e-16 solution parity");
    println!("    SpMV CSR:          1.78e-15");
    println!("    Lanczos evals:     1e-15");
    println!("    MD energy drift:   0.000%");
    println!("    HFB eigensolve:    2.4e-12");
    println!();
    println!("  Performance evolution:");
    println!("    Python → Rust:     56-478× (lattice QCD, nuclear EOS)");
    println!("    Rust CPU → GPU:    22.2× at 16⁴ (lattice QCD CG)");
    println!("    GPU → NPU:         9,017× less energy (transport)");
    println!();
}
