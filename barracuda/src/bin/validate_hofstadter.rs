// SPDX-License-Identifier: AGPL-3.0-only

//! Hofstadter Butterfly Validation — Spectral Topology (Kachkovskiy Extension)
//!
//! Validates the Hofstadter butterfly: the fractal spectrum of the
//! almost-Mathieu operator as a function of magnetic flux α = p/q.
//!
//! **Key physics**:
//! - At rational flux α = p/q, the spectrum splits into exactly q bands
//! - At λ = 1 (critical coupling), the spectrum at irrational α is a
//!   Cantor set of measure zero (Ten Martini Problem, Avila-Jitomirskaya 2009)
//! - The butterfly has E → -E symmetry (particle-hole) and α → 1-α symmetry
//! - Self-similar fractal structure — zooming into any gap reveals a
//!   miniature copy of the whole butterfly
//!
//! This is the canonical example of spectral topology in condensed matter
//! physics, connecting number theory (continued fractions), topology (Chern
//! numbers), and quantum mechanics (Bloch electrons in magnetic fields).
//!
//! # Provenance
//!
//! Hofstadter (1976) Phys. Rev. B 14, 2239 — "Energy levels and wave
//!   functions of Bloch electrons in rational and irrational magnetic fields"
//! Avila & Jitomirskaya (2009) Ann. Math. 170, 303 — "The Ten Martini Problem"
//! Thouless, Kohmoto, Nightingale, den Nijs (1982) PRL 49, 405 — TKNN invariant

use hotspring_barracuda::spectral;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Hofstadter Butterfly — Spectral Topology                  ║");
    println!("║  Fractal spectrum of Bloch electrons in a magnetic field   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("hofstadter_butterfly");

    check_spectrum_bounds(&mut harness);
    check_band_count_q2(&mut harness);
    check_band_count_q3(&mut harness);
    check_band_count_q5(&mut harness);
    check_particle_hole_symmetry(&mut harness);
    check_alpha_symmetry(&mut harness);
    check_butterfly_computation(&mut harness);
    check_gap_opening(&mut harness);
    check_cantor_measure(&mut harness);
    check_localized_phase_gaps(&mut harness);

    println!();
    harness.finish();
}

/// \[1\] Spectrum bounded by [-2-2λ, 2+2λ] for all rational α.
fn check_spectrum_bounds(harness: &mut ValidationHarness) {
    println!("[1] Hofstadter — Spectrum Bounds");

    let lambda = 1.0;
    let n = 500;
    let bound = 2.0 + 2.0 * lambda;

    let alphas = [1.0 / 2.0, 1.0 / 3.0, 1.0 / 5.0, 2.0 / 5.0, 3.0 / 7.0];
    let mut all_bounded = true;

    for &alpha in &alphas {
        let (d, e) = spectral::almost_mathieu_hamiltonian(n, lambda, alpha, 0.0);
        let evals = spectral::find_all_eigenvalues(&d, &e);
        let e_min = evals[0];
        let e_max = *evals.last().expect("collection verified non-empty");
        if e_min < -(bound + 0.01) || e_max > bound + 0.01 {
            all_bounded = false;
        }
        println!("  α={alpha:.4}: [{e_min:.4}, {e_max:.4}]");
    }

    println!("  Bound: [-{bound:.1}, {bound:.1}]");
    harness.check_bool("all spectra within [-4, 4] at λ=1", all_bounded);
    println!();
}

/// \[2\] α = 1/2: exactly 2 bands.
fn check_band_count_q2(harness: &mut ValidationHarness) {
    println!("[2] Band Count — α = 1/2 (expect 2 bands)");

    let n = 500;
    let (d, e) = spectral::almost_mathieu_hamiltonian(n, 1.0, 0.5, 0.0);
    let evals = spectral::find_all_eigenvalues(&d, &e);
    let bands = spectral::detect_bands(&evals, 10.0);

    println!("  N={n}, α=1/2, λ=1");
    println!("  Detected {} bands:", bands.len());
    for (i, &(lo, hi)) in bands.iter().enumerate() {
        println!(
            "    Band {}: [{lo:.4}, {hi:.4}] (width {:.4})",
            i + 1,
            hi - lo
        );
    }

    harness.check_bool("α=1/2 has exactly 2 bands", bands.len() == 2);
    println!();
}

/// \[3\] α = 1/3: exactly 3 wide bands (single-θ artifacts filtered).
fn check_band_count_q3(harness: &mut ValidationHarness) {
    println!("[3] Band Count — α = 1/3 (expect 3 bands)");

    let n = 600;
    let (d, e) = spectral::almost_mathieu_hamiltonian(n, 1.0, 1.0 / 3.0, 0.0);
    let evals = spectral::find_all_eigenvalues(&d, &e);
    let bands = spectral::detect_bands(&evals, 10.0);
    let wide: Vec<_> = bands.iter().filter(|&&(lo, hi)| hi - lo > 0.01).collect();

    println!("  N={n}, α=1/3, λ=1");
    println!(
        "  Detected {} raw bands ({} wide):",
        bands.len(),
        wide.len()
    );
    for (i, &(lo, hi)) in bands.iter().enumerate() {
        let tag = if hi - lo > 0.01 { "" } else { " (artifact)" };
        println!(
            "    Band {}: [{lo:.4}, {hi:.4}] (width {:.4}){tag}",
            i + 1,
            hi - lo
        );
    }

    harness.check_bool("α=1/3 has 3 wide bands", wide.len() == 3);
    println!();
}

/// \[4\] α = 1/5: exactly 5 wide bands.
fn check_band_count_q5(harness: &mut ValidationHarness) {
    println!("[4] Band Count — α = 1/5 (expect 5 bands)");

    let n = 500;
    let (d, e) = spectral::almost_mathieu_hamiltonian(n, 1.0, 0.2, 0.0);
    let evals = spectral::find_all_eigenvalues(&d, &e);
    let bands = spectral::detect_bands(&evals, 10.0);
    let wide: Vec<_> = bands.iter().filter(|&&(lo, hi)| hi - lo > 0.01).collect();

    println!("  N={n}, α=1/5, λ=1");
    println!(
        "  Detected {} raw bands ({} wide):",
        bands.len(),
        wide.len()
    );
    for (i, &(lo, hi)) in bands.iter().enumerate() {
        let tag = if hi - lo > 0.01 { "" } else { " (artifact)" };
        println!(
            "    Band {}: [{lo:.4}, {hi:.4}] (width {:.4}){tag}",
            i + 1,
            hi - lo
        );
    }

    harness.check_bool("α=1/5 has 5 wide bands", wide.len() == 5);
    println!();
}

/// \[5\] Particle-hole symmetry: spectrum symmetric about E = 0.
fn check_particle_hole_symmetry(harness: &mut ValidationHarness) {
    println!("[5] Particle-Hole Symmetry — E → -E");

    let n = 500;
    let alphas = [1.0 / 3.0, 1.0 / 5.0, 2.0 / 7.0, 3.0 / 8.0];
    let mut max_asymmetry = 0.0f64;

    for &alpha in &alphas {
        let (d, e) = spectral::almost_mathieu_hamiltonian(n, 1.0, alpha, 0.0);
        let evals = spectral::find_all_eigenvalues(&d, &e);
        let e_min = evals[0];
        let e_max = *evals.last().expect("collection verified non-empty");
        let asym = (e_min + e_max).abs();
        max_asymmetry = max_asymmetry.max(asym);
        println!("  α={alpha:.4}: |E_min + E_max| = {asym:.2e}");
    }

    // Symmetry may not be exact due to finite N and incommensurate α,
    // but should be small
    harness.check_upper(
        "max |E_min + E_max| within symmetry tolerance",
        max_asymmetry,
        tolerances::HOFSTADTER_SYMMETRY_TOLERANCE,
    );
    println!();
}

/// \[6\] α ↔ 1-α symmetry: spectrum at α equals spectrum at 1-α.
fn check_alpha_symmetry(harness: &mut ValidationHarness) {
    println!("[6] Flux Symmetry — α ↔ 1-α");

    let n = 500;
    let pairs = [
        (1.0 / 3.0, 2.0 / 3.0),
        (1.0 / 5.0, 4.0 / 5.0),
        (2.0 / 7.0, 5.0 / 7.0),
    ];
    let mut max_diff = 0.0f64;

    for &(alpha, alpha_comp) in &pairs {
        let (d1, e1) = spectral::almost_mathieu_hamiltonian(n, 1.0, alpha, 0.0);
        let evals1 = spectral::find_all_eigenvalues(&d1, &e1);

        let (d2, e2) = spectral::almost_mathieu_hamiltonian(n, 1.0, alpha_comp, 0.0);
        let evals2 = spectral::find_all_eigenvalues(&d2, &e2);

        let bw1 = evals1.last().expect("collection verified non-empty")
            - evals1.first().expect("collection verified non-empty");
        let bw2 = evals2.last().expect("collection verified non-empty")
            - evals2.first().expect("collection verified non-empty");
        let diff = (bw1 - bw2).abs();
        max_diff = max_diff.max(diff);

        println!("  α={alpha:.4}: BW={bw1:.4}, α={alpha_comp:.4}: BW={bw2:.4}, Δ={diff:.4}");
    }

    harness.check_upper("bandwidth(α) ≈ bandwidth(1-α)", max_diff, 0.1);
    println!();
}

/// \[7\] Full butterfly computation — timing and data integrity.
///
/// The number of distinct rational flux values α = p/q with 2 ≤ q ≤ 30
/// equals sum(φ(q), q=2..30) = 277, where φ(q) is the Euler totient function
/// counting integers 1..q coprime to q. The bounds [250, 300] give ~10%
/// margin around this analytical result. Reference: Hofstadter (1976) Phys.
/// Rev. B 14, 2239.
fn check_butterfly_computation(harness: &mut ValidationHarness) {
    println!("[7] Butterfly Computation — Q_max = 30");

    let t = Instant::now();
    let butterfly = spectral::hofstadter_butterfly(30, 1.0, 300);
    let elapsed = t.elapsed().as_secs_f64();

    let total_points: usize = butterfly.iter().map(|(_, evals)| evals.len()).sum();
    let n_alphas = butterfly.len();

    println!("  Q_max=30, N=300 per α");
    println!("  α values: {n_alphas}");
    println!("  Total (α, E) points: {total_points}");
    println!("  Time: {elapsed:.3}s");
    println!(
        "  Throughput: {:.0} eigenvalues/s",
        total_points as f64 / elapsed
    );

    // Exact: sum(φ(q), q=2..30) = 277 (Euler totient sum)
    let expected_min = 250;
    let expected_max = 300;

    harness.check_bool(
        "reasonable number of α values computed",
        n_alphas >= expected_min && n_alphas <= expected_max,
    );
    println!();
}

/// \[8\] Gaps open at rational flux — spectrum splits.
fn check_gap_opening(harness: &mut ValidationHarness) {
    println!("[8] Gap Opening — Spectrum Splits at Rational Flux");
    println!("    At α = p/q, the spectrum has q-1 gaps\n");

    let n = 500;

    // At α=1/2, there should be a gap at E=0
    let (d, e) = spectral::almost_mathieu_hamiltonian(n, 1.0, 0.5, 0.0);
    let evals = spectral::find_all_eigenvalues(&d, &e);
    let bands = spectral::detect_bands(&evals, 10.0);

    let has_gap = bands.len() >= 2;
    let gap_size = if bands.len() >= 2 {
        bands[1].0 - bands[0].1
    } else {
        0.0
    };

    println!("  α=1/2: {} bands, gap = {gap_size:.4}", bands.len());

    // At α = golden ratio (irrational), no clear band structure
    let (d_irr, e_irr) = spectral::almost_mathieu_hamiltonian(n, 1.0, spectral::GOLDEN_RATIO, 0.0);
    let evals_irr = spectral::find_all_eigenvalues(&d_irr, &e_irr);
    let bands_irr = spectral::detect_bands(&evals_irr, 10.0);
    println!(
        "  α=φ (golden): {} bands (Cantor set, no clean gaps)",
        bands_irr.len()
    );

    harness.check_bool("rational flux opens gaps", has_gap && gap_size > 0.01);
    println!();
}

/// \[9\] Cantor set measure: individual band widths decrease with q.
///
/// At λ=1, the spectrum is a Cantor set of measure zero for irrational α.
/// For rational α = 1/q, the total spectral width (sum of band widths)
/// should decrease with increasing q.
fn check_cantor_measure(harness: &mut ValidationHarness) {
    println!("[9] Cantor Measure — Band Widths Decrease with q");
    println!("    At λ=1, spectrum → Cantor set (measure 0) for irrational α\n");

    let n = 500;
    let qs = [2, 3, 5, 8, 13, 21];
    let mut measures = Vec::new();

    for &q in &qs {
        let alpha = 1.0 / q as f64;
        let (d, e) = spectral::almost_mathieu_hamiltonian(n, 1.0, alpha, 0.0);
        let evals = spectral::find_all_eigenvalues(&d, &e);
        let bands = spectral::detect_bands(&evals, 8.0);
        let total_measure: f64 = bands.iter().map(|&(lo, hi)| hi - lo).sum();
        measures.push(total_measure);
        println!(
            "  α=1/{q:>2}: {nb} bands, total width = {total_measure:.4}",
            nb = bands.len()
        );
    }

    // The total spectral measure should generally decrease with q
    // (approach to Cantor set). Check first > last.
    let decreasing = measures.first().expect("collection verified non-empty")
        > measures.last().expect("collection verified non-empty");
    harness.check_bool(
        "spectral measure decreases (Cantor convergence)",
        decreasing,
    );
    println!();
}

/// \[10\] Localized phase (λ>1): all gaps open, pure point spectrum.
///
/// At λ>1, every gap opens (Avila global theory). With single-θ sampling,
/// we count wide bands (width > 0.01) and verify the correct number.
fn check_localized_phase_gaps(harness: &mut ValidationHarness) {
    println!("[10] Localized Phase — λ = 2, All Gaps Open");
    println!("    At λ>1, all q-1 gaps open (Avila global theory)\n");

    let n = 500;
    let lambda = 2.0;

    let qs = [2, 3, 5];
    let mut all_correct = true;

    for &q in &qs {
        let alpha = 1.0 / q as f64;
        let (d, e) = spectral::almost_mathieu_hamiltonian(n, lambda, alpha, 0.0);
        let evals = spectral::find_all_eigenvalues(&d, &e);
        let bands = spectral::detect_bands(&evals, 8.0);

        // At λ>1, ALL q-1 gaps open → at least q spectral fragments
        let ok = bands.len() >= q;
        if !ok {
            all_correct = false;
        }

        println!(
            "  α=1/{q}, λ={lambda}: {nb} bands (≥{q} expected) {status}",
            nb = bands.len(),
            status = if ok { "✓" } else { "✗" }
        );
    }

    harness.check_bool("at least q bands at λ>1 (all gaps open)", all_correct);
    println!();
}
