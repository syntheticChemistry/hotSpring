// SPDX-License-Identifier: AGPL-3.0-only

//! GPU Physics Proxy Pipeline — Exp 025
//!
//! Runs cheap physics models on a secondary GPU (Titan V) or CPU to generate
//! training data for NPU heads 12-14 (RMT spectral, Potts phase, Anderson CG).
//!
//! The proxy pipeline runs **alongside** the primary dynamical HMC on the RTX
//! 3090, feeding predictions to the NPU with zero impact on the main GPU.
//!
//! # Architecture
//!
//! ```text
//! Titan V / CPU                     NPU (Akida / ESN)
//! ─────────────                     ─────────────────
//! Anderson 3D (disorder W)     →    Head 13: CG cost predictor
//! Z(3) Potts MC (β, L)        →    Head 12: Potts phase classifier
//! Level statistics (⟨r⟩, IPR) →    Head 11: RMT spectral predictor
//! ```
//!
//! # Usage
//!
//! ```bash
//! # Run on Titan V while 3090 does dynamical HMC
//! HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin gpu_physics_proxy -- \
//!   --output=results/exp025_proxy_training.jsonl
//! ```

use hotspring_barracuda::spectral::{
    anderson_3d, find_all_eigenvalues, lanczos_eigenvalues, level_spacing_ratio,
};
use std::io::Write;
use std::time::Instant;

fn main() {
    let mut output_path = "results/exp025_proxy_training.jsonl".to_string();
    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--output=") {
            output_path = val.to_string();
        }
    }

    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&output_path).expect("cannot create output file"),
    );

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Exp 025: Physics Proxy Pipeline — NPU Training Data   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("  Output: {output_path}");
    println!();

    // ── Phase 1: Anderson 3D → CG cost proxy ────────────────────────
    //
    // The Dirac operator in lattice QCD behaves like an Anderson Hamiltonian
    // where gauge fluctuations act as effective disorder. Plaquette variance
    // maps to disorder W. Level statistics (⟨r⟩) predict CG difficulty:
    // GOE (extended, easy CG) vs Poisson (localized, hard CG).

    println!("═══ Phase 1: Anderson 3D — CG Cost Proxy ═══");
    println!();

    let anderson_lattices: Vec<usize> = vec![8, 10, 12];
    let disorders: Vec<f64> = vec![
        1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 16.5, 17.0, 18.0, 20.0, 25.0,
    ];
    let anderson_seeds: Vec<u64> = vec![42, 137, 271];

    let total_anderson =
        anderson_lattices.len() * disorders.len() * anderson_seeds.len();
    let mut anderson_idx = 0;

    let phase1_start = Instant::now();

    for &l in &anderson_lattices {
        for &w in &disorders {
            for &seed in &anderson_seeds {
                anderson_idx += 1;
                let t0 = Instant::now();

                let h = anderson_3d(l, l, l, w, seed);
                let n = l * l * l;

                let eigenvalues = if n <= 1000 {
                    let tri = hotspring_barracuda::spectral::lanczos(&h, n.min(h.n), seed);
                    find_all_eigenvalues(&tri.alpha, &tri.beta)
                } else {
                    let k = 200.min(n);
                    let tri = hotspring_barracuda::spectral::lanczos(&h, k, seed);
                    lanczos_eigenvalues(&tri)
                };

                let r = level_spacing_ratio(&eigenvalues);
                let bandwidth = if eigenvalues.len() >= 2 {
                    eigenvalues.last().unwrap() - eigenvalues.first().unwrap()
                } else {
                    0.0
                };

                let lambda_min = eigenvalues.iter().map(|e| e.abs()).fold(f64::MAX, f64::min);

                let ipr = compute_ipr_from_stats(&eigenvalues);

                let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let phase_label = if r > 0.48 {
                    "extended"
                } else if r < 0.42 {
                    "localized"
                } else {
                    "critical"
                };

                let result = serde_json::json!({
                    "proxy": "anderson_3d",
                    "lattice": l,
                    "volume": n,
                    "disorder": w,
                    "seed": seed,
                    "level_spacing_ratio": r,
                    "bandwidth": bandwidth,
                    "lambda_min_abs": lambda_min,
                    "ipr_estimate": ipr,
                    "phase": phase_label,
                    "n_eigenvalues": eigenvalues.len(),
                    "wall_ms": wall_ms,
                });

                writeln!(out, "{}", serde_json::to_string(&result).unwrap()).ok();

                let status = match phase_label {
                    "extended" => "E",
                    "localized" => "L",
                    _ => "C",
                };

                if anderson_idx % 5 == 0 || anderson_idx == total_anderson {
                    println!(
                        "  [{anderson_idx:3}/{total_anderson}] L={l} W={w:5.1} ⟨r⟩={r:.3} |λ|_min={lambda_min:.3} [{status}] ({wall_ms:.0}ms)"
                    );
                }
            }
        }
    }

    out.flush().ok();
    let phase1_wall = phase1_start.elapsed().as_secs_f64();
    println!();
    println!("  Phase 1 complete: {anderson_idx} points in {phase1_wall:.1}s");
    println!();

    // ── Phase 2: Z(3) Potts Monte Carlo → Phase Classifier ─────────
    //
    // The SU(3) deconfinement transition maps onto the Z(3) Potts model
    // (Svetitsky-Yaffe 1982). A cheap Potts MC provides phase labels
    // and β_c estimates at negligible cost.

    println!("═══ Phase 2: Z(3) Potts Monte Carlo — Phase Classifier ═══");
    println!();

    let potts_lattices: Vec<usize> = vec![8, 16, 24];
    let potts_betas: Vec<f64> = vec![
        0.3, 0.4, 0.5, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.80, 0.90, 1.0,
    ];
    let potts_therm = 500;
    let potts_meas = 200;

    let total_potts = potts_lattices.len() * potts_betas.len();
    let mut potts_idx = 0;
    let phase2_start = Instant::now();

    for &l in &potts_lattices {
        for &beta_potts in &potts_betas {
            potts_idx += 1;
            let t0 = Instant::now();

            let (magnetization, susceptibility, energy, phase_label) =
                potts_z3_monte_carlo(l, beta_potts, potts_therm, potts_meas, 42);

            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let result = serde_json::json!({
                "proxy": "potts_z3",
                "lattice": l,
                "volume": l * l * l,
                "beta_potts": beta_potts,
                "magnetization": magnetization,
                "susceptibility": susceptibility,
                "energy": energy,
                "phase": phase_label,
                "therm": potts_therm,
                "meas": potts_meas,
                "wall_ms": wall_ms,
            });

            writeln!(out, "{}", serde_json::to_string(&result).unwrap()).ok();

            let status = match phase_label.as_str() {
                "ordered" => "O",
                "disordered" => "D",
                _ => "T",
            };

            println!(
                "  [{potts_idx:3}/{total_potts}] L={l:2} β_P={beta_potts:.2} |M|={magnetization:.3} χ={susceptibility:.1} [{status}] ({wall_ms:.0}ms)"
            );
        }
    }

    out.flush().ok();
    let phase2_wall = phase2_start.elapsed().as_secs_f64();
    println!();
    println!("  Phase 2 complete: {potts_idx} points in {phase2_wall:.1}s");
    println!();

    // ── Phase 3: Mapping Table (Potts β ↔ QCD β) ───────────────────

    println!("═══ Phase 3: Potts ↔ QCD Mapping Table ═══");
    println!();

    // Svetitsky-Yaffe: SU(3) Nt=4 deconfinement maps to 3D Z(3) Potts.
    // β_QCD ≈ 5.69 → β_Potts ≈ 0.55 (at the critical point).
    // The mapping is approximate; the NPU learns the correction.
    let mapping = vec![
        (4.0, 0.30, "confined/ordered"),
        (4.5, 0.35, "confined/ordered"),
        (5.0, 0.42, "confined/ordered"),
        (5.3, 0.48, "near-transition"),
        (5.5, 0.52, "transition"),
        (5.69, 0.55, "critical"),
        (5.8, 0.60, "near-transition"),
        (6.0, 0.65, "deconfined/disordered"),
        (6.5, 0.80, "deconfined/disordered"),
    ];

    for (beta_qcd, beta_potts, label) in &mapping {
        let result = serde_json::json!({
            "proxy": "potts_qcd_map",
            "beta_qcd": beta_qcd,
            "beta_potts": beta_potts,
            "expected_phase": label,
        });
        writeln!(out, "{}", serde_json::to_string(&result).unwrap()).ok();
        println!("  β_QCD={beta_qcd:.2} → β_Potts={beta_potts:.2} [{label}]");
    }

    out.flush().ok();
    let total_wall = phase1_start.elapsed().as_secs_f64();

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Proxy pipeline complete: {total_wall:.1}s ({:.1} min)", total_wall / 60.0);
    println!("  Anderson points: {anderson_idx}");
    println!("  Potts points:    {potts_idx}");
    println!("  Mapping entries: {}", mapping.len());
    println!("  Output: {output_path}");
    println!("═══════════════════════════════════════════════════════════");
}

/// Estimate IPR from eigenvalue density: closely-spaced eigenvalues
/// indicate extended states (low IPR), isolated ones indicate localized (high IPR).
fn compute_ipr_from_stats(eigenvalues: &[f64]) -> f64 {
    if eigenvalues.len() < 3 {
        return 1.0;
    }
    let n = eigenvalues.len() as f64;
    let mut spacings: Vec<f64> = eigenvalues
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .collect();
    spacings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = spacings[spacings.len() / 2];
    if median < 1e-15 {
        return 1.0 / n;
    }
    let var = spacings.iter().map(|s| (s - median).powi(2)).sum::<f64>() / spacings.len() as f64;
    let cv = var.sqrt() / median;
    // GOE: cv ≈ 0.52, Poisson: cv ≈ 1.0
    // Map to approximate IPR: extended → 1/N, localized → O(1)
    let ipr = (1.0 / n) + (1.0 - 1.0 / n) * (cv - 0.52).max(0.0) / 0.48;
    ipr.clamp(1.0 / n, 1.0)
}

/// 3D Z(3) Potts Monte Carlo on an L^3 cubic lattice.
///
/// Returns (mean |magnetization|, susceptibility, mean energy, phase label).
fn potts_z3_monte_carlo(
    l: usize,
    beta: f64,
    n_therm: usize,
    n_meas: usize,
    seed: u64,
) -> (f64, f64, f64, String) {
    let n = l * l * l;
    let mut spins: Vec<u8> = vec![0; n];
    let mut rng = LcgRng::new(seed);

    for s in &mut spins {
        *s = (rng.next_u64() % 3) as u8;
    }

    let idx = |x: usize, y: usize, z: usize| -> usize {
        (x % l) + (y % l) * l + (z % l) * l * l
    };

    let neighbors = |site: usize| -> [usize; 6] {
        let x = site % l;
        let y = (site / l) % l;
        let z = site / (l * l);
        [
            idx((x + 1) % l, y, z),
            idx((x + l - 1) % l, y, z),
            idx(x, (y + 1) % l, z),
            idx(x, (y + l - 1) % l, z),
            idx(x, y, (z + 1) % l),
            idx(x, y, (z + l - 1) % l),
        ]
    };

    // Metropolis sweeps
    for _ in 0..(n_therm + n_meas) * n {
        // Intentionally empty — we'll do sweeps below
    }

    // Actually do proper sweeps
    let mut mag_samples = Vec::with_capacity(n_meas);
    let mut energy_samples = Vec::with_capacity(n_meas);

    for sweep in 0..(n_therm + n_meas) {
        // One Metropolis sweep
        for site in 0..n {
            let old_spin = spins[site];
            let new_spin = ((old_spin as u64 + 1 + rng.next_u64() % 2) % 3) as u8;

            let nbrs = neighbors(site);
            let mut delta_e: i32 = 0;
            for &nb in &nbrs {
                if spins[nb] == new_spin {
                    delta_e -= 1;
                }
                if spins[nb] == old_spin {
                    delta_e += 1;
                }
            }

            let accept = if delta_e <= 0 {
                true
            } else {
                rng.uniform() < (-beta * delta_e as f64).exp()
            };

            if accept {
                spins[site] = new_spin;
            }
        }

        if sweep >= n_therm {
            // Measure magnetization (Z(3) order parameter)
            let mut m = [0.0f64; 2]; // real, imag of Z(3) magnetization
            let omega_re = [-0.5, -0.5, 1.0]; // Re(ω^q) for q=0,1,2
            let omega_im = [
                3.0_f64.sqrt() / 2.0,
                -(3.0_f64.sqrt()) / 2.0,
                0.0,
            ];
            for &s in &spins {
                m[0] += omega_re[s as usize];
                m[1] += omega_im[s as usize];
            }
            let mag = (m[0] * m[0] + m[1] * m[1]).sqrt() / n as f64;
            mag_samples.push(mag);

            // Measure energy
            let mut energy = 0.0f64;
            for site in 0..n {
                let nbrs = neighbors(site);
                for &nb in &nbrs {
                    if spins[nb] == spins[site] {
                        energy -= 1.0;
                    }
                }
            }
            energy /= 2.0 * n as f64; // double-counting + per-site
            energy_samples.push(energy);
        }
    }

    let mean_mag = mag_samples.iter().sum::<f64>() / mag_samples.len() as f64;
    let mean_mag2 = mag_samples.iter().map(|m| m * m).sum::<f64>() / mag_samples.len() as f64;
    let chi = (mean_mag2 - mean_mag * mean_mag) * n as f64;
    let mean_energy = energy_samples.iter().sum::<f64>() / energy_samples.len() as f64;

    let phase = if mean_mag > 0.5 {
        "ordered".to_string()
    } else if mean_mag < 0.2 {
        "disordered".to_string()
    } else {
        "transition".to_string()
    };

    (mean_mag, chi, mean_energy, phase)
}

struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(6364136223846793005).wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
