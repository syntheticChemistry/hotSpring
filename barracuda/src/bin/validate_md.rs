// SPDX-License-Identifier: AGPL-3.0-only

//! Molecular Dynamics Force & Integrator Validation
//!
//! Validates `BarraCUDA`'s WGSL GPU force kernels against analytical CPU references:
//!   - Lennard-Jones (van der Waals)
//!   - Coulomb (electrostatic)
//!   - Morse (bonded/anharmonic)
//!   - Velocity-Verlet (symplectic integrator)
//!
//! Each test computes forces on GPU via WGSL, then compares against exact
//! analytical formulas computed on CPU in f64.
//!
//! **Provenance**: All expected values are analytical force laws, not Python
//! baselines. See `provenance::MD_FORCE_REFS`.

use barracuda::device::WgpuDevice;
use barracuda::ops::md::forces::{CoulombForce, LennardJonesForce, MorseForce};
use barracuda::ops::md::integrators::VelocityVerlet;
use barracuda::tensor::Tensor;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════
// CPU Reference Implementations (f64 analytical)
// ═══════════════════════════════════════════════════════════════════

/// Analytical LJ force between two particles along x-axis
/// F = 24ε/r * [2(σ/r)¹² - (σ/r)⁶]
/// Force on particle 0 from particle 1 (positive = attraction toward 1)
fn lj_force_analytical(r: f64, sigma: f64, epsilon: f64) -> f64 {
    let sr = sigma / r;
    let sr6 = sr.powi(6);
    let sr12 = sr6 * sr6;
    24.0 * epsilon / r * (2.0 * sr12 - sr6)
}

/// Analytical Coulomb force between two particles along x-axis
/// F = k * q1 * q2 / r²
/// Force on particle 0 from particle 1
fn coulomb_force_analytical(r: f64, q1: f64, q2: f64, k: f64, epsilon: f64) -> f64 {
    // Softened: r² → r² + ε
    let r2_soft = r * r + epsilon;
    let r_soft = r2_soft.sqrt();
    // Force magnitude = k * q1 * q2 / r²_soft
    // Direction: like charges repel (negative force on particle 0)
    //            opposite charges attract (positive force on particle 0)
    // In the WGSL shader: force = -k*qi*qj/r² * r_hat (where r_hat points from i to j)
    // So for particle 0: force_x = -k*q1*q2/r²_soft * (r/r_soft)
    -k * q1 * q2 / r2_soft * (r / r_soft)
}

/// Analytical Morse force between two particles along x-axis
/// F = 2Dα[1 - exp(-α(r-r₀))] * exp(-α(r-r₀))
fn morse_force_analytical(r: f64, d: f64, alpha: f64, r0: f64) -> f64 {
    let dr = r - r0;
    let exp_term = (-alpha * dr).exp();
    2.0 * d * alpha * (1.0 - exp_term) * exp_term
}

// ═══════════════════════════════════════════════════════════════════
// Test Framework
// ═══════════════════════════════════════════════════════════════════

struct TestResult {
    name: String,
    passed: bool,
    detail: String,
}

impl TestResult {
    fn pass(name: &str, detail: String) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            detail,
        }
    }
    fn fail(name: &str, detail: String) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            detail,
        }
    }
}

fn check_close(gpu: f32, expected: f64, tol: f64, label: &str) -> TestResult {
    let err = (f64::from(gpu) - expected).abs();
    let rel_err = if expected.abs() > tolerances::EXACT_F64 {
        err / expected.abs()
    } else {
        err
    };
    if rel_err < tol || err < tolerances::MD_ABSOLUTE_FLOOR {
        TestResult::pass(
            label,
            format!("GPU={gpu:.6}, CPU={expected:.6}, rel_err={rel_err:.2e}"),
        )
    } else {
        TestResult::fail(
            label,
            format!("GPU={gpu:.6}, CPU={expected:.6}, rel_err={rel_err:.2e} > tol={tol:.2e}"),
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
// GPU Tests
// ═══════════════════════════════════════════════════════════════════

fn test_lennard_jones(device: &Arc<WgpuDevice>) -> Vec<TestResult> {
    let mut results = Vec::new();

    // ── Test 1: Two argon atoms at equilibrium (r = σ) ──
    // At r = σ: F = 24ε/σ * [2(1)¹² - (1)⁶] = 24ε/σ * (2-1) = 24ε/σ
    {
        let sigma = 3.4_f64; // Angstroms
        let epsilon = 1.0_f64;
        let r = sigma; // At σ

        let positions = vec![0.0_f32, 0.0, 0.0, r as f32, 0.0, 0.0];
        let sigmas = vec![sigma as f32, sigma as f32];
        let epsilons = vec![epsilon as f32, epsilon as f32];

        let pos_t = Tensor::from_data(&positions, vec![2, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let sig_t = Tensor::from_data(&sigmas, vec![2], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let eps_t = Tensor::from_data(&epsilons, vec![2], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");

        let cutoff = 10.0_f32;
        let lj = LennardJonesForce::new(pos_t, sig_t, eps_t, Some(cutoff))
            .expect("LennardJonesForce::new");
        let forces = lj.execute().expect("LennardJonesForce::execute");
        let f = forces.to_vec().expect("tensor read-back from GPU");

        // At r = σ, force is repulsive (positive toward j from j's perspective)
        // Force on particle 0 from particle 1: r_vec = (σ,0,0), r_hat = (1,0,0)
        // F = 24*1/σ * (2*1 - 1) * (1,0,0) = (24/σ, 0, 0)
        let expected_fx = lj_force_analytical(r, sigma, epsilon);

        results.push(check_close(
            f[0],
            expected_fx,
            tolerances::MD_FORCE_TOLERANCE,
            "LJ: 2 atoms at r=σ (Fx particle 0)",
        ));
        results.push(check_close(
            f[1],
            0.0,
            tolerances::MD_FORCE_TOLERANCE,
            "LJ: 2 atoms at r=σ (Fy = 0)",
        ));
        results.push(check_close(
            f[2],
            0.0,
            tolerances::MD_FORCE_TOLERANCE,
            "LJ: 2 atoms at r=σ (Fz = 0)",
        ));

        // Newton's third law
        let n3_err = (f[0] + f[3]).abs();
        results.push(if (n3_err as f64) < tolerances::NEWTON_3RD_LAW_ABS {
            TestResult::pass("LJ: Newton's 3rd law (Fx)", format!("F0+F1={n3_err:.6}"))
        } else {
            TestResult::fail("LJ: Newton's 3rd law (Fx)", format!("F0+F1={n3_err:.6}"))
        });
    }

    // ── Test 2: Two atoms at r = 2^(1/6)*σ (equilibrium, F=0) ──
    {
        let sigma = 3.4_f64;
        let epsilon = 1.0_f64;
        let r = sigma * 2.0_f64.powf(1.0 / 6.0); // ~3.816 Å

        let positions = vec![0.0_f32, 0.0, 0.0, r as f32, 0.0, 0.0];
        let sigmas = vec![sigma as f32, sigma as f32];
        let epsilons = vec![epsilon as f32, epsilon as f32];

        let pos_t = Tensor::from_data(&positions, vec![2, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let sig_t = Tensor::from_data(&sigmas, vec![2], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let eps_t = Tensor::from_data(&epsilons, vec![2], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");

        let lj = LennardJonesForce::new(pos_t, sig_t, eps_t, Some(10.0))
            .expect("LennardJonesForce::new");
        let forces = lj.execute().expect("LennardJonesForce::execute");
        let f = forces.to_vec().expect("tensor read-back from GPU");

        // At equilibrium, force should be ~0
        let f0_mag = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
        results.push(if (f0_mag as f64) < tolerances::MD_EQUILIBRIUM_FORCE_ABS {
            TestResult::pass(
                "LJ: equilibrium r=2^(1/6)σ → F≈0",
                format!("|F|={f0_mag:.6}"),
            )
        } else {
            TestResult::fail(
                "LJ: equilibrium r=2^(1/6)σ → F≈0",
                format!("|F|={f0_mag:.6}"),
            )
        });
    }

    // ── Test 3: Three particles, triangular arrangement ──
    {
        let sigma = 1.0_f64;
        let epsilon = 1.0_f64;
        // Equilateral triangle with side=2.0
        let positions = vec![
            0.0_f32,
            0.0,
            0.0, // particle 0
            2.0,
            0.0,
            0.0, // particle 1
            1.0,
            3.0_f32.sqrt(),
            0.0, // particle 2
        ];
        let sigmas = vec![sigma as f32; 3];
        let epsilons = vec![epsilon as f32; 3];

        let pos_t = Tensor::from_data(&positions, vec![3, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let sig_t = Tensor::from_data(&sigmas, vec![3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let eps_t = Tensor::from_data(&epsilons, vec![3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");

        let lj = LennardJonesForce::new(pos_t, sig_t, eps_t, Some(10.0))
            .expect("LennardJonesForce::new");
        let forces = lj.execute().expect("LennardJonesForce::execute");
        let f = forces.to_vec().expect("tensor read-back from GPU");

        // Total force should be zero (momentum conservation)
        let total_fx = f[0] + f[3] + f[6];
        let total_fy = f[1] + f[4] + f[7];
        let total_fz = f[2] + f[5] + f[8];
        let total_f = (total_fx * total_fx + total_fy * total_fy + total_fz * total_fz).sqrt();

        results.push(if (total_f as f64) < tolerances::MD_FORCE_TOLERANCE {
            TestResult::pass(
                "LJ: 3-particle momentum conservation",
                format!("|F_total|={total_f:.6}"),
            )
        } else {
            TestResult::fail(
                "LJ: 3-particle momentum conservation",
                format!("|F_total|={total_f:.6}"),
            )
        });
    }

    results
}

fn test_coulomb(device: &Arc<WgpuDevice>) -> Vec<TestResult> {
    let mut results = Vec::new();

    // ── Test 1: Two like charges (repulsion) ──
    {
        let r = 2.0_f64;
        let q1 = 1.0_f64;
        let q2 = 1.0_f64;
        let k = 1.0_f64;
        let eps = 1e-6_f64;

        let positions = vec![0.0_f32, 0.0, 0.0, r as f32, 0.0, 0.0];
        let charges = vec![q1 as f32, q2 as f32];

        let pos_t = Tensor::from_data(&positions, vec![2, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let chg_t = Tensor::from_data(&charges, vec![2], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");

        let coulomb = CoulombForce::new(pos_t, chg_t, Some(k as f32), Some(10.0), Some(eps as f32))
            .expect("CoulombForce::new");
        let forces = coulomb.execute().expect("CoulombForce::execute");
        let f = forces.to_vec().expect("tensor read-back from GPU");

        let expected_fx = coulomb_force_analytical(r, q1, q2, k, eps);

        results.push(check_close(
            f[0],
            expected_fx,
            tolerances::MD_FORCE_TOLERANCE,
            "Coulomb: like charges (Fx particle 0)",
        ));

        // Like charges: particle 0 should be repelled in -x direction
        results.push(if f[0] < 0.0 {
            TestResult::pass(
                "Coulomb: like charges repel (F₀ₓ < 0)",
                format!("F₀ₓ={:.6}", f[0]),
            )
        } else {
            TestResult::fail(
                "Coulomb: like charges repel (F₀ₓ < 0)",
                format!("F₀ₓ={:.6}", f[0]),
            )
        });

        // Newton's 3rd law
        let n3 = (f[0] + f[3]).abs();
        results.push(if (n3 as f64) < tolerances::NEWTON_3RD_LAW_ABS {
            TestResult::pass("Coulomb: Newton's 3rd law", format!("|F₀+F₁|={n3:.6}"))
        } else {
            TestResult::fail("Coulomb: Newton's 3rd law", format!("|F₀+F₁|={n3:.6}"))
        });
    }

    // ── Test 2: Opposite charges (attraction) ──
    {
        let r = 2.0_f64;
        let q1 = 1.0_f64;
        let q2 = -1.0_f64;
        let k = 1.0_f64;
        let eps = 1e-6_f64;

        let positions = vec![0.0_f32, 0.0, 0.0, r as f32, 0.0, 0.0];
        let charges = vec![q1 as f32, q2 as f32];

        let pos_t = Tensor::from_data(&positions, vec![2, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let chg_t = Tensor::from_data(&charges, vec![2], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");

        let coulomb = CoulombForce::new(pos_t, chg_t, Some(k as f32), Some(10.0), Some(eps as f32))
            .expect("CoulombForce::new");
        let forces = coulomb.execute().expect("CoulombForce::execute");
        let f = forces.to_vec().expect("tensor read-back from GPU");

        let expected_fx = coulomb_force_analytical(r, q1, q2, k, eps);

        results.push(check_close(
            f[0],
            expected_fx,
            tolerances::MD_FORCE_TOLERANCE,
            "Coulomb: opposite charges (Fx particle 0)",
        ));

        // Opposite charges: particle 0 should be attracted in +x direction
        results.push(if f[0] > 0.0 {
            TestResult::pass(
                "Coulomb: opposite charges attract (F₀ₓ > 0)",
                format!("F₀ₓ={:.6}", f[0]),
            )
        } else {
            TestResult::fail(
                "Coulomb: opposite charges attract (F₀ₓ > 0)",
                format!("F₀ₓ={:.6}", f[0]),
            )
        });
    }

    // ── Test 3: Inverse-square law scaling ──
    {
        let k = 1.0_f64;
        let eps = 1e-6_f64;
        let q = 1.0_f64;

        let r1 = 1.0_f64;
        let r2 = 2.0_f64;

        // Force at r1
        let pos1 = vec![0.0_f32, 0.0, 0.0, r1 as f32, 0.0, 0.0];
        let chg1 = vec![q as f32, q as f32];
        let pt1 = Tensor::from_data(&pos1, vec![2, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let ct1 = Tensor::from_data(&chg1, vec![2], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let c1 = CoulombForce::new(pt1, ct1, Some(k as f32), Some(10.0), Some(eps as f32))
            .expect("CoulombForce::new");
        let f1 = c1
            .execute()
            .expect("CoulombForce::execute")
            .to_vec()
            .expect("tensor read-back from GPU");

        // Force at r2
        let pos2 = vec![0.0_f32, 0.0, 0.0, r2 as f32, 0.0, 0.0];
        let chg2 = vec![q as f32, q as f32];
        let pt2 = Tensor::from_data(&pos2, vec![2, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let ct2 = Tensor::from_data(&chg2, vec![2], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let c2 = CoulombForce::new(pt2, ct2, Some(k as f32), Some(10.0), Some(eps as f32))
            .expect("CoulombForce::new");
        let f2 = c2
            .execute()
            .expect("CoulombForce::execute")
            .to_vec()
            .expect("tensor read-back from GPU");

        // |F(r1)/F(r2)| should ≈ (r2/r1)² = 4.0
        let ratio = (f1[0] / f2[0]).abs();
        let expected_ratio = (r2 / r1).powi(2) as f32;
        let ratio_err = ((ratio - expected_ratio) / expected_ratio).abs();

        results.push(
            if ratio_err < (2.0 * tolerances::MD_FORCE_TOLERANCE) as f32 {
                TestResult::pass(
                    "Coulomb: inverse-square law",
                    format!("|F(r1)/F(r2)|={ratio:.4}, expected={expected_ratio:.4}"),
                )
            } else {
                TestResult::fail(
                    "Coulomb: inverse-square law",
                    format!(
                    "|F(r1)/F(r2)|={ratio:.4}, expected={expected_ratio:.4}, err={ratio_err:.4}"
                ),
                )
            },
        );
    }

    results
}

fn test_morse(device: &Arc<WgpuDevice>) -> Vec<TestResult> {
    let mut results = Vec::new();

    // ── Test 1: At equilibrium (r = r₀, F ≈ 0) ──
    {
        let d = 100.0_f64;
        let alpha = 2.0_f64;
        let r0 = 1.5_f64;
        let r = r0; // At equilibrium

        let positions = vec![0.0_f32, 0.0, 0.0, r as f32, 0.0, 0.0];
        let bond_pairs = vec![0.0_f32, 1.0];
        let dissociation = vec![d as f32];
        let width = vec![alpha as f32];
        let eq_dist = vec![r0 as f32];

        let pos_t = Tensor::from_data(&positions, vec![2, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let pairs_t = Tensor::from_data(&bond_pairs, vec![1, 2], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let d_t = Tensor::from_data(&dissociation, vec![1], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let a_t = Tensor::from_data(&width, vec![1], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let r0_t = Tensor::from_data(&eq_dist, vec![1], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");

        let morse = MorseForce::new(pos_t, pairs_t, d_t, a_t, r0_t).expect("MorseForce::new");
        let forces = morse.execute().expect("MorseForce::execute");
        let f = forces.to_vec().expect("tensor read-back from GPU");

        let f0_mag = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
        let expected_f = morse_force_analytical(r, d, alpha, r0);

        results.push(if (f0_mag as f64) < tolerances::MD_EQUILIBRIUM_FORCE_ABS {
            TestResult::pass(
                "Morse: equilibrium r=r₀ → F≈0",
                format!("|F|={f0_mag:.4}, analytical={expected_f:.6}"),
            )
        } else {
            TestResult::fail(
                "Morse: equilibrium r=r₀ → F≈0",
                format!("|F|={f0_mag:.4}, analytical={expected_f:.6}"),
            )
        });
    }

    // ── Test 2: Stretched bond (r > r₀, attractive force) ──
    {
        let d = 10.0_f64;
        let alpha = 1.0_f64;
        let r0 = 1.0_f64;
        let r = 1.5_f64; // Stretched

        let positions = vec![0.0_f32, 0.0, 0.0, r as f32, 0.0, 0.0];
        let bond_pairs = vec![0.0_f32, 1.0];
        let dissociation = vec![d as f32];
        let width = vec![alpha as f32];
        let eq_dist = vec![r0 as f32];

        let pos_t = Tensor::from_data(&positions, vec![2, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let pairs_t = Tensor::from_data(&bond_pairs, vec![1, 2], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let d_t = Tensor::from_data(&dissociation, vec![1], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let a_t = Tensor::from_data(&width, vec![1], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let r0_t = Tensor::from_data(&eq_dist, vec![1], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");

        let morse = MorseForce::new(pos_t, pairs_t, d_t, a_t, r0_t).expect("MorseForce::new");
        let forces = morse.execute().expect("MorseForce::execute");
        let f = forces.to_vec().expect("tensor read-back from GPU");

        let expected_f = morse_force_analytical(r, d, alpha, r0);

        // Stretched: attractive force on particle 0 toward particle 1 (+x)
        results.push(if f[0] > 0.0 {
            TestResult::pass(
                "Morse: stretched bond attracts (F₀ₓ > 0)",
                format!("F₀ₓ={:.4}, analytical={:.6}", f[0], expected_f),
            )
        } else {
            TestResult::fail(
                "Morse: stretched bond attracts (F₀ₓ > 0)",
                format!("F₀ₓ={:.4}, analytical={:.6}", f[0], expected_f),
            )
        });

        // Newton's 3rd law (note: Morse uses atomic int accumulation with /1000 precision)
        let n3 = (f[0] + f[3]).abs();
        results.push(if (n3 as f64) < tolerances::NEWTON_3RD_LAW_ABS {
            TestResult::pass("Morse: Newton's 3rd law", format!("|F₀+F₁|={n3:.4}"))
        } else {
            TestResult::fail("Morse: Newton's 3rd law", format!("|F₀+F₁|={n3:.4}"))
        });
    }

    // ── Test 3: Compressed bond (r < r₀, repulsive force) ──
    {
        let d = 10.0_f64;
        let alpha = 1.0_f64;
        let r0 = 2.0_f64;
        let r = 1.0_f64; // Compressed

        let positions = vec![0.0_f32, 0.0, 0.0, r as f32, 0.0, 0.0];
        let bond_pairs = vec![0.0_f32, 1.0];
        let dissociation = vec![d as f32];
        let width = vec![alpha as f32];
        let eq_dist = vec![r0 as f32];

        let pos_t = Tensor::from_data(&positions, vec![2, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let pairs_t = Tensor::from_data(&bond_pairs, vec![1, 2], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let d_t = Tensor::from_data(&dissociation, vec![1], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let a_t = Tensor::from_data(&width, vec![1], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let r0_t = Tensor::from_data(&eq_dist, vec![1], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");

        let morse = MorseForce::new(pos_t, pairs_t, d_t, a_t, r0_t).expect("MorseForce::new");
        let forces = morse.execute().expect("MorseForce::execute");
        let f = forces.to_vec().expect("tensor read-back from GPU");

        let expected_f = morse_force_analytical(r, d, alpha, r0);

        // Compressed: repulsive force on particle 0 away from particle 1 (-x)
        results.push(if f[0] < 0.0 {
            TestResult::pass(
                "Morse: compressed bond repels (F₀ₓ < 0)",
                format!("F₀ₓ={:.4}, analytical={:.6}", f[0], expected_f),
            )
        } else {
            TestResult::fail(
                "Morse: compressed bond repels (F₀ₓ < 0)",
                format!("F₀ₓ={:.4}, analytical={:.6}", f[0], expected_f),
            )
        });
    }

    results
}

fn test_velocity_verlet(device: &Arc<WgpuDevice>) -> Vec<TestResult> {
    let mut results = Vec::new();

    // ── Test 1: Free particle (no force → constant velocity) ──
    {
        let dt = 0.01_f32;
        let x0 = 0.0_f32;
        let v0 = 1.0_f32; // 1 unit/step in x

        let positions = vec![x0, 0.0, 0.0];
        let velocities = vec![v0, 0.0, 0.0];
        let forces = vec![0.0_f32, 0.0, 0.0]; // Zero force
        let masses = vec![1.0_f32];

        let pos_t = Tensor::from_data(&positions, vec![1, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let vel_t = Tensor::from_data(&velocities, vec![1, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let f_old_t = Tensor::from_data(&forces, vec![1, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let f_new_t = Tensor::from_data(&forces, vec![1, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let mass_t = Tensor::from_data(&masses, vec![1], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");

        let vv = VelocityVerlet::new(pos_t, vel_t, f_old_t, f_new_t, mass_t, dt)
            .expect("VelocityVerlet::new");
        let (new_pos, new_vel) = vv.execute().expect("VelocityVerlet::execute");

        let p = new_pos
            .to_vec()
            .expect("position tensor read-back from GPU");
        let v = new_vel
            .to_vec()
            .expect("velocity tensor read-back from GPU");

        // After one step: x = x0 + v*dt = 0.01
        let expected_x = x0 + v0 * dt;
        results.push(check_close(
            p[0],
            f64::from(expected_x),
            tolerances::MD_FORCE_TOLERANCE,
            "VV: free particle position",
        ));
        results.push(check_close(
            v[0],
            f64::from(v0),
            tolerances::MD_FORCE_TOLERANCE,
            "VV: free particle velocity",
        ));
    }

    // ── Test 2: Constant force (uniform acceleration) ──
    {
        let dt = 0.01_f32;
        let force = 2.0_f32;
        let mass = 1.0_f32;

        let positions = vec![0.0_f32, 0.0, 0.0];
        let velocities = vec![0.0_f32, 0.0, 0.0];
        let forces_val = vec![force, 0.0, 0.0];
        let masses = vec![mass];

        let pos_t = Tensor::from_data(&positions, vec![1, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let vel_t = Tensor::from_data(&velocities, vec![1, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let f_old_t = Tensor::from_data(&forces_val, vec![1, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let f_new_t = Tensor::from_data(&forces_val, vec![1, 3], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");
        let mass_t = Tensor::from_data(&masses, vec![1], device.clone())
            .expect("Tensor::from_data: GPU buffer allocation");

        let vv = VelocityVerlet::new(pos_t, vel_t, f_old_t, f_new_t, mass_t, dt)
            .expect("VelocityVerlet::new");
        let (new_pos, new_vel) = vv.execute().expect("VelocityVerlet::execute");

        let p = new_pos
            .to_vec()
            .expect("position tensor read-back from GPU");
        let v = new_vel
            .to_vec()
            .expect("velocity tensor read-back from GPU");

        let a = force / mass;
        // VV position: x = x0 + v0*dt + 0.5*a*dt²
        let expected_x = 0.5 * a * dt * dt;
        // VV velocity: v = v0 + 0.5*(a_old + a_new)*dt = a*dt (const force)
        let expected_v = a * dt;

        results.push(check_close(
            p[0],
            f64::from(expected_x),
            tolerances::GPU_VS_CPU_F64,
            "VV: const-force position x=½at²",
        ));
        results.push(check_close(
            v[0],
            f64::from(expected_v),
            tolerances::GPU_VS_CPU_F64,
            "VV: const-force velocity v=at",
        ));
    }

    results
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  BarraCUDA MD Force & Integrator Validation                ║");
    println!("║  GPU WGSL kernels vs. CPU f64 analytical references        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize GPU
    let device = Arc::new(
        WgpuDevice::new()
            .await
            .expect("Failed to create GPU device"),
    );
    println!("  GPU: {}", device.name());
    println!();

    let mut all_results: Vec<TestResult> = Vec::new();

    // ── Lennard-Jones ──
    println!("── Lennard-Jones Forces ────────────────────────────────────");
    let lj_results = test_lennard_jones(&device);
    for r in &lj_results {
        let icon = if r.passed { "✅" } else { "❌" };
        println!("  {} {}: {}", icon, r.name, r.detail);
    }
    all_results.extend(lj_results);
    println!();

    // ── Coulomb ──
    println!("── Coulomb Forces ────────────────────────────────────────");
    let coulomb_results = test_coulomb(&device);
    for r in &coulomb_results {
        let icon = if r.passed { "✅" } else { "❌" };
        println!("  {} {}: {}", icon, r.name, r.detail);
    }
    all_results.extend(coulomb_results);
    println!();

    // ── Morse ──
    println!("── Morse Forces ────────────────────────────────────────────");
    let morse_results = test_morse(&device);
    for r in &morse_results {
        let icon = if r.passed { "✅" } else { "❌" };
        println!("  {} {}: {}", icon, r.name, r.detail);
    }
    all_results.extend(morse_results);
    println!();

    // ── Velocity Verlet ──
    println!("── Velocity-Verlet Integrator ─────────────────────────────");
    let vv_results = test_velocity_verlet(&device);
    for r in &vv_results {
        let icon = if r.passed { "✅" } else { "❌" };
        println!("  {} {}: {}", icon, r.name, r.detail);
    }
    all_results.extend(vv_results);
    println!();

    // ── Summary (with exit code) ──
    let mut harness = ValidationHarness::new("md_forces_integrators");
    for r in &all_results {
        harness.check_bool(&r.name, r.passed);
    }
    let failed: Vec<&TestResult> = all_results.iter().filter(|r| !r.passed).collect();
    if !failed.is_empty() {
        for f in &failed {
            println!("    ❌ {}: {}", f.name, f.detail);
        }
    }
    harness.finish();
}
