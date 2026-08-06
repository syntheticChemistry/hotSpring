// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reversibility test for HMC integrator.
//!
//! Runs N MD steps forward, then N steps backward (negated dt), and measures
//! ||U_final − U_initial|| to verify the integrator is time-reversible.
//!
//! Usage: cargo run --release --features barracuda-local --bin arxiv_reversibility_test

use hotspring_barracuda::lattice::hmc::{HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  HMC Integrator Reversibility Test (C9)                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let dims = [4, 4, 4, 4];
    let beta = 2.3;

    for &n_md in &[10, 20, 40] {
        for &dt in &[0.05, 0.02, 0.01] {
            let mut lattice = Lattice::hot_start(dims, beta, 42);

            // Thermalize briefly
            let mut config = HmcConfig {
                n_md_steps: 10,
                dt: 0.05,
                integrator: IntegratorType::Omelyan,
                seed: 42,
            };
            for _ in 0..20 {
                hotspring_barracuda::lattice::hmc::hmc_trajectory(&mut lattice, &mut config);
            }

            // Save initial state
            let initial_links = lattice.links.clone();
            let initial_plaq = lattice.average_plaquette();

            // Generate fixed momenta for the test
            let vol = lattice.volume();
            let mut momenta: Vec<hotspring_barracuda::lattice::su3::Su3Matrix> =
                Vec::with_capacity(vol * 4);
            let mut seed = 137u64;
            for _ in 0..vol * 4 {
                momenta.push(hotspring_barracuda::lattice::su3::Su3Matrix::random_algebra(
                    &mut seed,
                ));
            }
            let saved_momenta = momenta.clone();

            // Forward integration
            integrate_omelyan(&mut lattice, &mut momenta, n_md, dt);
            let mid_plaq = lattice.average_plaquette();

            // Negate momenta (time reversal)
            for p in &mut momenta {
                *p = p.scale(-1.0);
            }

            // Backward integration (same dt, negated momenta = reverse trajectory)
            integrate_omelyan(&mut lattice, &mut momenta, n_md, dt);

            // Measure deviation
            let final_plaq = lattice.average_plaquette();
            let plaq_diff = (initial_plaq - final_plaq).abs();

            let mut max_link_diff = 0.0f64;
            let mut mean_link_diff = 0.0f64;
            let n_links = lattice.links.len();
            for (i, (u_init, u_final)) in initial_links.iter().zip(lattice.links.iter()).enumerate()
            {
                let diff = *u_init + u_final.scale(-1.0);
                let d = diff.norm_sq().sqrt();
                mean_link_diff += d;
                if d > max_link_diff {
                    max_link_diff = d;
                }
                let _ = i;
            }
            mean_link_diff /= n_links as f64;

            // Also check momentum reversal
            // After forward + backward, momenta should be -P_initial
            // (we negated before backward, so final should be -(-P_init) = P_init... 
            //  but the negation was done manually, so final_P = -P_after_forward_backward)
            // For Omelyan: P_final_forward = P', then we negate to -P', integrate backward
            // giving P'' which should be -P_initial

            let mut max_mom_diff = 0.0f64;
            for (p_init, p_final) in saved_momenta.iter().zip(momenta.iter()) {
                // After: forward with P -> P', negate -> -P', backward -> P''
                // Reversibility: P'' should equal -P_initial (we negated)
                // But we negated once, so P'' should be +P_initial
                // Actually: forward(U,P) -> (U',P'), then backward(U', -P') -> (U, -(-P)) = (U, P)
                // Wait, let's just check:
                let diff = *p_init + p_final.scale(-1.0);
                let d = diff.norm_sq().sqrt();
                if d > max_mom_diff {
                    max_mom_diff = d;
                }
            }

            println!(
                "  N_md={:>2} dt={:.3} | ΔP={:.2e}  max||ΔU||={:.2e}  mean||ΔU||={:.2e}  max||Δπ||={:.2e}",
                n_md, dt, plaq_diff, max_link_diff, mean_link_diff, max_mom_diff
            );
        }
    }

    println!("\n  Interpretation: ||ΔU|| should be O(ε_machine) ≈ 10⁻¹⁵ for f64.");
    println!("  Reversibility is exact to machine precision if the integrator");
    println!("  is symplectic and time-reversible (Omelyan 2MN satisfies both).");
}

fn integrate_omelyan(
    lattice: &mut Lattice,
    momenta: &mut [hotspring_barracuda::lattice::su3::Su3Matrix],
    n_steps: usize,
    dt: f64,
) {
    use hotspring_barracuda::lattice::su3::Su3Matrix;

    let vol = lattice.volume();
    let lam = 0.1931833275037836;
    let half_dt = 0.5 * dt;

    for _step in 0..n_steps {
        update_momenta(lattice, momenta, lam * dt);

        for idx in 0..vol {
            let x = lattice.site_coords(idx);
            for mu in 0..4 {
                let p = momenta[idx * 4 + mu];
                let u = lattice.link(x, mu);
                let exp_p = hotspring_barracuda::lattice::hmc::exp_su3_cayley_pub(&p, half_dt);
                let new_u = (exp_p * u).reunitarize();
                lattice.set_link(x, mu, new_u);
            }
        }

        update_momenta(lattice, momenta, 2.0f64.mul_add(-lam, 1.0) * dt);

        for idx in 0..vol {
            let x = lattice.site_coords(idx);
            for mu in 0..4 {
                let p = momenta[idx * 4 + mu];
                let u = lattice.link(x, mu);
                let exp_p = hotspring_barracuda::lattice::hmc::exp_su3_cayley_pub(&p, half_dt);
                let new_u = (exp_p * u).reunitarize();
                lattice.set_link(x, mu, new_u);
            }
        }

        update_momenta(lattice, momenta, lam * dt);
    }
}

fn update_momenta(
    lattice: &Lattice,
    momenta: &mut [hotspring_barracuda::lattice::su3::Su3Matrix],
    eps: f64,
) {
    let vol = lattice.volume();
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        for mu in 0..4 {
            let force = lattice.gauge_force(x, mu);
            let scaled = force.scale(eps);
            momenta[idx * 4 + mu] = momenta[idx * 4 + mu] + scaled;
        }
    }
}
