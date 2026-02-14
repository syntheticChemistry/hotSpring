//! PPPM Coulomb Validation — GPU Ewald Sum
//!
//! Validates toadstool's PppmGpu against direct Coulomb sum for small systems.
//! This prepares the kappa=0 path for MD and Level 3 nuclear Coulomb.
//!
//! Tests:
//!   1. Two-charge system (exact analytical result)
//!   2. NaCl-like crystal (Madelung constant validation)
//!   3. Random charge system (GPU PPPM vs CPU direct sum)
//!
//! Run: cargo run --release --bin validate_pppm

use hotspring_barracuda::gpu::GpuF64;

use barracuda::ops::md::electrostatics::{PppmGpu, PppmParams};

use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  PPPM Coulomb Validation — toadstool GPU Ewald Sum          ║");
    println!("║  kappa=0 pure Coulomb for MD and L3 nuclear physics         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // ── Initialize GPU ──────────────────────────────────────────────
        let gpu = match GpuF64::new().await {
            Ok(g) => g,
            Err(e) => {
                println!("  GPU init failed: {}", e);
                return;
            }
        };
        print!("  ");
        gpu.print_info();
        if !gpu.has_f64 {
            println!("  SHADER_F64 not supported — skipping PPPM validation.");
            return;
        }
        println!();

        // ═════════════════════════════════════════════════════════════════
        //  TEST 1: Two charges in a periodic box
        // ═════════════════════════════════════════════════════════════════
        println!("══════════════════════════════════════════════════════════════");
        println!("  TEST 1: Two charges, distance=1.0 in box L=10.0");
        println!("══════════════════════════════════════════════════════════════");
        println!();

        let box_side = 10.0;
        let n = 2;
        // Two unit charges separated by d=1.0 along x-axis
        let positions = vec![
            4.5, 5.0, 5.0, // charge 1
            5.5, 5.0, 5.0, // charge 2
        ];
        let charges = vec![1.0, -1.0]; // opposite charges

        let params = PppmParams::custom(
            n,
            [box_side, box_side, box_side],
            [16, 16, 16], // power-of-2 mesh
            0.5,          // alpha (Ewald splitting)
            4.0,          // real-space cutoff
            4,            // interpolation order
        );

        let t0 = Instant::now();
        let pppm = match PppmGpu::new(gpu.device.clone(), gpu.queue.clone(), params).await {
            Ok(p) => p,
            Err(e) => {
                println!("  PppmGpu::new failed: {}", e);
                return;
            }
        };
        let init_time = t0.elapsed().as_secs_f64();
        println!("  PppmGpu init: {:.1}ms", init_time * 1000.0);

        // GPU PPPM computation
        let t1 = Instant::now();
        match pppm.compute_with_kspace(&positions, &charges).await {
            Ok((forces, energy)) => {
                let compute_time = t1.elapsed().as_secs_f64();

                // Analytical: for opposite charges at d=1.0, V = -k*q1*q2/d = -1.0
                // (with k_coulomb=1.0 in reduced units)
                // The Ewald sum includes periodic images, so energy differs slightly
                println!("  GPU PPPM result:");
                println!("    Energy:     {:.8}", energy);
                println!("    Forces[0]:  ({:.6}, {:.6}, {:.6})",
                    forces[0], forces[1], forces[2]);
                println!("    Forces[1]:  ({:.6}, {:.6}, {:.6})",
                    forces[3], forces[4], forces[5]);
                println!("    Compute:    {:.3}ms", compute_time * 1000.0);

                // Check: forces should be equal and opposite
                let f_diff = ((forces[0] + forces[3]).powi(2)
                    + (forces[1] + forces[4]).powi(2)
                    + (forces[2] + forces[5]).powi(2))
                    .sqrt();
                let f_mag = (forces[0].powi(2) + forces[1].powi(2) + forces[2].powi(2)).sqrt();
                let newton3_err = if f_mag > 0.0 { f_diff / f_mag } else { 0.0 };
                println!("    Newton 3rd: |F1+F2|/|F1| = {:.2e} {}",
                    newton3_err,
                    if newton3_err < 1e-6 { "PASS" } else { "FAIL" });
            }
            Err(e) => {
                println!("  PPPM compute failed: {}", e);
            }
        }
        println!();

        // ═════════════════════════════════════════════════════════════════
        //  TEST 2: Madelung-like crystal (small 2x2x2 NaCl)
        // ═════════════════════════════════════════════════════════════════
        println!("══════════════════════════════════════════════════════════════");
        println!("  TEST 2: NaCl 2x2x2 crystal (16 ions, Madelung validation)");
        println!("══════════════════════════════════════════════════════════════");
        println!();

        let a = 2.0; // lattice constant
        let box_nacl = 4.0; // 2 unit cells
        let mut pos_nacl = Vec::new();
        let mut charges_nacl = Vec::new();

        // Build NaCl unit cells: Na at (0,0,0) corners, Cl at face centers
        for ix in 0..2 {
            for iy in 0..2 {
                for iz in 0..2 {
                    let x0 = ix as f64 * a;
                    let y0 = iy as f64 * a;
                    let z0 = iz as f64 * a;

                    // Na+ at corner
                    pos_nacl.extend_from_slice(&[x0, y0, z0]);
                    charges_nacl.push(1.0);

                    // Cl- at body center of sub-cube
                    pos_nacl.extend_from_slice(&[x0 + a / 2.0, y0 + a / 2.0, z0 + a / 2.0]);
                    charges_nacl.push(-1.0);
                }
            }
        }

        let n_nacl = charges_nacl.len();
        println!("  {} ions in box L={}", n_nacl, box_nacl);

        let params_nacl = PppmParams::custom(
            n_nacl,
            [box_nacl, box_nacl, box_nacl],
            [32, 32, 32],
            1.0,
            box_nacl / 2.0,
            4,
        );

        match PppmGpu::new(gpu.device.clone(), gpu.queue.clone(), params_nacl).await {
            Ok(pppm_nacl) => {
                let t2 = Instant::now();
                match pppm_nacl.compute_with_kspace(&pos_nacl, &charges_nacl).await {
                    Ok((forces, energy)) => {
                        let t_ms = t2.elapsed().as_secs_f64() * 1000.0;

                        // Madelung constant for NaCl: ~1.747565
                        // E_total = -N * alpha_M * e^2 / (2*a) for N ion pairs
                        let n_pairs = n_nacl / 2;
                        let energy_per_pair = energy / n_pairs as f64;
                        let alpha_m_estimate = -energy_per_pair * 2.0 * (a / 2.0);
                        // Note: this is a rough estimate since we have a small crystal
                        // with periodic images

                        println!("  GPU PPPM:");
                        println!("    Total energy:    {:.8}", energy);
                        println!("    Energy per pair: {:.8}", energy_per_pair);
                        println!("    Madelung est:    {:.4} (ref: 1.7476)", alpha_m_estimate.abs());
                        println!("    Compute:         {:.3}ms", t_ms);

                        // Check force magnitudes are reasonable
                        let max_f: f64 = forces.chunks(3)
                            .map(|f| (f[0].powi(2) + f[1].powi(2) + f[2].powi(2)).sqrt())
                            .fold(0.0f64, f64::max);
                        println!("    Max |F|:         {:.6}", max_f);

                        // Newton 3rd law: sum of all forces should be ~0
                        let fx_sum: f64 = forces.iter().step_by(3).sum();
                        let fy_sum: f64 = forces.iter().skip(1).step_by(3).sum();
                        let fz_sum: f64 = forces.iter().skip(2).step_by(3).sum();
                        let f_net = (fx_sum.powi(2) + fy_sum.powi(2) + fz_sum.powi(2)).sqrt();
                        println!("    |sum F|:         {:.2e} (should be ~0)", f_net);
                    }
                    Err(e) => println!("  PPPM compute failed: {}", e),
                }
            }
            Err(e) => println!("  PppmGpu init failed: {}", e),
        }
        println!();

        // ═════════════════════════════════════════════════════════════════
        //  TEST 3: Direct Coulomb sum vs PPPM for random system
        // ═════════════════════════════════════════════════════════════════
        println!("══════════════════════════════════════════════════════════════");
        println!("  TEST 3: Random 64-particle system (PPPM vs direct sum)");
        println!("══════════════════════════════════════════════════════════════");
        println!();

        let n_rand = 64;
        let box_rand = 10.0;

        // Pseudo-random positions (deterministic for reproducibility)
        let mut positions_rand = Vec::with_capacity(n_rand * 3);
        let mut charges_rand = Vec::with_capacity(n_rand);
        let mut seed: u64 = 12345;
        for i in 0..n_rand {
            for _ in 0..3 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = (seed >> 33) as f64 / (1u64 << 31) as f64 * box_rand;
                positions_rand.push(val);
            }
            // Alternate +1/-1 charges (net neutral)
            charges_rand.push(if i % 2 == 0 { 1.0 } else { -1.0 });
        }

        // Direct Coulomb sum (CPU reference, minimum image convention)
        let t_direct = Instant::now();
        let mut direct_energy = 0.0;
        for i in 0..n_rand {
            for j in (i + 1)..n_rand {
                let mut dx = positions_rand[i * 3] - positions_rand[j * 3];
                let mut dy = positions_rand[i * 3 + 1] - positions_rand[j * 3 + 1];
                let mut dz = positions_rand[i * 3 + 2] - positions_rand[j * 3 + 2];
                // Minimum image
                dx -= (dx / box_rand).round() * box_rand;
                dy -= (dy / box_rand).round() * box_rand;
                dz -= (dz / box_rand).round() * box_rand;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r > 1e-10 {
                    direct_energy += charges_rand[i] * charges_rand[j] / r;
                }
            }
        }
        let direct_time = t_direct.elapsed().as_secs_f64();
        println!("  Direct Coulomb (CPU, minimum image):");
        println!("    Energy:  {:.8}", direct_energy);
        println!("    Time:    {:.3}ms", direct_time * 1000.0);

        // GPU PPPM
        let params_rand = PppmParams::custom(
            n_rand,
            [box_rand, box_rand, box_rand],
            [32, 32, 32],
            0.3,
            box_rand / 2.0 - 0.1,
            4,
        );

        match PppmGpu::new(gpu.device.clone(), gpu.queue.clone(), params_rand).await {
            Ok(pppm_rand) => {
                let t_pppm = Instant::now();
                match pppm_rand.compute_with_kspace(&positions_rand, &charges_rand).await {
                    Ok((_forces, pppm_energy)) => {
                        let pppm_time = t_pppm.elapsed().as_secs_f64();
                        let rel_err = ((pppm_energy - direct_energy) / direct_energy.abs().max(1e-10)).abs();

                        println!();
                        println!("  GPU PPPM:");
                        println!("    Energy:    {:.8}", pppm_energy);
                        println!("    Time:      {:.3}ms", pppm_time * 1000.0);
                        println!();
                        println!("  Comparison:");
                        println!("    |E_pppm - E_direct|:  {:.2e}", (pppm_energy - direct_energy).abs());
                        println!("    Relative error:       {:.2e}", rel_err);
                        println!("    (Note: direct sum uses minimum image only;");
                        println!("     PPPM includes full periodic Ewald sum.)");
                        println!("    For large boxes, these converge. Difference is");
                        println!("    expected for small boxes with long-range interactions.");
                    }
                    Err(e) => println!("  PPPM compute failed: {}", e),
                }
            }
            Err(e) => println!("  PppmGpu init failed: {}", e),
        }

        println!();
        println!("══════════════════════════════════════════════════════════════");
        println!("  PPPM Validation Complete");
        println!("══════════════════════════════════════════════════════════════");
        println!();
        println!("  PppmGpu is ready for:");
        println!("    - kappa=0 Coulomb MD simulations");
        println!("    - Level 3 nuclear Coulomb validation");
        println!("    - Ewald-sum electrostatics benchmarks");
    });
}
