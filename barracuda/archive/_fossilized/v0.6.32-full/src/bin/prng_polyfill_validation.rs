// SPDX-License-Identifier: AGPL-3.0-or-later

//! PRNG Polyfill Validation — Box-Muller distribution characterization
//!
//! Dispatches the GPU PRNG shader directly on both NVIDIA and AMD GPUs,
//! reads back raw momenta, and analyzes the statistical distribution.
//! Compares against expected N(0, 1/√2) for off-diagonal su(3) components.
//!
//! Purpose: Quantify the polyfill bias identified in the plaquette divergence
//! root-cause (AAR: STRANDGATE_PLAQUETTE_DIVERGENCE_ROOT_CAUSE_AAR.md).

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::GpuHmcStreamingPipelines;
use std::time::Instant;

const WGSL_PRNG_STANDALONE: &str = include_str!(
    "../../../../../primals/barraCuda/crates/barracuda/src/shaders/lattice/su3_random_momenta_f64.wgsl"
);

fn analyze_distribution(data: &[f64], label: &str, expected_sigma: f64) {
    let n = data.len() as f64;
    let mean: f64 = data.iter().sum::<f64>() / n;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let sigma = var.sqrt();

    let skewness: f64 = data.iter().map(|x| ((x - mean) / sigma).powi(3)).sum::<f64>() / n;
    let kurtosis: f64 =
        data.iter().map(|x| ((x - mean) / sigma).powi(4)).sum::<f64>() / n - 3.0;

    let sigma_err = expected_sigma / (2.0 * n).sqrt();
    let mean_err = expected_sigma / n.sqrt();

    let mean_z = mean / mean_err;
    let sigma_z = (sigma - expected_sigma) / sigma_err;

    println!("    {label}:");
    println!("      N = {}", data.len());
    println!("      mean     = {mean:+.8e}  (expected 0, z = {mean_z:+.2})");
    println!(
        "      σ        = {sigma:.8}  (expected {expected_sigma:.8}, z = {sigma_z:+.2})"
    );
    println!("      variance = {var:.8}  (expected {:.8})", expected_sigma.powi(2));
    println!("      skewness = {skewness:+.6}  (expected 0)");
    println!("      kurtosis = {kurtosis:+.6}  (expected 0)");
    println!();

    // Bin test: count samples in [-3σ, -2σ, -σ, 0, σ, 2σ, 3σ]
    let bins = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let expected_fracs = [0.0013, 0.0215, 0.1359, 0.3413, 0.3413, 0.1359, 0.0215, 0.0013];
    let mut counts = vec![0usize; 8];
    for &x in data {
        let z = (x - mean) / sigma;
        let bin = bins.iter().position(|&b| z < b).unwrap_or(7);
        counts[bin] += 1;
    }
    let n_total = data.len() as f64;
    println!("      Bin distribution (z-score intervals):");
    println!("      {:>8} {:>8} {:>8} {:>8}", "Range", "Count", "Observed", "Expected");
    let labels = ["<-3σ", "-3..-2σ", "-2..-1σ", "-1..0σ", "0..1σ", "1..2σ", "2..3σ", ">3σ"];
    for i in 0..8 {
        let obs = counts[i] as f64 / n_total;
        println!(
            "      {:>8} {:>8} {:>8.4} {:>8.4}",
            labels[i], counts[i], obs, expected_fracs[i]
        );
    }
    println!();
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  PRNG Polyfill Validation — Box-Muller Distribution Test   ║");
    println!("║  GPU PRNG vs expected N(0, 1/√2) for su(3) momenta         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));

    let gpu_hints = &["nvidia", "amd"];
    let n_links: u32 = 1024; // 4⁴ lattice
    let n_dispatches: u32 = 5;
    let expected_sigma = std::f64::consts::FRAC_1_SQRT_2; // 1/√2 ≈ 0.7071

    for hint in gpu_hints {
        let gpu = match rt.block_on(GpuF64::with_adapter(hint)) {
            Ok(g) => g,
            Err(e) => {
                println!("  GPU ({hint}) not available: {e}");
                continue;
            }
        };

        println!("═══════════════════════════════════════════════════════════════");
        println!("  GPU: {}", gpu.adapter_name);
        println!("  f64 support: {}, backend: {:?}", gpu.has_f64, gpu.capabilities.backend);
        println!("═══════════════════════════════════════════════════════════════");
        println!();

        let _pipelines = GpuHmcStreamingPipelines::new(&gpu);

        // Use standalone shader directly (avoid duplicate-definition issue in composed pipeline)
        let standalone_pipeline = gpu.create_pipeline_f64(WGSL_PRNG_STANDALONE, "prng_standalone");

        // Dispatch PRNG shader multiple times with different traj_ids
        let mom_buf = gpu.create_f64_output_buffer((n_links * 18) as usize, "prng_test_mom");
        let wg_links = (n_links + 63) / 64;

        let mut all_offdiag: Vec<f64> = Vec::with_capacity((n_links * 12 * n_dispatches) as usize);
        let mut all_diag: Vec<f64> = Vec::with_capacity((n_links * 6 * n_dispatches) as usize);

        // First: write sentinel values to buffer and verify readback works
        let sentinel: Vec<f64> = vec![42.0; (n_links * 18) as usize];
        let sentinel_bytes: &[u8] = bytemuck::cast_slice(&sentinel);
        gpu.queue().write_buffer(&mom_buf, 0, sentinel_bytes);
        let check = gpu.read_back_f64(&mom_buf, 10).expect("sentinel readback");
        println!("  Sentinel check (should be 42.0): {:?}", &check[..5]);

        let start = Instant::now();
        for traj_id in 0..n_dispatches {
            // Create params buffer
            let params: [u32; 4] = [n_links, traj_id, 42, 7777];
            let params_bytes = bytemuck::cast_slice(&params);
            let params_buf = gpu.create_uniform_buffer(params_bytes, "prng_params");

            // Dispatch using standalone pipeline (not the composed one)
            let bg = gpu.create_bind_group(&standalone_pipeline, &[&params_buf, &mom_buf]);
            let mut encoder = gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("prng_dispatch"),
                });
            GpuF64::encode_pass(&mut encoder, &standalone_pipeline, &bg, wg_links);
            gpu.queue().submit(std::iter::once(encoder.finish()));
            // Ensure GPU completes before readback
            let _ = gpu.device().poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            // Read back
            let data = gpu
                .read_back_f64(&mom_buf, (n_links * 18) as usize)
                .expect("readback failed");

            // Debug: print first few values on first dispatch
            if traj_id == 0 {
                println!("  First 10 momenta values: {:?}", &data[..10]);
            }

            // Extract components per link (18 f64 per link)
            for link in 0..n_links as usize {
                let base = link * 18;
                // Diagonal: imag parts at indices 1, 7 (re of (1,1)), 13 (re of (2,2))
                // Actually: [0,1] = (0, h00), [6,7] = (0, h11), [12,13] = (0, h22)
                // where h00 = a3 + a8/√3, h11 = -a3 + a8/√3, h22 = -2a8/√3
                all_diag.push(data[base + 1]); // h00
                all_diag.push(data[base + 7]); // h11
                all_diag.push(data[base + 13]); // h22

                // Off-diagonal: indices 2,3 = (-im_01, re_01), each ~ N(0, 1/√2)
                // indices 4,5 = (re_02 stuff), 8,9, 10,11, 14,15, 16,17
                all_offdiag.push(data[base + 2]); // -im_01
                all_offdiag.push(data[base + 3]); // re_01
                all_offdiag.push(data[base + 4]); // -im_02
                all_offdiag.push(data[base + 5]); // re_02
                all_offdiag.push(data[base + 8]); // conjugate part
                all_offdiag.push(data[base + 9]);
                all_offdiag.push(data[base + 10]); // -im_12
                all_offdiag.push(data[base + 11]); // re_12
                all_offdiag.push(data[base + 14]);
                all_offdiag.push(data[base + 15]);
                all_offdiag.push(data[base + 16]);
                all_offdiag.push(data[base + 17]);
            }
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  Dispatched {n_dispatches} × {n_links} links = {} momenta in {elapsed_ms:.1} ms",
            all_offdiag.len() + all_diag.len()
        );
        println!();

        // Analyze off-diagonal (should be N(0, 1/√2))
        analyze_distribution(&all_offdiag, "Off-diagonal components (expected σ = 1/√2)", expected_sigma);

        // Analyze diagonal (more complex: h00 = scale*(g0 + g1/√3), not pure Gaussian)
        // But each diagonal should have variance = scale² * (1 + 1/3) = (1/2)*(4/3) = 2/3
        let diag_sigma = (2.0_f64 / 3.0).sqrt();
        analyze_distribution(&all_diag, "Diagonal components (expected σ = √(2/3))", diag_sigma);

        // Key metric: kinetic energy per component
        let ke_offdiag: f64 = all_offdiag.iter().map(|x| x * x).sum::<f64>() / all_offdiag.len() as f64;
        let ke_expected = expected_sigma.powi(2); // = 0.5

        // Also do CPU reference for comparison
        let n_cpu = all_offdiag.len();
        let mut cpu_offdiag: Vec<f64> = Vec::with_capacity(n_cpu);
        let mut rng_state = 42u64;
        let lcg_a: u64 = 6364136223846793005;
        let lcg_c: u64 = 1442695040888963407;
        while cpu_offdiag.len() < n_cpu {
            // LCG uniform [0,1)
            rng_state = rng_state.wrapping_mul(lcg_a).wrapping_add(lcg_c);
            let u1 = (rng_state >> 11) as f64 * 2.0f64.powi(-53);
            rng_state = rng_state.wrapping_mul(lcg_a).wrapping_add(lcg_c);
            let u2 = (rng_state >> 11) as f64 * 2.0f64.powi(-53);
            // Box-Muller (CPU native transcendentals)
            let r = (-2.0 * u1.max(1e-20).ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            let g = r * theta.cos();
            cpu_offdiag.push(g * expected_sigma);
        }
        let ke_cpu: f64 = cpu_offdiag.iter().map(|x| x * x).sum::<f64>() / cpu_offdiag.len() as f64;

        println!("  ┌─ KINETIC ENERGY COMPARISON ────────────────────────────────┐");
        println!("  │ GPU ⟨p²⟩   = {ke_offdiag:.8}  (per off-diag component)     │");
        println!("  │ CPU ⟨p²⟩   = {ke_cpu:.8}  (LCG + native Box-Muller)    │");
        println!("  │ Expected   = {ke_expected:.8}  (= σ² = 1/2)                 │");
        println!("  │ GPU bias   = {:+.6e}  ({:+.2}%)                    │",
            ke_offdiag - ke_expected, (ke_offdiag - ke_expected) / ke_expected * 100.0);
        println!("  │ CPU bias   = {:+.6e}  ({:+.2}%)                    │",
            ke_cpu - ke_expected, (ke_cpu - ke_expected) / ke_expected * 100.0);
        println!("  └────────────────────────────────────────────────────────────┘");
        println!();

        analyze_distribution(&cpu_offdiag, "CPU reference (LCG + native Box-Muller, σ = 1/√2)", expected_sigma);
    }
}
