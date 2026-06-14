// SPDX-License-Identifier: AGPL-3.0-or-later

use std::time::Instant;

pub fn bench_spectral_stats_cpu() {
    println!("═══ Phase 1b: Spectral Statistics (CPU) ═══");
    println!("  Provenance: hotSpring Anderson proxy → toadStool S78 (spectral/stats.rs)");
    println!("  Cross-spring: level_spacing_ratio born in hotSpring, absorbed into toadStool,");
    println!("    now used by wetSpring (bio spectral) and neuralSpring (RMT classifier)");
    println!();

    use barracuda::spectral::{
        GOE_R, POISSON_R, SpectralAnalysis, anderson_3d, find_all_eigenvalues, lanczos,
        level_spacing_ratio, spectral_bandwidth, spectral_condition_number,
    };

    for &(l, w, label) in &[
        (6_usize, 4.0_f64, "weak disorder (extended)"),
        (6, 16.0, "moderate disorder (critical)"),
        (6, 30.0, "strong disorder (localized)"),
        (8, 4.0, "8^3 weak disorder"),
        (10, 10.0, "10^3 moderate disorder"),
    ] {
        let t = Instant::now();
        let h = anderson_3d(l, l, l, w, 42);
        let n = l * l * l;
        let tri = lanczos(&h, n.min(h.n), 42);
        let eigs = find_all_eigenvalues(&tri.alpha, &tri.beta);
        let lanczos_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let r = level_spacing_ratio(&eigs);
        let bw = spectral_bandwidth(&eigs);
        let kappa = spectral_condition_number(&eigs);
        let gamma_est = if bw > 0.0 { n as f64 / bw } else { 1.0 };
        let analysis = SpectralAnalysis::from_eigenvalues(eigs.clone(), gamma_est);
        let stats_us = t.elapsed().as_micros();

        let phase_r = if r > 0.48 {
            "extended"
        } else if r < 0.42 {
            "localized"
        } else {
            "critical"
        };
        let phase_mp = format!("{:?}", analysis.phase);

        println!(
            "  L={l}, W={w:>4.0} ({label:30}): <r>={r:.4} (GOE={GOE_R:.4}, Poisson={POISSON_R:.4})"
        );
        println!("    BW={bw:.2}, kappa={kappa:.1e}, phase_r={phase_r}, phase_MP={phase_mp}");
        println!(
            "    Lanczos: {lanczos_ms:.1}ms | Stats: {stats_us}us | n_eig={}",
            eigs.len()
        );
    }
    println!();
}
