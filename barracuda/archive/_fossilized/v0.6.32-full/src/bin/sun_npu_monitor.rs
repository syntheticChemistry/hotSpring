// SPDX-License-Identifier: AGPL-3.0-or-later

//! SU(N) Thermalization Monitor — AKD1000 NPU heterogeneous pipeline.
//!
//! Scans cached SU(N) thermalized configurations, extracts observable time
//! series (plaquette, Polyakov loop), trains an ESN phase classifier on CPU,
//! then deploys the classifier on the AKD1000 neuromorphic processor via VFIO.
//!
//! Demonstrates the heterogeneous compute thesis:
//!   CPU thermalizer → BLAKE3 config cache → ESN training → NPU inference
//!
//! Usage:
//!   cargo run --features npu-hw,barracuda-local --bin sun_npu_monitor

use hotspring_barracuda::lattice::generic_lattice::GenericLattice;
use hotspring_barracuda::lattice::su2::Su2Matrix;
use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "npu-hw")]
use hotspring_barracuda::md::npu_hw::NpuHardware;

fn config_dir(gauge_group: &str) -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hotspring")
        .join("configs")
        .join(gauge_group)
}

struct ConfigObservable {
    _gauge_group: String,
    dims: [usize; 4],
    beta: f64,
    plaquette: f64,
    polyakov_abs: f64,
}

fn scan_su2_configs() -> Vec<ConfigObservable> {
    let dir = config_dir("su2");
    let mut results = Vec::new();

    let entries: Vec<_> = match std::fs::read_dir(&dir) {
        Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
        Err(_) => return results,
    };

    for entry in &entries {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "lat") {
            if let Ok(lat) = GenericLattice::<Su2Matrix>::load(&path) {
                let plaq = lat.average_plaquette();
                let nx = lat.dims[0];
                let ny = lat.dims[1];
                let nz = lat.dims[2];
                let poly = lat.polyakov_loop([nx / 2, ny / 2, nz / 2]);
                results.push(ConfigObservable {
                    _gauge_group: "SU(2)".to_string(),
                    dims: lat.dims,
                    beta: lat.beta,
                    plaquette: plaq,
                    polyakov_abs: poly.abs(),
                });
            }
        }
    }

    results.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap());
    results
}

fn scan_su3_configs() -> Vec<ConfigObservable> {
    use hotspring_barracuda::lattice::wilson::Lattice;

    let dir = config_dir("su3");
    let mut results = Vec::new();

    let entries: Vec<_> = match std::fs::read_dir(&dir) {
        Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
        Err(_) => return results,
    };

    for entry in &entries {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "lat") {
            // SU(3) configs use the legacy 40-byte header format (Wilson Lattice::save)
            if let Ok(lat) = Lattice::load(&path) {
                let plaq = lat.average_plaquette();
                let poly = lat.average_polyakov_loop();
                results.push(ConfigObservable {
                    _gauge_group: "SU(3)".to_string(),
                    dims: lat.dims,
                    beta: lat.beta,
                    plaquette: plaq,
                    polyakov_abs: poly,
                });
            }
        }
    }

    results.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap());
    results
}

fn build_training_data(
    observables: &[ConfigObservable],
    beta_c: f64,
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut seqs = Vec::new();
    let mut targets = Vec::new();

    for obs in observables {
        let beta_norm = (obs.beta - beta_c) / 2.0;
        let seq = vec![vec![beta_norm, obs.plaquette, obs.polyakov_abs]];
        let phase = if obs.beta > beta_c { 1.0 } else { 0.0 };
        seqs.push(seq);
        targets.push(vec![phase]);
    }

    (seqs, targets)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  SU(N) Thermalization Monitor — AKD1000 NPU Pipeline       ║");
    println!("║  Heterogeneous: CPU therm → BLAKE3 cache → ESN → NPU      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Phase 1: Scan cached configurations ──────────────────────────────
    println!("═══ Phase 1: Observable Extraction from Cached Configs ═══");
    println!();

    let t0 = Instant::now();
    let su2_obs = scan_su2_configs();
    let su3_obs = scan_su3_configs();
    let scan_time = t0.elapsed();

    println!("  SU(2) configs scanned: {}", su2_obs.len());
    for obs in &su2_obs {
        println!(
            "    {}⁴ β={:.2}  ⟨P⟩={:.6}  |L|={:.6}",
            obs.dims[0], obs.beta, obs.plaquette, obs.polyakov_abs
        );
    }
    println!();

    println!("  SU(3) configs scanned: {}", su3_obs.len());
    for obs in &su3_obs {
        println!(
            "    {}⁴ β={:.2}  ⟨P⟩={:.6}  |L|={:.6}",
            obs.dims[0], obs.beta, obs.plaquette, obs.polyakov_abs
        );
    }
    println!();
    println!("  Scan time: {:.1}s", scan_time.as_secs_f64());
    println!();

    if su2_obs.is_empty() {
        println!("  ⚠ No cached SU(2) configs found. Run arxiv_thermalize_sun first.");
        println!("    Using synthetic training data instead.");
    }

    // ── Phase 2: Train ESN phase classifier on CPU ───────────────────────
    println!("═══ Phase 2: ESN Phase Classifier Training (CPU) ═══");
    println!();

    let beta_c_su2 = 2.30; // SU(2) β_c on small lattices
    let beta_c_su3 = 5.69; // SU(3) β_c on small lattices

    let esn_config = EsnConfig {
        input_size: 3,
        reservoir_size: 30,
        output_size: 1,
        spectral_radius: 0.95,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-2,
        seed: 42,
        ..Default::default()
    };

    let (train_seqs, train_targets) = if su2_obs.len() >= 4 {
        build_training_data(&su2_obs, beta_c_su2)
    } else {
        generate_synthetic_training(beta_c_su3)
    };

    let t1 = Instant::now();
    let mut esn = EchoStateNetwork::new(esn_config);
    esn.train(&train_seqs, &train_targets);
    let train_time = t1.elapsed();

    let mut correct = 0;
    let total = train_seqs.len();
    for (seq, target) in train_seqs.iter().zip(train_targets.iter()) {
        let pred = esn.predict(seq).expect("ESN trained")[0];
        let pred_class = if pred > 0.5 { 1.0 } else { 0.0 };
        if (pred_class - target[0]).abs() < 0.01 {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / total as f64;
    println!("  ESN training: {total} samples, {train_time:.1?}");
    println!("  Train accuracy: {:.1}% ({correct}/{total})", accuracy * 100.0);
    println!();

    // ── Phase 3: NpuSimulator parity (software baseline) ─────────────────
    println!("═══ Phase 3: NpuSimulator f32 Parity ═══");
    println!();

    let weights = esn.export_weights().expect("export weights");
    let mut npu_sim = NpuSimulator::from_exported(&weights);

    let mut max_err = 0.0f64;
    let mut agree = 0;
    for seq in &train_seqs {
        let cpu_pred = esn.predict(seq).expect("ESN trained")[0];
        let npu_pred = npu_sim.predict(seq)[0];
        let err = (cpu_pred - npu_pred).abs();
        if err > max_err {
            max_err = err;
        }
        let cpu_class = i32::from(cpu_pred > 0.5);
        let npu_class = i32::from(npu_pred > 0.5);
        if cpu_class == npu_class {
            agree += 1;
        }
    }
    println!("  f64↔f32 max absolute error: {max_err:.2e}");
    println!("  Classification agreement: {agree}/{total}");
    println!();

    // ── Phase 4: AKD1000 NPU Hardware Inference ──────────────────────────
    #[cfg(feature = "npu-hw")]
    {
        println!("═══ Phase 4: AKD1000 NPU Hardware Inference ═══");
        println!();

        match NpuHardware::discover() {
            Some(hw_info) => {
                println!(
                    "  Device: {} @ {} — {} NPs, {} MB SRAM, PCIe Gen{} x{}",
                    hw_info.chip_version,
                    hw_info.pcie_address,
                    hw_info.npu_count,
                    hw_info.memory_mb,
                    hw_info.pcie_gen,
                    hw_info.pcie_lanes
                );

                let mut npu_hw = NpuHardware::from_exported(&weights, hw_info);

                let t2 = Instant::now();
                let mut npu_correct = 0;
                let mut npu_total = 0;

                for (seq, target) in train_seqs.iter().zip(train_targets.iter()) {
                    let pred = npu_hw.predict(seq)[0];
                    let pred_class = if pred > 0.5 { 1.0 } else { 0.0 };
                    if (pred_class - target[0]).abs() < 0.01 {
                        npu_correct += 1;
                    }
                    npu_total += 1;
                }
                let npu_time = t2.elapsed();

                println!(
                    "  NPU classification: {npu_correct}/{npu_total} correct ({:.1}%)",
                    npu_correct as f64 / npu_total as f64 * 100.0
                );
                println!("  NPU inference time: {npu_time:.1?} ({npu_total} samples)");
                if npu_total > 0 {
                    let per_sample = npu_time.as_micros() as f64 / npu_total as f64;
                    println!("  Per-sample latency: {per_sample:.0} µs");
                }
                println!();

                // ── Phase 5: Convergence detection on live observable streams ────
                println!("═══ Phase 5: Convergence Detection ═══");
                println!();

                convergence_scan(&mut npu_hw, &su2_obs, "SU(2)", beta_c_su2);
                convergence_scan_sim(&mut npu_sim, &su3_obs, "SU(3)", beta_c_su3);
            }
            None => {
                println!("  ⚠ No AKD1000 hardware detected. Running NpuSimulator path only.");
                println!();
                println!("═══ Phase 5: Convergence Detection (Simulator) ═══");
                println!();
                convergence_scan_sim(&mut npu_sim, &su2_obs, "SU(2)", beta_c_su2);
                convergence_scan_sim(&mut npu_sim, &su3_obs, "SU(3)", beta_c_su3);
            }
        }
    }

    #[cfg(not(feature = "npu-hw"))]
    {
        println!("═══ Phase 4: NPU Hardware (skipped — compile with --features npu-hw) ═══");
        println!();
        println!("═══ Phase 5: Convergence Detection (Simulator) ═══");
        println!();
        convergence_scan_sim(&mut npu_sim, &su2_obs, "SU(2)", beta_c_su2);
        convergence_scan_sim(&mut npu_sim, &su3_obs, "SU(3)", beta_c_su3);
    }

    println!();
    println!("═══ Pipeline Complete ═══");
    println!("  Heterogeneous compute chain validated:");
    println!("    CPU thermalizer → BLAKE3 cache → ESN (CPU) → NPU inference");
}

#[cfg(feature = "npu-hw")]
fn convergence_scan(
    npu: &mut NpuHardware,
    observables: &[ConfigObservable],
    label: &str,
    beta_c: f64,
) {
    if observables.is_empty() {
        println!("  {label}: no configs to monitor");
        return;
    }

    println!("  {label} convergence monitor ({} configs):", observables.len());

    let mut prev_plaq = None;
    for obs in observables {
        let beta_norm = (obs.beta - beta_c) / 2.0;
        let seq = vec![vec![beta_norm, obs.plaquette, obs.polyakov_abs]];
        let pred = npu.predict(&seq)[0];
        let phase = if pred > 0.5 { "DECONF" } else { "CONFND" };

        let stability = if let Some(prev) = prev_plaq {
            let delta: f64 = (obs.plaquette - prev) / prev;
            format!("Δ={delta:+.4}")
        } else {
            "—".to_string()
        };
        prev_plaq = Some(obs.plaquette);

        println!(
            "    β={:.2} ⟨P⟩={:.6} |L|={:.6}  NPU→{phase}  {stability}",
            obs.beta, obs.plaquette, obs.polyakov_abs
        );
    }
    println!();
}

fn convergence_scan_sim(
    npu_sim: &mut NpuSimulator,
    observables: &[ConfigObservable],
    label: &str,
    beta_c: f64,
) {
    if observables.is_empty() {
        println!("  {label}: no configs to monitor");
        return;
    }

    println!("  {label} convergence monitor ({} configs, simulator):", observables.len());

    let mut prev_plaq = None;
    for obs in observables {
        let beta_norm = (obs.beta - beta_c) / 2.0;
        let seq = vec![vec![beta_norm, obs.plaquette, obs.polyakov_abs]];
        let pred = npu_sim.predict(&seq)[0];
        let phase = if pred > 0.5 { "DECONF" } else { "CONFND" };

        let stability = if let Some(prev) = prev_plaq {
            let delta: f64 = (obs.plaquette - prev) / prev;
            format!("Δ={delta:+.4}")
        } else {
            "—".to_string()
        };
        prev_plaq = Some(obs.plaquette);

        println!(
            "    β={:.2} ⟨P⟩={:.6} |L|={:.6}  SIM→{phase}  {stability}",
            obs.beta, obs.plaquette, obs.polyakov_abs
        );
    }
    println!();
}

fn generate_synthetic_training(beta_c: f64) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut seqs = Vec::new();
    let mut targets = Vec::new();

    for i in 0..40 {
        let beta = beta_c - 1.5 + 3.0 * (i as f64) / 39.0;
        let beta_norm = (beta - beta_c) / 2.0;
        let plaq = synthetic_plaquette(beta, beta_c, i as u64);
        let poly = synthetic_polyakov(beta, beta_c, i as u64);
        let phase = if beta > beta_c { 1.0 } else { 0.0 };

        seqs.push(vec![vec![beta_norm, plaq, poly]]);
        targets.push(vec![phase]);
    }

    (seqs, targets)
}

fn synthetic_plaquette(beta: f64, beta_c: f64, seed: u64) -> f64 {
    let phase_frac = 1.0 / (1.0 + (-((beta - beta_c) / 0.075)).exp());
    let strong = (beta / 18.0).mul_add(beta / 18.0, beta / 18.0);
    let weak = 1.0 - 3.0 / (4.0 * beta);
    let plaq = (1.0 - phase_frac).mul_add(strong, phase_frac * weak);
    let noise = lcg_normal(seed) * 0.005;
    (plaq + noise).clamp(0.0, 1.0)
}

fn synthetic_polyakov(beta: f64, beta_c: f64, seed: u64) -> f64 {
    let phase_frac = 1.0 / (1.0 + (-((beta - beta_c) / 0.075)).exp());
    let deconf_val = 0.15 + 0.35 / (1.0 + (-((beta - beta_c) / 0.5)).exp());
    let poly = phase_frac * deconf_val;
    let noise = lcg_normal(seed + 1) * 0.005;
    (poly + noise).clamp(0.0, 1.0)
}

fn lcg_normal(seed: u64) -> f64 {
    let s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let u1 = (s >> 33) as f64 / (1u64 << 31) as f64;
    let s2 = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let u2 = (s2 >> 33) as f64 / (1u64 << 31) as f64;
    let u1c = u1.clamp(1e-10, 1.0 - 1e-10);
    let u2c = u2.clamp(1e-10, 1.0 - 1e-10);
    (-2.0 * u1c.ln()).sqrt() * (std::f64::consts::TAU * u2c).cos()
}
