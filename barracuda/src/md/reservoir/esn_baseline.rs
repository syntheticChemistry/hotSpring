// SPDX-License-Identifier: AGPL-3.0-or-later

//! ESN baseline validation helpers.
//!
//! Shared logic for the `esn_baseline_validation` binary: JSONL loading,
//! input/target construction, synthetic datasets, training, and result formatting.

use super::heads;
use super::{Activation, EchoStateNetwork, EsnConfig};
use std::collections::HashMap;

/// Critical β for phase classification (SU(3) deconfinement).
pub const KNOWN_BETA_C: f64 = 5.69;

/// Aggregated lattice point for ESN training (beta, mass, lattice, observables).
#[derive(Debug, Clone)]
pub struct AggPoint {
    /// Coupling parameter.
    pub beta: f64,
    /// Fermion mass.
    pub mass: f64,
    /// Lattice extent L (L⁴).
    pub lattice: usize,
    /// Mean plaquette.
    pub mean_plaq: f64,
    /// Plaquette standard deviation.
    pub std_plaq: f64,
    /// HMC acceptance rate.
    pub acceptance: f64,
    /// Mean CG iterations.
    pub mean_cg: f64,
    /// Mean |ΔH| (action density proxy).
    pub mean_delta_h: f64,
    /// Polyakov loop magnitude.
    pub polyakov: f64,
    /// Number of trajectories.
    pub n_traj: usize,
    /// Experiment identifier.
    pub experiment: String,
}

/// Load aggregated points from JSONL files in a results directory.
#[must_use]
pub fn load_jsonl_files(results_dir: &str) -> Vec<AggPoint> {
    let mut raw: HashMap<(String, usize, String), Vec<serde_json::Value>> = HashMap::new();
    let mut schema_b_points: Vec<AggPoint> = Vec::new();

    let Ok(entries) = std::fs::read_dir(results_dir) else {
        eprintln!("Cannot read {results_dir}");
        return vec![];
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if !name.to_ascii_lowercase().ends_with(".jsonl") || !name.starts_with("exp0") {
            continue;
        }

        let stem = name.trim_end_matches(".jsonl");
        let exp = stem.split('_').next().unwrap_or(stem).to_string();
        let lattice = infer_lattice_from_filename(stem);

        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };

        if let Ok(root) = serde_json::from_str::<serde_json::Value>(&content)
            && let Some(pts) = root.get("points").and_then(|p| p.as_array())
        {
            let file_lattice = root
                .get("lattice")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(lattice as u64) as usize;
            let file_mass = root
                .get("mass")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.1);
            for pt in pts {
                let beta = pt
                    .get("beta")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);
                let mass = pt
                    .get("mass")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(file_mass);
                let mean_plaq = pt
                    .get("mean_plaquette")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);
                let acceptance = pt
                    .get("acceptance")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);
                let mean_cg = pt
                    .get("mean_cg_iterations")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);
                let n_traj = pt
                    .get("n_trajectories")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(1) as usize;
                let std_plaq = pt
                    .get("std_plaquette")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.005);
                let polyakov = pt
                    .get("polyakov_re")
                    .and_then(serde_json::Value::as_f64)
                    .or_else(|| pt.get("polyakov").and_then(serde_json::Value::as_f64))
                    .unwrap_or(0.0)
                    .abs();
                let action_density = pt
                    .get("action_density")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);

                schema_b_points.push(AggPoint {
                    beta,
                    mass,
                    lattice: file_lattice,
                    mean_plaq,
                    std_plaq,
                    acceptance,
                    mean_cg,
                    mean_delta_h: action_density,
                    polyakov,
                    n_traj,
                    experiment: exp.clone(),
                });
            }
            continue;
        }

        for line in content.lines() {
            let Ok(val) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if val.get("phase").and_then(|p| p.as_str()) != Some("measurement") {
                continue;
            }
            let beta = val
                .get("beta")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            let mass = val
                .get("mass")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.1);
            let key = format!("{beta:.4}_{mass:.4}");
            raw.entry((exp.clone(), lattice, key))
                .or_default()
                .push(val);
        }
    }

    let mut points = schema_b_points;

    for ((exp, lattice, _key), records) in &raw {
        if records.is_empty() {
            continue;
        }
        let n = records.len() as f64;

        let plaqs: Vec<f64> = records
            .iter()
            .filter_map(|r| r.get("plaquette").and_then(serde_json::Value::as_f64))
            .collect();
        let cgs: Vec<f64> = records
            .iter()
            .filter_map(|r| r.get("cg_iters").and_then(serde_json::Value::as_f64))
            .collect();
        let accs: Vec<f64> = records
            .iter()
            .map(|r| {
                if r.get("accepted")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let dhs: Vec<f64> = records
            .iter()
            .filter_map(|r| r.get("delta_h").and_then(serde_json::Value::as_f64))
            .collect();
        let polys: Vec<f64> = records
            .iter()
            .filter_map(|r| r.get("polyakov_re").and_then(serde_json::Value::as_f64))
            .collect();

        let mean_plaq = plaqs.iter().sum::<f64>() / plaqs.len().max(1) as f64;
        let std_plaq = if plaqs.len() > 1 {
            let var: f64 = plaqs.iter().map(|p| (p - mean_plaq).powi(2)).sum::<f64>()
                / (plaqs.len() - 1) as f64;
            var.sqrt()
        } else {
            0.0
        };
        let mean_cg = cgs.iter().sum::<f64>() / cgs.len().max(1) as f64;
        let acceptance = accs.iter().sum::<f64>() / n;
        let mean_dh = dhs.iter().map(|d| d.abs()).sum::<f64>() / dhs.len().max(1) as f64;
        let polyakov = if polys.is_empty() {
            0.0
        } else {
            polys.iter().map(|p| p.abs()).sum::<f64>() / polys.len() as f64
        };

        let beta = records[0]
            .get("beta")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let mass = records[0]
            .get("mass")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.1);

        points.push(AggPoint {
            beta,
            mass,
            lattice: *lattice,
            mean_plaq,
            std_plaq,
            acceptance,
            mean_cg,
            mean_delta_h: mean_dh,
            polyakov,
            n_traj: records.len(),
            experiment: exp.clone(),
        });
    }

    points.sort_by(|a, b| {
        a.beta
            .partial_cmp(&b.beta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    points
}

/// Infer lattice size from filename stem (e.g. "exp0_2x2" → 2).
#[must_use]
pub fn infer_lattice_from_filename(stem: &str) -> usize {
    if stem.contains("2x2") {
        return 2;
    }
    if stem.contains("6x6") {
        return 6;
    }
    if stem.contains("8x8") {
        return 8;
    }
    4
}

/// Canonical 6D input for v2 ESN (beta_norm, plaq, mass, susceptibility, acceptance, lattice).
#[must_use]
pub fn canonical_input_v2(p: &AggPoint) -> Vec<f64> {
    let beta_norm = (p.beta - 5.0) / 2.0;
    let susceptibility = p.std_plaq.powi(2) * (p.lattice as f64).powi(4);
    vec![
        beta_norm,
        p.mean_plaq,
        p.mass,
        susceptibility / 1000.0,
        p.acceptance,
        p.lattice as f64 / 8.0,
    ]
}

/// Build input sequence with small noise for v2 ESN.
#[must_use]
pub fn build_sequence_v2(p: &AggPoint, seq_len: usize) -> Vec<Vec<f64>> {
    let base = canonical_input_v2(p);
    (0..seq_len)
        .map(|j| {
            let noise = 0.005 * ((j as f64) * 0.7).sin();
            let mut v = base.clone();
            v[1] += noise * p.std_plaq;
            v
        })
        .collect()
}

/// Head specification: name, output index, target function.
pub struct HeadSpec {
    /// Display name.
    pub name: &'static str,
    /// Output head index.
    pub index: usize,
    /// Target extraction from AggPoint.
    pub target_fn: fn(&AggPoint) -> f64,
}

/// v2 head specs: log-CG, phase, reject, plaquette, beta priority, param suggest, acceptance, action.
#[must_use]
pub fn head_specs_v2() -> Vec<HeadSpec> {
    vec![
        HeadSpec {
            name: "LOG_CG",
            index: heads::CG_ESTIMATE,
            target_fn: |p| (p.mean_cg.max(1.0).log10()) / 6.0,
        },
        HeadSpec {
            name: "PHASE_CLASSIFY",
            index: heads::PHASE_CLASSIFY,
            target_fn: |p| if p.beta > KNOWN_BETA_C { 1.0 } else { 0.0 },
        },
        HeadSpec {
            name: "REJECT_PREDICT",
            index: heads::REJECT_PREDICT,
            target_fn: |p| 1.0 - p.acceptance,
        },
        HeadSpec {
            name: "PLAQUETTE",
            index: heads::QUALITY_SCORE,
            target_fn: |p| p.mean_plaq,
        },
        HeadSpec {
            name: "BETA_PRIORITY",
            index: heads::BETA_PRIORITY,
            target_fn: |p| {
                let chi = p.std_plaq.powi(2) * (p.lattice as f64).powi(4);
                chi.log10().max(-4.0) / 4.0 + 1.0
            },
        },
        HeadSpec {
            name: "PARAM_SUGGEST",
            index: heads::PARAM_SUGGEST,
            target_fn: |p| {
                if p.acceptance > 0.85 {
                    0.015
                } else if p.acceptance > 0.5 {
                    0.01
                } else if p.acceptance > 0.2 {
                    0.005
                } else {
                    0.002
                }
            },
        },
        HeadSpec {
            name: "ACCEPTANCE",
            index: heads::B2_QCD_ACCEPTANCE,
            target_fn: |p| p.acceptance,
        },
        HeadSpec {
            name: "ACTION_DENSITY",
            index: heads::A2_ANDERSON_LAMBDA_MIN,
            target_fn: |p| p.mean_delta_h.min(10.0) / 10.0,
        },
    ]
}

/// Train ESN and evaluate on test set (tanh activation).
#[must_use]
pub fn train_and_evaluate(
    train_data: &[AggPoint],
    test_data: &[AggPoint],
    specs: &[HeadSpec],
    reservoir_size: usize,
    input_size: usize,
) -> Vec<(String, f64, f64, f64)> {
    train_and_evaluate_with(
        train_data,
        test_data,
        specs,
        reservoir_size,
        input_size,
        Activation::Tanh,
    )
}

/// Train ESN with given activation and evaluate.
#[must_use]
pub fn train_and_evaluate_with(
    train_data: &[AggPoint],
    test_data: &[AggPoint],
    specs: &[HeadSpec],
    reservoir_size: usize,
    input_size: usize,
    activation: Activation,
) -> Vec<(String, f64, f64, f64)> {
    let seq_len = 10;
    let n_heads = specs.len();

    let train_seqs: Vec<Vec<Vec<f64>>> = train_data
        .iter()
        .map(|p| build_sequence_v2(p, seq_len))
        .collect();
    let train_targets: Vec<Vec<f64>> = train_data
        .iter()
        .map(|p| specs.iter().map(|s| (s.target_fn)(p)).collect())
        .collect();

    let config = EsnConfig {
        input_size,
        reservoir_size,
        output_size: n_heads,
        spectral_radius: 0.95,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-4,
        seed: 42,
        activation,
    };

    let mut esn = EchoStateNetwork::new(config);
    esn.train(&train_seqs, &train_targets);

    let mut results = Vec::new();
    for (hi, spec) in specs.iter().enumerate() {
        let mut actual = Vec::new();
        let mut predicted = Vec::new();

        for p in test_data {
            let seq = build_sequence_v2(p, seq_len);
            if let Ok(out) = esn.predict(&seq)
                && hi < out.len()
            {
                actual.push((spec.target_fn)(p));
                predicted.push(out[hi]);
            }
        }

        if actual.is_empty() {
            results.push((spec.name.to_string(), f64::NAN, f64::NAN, f64::NAN));
            continue;
        }

        let n = actual.len() as f64;
        let mae: f64 = actual
            .iter()
            .zip(&predicted)
            .map(|(a, p)| (a - p).abs())
            .sum::<f64>()
            / n;
        let mean_actual = actual.iter().sum::<f64>() / n;
        let ss_res: f64 = actual
            .iter()
            .zip(&predicted)
            .map(|(a, p)| (a - p).powi(2))
            .sum();
        let ss_tot: f64 = actual.iter().map(|a| (a - mean_actual).powi(2)).sum();
        let r2 = if ss_tot > 1e-15 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };
        let mean_abs_actual = actual.iter().map(|a| a.abs()).sum::<f64>() / n;
        let rel_err = if mean_abs_actual > 1e-15 {
            mae / mean_abs_actual
        } else {
            f64::NAN
        };

        results.push((spec.name.to_string(), mae, r2, rel_err));
    }
    results
}

/// Grade R² into a string label.
#[must_use]
pub fn grade(r2: f64) -> &'static str {
    if r2 > 0.9 {
        "EXCELLENT"
    } else if r2 > 0.7 {
        "GOOD"
    } else if r2 > 0.4 {
        "FAIR"
    } else if r2 > 0.0 {
        "WEAK"
    } else {
        "FAIL"
    }
}

/// Print results table (Head, MAE, R², RelErr, Grade).
pub fn print_results_table(results: &[(String, f64, f64, f64)]) {
    println!(
        "  {:<18} {:>8} {:>8} {:>10} Grade",
        "Head", "MAE", "R²", "RelErr"
    );
    println!(
        "  {:-<18} {:->8} {:->8} {:->10} {:->10}",
        "", "", "", "", ""
    );
    for (name, mae, r2, rel) in results {
        println!(
            "  {:<18} {:>8.4} {:>8.3} {:>9.1}%  {}",
            name,
            mae,
            r2,
            rel * 100.0,
            grade(*r2)
        );
    }
}

/// Generate sine-wave synthetic dataset.
#[must_use]
pub fn generate_sine_dataset(n: usize) -> (Vec<AggPoint>, &'static str) {
    let points: Vec<AggPoint> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            let x = t * 4.0 * std::f64::consts::PI;
            AggPoint {
                beta: t.mul_add(2.0, 5.0),
                mass: 0.1,
                lattice: 4,
                mean_plaq: 0.2f64.mul_add(x.sin(), 0.5),
                std_plaq: 0.01,
                acceptance: 0.4f64.mul_add((x * 0.7).cos(), 0.5),
                mean_cg: 900.0f64.mul_add(0.5f64.mul_add(x.sin(), 0.5), 100.0),
                mean_delta_h: 3.0 + x.sin(),
                polyakov: 0.3,
                n_traj: 10,
                experiment: format!("syn_sin_{i}"),
            }
        })
        .collect();
    (
        points,
        "Sine: smooth periodic targets, tests basic function approximation",
    )
}

/// Generate step-function synthetic dataset.
#[must_use]
pub fn generate_step_dataset(n: usize) -> (Vec<AggPoint>, &'static str) {
    let points: Vec<AggPoint> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            let phase = if t > 0.5 { 1.0 } else { 0.0 };
            AggPoint {
                beta: t.mul_add(2.0, 5.0),
                mass: 0.1,
                lattice: 4,
                mean_plaq: 0.2f64.mul_add(phase, 0.4),
                std_plaq: 0.01f64.mul_add(-phase, 0.02),
                acceptance: 0.5f64.mul_add(phase, 0.3),
                mean_cg: if phase > 0.5 { 500.0 } else { 50000.0 },
                mean_delta_h: 3.0,
                polyakov: 0.3,
                n_traj: 10,
                experiment: format!("syn_step_{i}"),
            }
        })
        .collect();
    (
        points,
        "Step: sharp phase transition, tests classification boundary",
    )
}

/// Generate power-law synthetic dataset (CG ~ 1/m²).
#[must_use]
pub fn generate_power_law_dataset(n: usize) -> (Vec<AggPoint>, &'static str) {
    let points: Vec<AggPoint> = (0..n)
        .map(|i| {
            let mass = (i as f64 / n as f64).mul_add(0.49, 0.01);
            let cg = 100.0 / (mass * mass);
            AggPoint {
                beta: 5.69,
                mass,
                lattice: 4,
                mean_plaq: 0.55,
                std_plaq: 0.005,
                acceptance: (1.0 - mass * 0.5).max(0.1),
                mean_cg: cg,
                mean_delta_h: 3.0,
                polyakov: 0.3,
                n_traj: 10,
                experiment: format!("syn_pow_{i}"),
            }
        })
        .collect();
    (
        points,
        "Power-law: CG ~ 1/m², tests multi-scale regression (the hard case)",
    )
}

/// Generate volume-scaling synthetic dataset.
#[must_use]
pub fn generate_volume_scaling_dataset(n: usize) -> (Vec<AggPoint>, &'static str) {
    let lattices = [2, 4, 6, 8];
    let points: Vec<AggPoint> = (0..n)
        .map(|i| {
            let lat = lattices[i % 4];
            let vol = (lat as f64).powi(4);
            let beta = (i as f64 / n as f64).mul_add(2.0, 5.0);
            AggPoint {
                beta,
                mass: 0.1,
                lattice: lat,
                mean_plaq: 0.1f64.mul_add(beta - 5.0, 0.4),
                std_plaq: 0.1 / vol.sqrt(),
                acceptance: vol.log10().mul_add(-0.1, 0.9).max(0.1),
                mean_cg: 100.0 * vol.sqrt(),
                mean_delta_h: 3.0,
                polyakov: 0.3,
                n_traj: 10,
                experiment: format!("syn_vol_{i}"),
            }
        })
        .collect();
    (
        points,
        "Volume-scaling: observables scale with L^4, tests volume awareness",
    )
}

/// Generate noisy synthetic dataset.
#[must_use]
pub fn generate_noisy_dataset(n: usize) -> (Vec<AggPoint>, &'static str) {
    let mut rng_state: u64 = 12345;
    let mut next_f64 = move || -> f64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64) / (u64::MAX as f64)
    };
    let points: Vec<AggPoint> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            let noise = next_f64() * 0.3;
            AggPoint {
                beta: t.mul_add(2.0, 5.0),
                mass: 0.1,
                lattice: 4,
                mean_plaq: 0.1f64.mul_add(t, 0.5) + noise * 0.1,
                std_plaq: 0.01 + noise * 0.02,
                acceptance: (0.7 + noise * 0.3).clamp(0.0, 1.0),
                mean_cg: 1000.0 + noise * 5000.0,
                mean_delta_h: 3.0,
                polyakov: 0.3,
                n_traj: 10,
                experiment: format!("syn_noise_{i}"),
            }
        })
        .collect();
    (
        points,
        "Noisy: 30% uniform noise on targets, tests noise robustness",
    )
}

/// Run synthetic test: 80/20 split, train, evaluate, print table.
pub fn run_synthetic_test(name: &str, description: &str, data: &[AggPoint], specs: &[HeadSpec]) {
    let split = data.len() * 4 / 5;
    let train = &data[..split];
    let test = &data[split..];
    println!("\n  ── {name} ──");
    println!("  {description}");
    println!("  {} train, {} test points", train.len(), test.len());
    let results = train_and_evaluate(train, test, specs, 100, 6);
    print_results_table(&results);
}

/// Run activation comparison: tanh vs ReLU-approx-tanh.
pub fn run_activation_comparison(name: &str, data: &[AggPoint], specs: &[HeadSpec]) {
    let split = data.len() * 4 / 5;
    let train = &data[..split];
    let test = &data[split..];

    let tanh_results = train_and_evaluate_with(train, test, specs, 100, 6, Activation::Tanh);
    let relu_results =
        train_and_evaluate_with(train, test, specs, 100, 6, Activation::ReluTanhApprox);

    println!("\n  ── {name} ──");
    println!("  {} train, {} test points\n", train.len(), test.len());
    println!(
        "  {:<18} {:>8} {:>8}   {:>8} {:>8}   {:>7}",
        "Head", "tanh R²", "Grade", "ReLU R²", "Grade", "Δ R²"
    );
    println!(
        "  {:-<18} {:->8} {:->8}   {:->8} {:->8}   {:->7}",
        "", "", "", "", "", ""
    );
    for (i, spec) in specs.iter().enumerate() {
        let t_r2 = tanh_results.get(i).map_or(f64::NAN, |r| r.2);
        let r_r2 = relu_results.get(i).map_or(f64::NAN, |r| r.2);
        let delta = r_r2 - t_r2;
        let marker = if delta.abs() < 0.01 {
            "  ≈"
        } else if delta > 0.0 {
            " ▲"
        } else {
            " ▼"
        };
        println!(
            "  {:<18} {:>8.3} {:>8}   {:>8.3} {:>8}   {:>+6.3}{}",
            spec.name,
            t_r2,
            grade(t_r2),
            r_r2,
            grade(r_r2),
            delta,
            marker
        );
    }
}
