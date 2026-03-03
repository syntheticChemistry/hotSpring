// SPDX-License-Identifier: AGPL-3.0-only

//! CPU ESN baseline validation harness (v2).
//!
//! Loads historical JSONL, trains with improved targets (log-CG, 6D input with
//! volume), runs synthetic dataset probes, and sweeps reservoir hyperparameters.

use hotspring_barracuda::md::reservoir::{heads, Activation, EchoStateNetwork, EsnConfig};
use std::collections::HashMap;
use std::path::PathBuf;

const KNOWN_BETA_C: f64 = 5.69;

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AggPoint {
    beta: f64,
    mass: f64,
    lattice: usize,
    mean_plaq: f64,
    std_plaq: f64,
    acceptance: f64,
    mean_cg: f64,
    mean_delta_h: f64,
    polyakov: f64,
    n_traj: usize,
    experiment: String,
}

// ═══════════════════════════════════════════════════════════════════
//  Data Loading
// ═══════════════════════════════════════════════════════════════════

fn load_jsonl_files(results_dir: &str) -> Vec<AggPoint> {
    let mut raw: HashMap<(String, usize, String), Vec<serde_json::Value>> = HashMap::new();
    let mut schema_b_points: Vec<AggPoint> = Vec::new();

    let Ok(entries) = std::fs::read_dir(results_dir) else {
        eprintln!("Cannot read {results_dir}");
        return vec![];
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        if !name.ends_with(".jsonl") || !name.starts_with("exp0") {
            continue;
        }

        let stem = name.trim_end_matches(".jsonl");
        let exp = stem.split('_').next().unwrap_or(stem).to_string();
        let lattice = infer_lattice_from_filename(stem);

        let Ok(content) = std::fs::read_to_string(&path) else { continue };

        if let Ok(root) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(pts) = root.get("points").and_then(|p| p.as_array()) {
                let file_lattice = root.get("lattice").and_then(|v| v.as_u64()).unwrap_or(lattice as u64) as usize;
                let file_mass = root.get("mass").and_then(|v| v.as_f64()).unwrap_or(0.1);
                for pt in pts {
                    let beta = pt.get("beta").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let mass = pt.get("mass").and_then(|v| v.as_f64()).unwrap_or(file_mass);
                    let mean_plaq = pt.get("mean_plaquette").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let acceptance = pt.get("acceptance").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let mean_cg = pt.get("mean_cg_iterations").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let n_traj = pt.get("n_trajectories").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                    let std_plaq = pt.get("std_plaquette").and_then(|v| v.as_f64()).unwrap_or(0.005);
                    let polyakov = pt.get("polyakov_re").and_then(|v| v.as_f64())
                        .or_else(|| pt.get("polyakov").and_then(|v| v.as_f64()))
                        .unwrap_or(0.0).abs();
                    let action_density = pt.get("action_density").and_then(|v| v.as_f64()).unwrap_or(0.0);

                    schema_b_points.push(AggPoint {
                        beta, mass, lattice: file_lattice, mean_plaq, std_plaq, acceptance,
                        mean_cg, mean_delta_h: action_density, polyakov,
                        n_traj, experiment: exp.clone(),
                    });
                }
                continue;
            }
        }

        for line in content.lines() {
            let Ok(val) = serde_json::from_str::<serde_json::Value>(line) else { continue };
            if val.get("phase").and_then(|p| p.as_str()) != Some("measurement") {
                continue;
            }
            let beta = val.get("beta").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let mass = val.get("mass").and_then(|v| v.as_f64()).unwrap_or(0.1);
            let key = format!("{beta:.4}_{mass:.4}");
            raw.entry((exp.clone(), lattice, key)).or_default().push(val);
        }
    }

    let mut points = schema_b_points;

    for ((exp, lattice, _key), records) in &raw {
        if records.is_empty() { continue; }
        let n = records.len() as f64;

        let plaqs: Vec<f64> = records.iter()
            .filter_map(|r| r.get("plaquette").and_then(|v| v.as_f64()))
            .collect();
        let cgs: Vec<f64> = records.iter()
            .filter_map(|r| r.get("cg_iters").and_then(|v| v.as_f64()))
            .collect();
        let accs: Vec<f64> = records.iter()
            .map(|r| if r.get("accepted").and_then(|v| v.as_bool()).unwrap_or(false) { 1.0 } else { 0.0 })
            .collect();
        let dhs: Vec<f64> = records.iter()
            .filter_map(|r| r.get("delta_h").and_then(|v| v.as_f64()))
            .collect();
        let polys: Vec<f64> = records.iter()
            .filter_map(|r| r.get("polyakov_re").and_then(|v| v.as_f64()))
            .collect();

        let mean_plaq = plaqs.iter().sum::<f64>() / plaqs.len().max(1) as f64;
        let std_plaq = if plaqs.len() > 1 {
            let var: f64 = plaqs.iter().map(|p| (p - mean_plaq).powi(2)).sum::<f64>() / (plaqs.len() - 1) as f64;
            var.sqrt()
        } else { 0.0 };
        let mean_cg = cgs.iter().sum::<f64>() / cgs.len().max(1) as f64;
        let acceptance = accs.iter().sum::<f64>() / n;
        let mean_dh = dhs.iter().map(|d| d.abs()).sum::<f64>() / dhs.len().max(1) as f64;
        let polyakov = if polys.is_empty() { 0.0 } else { polys.iter().map(|p| p.abs()).sum::<f64>() / polys.len() as f64 };

        let beta = records[0].get("beta").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let mass = records[0].get("mass").and_then(|v| v.as_f64()).unwrap_or(0.1);

        points.push(AggPoint {
            beta, mass, lattice: *lattice, mean_plaq, std_plaq, acceptance,
            mean_cg, mean_delta_h: mean_dh, polyakov,
            n_traj: records.len(), experiment: exp.clone(),
        });
    }

    points.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap_or(std::cmp::Ordering::Equal));
    points
}

fn infer_lattice_from_filename(stem: &str) -> usize {
    if stem.contains("2x2") { return 2; }
    if stem.contains("6x6") { return 6; }
    if stem.contains("8x8") { return 8; }
    4
}

// ═══════════════════════════════════════════════════════════════════
//  Input / Target Construction (v2: 6D with volume, log-CG targets)
// ═══════════════════════════════════════════════════════════════════

fn canonical_input_v2(p: &AggPoint) -> Vec<f64> {
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

fn build_sequence_v2(p: &AggPoint, seq_len: usize) -> Vec<Vec<f64>> {
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

struct HeadSpec {
    name: &'static str,
    index: usize,
    target_fn: fn(&AggPoint) -> f64,
}

fn head_specs_v2() -> Vec<HeadSpec> {
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
                if p.acceptance > 0.85 { 0.015 }
                else if p.acceptance > 0.5 { 0.01 }
                else if p.acceptance > 0.2 { 0.005 }
                else { 0.002 }
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

fn train_and_evaluate(
    train_data: &[AggPoint],
    test_data: &[AggPoint],
    specs: &[HeadSpec],
    reservoir_size: usize,
    input_size: usize,
) -> Vec<(String, f64, f64, f64)> {
    train_and_evaluate_with(train_data, test_data, specs, reservoir_size, input_size, Activation::Tanh)
}

fn train_and_evaluate_with(
    train_data: &[AggPoint],
    test_data: &[AggPoint],
    specs: &[HeadSpec],
    reservoir_size: usize,
    input_size: usize,
    activation: Activation,
) -> Vec<(String, f64, f64, f64)> {
    let seq_len = 10;
    let n_heads = specs.len();

    let train_seqs: Vec<Vec<Vec<f64>>> = train_data.iter().map(|p| build_sequence_v2(p, seq_len)).collect();
    let train_targets: Vec<Vec<f64>> = train_data.iter().map(|p| {
        specs.iter().map(|s| (s.target_fn)(p)).collect()
    }).collect();

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
            if let Ok(out) = esn.predict(&seq) {
                if hi < out.len() {
                    actual.push((spec.target_fn)(p));
                    predicted.push(out[hi]);
                }
            }
        }

        if actual.is_empty() {
            results.push((spec.name.to_string(), f64::NAN, f64::NAN, f64::NAN));
            continue;
        }

        let n = actual.len() as f64;
        let mae: f64 = actual.iter().zip(&predicted)
            .map(|(a, p)| (a - p).abs())
            .sum::<f64>() / n;
        let mean_actual = actual.iter().sum::<f64>() / n;
        let ss_res: f64 = actual.iter().zip(&predicted)
            .map(|(a, p)| (a - p).powi(2)).sum();
        let ss_tot: f64 = actual.iter()
            .map(|a| (a - mean_actual).powi(2)).sum();
        let r2 = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 };
        let mean_abs_actual = actual.iter().map(|a| a.abs()).sum::<f64>() / n;
        let rel_err = if mean_abs_actual > 1e-15 { mae / mean_abs_actual } else { f64::NAN };

        results.push((spec.name.to_string(), mae, r2, rel_err));
    }
    results
}

fn grade(r2: f64) -> &'static str {
    if r2 > 0.9 { "EXCELLENT" }
    else if r2 > 0.7 { "GOOD" }
    else if r2 > 0.4 { "FAIR" }
    else if r2 > 0.0 { "WEAK" }
    else { "FAIL" }
}

fn print_results_table(results: &[(String, f64, f64, f64)]) {
    println!("  {:<18} {:>8} {:>8} {:>10} {}", "Head", "MAE", "R²", "RelErr", "Grade");
    println!("  {:-<18} {:->8} {:->8} {:->10} {:->10}", "", "", "", "", "");
    for (name, mae, r2, rel) in results {
        println!("  {:<18} {:>8.4} {:>8.3} {:>9.1}%  {}", name, mae, r2, rel * 100.0, grade(*r2));
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Synthetic Datasets
// ═══════════════════════════════════════════════════════════════════

fn generate_sine_dataset(n: usize) -> (Vec<AggPoint>, &'static str) {
    let points: Vec<AggPoint> = (0..n).map(|i| {
        let t = i as f64 / n as f64;
        let x = t * 4.0 * std::f64::consts::PI;
        AggPoint {
            beta: 5.0 + t * 2.0,
            mass: 0.1,
            lattice: 4,
            mean_plaq: 0.5 + 0.2 * x.sin(),
            std_plaq: 0.01,
            acceptance: 0.5 + 0.4 * (x * 0.7).cos(),
            mean_cg: 100.0 + 900.0 * (0.5 + 0.5 * x.sin()),
            mean_delta_h: 3.0 + x.sin(),
            polyakov: 0.3,
            n_traj: 10,
            experiment: format!("syn_sin_{i}"),
        }
    }).collect();
    (points, "Sine: smooth periodic targets, tests basic function approximation")
}

fn generate_step_dataset(n: usize) -> (Vec<AggPoint>, &'static str) {
    let points: Vec<AggPoint> = (0..n).map(|i| {
        let t = i as f64 / n as f64;
        let phase = if t > 0.5 { 1.0 } else { 0.0 };
        AggPoint {
            beta: 5.0 + t * 2.0,
            mass: 0.1,
            lattice: 4,
            mean_plaq: 0.4 + 0.2 * phase,
            std_plaq: 0.02 - 0.01 * phase,
            acceptance: 0.3 + 0.5 * phase,
            mean_cg: if phase > 0.5 { 500.0 } else { 50000.0 },
            mean_delta_h: 3.0,
            polyakov: 0.3,
            n_traj: 10,
            experiment: format!("syn_step_{i}"),
        }
    }).collect();
    (points, "Step: sharp phase transition, tests classification boundary")
}

fn generate_power_law_dataset(n: usize) -> (Vec<AggPoint>, &'static str) {
    let points: Vec<AggPoint> = (0..n).map(|i| {
        let mass = 0.01 + (i as f64 / n as f64) * 0.49;
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
    }).collect();
    (points, "Power-law: CG ~ 1/m², tests multi-scale regression (the hard case)")
}

fn generate_volume_scaling_dataset(n: usize) -> (Vec<AggPoint>, &'static str) {
    let lattices = [2, 4, 6, 8];
    let points: Vec<AggPoint> = (0..n).map(|i| {
        let lat = lattices[i % 4];
        let vol = (lat as f64).powi(4);
        let beta = 5.0 + (i as f64 / n as f64) * 2.0;
        AggPoint {
            beta,
            mass: 0.1,
            lattice: lat,
            mean_plaq: 0.4 + 0.1 * (beta - 5.0),
            std_plaq: 0.1 / vol.sqrt(),
            acceptance: (0.9 - vol.log10() * 0.1).max(0.1),
            mean_cg: 100.0 * vol.sqrt(),
            mean_delta_h: 3.0,
            polyakov: 0.3,
            n_traj: 10,
            experiment: format!("syn_vol_{i}"),
        }
    }).collect();
    (points, "Volume-scaling: observables scale with L^4, tests volume awareness")
}

fn generate_noisy_dataset(n: usize) -> (Vec<AggPoint>, &'static str) {
    let mut rng_state: u64 = 12345;
    let mut next_f64 = move || -> f64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64) / (u64::MAX as f64)
    };
    let points: Vec<AggPoint> = (0..n).map(|i| {
        let t = i as f64 / n as f64;
        let noise = next_f64() * 0.3;
        AggPoint {
            beta: 5.0 + t * 2.0,
            mass: 0.1,
            lattice: 4,
            mean_plaq: 0.5 + 0.1 * t + noise * 0.1,
            std_plaq: 0.01 + noise * 0.02,
            acceptance: (0.7 + noise * 0.3).clamp(0.0, 1.0),
            mean_cg: 1000.0 + noise * 5000.0,
            mean_delta_h: 3.0,
            polyakov: 0.3,
            n_traj: 10,
            experiment: format!("syn_noise_{i}"),
        }
    }).collect();
    (points, "Noisy: 30% uniform noise on targets, tests noise robustness")
}

fn run_synthetic_test(
    name: &str,
    description: &str,
    data: &[AggPoint],
    specs: &[HeadSpec],
) {
    let split = data.len() * 4 / 5;
    let train = &data[..split];
    let test = &data[split..];
    println!("\n  ── {} ──", name);
    println!("  {}", description);
    println!("  {} train, {} test points", train.len(), test.len());
    let results = train_and_evaluate(train, test, specs, 100, 6);
    print_results_table(&results);
}

fn run_activation_comparison(
    name: &str,
    data: &[AggPoint],
    specs: &[HeadSpec],
) {
    let split = data.len() * 4 / 5;
    let train = &data[..split];
    let test = &data[split..];

    let tanh_results = train_and_evaluate_with(train, test, specs, 100, 6, Activation::Tanh);
    let relu_results = train_and_evaluate_with(train, test, specs, 100, 6, Activation::ReluTanhApprox);

    println!("\n  ── {} ──", name);
    println!("  {} train, {} test points\n", train.len(), test.len());
    println!("  {:<18} {:>8} {:>8}   {:>8} {:>8}   {:>7}", "Head", "tanh R²", "Grade", "ReLU R²", "Grade", "Δ R²");
    println!("  {:-<18} {:->8} {:->8}   {:->8} {:->8}   {:->7}", "", "", "", "", "", "");
    for (i, spec) in specs.iter().enumerate() {
        let t_r2 = tanh_results.get(i).map(|r| r.2).unwrap_or(f64::NAN);
        let r_r2 = relu_results.get(i).map(|r| r.2).unwrap_or(f64::NAN);
        let delta = r_r2 - t_r2;
        let marker = if delta.abs() < 0.01 { "  ≈" }
            else if delta > 0.0 { " ▲" }
            else { " ▼" };
        println!("  {:<18} {:>8.3} {:>8}   {:>8.3} {:>8}   {:>+6.3}{}",
            spec.name, t_r2, grade(t_r2), r_r2, grade(r_r2), delta, marker);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    println!("══════════════════════════════════════════════════════");
    println!("  ESN Baseline Validation Harness v2");
    println!("  6D input (+ volume), log-CG, synthetic probes");
    println!("══════════════════════════════════════════════════════");

    let results_dir = std::env::args().nth(1).unwrap_or_else(|| "results".to_string());
    let results_path = if PathBuf::from(&results_dir).is_absolute() {
        PathBuf::from(&results_dir)
    } else {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.push(&results_dir);
        p
    };

    println!("\n  Loading JSONL from: {}", results_path.display());
    let all_points = load_jsonl_files(&results_path.to_string_lossy());
    println!("  Loaded {} aggregated (beta, mass, lattice) points", all_points.len());

    if all_points.len() < 5 {
        eprintln!("  ERROR: need at least 5 data points for cross-validation");
        std::process::exit(1);
    }

    let total_traj: usize = all_points.iter().map(|p| p.n_traj).sum();
    println!("  Total measurement trajectories: {total_traj}");

    let experiments: Vec<String> = {
        let mut e: Vec<String> = all_points.iter().map(|p| p.experiment.clone()).collect();
        e.sort(); e.dedup(); e
    };
    println!("  Experiments: {}", experiments.join(", "));

    let lattices: Vec<usize> = {
        let mut l: Vec<usize> = all_points.iter().map(|p| p.lattice).collect();
        l.sort(); l.dedup(); l
    };
    println!("  Lattice sizes: {:?}", lattices);

    let masses: Vec<String> = {
        let mut m: Vec<String> = all_points.iter().map(|p| format!("{:.3}", p.mass)).collect();
        m.sort(); m.dedup(); m
    };
    println!("  Masses: {}", masses.join(", "));

    let specs = head_specs_v2();

    // ════════════════════════════════════════════════════════════
    // PART 0: ACTIVATION COMPARISON — tanh vs ReLU-approx-tanh
    // ════════════════════════════════════════════════════════════
    println!("\n══════════════════════════════════════════════════════");
    println!("  PART 0: ACTIVATION COMPARISON");
    println!("  tanh (native) vs ReLU-approx-tanh (AKD1000-deployable)");
    println!("  Same reservoir, same weights, different activation");
    println!("══════════════════════════════════════════════════════");

    let (sine_data, sine_desc) = generate_sine_dataset(100);
    let (step_data, step_desc) = generate_step_dataset(100);
    let (pow_data, pow_desc) = generate_power_law_dataset(100);
    let (vol_data, vol_desc) = generate_volume_scaling_dataset(100);
    let (noisy_data, noisy_desc) = generate_noisy_dataset(100);

    run_activation_comparison("Sine Wave (smooth periodic)", &sine_data, &specs);
    run_activation_comparison("Step Function (sharp boundary)", &step_data, &specs);
    run_activation_comparison("Power-Law CG (multi-scale)", &pow_data, &specs);
    run_activation_comparison("Volume Scaling (L^4)", &vol_data, &specs);
    run_activation_comparison("Noisy Signal (robustness)", &noisy_data, &specs);

    if all_points.len() >= 10 {
        run_activation_comparison("Real Physics Data", &all_points, &specs);
    }

    // ════════════════════════════════════════════════════════════
    // PART 1: SYNTHETIC DATASETS (tanh baseline)
    // ════════════════════════════════════════════════════════════
    println!("\n══════════════════════════════════════════════════════");
    println!("  PART 1: SYNTHETIC DATASET PROBES (tanh baseline)");
    println!("  (Tests ESN capability independent of real data)");
    println!("══════════════════════════════════════════════════════");

    run_synthetic_test("Sine Wave", sine_desc, &sine_data, &specs);

    run_synthetic_test("Step Function", step_desc, &step_data, &specs);

    run_synthetic_test("Power-Law CG", pow_desc, &pow_data, &specs);

    run_synthetic_test("Volume Scaling", vol_desc, &vol_data, &specs);

    run_synthetic_test("Noisy Signal", noisy_desc, &noisy_data, &specs);

    // ════════════════════════════════════════════════════════════
    // PART 2: REAL DATA CROSS-VALIDATION
    // ════════════════════════════════════════════════════════════
    println!("\n══════════════════════════════════════════════════════");
    println!("  PART 2: REAL DATA CROSS-VALIDATION (v2 targets)");
    println!("══════════════════════════════════════════════════════");

    let mut all_cv_results: Vec<Vec<(String, f64, f64, f64)>> = Vec::new();

    let exp_groups: Vec<Vec<String>> = if experiments.len() <= 5 {
        experiments.iter().map(|e| vec![e.clone()]).collect()
    } else {
        let k = 5;
        let chunk = (experiments.len() + k - 1) / k;
        experiments.chunks(chunk).map(|c| c.to_vec()).collect()
    };

    for (fold, held_out) in exp_groups.iter().enumerate() {
        let train: Vec<AggPoint> = all_points.iter()
            .filter(|p| !held_out.contains(&p.experiment))
            .cloned().collect();
        let test: Vec<AggPoint> = all_points.iter()
            .filter(|p| held_out.contains(&p.experiment))
            .cloned().collect();

        if train.is_empty() || test.is_empty() { continue; }

        let results = train_and_evaluate(&train, &test, &specs, 100, 6);
        println!("\n  Fold {}: train={} pts (excl {:?}), test={} pts",
            fold + 1, train.len(), held_out, test.len());
        print_results_table(&results);
        all_cv_results.push(results);
    }

    if !all_cv_results.is_empty() {
        println!("\n══════════════════════════════════════════════════════");
        println!("  MEAN CROSS-VALIDATION RESULTS (v2)");
        println!("══════════════════════════════════════════════════════");

        let n_folds = all_cv_results.len() as f64;
        let mut mean_results = Vec::new();
        for hi in 0..specs.len() {
            let name = specs[hi].name.to_string();
            let mae_avg: f64 = all_cv_results.iter()
                .filter_map(|r| r.get(hi).map(|x| x.1))
                .filter(|v| v.is_finite()).sum::<f64>() / n_folds;
            let r2_avg: f64 = all_cv_results.iter()
                .filter_map(|r| r.get(hi).map(|x| x.2))
                .filter(|v| v.is_finite()).sum::<f64>() / n_folds;
            let rel_avg: f64 = all_cv_results.iter()
                .filter_map(|r| r.get(hi).map(|x| x.3))
                .filter(|v| v.is_finite()).sum::<f64>() / n_folds;
            mean_results.push((name, mae_avg, r2_avg, rel_avg));
        }
        print_results_table(&mean_results);
    }

    // ════════════════════════════════════════════════════════════
    // PART 3: RESERVOIR HYPERPARAMETER SWEEP
    // ════════════════════════════════════════════════════════════
    println!("\n══════════════════════════════════════════════════════");
    println!("  PART 3: RESERVOIR SIZE SWEEP");
    println!("══════════════════════════════════════════════════════");

    let n = all_points.len();
    let test_size = n / 5;
    let train_pool: Vec<AggPoint> = all_points[test_size..].to_vec();
    let test_set: Vec<AggPoint> = all_points[..test_size].to_vec();

    if !test_set.is_empty() && !train_pool.is_empty() {
        let res_sizes = [25, 50, 100, 200, 400];

        print!("  {:<18}", "Head");
        for &rs in &res_sizes {
            print!("  N={:<4}", rs);
        }
        println!("   (R² by reservoir size)");
        print!("  {:-<18}", "");
        for _ in &res_sizes { print!(" {:->6}", ""); }
        println!();

        for spec in &specs {
            print!("  {:<18}", spec.name);
            for &rs in &res_sizes {
                let results = train_and_evaluate(&train_pool, &test_set, &[HeadSpec {
                    name: spec.name, index: spec.index, target_fn: spec.target_fn,
                }], rs, 6);
                let r2 = results.first().map(|r| r.2).unwrap_or(f64::NAN);
                print!(" {:>6.3}", r2);
            }
            println!();
        }
    }

    // ════════════════════════════════════════════════════════════
    // PART 4: CROSS-VOLUME GENERALIZATION (with volume input)
    // ════════════════════════════════════════════════════════════
    if lattices.len() > 1 {
        println!("\n══════════════════════════════════════════════════════");
        println!("  PART 4: CROSS-VOLUME GENERALIZATION (6D input)");
        println!("══════════════════════════════════════════════════════");

        for &test_lat in &lattices {
            let train: Vec<AggPoint> = all_points.iter()
                .filter(|p| p.lattice != test_lat).cloned().collect();
            let test: Vec<AggPoint> = all_points.iter()
                .filter(|p| p.lattice == test_lat).cloned().collect();

            if train.len() < 3 || test.is_empty() { continue; }

            let results = train_and_evaluate(&train, &test, &specs, 100, 6);
            println!("\n  Train: all except {}^4 ({} pts), Test: {}^4 ({} pts)",
                test_lat, train.len(), test_lat, test.len());
            print_results_table(&results);
        }
    }

    // ════════════════════════════════════════════════════════════
    // PART 5: LEARNING CURVES
    // ════════════════════════════════════════════════════════════
    println!("\n══════════════════════════════════════════════════════");
    println!("  PART 5: LEARNING CURVES (data volume sweep)");
    println!("══════════════════════════════════════════════════════");

    if !test_set.is_empty() && !train_pool.is_empty() {
        let fractions = [0.1, 0.25, 0.5, 0.75, 1.0];

        println!("  Test set: {} pts, Train pool: {} pts\n", test_set.len(), train_pool.len());
        print!("  {:<18}", "Head");
        for f in &fractions { print!(" {:>6.0}%", f * 100.0); }
        println!("   (R² at each fraction)");
        print!("  {:-<18}", "");
        for _ in &fractions { print!(" {:->6}", ""); }
        println!();

        for spec in &specs {
            print!("  {:<18}", spec.name);
            for &frac in &fractions {
                let n_train = ((train_pool.len() as f64 * frac) as usize).max(3);
                let subset: Vec<AggPoint> = train_pool[..n_train].to_vec();
                let results = train_and_evaluate(&subset, &test_set, &[HeadSpec {
                    name: spec.name, index: spec.index, target_fn: spec.target_fn,
                }], 100, 6);
                let r2 = results.first().map(|r| r.2).unwrap_or(f64::NAN);
                print!(" {:>6.3}", r2);
            }
            println!();
        }
    }

    // ════════════════════════════════════════════════════════════
    // PART 6: PER-MASS REGIME ANALYSIS
    // ════════════════════════════════════════════════════════════
    println!("\n══════════════════════════════════════════════════════");
    println!("  PART 6: PER-MASS REGIME ANALYSIS");
    println!("══════════════════════════════════════════════════════");

    let mass_groups: Vec<(String, Vec<AggPoint>)> = {
        let mut groups: HashMap<String, Vec<AggPoint>> = HashMap::new();
        for p in &all_points {
            let key = format!("{:.3}", p.mass);
            groups.entry(key).or_default().push(p.clone());
        }
        let mut sorted: Vec<_> = groups.into_iter().collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        sorted
    };

    println!("  {:<8} {:>5} {:>10} {:>10} {:>10} {:>10}", "Mass", "N", "mean_plaq", "mean_CG", "mean_acc", "mean_beta");
    println!("  {:-<8} {:->5} {:->10} {:->10} {:->10} {:->10}", "", "", "", "", "", "");
    for (mass_str, pts) in &mass_groups {
        let n = pts.len() as f64;
        let mean_p: f64 = pts.iter().map(|p| p.mean_plaq).sum::<f64>() / n;
        let mean_c: f64 = pts.iter().map(|p| p.mean_cg).sum::<f64>() / n;
        let mean_a: f64 = pts.iter().map(|p| p.acceptance).sum::<f64>() / n;
        let mean_b: f64 = pts.iter().map(|p| p.beta).sum::<f64>() / n;
        println!("  {:<8} {:>5} {:>10.4} {:>10.0} {:>10.2} {:>10.2}",
            mass_str, pts.len(), mean_p, mean_c, mean_a, mean_b);
    }

    // Train on heavy masses, test on light (the hard direction)
    let heavy: Vec<AggPoint> = all_points.iter().filter(|p| p.mass >= 0.1).cloned().collect();
    let light: Vec<AggPoint> = all_points.iter().filter(|p| p.mass < 0.1).cloned().collect();
    if heavy.len() >= 5 && light.len() >= 3 {
        println!("\n  Train: heavy masses (m>=0.1, {} pts), Test: light (m<0.1, {} pts)", heavy.len(), light.len());
        let results = train_and_evaluate(&heavy, &light, &specs, 100, 6);
        print_results_table(&results);
    }

    let light_train: Vec<AggPoint> = all_points.iter().filter(|p| p.mass <= 0.1).cloned().collect();
    let heavy_test: Vec<AggPoint> = all_points.iter().filter(|p| p.mass > 0.1).cloned().collect();
    if light_train.len() >= 5 && heavy_test.len() >= 3 {
        println!("\n  Train: light masses (m<=0.1, {} pts), Test: heavy (m>0.1, {} pts)", light_train.len(), heavy_test.len());
        let results = train_and_evaluate(&light_train, &heavy_test, &specs, 100, 6);
        print_results_table(&results);
    }

    println!("\n══════════════════════════════════════════════════════");
    println!("  DONE — v2 harness complete");
    println!("══════════════════════════════════════════════════════");
}
