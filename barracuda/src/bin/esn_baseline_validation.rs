// SPDX-License-Identifier: AGPL-3.0-only

//! CPU ESN baseline validation harness (v2).
//!
//! Loads historical JSONL, trains with improved targets (log-CG, 6D input with
//! volume), runs synthetic dataset probes, and sweeps reservoir hyperparameters.

use hotspring_barracuda::md::reservoir::esn_baseline::{
    self, head_specs_v2, load_jsonl_files, print_results_table, run_activation_comparison,
    run_synthetic_test, train_and_evaluate, AggPoint, HeadSpec,
};
use std::path::PathBuf;

fn main() {
    println!("══════════════════════════════════════════════════════");
    println!("  ESN Baseline Validation Harness v2");
    println!("  6D input (+ volume), log-CG, synthetic probes");
    println!("══════════════════════════════════════════════════════");

    let results_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "results".to_string());
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
    println!(
        "  Loaded {} aggregated (beta, mass, lattice) points",
        all_points.len()
    );

    if all_points.len() < 5 {
        eprintln!("  ERROR: need at least 5 data points for cross-validation");
        std::process::exit(1);
    }

    let total_traj: usize = all_points.iter().map(|p| p.n_traj).sum();
    println!("  Total measurement trajectories: {total_traj}");

    let experiments: Vec<String> = {
        let mut e: Vec<String> = all_points.iter().map(|p| p.experiment.clone()).collect();
        e.sort();
        e.dedup();
        e
    };
    println!("  Experiments: {}", experiments.join(", "));

    let lattices: Vec<usize> = {
        let mut l: Vec<usize> = all_points.iter().map(|p| p.lattice).collect();
        l.sort_unstable();
        l.dedup();
        l
    };
    println!("  Lattice sizes: {lattices:?}");

    let masses: Vec<String> = {
        let mut m: Vec<String> = all_points
            .iter()
            .map(|p| format!("{:.3}", p.mass))
            .collect();
        m.sort();
        m.dedup();
        m
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

    let (sine_data, sine_desc) = esn_baseline::generate_sine_dataset(100);
    let (step_data, step_desc) = esn_baseline::generate_step_dataset(100);
    let (pow_data, pow_desc) = esn_baseline::generate_power_law_dataset(100);
    let (vol_data, vol_desc) = esn_baseline::generate_volume_scaling_dataset(100);
    let (noisy_data, noisy_desc) = esn_baseline::generate_noisy_dataset(100);

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
        let chunk = experiments.len().div_ceil(k);
        experiments
            .chunks(chunk)
            .map(<[std::string::String]>::to_vec)
            .collect()
    };

    for (fold, held_out) in exp_groups.iter().enumerate() {
        let train: Vec<AggPoint> = all_points
            .iter()
            .filter(|p| !held_out.contains(&p.experiment))
            .cloned()
            .collect();
        let test: Vec<AggPoint> = all_points
            .iter()
            .filter(|p| held_out.contains(&p.experiment))
            .cloned()
            .collect();

        if train.is_empty() || test.is_empty() {
            continue;
        }

        let results = train_and_evaluate(&train, &test, &specs, 100, 6);
        println!(
            "\n  Fold {}: train={} pts (excl {:?}), test={} pts",
            fold + 1,
            train.len(),
            held_out,
            test.len()
        );
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
            let mae_avg: f64 = all_cv_results
                .iter()
                .filter_map(|r| r.get(hi).map(|x| x.1))
                .filter(|v| v.is_finite())
                .sum::<f64>()
                / n_folds;
            let r2_avg: f64 = all_cv_results
                .iter()
                .filter_map(|r| r.get(hi).map(|x| x.2))
                .filter(|v| v.is_finite())
                .sum::<f64>()
                / n_folds;
            let rel_avg: f64 = all_cv_results
                .iter()
                .filter_map(|r| r.get(hi).map(|x| x.3))
                .filter(|v| v.is_finite())
                .sum::<f64>()
                / n_folds;
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
            print!("  N={rs:<4}");
        }
        println!("   (R² by reservoir size)");
        print!("  {:-<18}", "");
        for _ in &res_sizes {
            print!(" {:->6}", "");
        }
        println!();

        for spec in &specs {
            print!("  {:<18}", spec.name);
            for &rs in &res_sizes {
                let results = train_and_evaluate(
                    &train_pool,
                    &test_set,
                    &[HeadSpec {
                        name: spec.name,
                        index: spec.index,
                        target_fn: spec.target_fn,
                    }],
                    rs,
                    6,
                );
                let r2 = results.first().map_or(f64::NAN, |r| r.2);
                print!(" {r2:>6.3}");
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
            let train: Vec<AggPoint> = all_points
                .iter()
                .filter(|p| p.lattice != test_lat)
                .cloned()
                .collect();
            let test: Vec<AggPoint> = all_points
                .iter()
                .filter(|p| p.lattice == test_lat)
                .cloned()
                .collect();

            if train.len() < 3 || test.is_empty() {
                continue;
            }

            let results = train_and_evaluate(&train, &test, &specs, 100, 6);
            println!(
                "\n  Train: all except {}^4 ({} pts), Test: {}^4 ({} pts)",
                test_lat,
                train.len(),
                test_lat,
                test.len()
            );
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

        println!(
            "  Test set: {} pts, Train pool: {} pts\n",
            test_set.len(),
            train_pool.len()
        );
        print!("  {:<18}", "Head");
        for f in &fractions {
            print!(" {:>6.0}%", f * 100.0);
        }
        println!("   (R² at each fraction)");
        print!("  {:-<18}", "");
        for _ in &fractions {
            print!(" {:->6}", "");
        }
        println!();

        for spec in &specs {
            print!("  {:<18}", spec.name);
            for &frac in &fractions {
                let n_train = ((train_pool.len() as f64 * frac) as usize).max(3);
                let subset: Vec<AggPoint> = train_pool[..n_train].to_vec();
                let results = train_and_evaluate(
                    &subset,
                    &test_set,
                    &[HeadSpec {
                        name: spec.name,
                        index: spec.index,
                        target_fn: spec.target_fn,
                    }],
                    100,
                    6,
                );
                let r2 = results.first().map_or(f64::NAN, |r| r.2);
                print!(" {r2:>6.3}");
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
        let mut groups: std::collections::HashMap<String, Vec<AggPoint>> =
            std::collections::HashMap::new();
        for p in &all_points {
            let key = format!("{:.3}", p.mass);
            groups.entry(key).or_default().push(p.clone());
        }
        let mut sorted: Vec<_> = groups.into_iter().collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        sorted
    };

    println!(
        "  {:<8} {:>5} {:>10} {:>10} {:>10} {:>10}",
        "Mass", "N", "mean_plaq", "mean_CG", "mean_acc", "mean_beta"
    );
    println!(
        "  {:-<8} {:->5} {:->10} {:->10} {:->10} {:->10}",
        "", "", "", "", "", ""
    );
    for (mass_str, pts) in &mass_groups {
        let n = pts.len() as f64;
        let mean_p: f64 = pts.iter().map(|p| p.mean_plaq).sum::<f64>() / n;
        let mean_c: f64 = pts.iter().map(|p| p.mean_cg).sum::<f64>() / n;
        let mean_a: f64 = pts.iter().map(|p| p.acceptance).sum::<f64>() / n;
        let mean_b: f64 = pts.iter().map(|p| p.beta).sum::<f64>() / n;
        println!(
            "  {:<8} {:>5} {:>10.4} {:>10.0} {:>10.2} {:>10.2}",
            mass_str,
            pts.len(),
            mean_p,
            mean_c,
            mean_a,
            mean_b
        );
    }

    // Train on heavy masses, test on light (the hard direction)
    let heavy: Vec<AggPoint> = all_points
        .iter()
        .filter(|p| p.mass >= 0.1)
        .cloned()
        .collect();
    let light: Vec<AggPoint> = all_points
        .iter()
        .filter(|p| p.mass < 0.1)
        .cloned()
        .collect();
    if heavy.len() >= 5 && light.len() >= 3 {
        println!(
            "\n  Train: heavy masses (m>=0.1, {} pts), Test: light (m<0.1, {} pts)",
            heavy.len(),
            light.len()
        );
        let results = train_and_evaluate(&heavy, &light, &specs, 100, 6);
        print_results_table(&results);
    }

    let light_train: Vec<AggPoint> = all_points
        .iter()
        .filter(|p| p.mass <= 0.1)
        .cloned()
        .collect();
    let heavy_test: Vec<AggPoint> = all_points
        .iter()
        .filter(|p| p.mass > 0.1)
        .cloned()
        .collect();
    if light_train.len() >= 5 && heavy_test.len() >= 3 {
        println!(
            "\n  Train: light masses (m<=0.1, {} pts), Test: heavy (m>0.1, {} pts)",
            light_train.len(),
            heavy_test.len()
        );
        let results = train_and_evaluate(&light_train, &heavy_test, &specs, 100, 6);
        print_results_table(&results);
    }

    println!("\n══════════════════════════════════════════════════════");
    println!("  DONE — v2 harness complete");
    println!("══════════════════════════════════════════════════════");
}
