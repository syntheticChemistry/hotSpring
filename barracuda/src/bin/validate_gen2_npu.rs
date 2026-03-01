// SPDX-License-Identifier: AGPL-3.0-only

//! Gen 2 NPU Validation — Overlapping Head Groups & Disagreement Map
//!
//! CPU-only test (no GPU required). Loads trajectory data from past
//! production runs (Exp 024, 028), trains the 36-head ESN, sweeps
//! β-space, and reports:
//!
//! 1. Per-head predictions across all groups
//! 2. Cross-group disagreement (Δ_cg, Δ_phase, Δ_anomaly, Δ_priority)
//! 3. "Concept edge" map: where in β-space do physics models diverge?
//!
//! Run while production experiments are in progress — this is pure CPU.

use hotspring_barracuda::md::reservoir::{
    heads, EchoStateNetwork, EsnConfig, ExportedWeights, HeadGroupDisagreement, MultiHeadNpu,
};
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(serde::Deserialize)]
struct TrajRecord {
    beta: f64,
    #[serde(default)]
    mass: f64,
    plaquette: f64,
    accepted: bool,
    #[serde(default)]
    cg_iters: usize,
    #[serde(default)]
    phase: String,
}

struct BetaSummary {
    beta: f64,
    mean_plaq: f64,
    std_plaq: f64,
    acceptance: f64,
    mean_cg: f64,
    susceptibility: f64,
    n_meas: usize,
}

fn load_and_aggregate(path: &str) -> Vec<BetaSummary> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Warning: cannot read {path}: {e}");
            return Vec::new();
        }
    };

    let records: Vec<TrajRecord> = content
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    let meas: Vec<&TrajRecord> = records
        .iter()
        .filter(|r| r.phase == "measurement")
        .collect();
    let effective = if meas.is_empty() { &records } else {
        // Use measurement-phase records
        // Safety: meas borrows from records which is alive
        &records // fallback handled below
    };
    let use_meas = !meas.is_empty();

    let mut by_beta: BTreeMap<i64, Vec<&TrajRecord>> = BTreeMap::new();
    let source = if use_meas {
        meas.iter().map(|r| *r).collect::<Vec<_>>()
    } else {
        records.iter().collect::<Vec<_>>()
    };

    for r in &source {
        let key = (r.beta * 10000.0).round() as i64;
        by_beta.entry(key).or_default().push(r);
    }

    let _ = effective;
    let mut summaries = Vec::new();
    for (_key, group) in &by_beta {
        let n = group.len();
        if n == 0 { continue; }

        let plaqs: Vec<f64> = group.iter().map(|r| r.plaquette).collect();
        let mean_plaq = plaqs.iter().sum::<f64>() / n as f64;
        let variance = if n > 1 {
            plaqs.iter().map(|p| (p - mean_plaq).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let std_plaq = variance.sqrt();
        let acceptance = group.iter().filter(|r| r.accepted).count() as f64 / n as f64;
        let cgs: Vec<f64> = group.iter().filter(|r| r.cg_iters > 0).map(|r| r.cg_iters as f64).collect();
        let mean_cg = if cgs.is_empty() { 0.0 } else { cgs.iter().sum::<f64>() / cgs.len() as f64 };

        summaries.push(BetaSummary {
            beta: group[0].beta,
            mean_plaq,
            std_plaq,
            acceptance,
            mean_cg,
            susceptibility: variance * n as f64,
            n_meas: n,
        });
    }

    println!("  {path}: {} beta points from {} records", summaries.len(), source.len());
    summaries
}

fn build_gen2_training_data(summaries: &[BetaSummary]) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut seqs = Vec::new();
    let mut targets = Vec::new();

    for r in summaries {
        let beta_norm = (r.beta - 5.0) / 2.0;
        let phase_val = if r.mean_plaq > 0.56 { 1.0 } else if r.mean_plaq < 0.40 { 0.0 } else { 0.5 };
        let proximity = (-(r.beta - 5.69_f64).powi(2) / 0.1).exp();
        let cg_norm = r.mean_cg / 500.0;
        let anomaly_val = if r.acceptance < 0.05 || r.mean_cg > 3000.0 { 1.0 } else { 0.0 };
        let quality = (r.acceptance * 0.4
            + (1.0 - r.std_plaq / r.mean_plaq.abs().max(1e-10)).clamp(0.0, 1.0) * 0.3
            + (r.n_meas as f64 / 100.0).min(1.0) * 0.3)
            .clamp(0.0, 1.0);

        let anderson_phase = if r.beta > 5.5 { 1.0 } else if r.beta < 5.0 { 0.0 } else { 0.5 };
        let potts_phase = if r.beta > 5.8 { 1.0 } else if r.beta < 5.2 { 0.0 } else { 0.5 };
        let optimal_dt = 0.01 + r.acceptance.abs() * 0.04;
        let optimal_nmd = ((1.0 / optimal_dt).round() / 200.0).clamp(0.0, 1.0);

        let seq: Vec<Vec<f64>> = (0..10)
            .map(|j| {
                let noise = 0.005 * ((j as f64) * 0.7).sin();
                vec![beta_norm, r.mean_plaq + noise * r.std_plaq, r.acceptance, r.susceptibility / 1000.0, r.acceptance]
            })
            .collect();
        seqs.push(seq);

        let mut t = vec![0.0; heads::NUM_HEADS];

        // Group A: Anderson-informed
        t[heads::A0_ANDERSON_CG_COST] = cg_norm;
        t[heads::A1_ANDERSON_PHASE] = anderson_phase;
        t[heads::A2_ANDERSON_LAMBDA_MIN] = (1.0 / (cg_norm + 0.01)).clamp(0.0, 1.0);
        t[heads::A3_ANDERSON_ANOMALY] = anomaly_val;
        t[heads::A4_ANDERSON_THERM] = if r.acceptance > 0.3 { 1.0 } else { 0.0 };
        t[heads::A5_ANDERSON_PRIORITY] = proximity;

        // Group B: QCD-empirical
        t[heads::B0_QCD_CG_COST] = cg_norm;
        t[heads::B1_QCD_PHASE] = phase_val;
        t[heads::B2_QCD_ACCEPTANCE] = 1.0 - r.acceptance;
        t[heads::B3_QCD_ANOMALY] = anomaly_val;
        t[heads::B4_QCD_THERM] = if r.acceptance > 0.3 { 1.0 } else { 0.0 };
        t[heads::B5_QCD_PRIORITY] = proximity;

        // Group C: Potts-informed
        t[heads::C0_POTTS_CG_COST] = cg_norm;
        t[heads::C1_POTTS_PHASE] = potts_phase;
        t[heads::C2_POTTS_BETA_C] = (r.beta - 5.69).abs().min(1.0);
        t[heads::C3_POTTS_ANOMALY] = anomaly_val;
        t[heads::C4_POTTS_ORDER] = 0.0;
        t[heads::C5_POTTS_PRIORITY] = proximity;

        // Group D: Steering/Control
        t[heads::D0_NEXT_BETA] = proximity;
        t[heads::D1_OPTIMAL_DT] = optimal_dt;
        t[heads::D2_OPTIMAL_NMD] = optimal_nmd;
        t[heads::D3_CHECK_INTERVAL] = if cg_norm > 0.5 { 0.2 } else { 0.8 };
        t[heads::D4_KILL_DECISION] = if r.mean_cg > 400.0 { 0.8 } else { 0.1 };
        t[heads::D5_SKIP_DECISION] = if quality < 0.2 { 0.8 } else { 0.1 };

        // Group E: Brain/Monitor
        t[heads::E0_RESIDUAL_ETA] = cg_norm;
        t[heads::E1_RESIDUAL_ANOMALY] = anomaly_val;
        t[heads::E2_CONVERGENCE_RATE] = (1.0 - cg_norm).clamp(0.0, 1.0);
        t[heads::E3_STALL_DETECTOR] = if r.mean_cg > 300.0 { 0.5 } else { 0.0 };
        t[heads::E4_DIVERGENCE_DETECTOR] = if anomaly_val > 0.5 { 0.8 } else { 0.0 };
        t[heads::E5_QUALITY_FORECAST] = quality;

        // Group M: Meta-mixer
        t[heads::M0_CG_CONSENSUS] = cg_norm;
        t[heads::M1_PHASE_CONSENSUS] = phase_val;
        t[heads::M2_CG_UNCERTAINTY] = (anderson_phase - phase_val).abs();
        t[heads::M3_PHASE_UNCERTAINTY] = (potts_phase - phase_val).abs();
        t[heads::M4_PROXY_TRUST] = if (anderson_phase - phase_val).abs() < 0.3 { 0.8 } else { 0.3 };
        t[heads::M5_ATTENTION_LEVEL] = if anomaly_val > 0.5 { 0.8 } else if cg_norm > 0.5 { 0.4 } else { 0.1 };

        targets.push(t);
    }

    (seqs, targets)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Gen 2 NPU Validation — Overlapping Heads & Disagreement Map  ║");
    println!("║  CPU-only: 36-head ESN, concept edge detection                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ═══ Load trajectory data from past runs ═══
    println!("═══ Loading Past Production Data ═══");
    let mut all_summaries = Vec::new();

    for path in &[
        "results/exp024_production_8x8.jsonl",
        "results/exp028_brain_production_8x8.jsonl",
    ] {
        let mut s = load_and_aggregate(path);
        all_summaries.append(&mut s);
    }

    // Deduplicate by beta (keep the one with more measurements)
    let mut by_beta: BTreeMap<i64, BetaSummary> = BTreeMap::new();
    for s in all_summaries {
        let key = (s.beta * 1000.0).round() as i64;
        let entry = by_beta.entry(key).or_insert_with(|| BetaSummary {
            beta: s.beta, mean_plaq: 0.0, std_plaq: 0.0, acceptance: 0.0,
            mean_cg: 0.0, susceptibility: 0.0, n_meas: 0,
        });
        if s.n_meas > entry.n_meas {
            *entry = s;
        }
    }
    let summaries: Vec<BetaSummary> = by_beta.into_values().collect();
    println!("  Combined: {} unique beta points", summaries.len());
    println!();

    // ═══ Train 36-head ESN ═══
    println!("═══ Training 36-Head ESN ═══");
    let t0 = Instant::now();

    let (seqs, targets) = build_gen2_training_data(&summaries);
    let mut esn = EchoStateNetwork::new(EsnConfig {
        input_size: 5,
        reservoir_size: 50,
        output_size: heads::NUM_HEADS,
        spectral_radius: 0.95,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-3,
        seed: 42,
    });
    esn.train(&seqs, &targets);
    let weights = esn.export_weights().expect("ESN export failed");
    let mut npu = MultiHeadNpu::from_exported(&weights);

    let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Trained in {train_ms:.1}ms");
    println!("  Readout weights: {} floats ({:.1} KB)",
        weights.w_out.len(), weights.w_out.len() as f64 * 4.0 / 1024.0);
    println!();

    // ═══ Sweep β-space: fine grid for disagreement map ═══
    println!("═══ Disagreement Map (β sweep) ═══");
    println!();

    let n_sweep = 100;
    let beta_min = 4.2;
    let beta_max = 6.6;
    let step = (beta_max - beta_min) / n_sweep as f64;

    struct SweepPoint {
        beta: f64,
        outputs: Vec<f64>,
        disagreement: HeadGroupDisagreement,
    }

    // Build interpolation tables from measured data for realistic sweep inputs
    let measured_betas: Vec<f64> = summaries.iter().map(|s| s.beta).collect();
    let measured_plaqs: Vec<f64> = summaries.iter().map(|s| s.mean_plaq).collect();
    let measured_acc: Vec<f64> = summaries.iter().map(|s| s.acceptance).collect();
    let measured_chi: Vec<f64> = summaries.iter().map(|s| s.susceptibility / 1000.0).collect();

    let interpolate = |beta: f64, xs: &[f64], ys: &[f64]| -> f64 {
        if beta <= xs[0] { return ys[0]; }
        if beta >= xs[xs.len() - 1] { return ys[ys.len() - 1]; }
        for i in 0..xs.len() - 1 {
            if beta >= xs[i] && beta <= xs[i + 1] {
                let t = (beta - xs[i]) / (xs[i + 1] - xs[i]);
                return ys[i] * (1.0 - t) + ys[i + 1] * t;
            }
        }
        ys[ys.len() / 2]
    };

    let mut sweep: Vec<SweepPoint> = Vec::new();

    for i in 0..=n_sweep {
        let beta = beta_min + i as f64 * step;
        let beta_norm = (beta - 5.0) / 2.0;
        let plaq = interpolate(beta, &measured_betas, &measured_plaqs);
        let acc = interpolate(beta, &measured_betas, &measured_acc);
        let chi = interpolate(beta, &measured_betas, &measured_chi);

        let input: Vec<Vec<f64>> = (0..10)
            .map(|_| vec![beta_norm, plaq, acc, chi, acc])
            .collect();

        let (outputs, disagreement) = npu.predict_with_disagreement(&input);
        sweep.push(SweepPoint { beta, outputs, disagreement });
    }

    // Print the disagreement map
    println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>7}",
        "β", "Δ_cg", "Δ_phase", "Δ_anom", "Δ_prior", "urgency", "edge?");
    println!("  {:─>6}  {:─>8}  {:─>8}  {:─>8}  {:─>8}  {:─>8}  {:─>7}",
        "", "", "", "", "", "", "");

    let mut concept_edges: Vec<(f64, f64)> = Vec::new();
    let urgency_threshold = 0.15;

    for pt in &sweep {
        let u = pt.disagreement.urgency();
        let is_edge = u > urgency_threshold;
        if is_edge {
            concept_edges.push((pt.beta, u));
        }
        // Print every 5th point, or if it's an edge
        let idx = ((pt.beta - beta_min) / step).round() as usize;
        if idx % 5 == 0 || is_edge {
            println!("  {:6.3}  {:8.4}  {:8.4}  {:8.4}  {:8.4}  {:8.4}  {}",
                pt.beta,
                pt.disagreement.delta_cg,
                pt.disagreement.delta_phase,
                pt.disagreement.delta_anomaly,
                pt.disagreement.delta_priority,
                u,
                if is_edge { "  <<<" } else { "" });
        }
    }
    println!();

    // ═══ Concept Edge Summary ═══
    println!("═══ Concept Edges (Disagreement Peaks) ═══");
    println!();

    if concept_edges.is_empty() {
        println!("  No concept edges detected (urgency threshold = {urgency_threshold}).");
        println!("  This may indicate the ESN needs more diverse training data.");
    } else {
        // Cluster nearby edges
        let mut clusters: Vec<(f64, f64, f64)> = Vec::new(); // (start, end, peak_urgency)
        let cluster_width = 0.15;

        for &(beta, urgency) in &concept_edges {
            if let Some(last) = clusters.last_mut() {
                if beta - last.1 < cluster_width {
                    last.1 = beta;
                    last.2 = last.2.max(urgency);
                    continue;
                }
            }
            clusters.push((beta, beta, urgency));
        }

        println!("  Found {} concept edge region(s):", clusters.len());
        println!();
        for (i, (start, end, peak)) in clusters.iter().enumerate() {
            let mid = (start + end) / 2.0;
            let width = end - start;
            println!("  Edge {}: β ∈ [{:.3}, {:.3}]  midpoint={:.3}  width={:.3}  peak_urgency={:.4}",
                i + 1, start, end, mid, width, peak);

            // Interpret the edge
            if mid > 5.0 && mid < 5.8 {
                println!("         → Crossover region: Anderson/Potts/QCD models diverge here");
                println!("           This is where the deconfinement crossover lives.");
                println!("           The NPU should sample this region MOST densely.");
            } else if mid < 4.5 {
                println!("         → Strong coupling edge: proxy models may be unreliable");
            } else if mid > 6.0 {
                println!("         → Weak coupling edge: Potts universality may break down");
            }
            println!();
        }

        // Visual map
        println!("  β-space disagreement map:");
        println!();
        let bar_width = 60;
        print!("  ");
        for i in 0..=n_sweep {
            let beta = beta_min + i as f64 * step;
            let u = sweep[i].disagreement.urgency();
            let col = (i as f64 / n_sweep as f64 * bar_width as f64) as usize;
            if col != ((i.max(1) - 1) as f64 / n_sweep as f64 * bar_width as f64) as usize {
                let ch = if u > 0.3 { '█' } else if u > 0.2 { '▓' } else if u > 0.1 { '▒' } else { '░' };
                print!("{ch}");
            }
            let _ = beta;
        }
        println!();
        println!("  {:<30}{:>30}", format!("β={:.1}", beta_min), format!("β={:.1}", beta_max));
        println!("  Legend: ░ agreement  ▒ mild  ▓ moderate  █ strong disagreement");
    }
    println!();

    // ═══ Per-Group Head Comparison at Training Points ═══
    println!("═══ Per-Group Predictions at Measured β Points ═══");
    println!();
    println!("  {:>6}  {:>7} {:>7} {:>7}  {:>7} {:>7} {:>7}  {:>7} {:>7} {:>7}  {:>8}",
        "β", "A:CG", "B:CG", "C:CG", "A:φ", "B:φ", "C:φ", "A:anom", "B:anom", "C:anom", "urgency");

    for s in &summaries {
        let beta_norm = (s.beta - 5.0) / 2.0;
        let input: Vec<Vec<f64>> = (0..10)
            .map(|_| vec![beta_norm, s.mean_plaq, s.acceptance, s.susceptibility / 1000.0, s.acceptance])
            .collect();

        let (out, dis) = npu.predict_with_disagreement(&input);
        println!("  {:6.3}  {:7.3} {:7.3} {:7.3}  {:7.3} {:7.3} {:7.3}  {:7.3} {:7.3} {:7.3}  {:8.4}",
            s.beta,
            out[heads::A0_ANDERSON_CG_COST], out[heads::B0_QCD_CG_COST], out[heads::C0_POTTS_CG_COST],
            out[heads::A1_ANDERSON_PHASE], out[heads::B1_QCD_PHASE], out[heads::C1_POTTS_PHASE],
            out[heads::A3_ANDERSON_ANOMALY], out[heads::B3_QCD_ANOMALY], out[heads::C3_POTTS_ANOMALY],
            dis.urgency());
    }
    println!();

    // ═══ Meta-head outputs ═══
    println!("═══ Meta-Mixer Head Outputs at Measured β Points ═══");
    println!();
    println!("  {:>6}  {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "β", "M:CG_con", "M:φ_con", "M:CG_Δ", "M:φ_Δ", "M:trust", "M:attn");

    for s in &summaries {
        let beta_norm = (s.beta - 5.0) / 2.0;
        let input: Vec<Vec<f64>> = (0..10)
            .map(|_| vec![beta_norm, s.mean_plaq, s.acceptance, s.susceptibility / 1000.0, s.acceptance])
            .collect();

        let out = npu.predict_all_heads(&input);
        println!("  {:6.3}  {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4}",
            s.beta,
            out[heads::M0_CG_CONSENSUS], out[heads::M1_PHASE_CONSENSUS],
            out[heads::M2_CG_UNCERTAINTY], out[heads::M3_PHASE_UNCERTAINTY],
            out[heads::M4_PROXY_TRUST], out[heads::M5_ATTENTION_LEVEL]);
    }
    println!();

    // ═══ Backward compatibility: load Gen 1 weights ═══
    println!("═══ Backward Compatibility: Gen 1 Weight Loading ═══");
    let gen1_path = "results/exp028_brain_weights.json";
    match std::fs::read_to_string(gen1_path) {
        Ok(json) => match serde_json::from_str::<ExportedWeights>(&json) {
            Ok(w) => {
                println!("  Loaded Gen 1 weights: {} heads (output_size={})",
                    w.output_size, w.output_size);
                let mut gen1_npu = MultiHeadNpu::from_exported(&w);
                let test_input: Vec<Vec<f64>> = (0..10)
                    .map(|_| vec![0.0, 0.5, 0.1, 0.01, 0.5])
                    .collect();
                let (out, dis) = gen1_npu.predict_with_disagreement(&test_input);
                println!("  Gen 1 output length: {} (< {} = Gen 2)", out.len(), heads::NUM_HEADS);
                println!("  Disagreement (should be zeros): Δ_cg={:.4} Δ_phase={:.4}",
                    dis.delta_cg, dis.delta_phase);
                println!("  ✓ Gen 1 weights load safely in Gen 2 code");
            }
            Err(e) => println!("  Cannot parse {gen1_path}: {e}"),
        },
        Err(e) => println!("  Cannot read {gen1_path}: {e} (expected — not a blocker)"),
    }
    println!();

    println!("═══ Summary ═══");
    println!("  Heads:           {}", heads::NUM_HEADS);
    println!("  Head groups:     6 (A:Anderson, B:QCD, C:Potts, D:Steering, E:Monitor, M:Meta)");
    println!("  Training points: {}", summaries.len());
    println!("  Sweep points:    {}", n_sweep + 1);
    println!("  Concept edges:   {}", concept_edges.len());
    println!("  Training time:   {train_ms:.1}ms");
    println!();
    println!("  The concept edges show WHERE physics models disagree —");
    println!("  these are the boundaries of model validity, and the most");
    println!("  scientifically interesting regions to sample densely.");
}
