// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-case persistent brain experiments.

use std::time::Instant;

use hotspring_barracuda::bench::{
    BenchReport, PhaseResult, PowerMonitor, peak_rss_mb,
};
use hotspring_barracuda::md::brain::MD_NUM_HEADS;
use hotspring_barracuda::md::config::{self, MdConfig};
use hotspring_barracuda::md::observables;
use hotspring_barracuda::md::sarkas_harness::{BrainState, run_single_case_with_brain};
use hotspring_barracuda::md::simulation;
use hotspring_barracuda::validation::ValidationHarness;

pub struct BrainEvolutionRow {
    pub primary: String,
    pub steps_per_sec: Option<f64>,
    pub trusted_heads: usize,
    pub confidence: f64,
    pub head_r2: Vec<f64>,
    pub nautilus_observations: usize,
    pub nautilus_generations: usize,
}

fn format_best_heads(r2s: &[f64]) -> String {
    let best: Vec<String> = r2s
        .iter()
        .enumerate()
        .filter(|(_, r)| **r > 0.1)
        .map(|(i, r)| format!("H{i}={r:.3}"))
        .collect();
    if best.is_empty() {
        "learning...".into()
    } else {
        best.join(", ")
    }
}

pub fn print_brain_evolution_table(title: &str, skin_mode: bool, rows: &[BrainEvolutionRow]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  {title:<56}  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    if skin_mode {
        println!(
            "  {:>6} {:>10} {:>8} {:>8} {:>8} {:>6}  Best heads",
            "Skin%", "Steps/s", "Trusted", "Conf", "NautObs", "Gens"
        );
        println!(
            "  {:>6} {:>10} {:>8} {:>8} {:>8} {:>6}  ─────────",
            "─────", "───────", "───────", "────", "───────", "────"
        );
        for row in rows {
            let sps = row.steps_per_sec.unwrap_or(0.0);
            println!(
                "  {:>6} {:>10.1} {:>5}/{} {:>8.3} {:>8} {:>6}  [{}]",
                row.primary,
                sps,
                row.trusted_heads,
                MD_NUM_HEADS,
                row.confidence,
                row.nautilus_observations,
                row.nautilus_generations,
                format_best_heads(&row.head_r2),
            );
        }
    } else {
        let primary_header = if rows
            .first()
            .is_some_and(|r| r.primary.chars().all(|c| c.is_ascii_digit()))
        {
            "N"
        } else {
            "Case"
        };
        println!(
            "  {:>12} {:>8} {:>8} {:>8} {:>6}  Best heads",
            primary_header, "Trusted", "Conf", "NautObs", "Gens"
        );
        println!(
            "  {:>12} {:>8} {:>8} {:>8} {:>6}  ─────────",
            "────", "───────", "────", "───────", "────"
        );
        for row in rows {
            println!(
                "  {:>12} {:>5}/{} {:>8.3} {:>8} {:>6}  [{}]",
                row.primary,
                row.trusted_heads,
                MD_NUM_HEADS,
                row.confidence,
                row.nautilus_observations,
                row.nautilus_generations,
                format_best_heads(&row.head_r2),
            );
        }
    }
    println!();
}

pub async fn run_brain_sweep(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Experiment: Cross-Case Persistent Brain                   ║");
    println!("║  9 PP Yukawa cases, N=2000, brain weights carried forward  ║");
    println!("║  Goal: teach brain to predict across (κ,Γ) regimes         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let cases = config::dsf_pp_cases(2000, true);
    let mut brain_state: Option<BrainState> = None;
    let mut passed = 0;
    let total = cases.len();
    let mut evolution: Vec<BrainEvolutionRow> = Vec::new();

    for (i, case_config) in cases.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "  Case {}/{}: {} [brain: {}]",
            i + 1,
            total,
            case_config.label,
            if brain_state.is_some() {
                "inherited"
            } else {
                "fresh"
            }
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let (ok, summary) =
            run_single_case_with_brain(case_config, report, brain_state.as_ref()).await;
        harness.check_bool(&format!("brain_sweep_{}", case_config.label), ok);
        if ok {
            passed += 1;
        }

        if let Some(bs) = summary {
            evolution.push(BrainEvolutionRow {
                primary: case_config.label.clone(),
                steps_per_sec: None,
                trusted_heads: bs.trusted_heads,
                confidence: bs.confidence,
                head_r2: bs.head_r2.clone(),
                nautilus_observations: bs.nautilus_observations,
                nautilus_generations: bs.nautilus_generations,
            });
            brain_state = bs.nautilus_json.map(|json| BrainState {
                nautilus_json: json,
            });
        }
        println!();
    }

    print_brain_evolution_table("BRAIN EVOLUTION ACROSS CASES", false, &evolution);

    println!("═══════════════════════════════════════════════════════════");
    println!("  BRAIN SWEEP: {passed}/{total} cases passed");
    println!("═══════════════════════════════════════════════════════════");
}

pub async fn run_brain_nscale(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Experiment: N-Scaling Persistent Brain                    ║");
    println!("║  κ=2, Γ=158 at N=500→10000, brain weights carried forward ║");
    println!("║  Goal: teach brain steps/s prediction and rebuild scaling  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let n_sizes = [500, 1000, 2000, 5000, 10_000];
    let mut brain_state: Option<BrainState> = None;
    let mut passed = 0;
    let total = n_sizes.len();
    let mut evolution: Vec<BrainEvolutionRow> = Vec::new();

    for (i, &n) in n_sizes.iter().enumerate() {
        let case_config = MdConfig {
            label: format!("brain_nscale_N{n}"),
            n_particles: n,
            kappa: 2.0,
            gamma: 158.0,
            dt: 0.01,
            rc: 6.5,
            equil_steps: 5_000,
            prod_steps: 30_000,
            dump_step: 10,
            berendsen_tau: 5.0,
            rdf_bins: 500,
            vel_snapshot_interval: 100,
        };

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "  N = {} ({}/{}) [brain: {}]",
            n,
            i + 1,
            total,
            if brain_state.is_some() {
                "inherited"
            } else {
                "fresh"
            }
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let (ok, summary) =
            run_single_case_with_brain(&case_config, report, brain_state.as_ref()).await;
        harness.check_bool(&format!("brain_nscale_N{n}"), ok);
        if ok {
            passed += 1;
        }

        if let Some(bs) = summary {
            evolution.push(BrainEvolutionRow {
                primary: n.to_string(),
                steps_per_sec: None,
                trusted_heads: bs.trusted_heads,
                confidence: bs.confidence,
                head_r2: bs.head_r2.clone(),
                nautilus_observations: bs.nautilus_observations,
                nautilus_generations: bs.nautilus_generations,
            });
            brain_state = bs.nautilus_json.map(|json| BrainState {
                nautilus_json: json,
            });
        }
        println!();
    }

    print_brain_evolution_table("BRAIN EVOLUTION ACROSS N", false, &evolution);

    println!("═══════════════════════════════════════════════════════════");
    println!("  BRAIN N-SCALE: {passed}/{total} cases passed");
    println!("═══════════════════════════════════════════════════════════");
}

pub async fn run_brain_skin_sweep(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Experiment: Skin Fraction Sweep with Persistent Brain     ║");
    println!("║  κ=2, Γ=158, N=2000 — skin from 0.05 to 0.40             ║");
    println!("║  Goal: teach brain optimal skin steering (C0)              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let skin_fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40];
    let mut brain_state: Option<BrainState> = None;
    let mut passed = 0;
    let total = skin_fractions.len();
    let mut evolution: Vec<BrainEvolutionRow> = Vec::new();

    for (i, &sf) in skin_fractions.iter().enumerate() {
        let rc = 6.5;
        let skin = sf * rc;
        let case_config = MdConfig {
            label: format!("brain_skin_{:.0}pct", sf * 100.0),
            n_particles: 2000,
            kappa: 2.0,
            gamma: 158.0,
            dt: 0.01,
            rc,
            equil_steps: 5_000,
            prod_steps: 30_000,
            dump_step: 10,
            berendsen_tau: 5.0,
            rdf_bins: 500,
            vel_snapshot_interval: 100,
        };

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "  skin_fraction={:.2} (skin={:.3} a_ws) [{}/{}] [brain: {}]",
            sf,
            skin,
            i + 1,
            total,
            if brain_state.is_some() {
                "inherited"
            } else {
                "fresh"
            }
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let monitor = PowerMonitor::start();
        let t0 = Instant::now();
        let nautilus_json = brain_state.as_ref().map(|bs| bs.nautilus_json.as_str());
        let result =
            simulation::run_simulation_verlet_with_brain(&case_config, skin, nautilus_json).await;
        let energy = monitor.stop();
        let wall_time = t0.elapsed().as_secs_f64();

        match result {
            Ok(sim) => {
                let energy_val = observables::validate_energy(&sim.energy_history, &case_config);
                let ssf_device = match hotspring_barracuda::gpu::GpuF64::new().await {
                    Ok(gpu) if gpu.has_f64 => Some(gpu.to_wgpu_device()),
                    _ => None,
                };
                observables::print_observable_summary_with_gpu(&sim, &case_config, ssf_device.as_ref())
                    .unwrap_or_else(|e| eprintln!("  VACF GPU failed: {e}"));

                if let Some(ref bs) = sim.brain_summary {
                    if bs.retrain_count > 0 {
                        let r2_strs: Vec<String> = bs
                            .head_r2
                            .iter()
                            .enumerate()
                            .filter(|(_, r)| **r > 0.1)
                            .map(|(i, r)| format!("H{i}={r:.3}"))
                            .collect();
                        println!(
                            "  Brain: {} retrains, {}/{} heads trusted, conf={:.2} [{}]",
                            bs.retrain_count,
                            bs.trusted_heads,
                            MD_NUM_HEADS,
                            bs.confidence,
                            if r2_strs.is_empty() {
                                "learning...".into()
                            } else {
                                r2_strs.join(", ")
                            },
                        );
                    }
                    evolution.push(BrainEvolutionRow {
                        primary: format!("{:.0}%", sf * 100.0),
                        steps_per_sec: Some(sim.steps_per_sec),
                        trusted_heads: bs.trusted_heads,
                        confidence: bs.confidence,
                        head_r2: bs.head_r2.clone(),
                        nautilus_observations: bs.nautilus_observations,
                        nautilus_generations: bs.nautilus_generations,
                    });
                    brain_state = bs.nautilus_json.clone().map(|json| BrainState {
                        nautilus_json: json,
                    });
                }

                let total_steps = case_config.equil_steps + case_config.prod_steps;
                report.add_phase(PhaseResult {
                    phase: format!("Sarkas GPU skin={:.0}%", sf * 100.0),
                    substrate: "GPU f64 WGSL".into(),
                    wall_time_s: wall_time,
                    per_eval_us: wall_time * 1e6 / total_steps as f64,
                    n_evals: total_steps,
                    energy,
                    peak_rss_mb: peak_rss_mb(),
                    chi2: energy_val.drift_pct,
                    precision_mev: 0.0,
                    notes: format!(
                        "skin={:.2}, {:.1} steps/s, drift={:.3}%",
                        sf, sim.steps_per_sec, energy_val.drift_pct
                    ),
                });

                harness.check_bool(&format!("brain_skin_{sf:.2}"), energy_val.passed);
                if energy_val.passed {
                    passed += 1;
                }
            }
            Err(e) => {
                println!("  ERROR: {e}");
                harness.check_bool(&format!("brain_skin_{sf:.2}"), false);
            }
        }
        println!();
    }

    print_brain_evolution_table("BRAIN EVOLUTION ACROSS SKIN FRACTIONS", true, &evolution);

    println!("═══════════════════════════════════════════════════════════");
    println!("  BRAIN SKIN SWEEP: {passed}/{total} cases passed");
    println!("═══════════════════════════════════════════════════════════");
}
