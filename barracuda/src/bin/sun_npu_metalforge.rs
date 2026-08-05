// SPDX-License-Identifier: AGPL-3.0-or-later

//! SU(N) NPU MetalForge — Comprehensive Phase Classification Experiment
//!
//! Expanded from sun_npu_monitor: richer observables, momentum-space features
//! (DFT of Polyakov spatial distribution), full confusion matrix with
//! precision/recall/F1/MCC, size-stratified analysis, and multi-backend
//! comparison (CPU f64 / NpuSimulator f32 / AKD1000 hardware).
//!
//! Usage:
//!   cargo run --release --features npu-hw,barracuda-local --bin sun_npu_metalforge

use hotspring_barracuda::lattice::generic_lattice::GenericLattice;
use hotspring_barracuda::lattice::su2::Su2Matrix;
use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use std::collections::BTreeMap;
use std::f64::consts::PI;
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

#[derive(Clone)]
struct RichObservable {
    gauge_group: String,
    dims: [usize; 4],
    beta: f64,
    plaquette: f64,
    polyakov_abs: f64,
    polyakov_re: f64,
    polyakov_im: f64,
    wilson_1_1: f64,
    wilson_2_1: f64,
    creutz_2_2: Option<f64>,
    polyakov_spatial_dft: Vec<f64>,
    volume: usize,
}

impl RichObservable {
    fn lattice_l(&self) -> usize {
        self.dims[0]
    }
}

/// DFT of the spatial Polyakov loop distribution.
/// Returns |P̃(k)|² for the lowest momentum modes.
fn polyakov_spatial_dft<G: hotspring_barracuda::lattice::gauge_group::GaugeGroup>(
    lat: &GenericLattice<G>,
) -> Vec<f64> {
    let ns = [lat.dims[0], lat.dims[1], lat.dims[2]];
    let spatial_vol = ns[0] * ns[1] * ns[2];

    let mut poly_re = Vec::with_capacity(spatial_vol);
    let mut poly_im = Vec::with_capacity(spatial_vol);

    for iz in 0..ns[2] {
        for iy in 0..ns[1] {
            for ix in 0..ns[0] {
                let c = lat.polyakov_loop([ix, iy, iz]);
                poly_re.push(c.re);
                poly_im.push(c.im);
            }
        }
    }

    let n_modes = 4.min(ns[0] / 2 + 1);
    let mut power_spectrum = Vec::with_capacity(n_modes);

    for kx in 0..n_modes {
        let mut ft_re = 0.0;
        let mut ft_im = 0.0;
        for (idx, (pre, pim)) in poly_re.iter().zip(poly_im.iter()).enumerate() {
            let ix = idx % ns[0];
            let phase = 2.0 * PI * (kx as f64) * (ix as f64) / (ns[0] as f64);
            let (s, c) = phase.sin_cos();
            ft_re += pre * c + pim * s;
            ft_im += pim * c - pre * s;
        }
        let norm = spatial_vol as f64;
        power_spectrum.push((ft_re * ft_re + ft_im * ft_im) / (norm * norm));
    }

    power_spectrum
}

fn scan_su2_configs() -> Vec<RichObservable> {
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
                let (poly_re, poly_im) = lat.complex_polyakov_average();
                let poly_abs = lat.average_polyakov_loop();
                let w11 = lat.spatial_temporal_wilson_loop(1, 1);
                let w21 = lat.spatial_temporal_wilson_loop(2, 1);
                let creutz = lat.creutz_ratio(2, 2);
                let dft = polyakov_spatial_dft(&lat);
                let volume = lat.volume();

                results.push(RichObservable {
                    gauge_group: "SU(2)".to_string(),
                    dims: lat.dims,
                    beta: lat.beta,
                    plaquette: plaq,
                    polyakov_abs: poly_abs,
                    polyakov_re: poly_re,
                    polyakov_im: poly_im,
                    wilson_1_1: w11,
                    wilson_2_1: w21,
                    creutz_2_2: creutz,
                    polyakov_spatial_dft: dft,
                    volume,
                });
            }
        }
    }

    results.sort_by(|a, b| {
        a.dims[0]
            .cmp(&b.dims[0])
            .then(a.beta.partial_cmp(&b.beta).unwrap())
    });
    results
}

fn scan_su3_configs() -> Vec<RichObservable> {
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
            if let Ok(lat) = Lattice::load(&path) {
                let plaq = lat.average_plaquette();
                let (poly_re, poly_im) = lat.complex_polyakov_average();
                let poly_abs = lat.average_polyakov_loop();
                let w11 = lat.spatial_temporal_wilson_loop(1, 1);
                let w21 = lat.spatial_temporal_wilson_loop(2, 1);
                let creutz = lat.creutz_ratio(2, 2);
                let volume = lat.volume();

                let ns = [lat.dims[0], lat.dims[1], lat.dims[2]];
                let spatial_vol = ns[0] * ns[1] * ns[2];
                let n_modes = 4.min(ns[0] / 2 + 1);

                let mut poly_spatial_re = Vec::with_capacity(spatial_vol);
                let mut poly_spatial_im = Vec::with_capacity(spatial_vol);
                for iz in 0..ns[2] {
                    for iy in 0..ns[1] {
                        for ix in 0..ns[0] {
                            let c = lat.polyakov_loop([ix, iy, iz]);
                            poly_spatial_re.push(c.re);
                            poly_spatial_im.push(c.im);
                        }
                    }
                }

                let mut dft = Vec::with_capacity(n_modes);
                for kx in 0..n_modes {
                    let mut ft_re = 0.0;
                    let mut ft_im = 0.0;
                    for (idx, (pre, pim)) in
                        poly_spatial_re.iter().zip(poly_spatial_im.iter()).enumerate()
                    {
                        let ix = idx % ns[0];
                        let phase = 2.0 * PI * (kx as f64) * (ix as f64) / (ns[0] as f64);
                        let (s, c) = phase.sin_cos();
                        ft_re += pre * c + pim * s;
                        ft_im += pim * c - pre * s;
                    }
                    let norm = spatial_vol as f64;
                    dft.push((ft_re * ft_re + ft_im * ft_im) / (norm * norm));
                }

                results.push(RichObservable {
                    gauge_group: "SU(3)".to_string(),
                    dims: lat.dims,
                    beta: lat.beta,
                    plaquette: plaq,
                    polyakov_abs: poly_abs,
                    polyakov_re: poly_re,
                    polyakov_im: poly_im,
                    wilson_1_1: w11,
                    wilson_2_1: w21,
                    creutz_2_2: creutz,
                    polyakov_spatial_dft: dft,
                    volume,
                });
            }
        }
    }

    results.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap());
    results
}

// ── Feature engineering ────────────────────────────────────────────────

const N_BASE_FEATURES: usize = 8;
const N_DFT_FEATURES: usize = 3;
const N_FEATURES: usize = N_BASE_FEATURES + N_DFT_FEATURES;

fn feature_vector(obs: &RichObservable, beta_c: f64) -> Vec<f64> {
    let beta_norm = (obs.beta - beta_c) / 2.0;
    let vol_norm = (obs.volume as f64).ln() / 10.0;

    let mut feat = vec![
        beta_norm,
        obs.plaquette,
        obs.polyakov_abs,
        obs.polyakov_re,
        obs.polyakov_im,
        obs.wilson_1_1,
        obs.wilson_2_1,
        vol_norm,
    ];

    for i in 1..=N_DFT_FEATURES {
        let val = obs.polyakov_spatial_dft.get(i).copied().unwrap_or(0.0);
        feat.push(val.sqrt());
    }

    feat
}

fn ground_truth(beta: f64, beta_c: f64) -> usize {
    if beta > beta_c {
        1
    } else {
        0
    }
}

// ── Confusion matrix & statistics ──────────────────────────────────────

#[derive(Default, Clone)]
struct ConfusionMatrix {
    tp: usize,
    tn: usize,
    fp: usize,
    r#fn: usize,
}

impl ConfusionMatrix {
    fn add(&mut self, predicted: usize, actual: usize) {
        match (predicted, actual) {
            (1, 1) => self.tp += 1,
            (0, 0) => self.tn += 1,
            (1, 0) => self.fp += 1,
            (0, 1) => self.r#fn += 1,
            _ => {}
        }
    }

    fn total(&self) -> usize {
        self.tp + self.tn + self.fp + self.r#fn
    }

    fn accuracy(&self) -> f64 {
        let t = self.total();
        if t == 0 {
            return 0.0;
        }
        (self.tp + self.tn) as f64 / t as f64
    }

    fn precision(&self) -> f64 {
        let d = self.tp + self.fp;
        if d == 0 {
            return 0.0;
        }
        self.tp as f64 / d as f64
    }

    fn recall(&self) -> f64 {
        let d = self.tp + self.r#fn;
        if d == 0 {
            return 0.0;
        }
        self.tp as f64 / d as f64
    }

    fn specificity(&self) -> f64 {
        let d = self.tn + self.fp;
        if d == 0 {
            return 0.0;
        }
        self.tn as f64 / d as f64
    }

    fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }

    fn mcc(&self) -> f64 {
        let tp = self.tp as f64;
        let tn = self.tn as f64;
        let fp = self.fp as f64;
        let fn_ = self.r#fn as f64;
        let numer = tp * tn - fp * fn_;
        let denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if denom == 0.0 {
            return 0.0;
        }
        numer / denom
    }

    fn print(&self, label: &str) {
        println!("  ┌─ {label} ─────────────────────────────────────┐");
        println!(
            "  │  Confusion Matrix:                               │"
        );
        println!(
            "  │              Predicted                            │"
        );
        println!(
            "  │              CONFND   DECONF                      │"
        );
        println!(
            "  │  Actual CONFND  {:>4}     {:>4}                       │",
            self.tn, self.fp
        );
        println!(
            "  │         DECONF  {:>4}     {:>4}                       │",
            self.r#fn, self.tp
        );
        println!(
            "  │                                                   │"
        );
        println!(
            "  │  Accuracy:    {:.1}%  ({}/{})",
            self.accuracy() * 100.0,
            self.tp + self.tn,
            self.total()
        );
        println!("  │  Precision:   {:.1}%", self.precision() * 100.0);
        println!("  │  Recall:      {:.1}%  (sensitivity)", self.recall() * 100.0);
        println!(
            "  │  Specificity: {:.1}%",
            self.specificity() * 100.0
        );
        println!("  │  F1 Score:    {:.4}", self.f1());
        println!("  │  MCC:         {:.4}", self.mcc());
        println!(
            "  └────────────────────────────────────────────────────┘"
        );
    }
}

// ── Per-sample detailed result ─────────────────────────────────────────

struct SampleResult {
    obs: RichObservable,
    truth: usize,
    cpu_raw: f64,
    cpu_class: usize,
    sim_raw: f64,
    sim_class: usize,
    #[cfg(feature = "npu-hw")]
    npu_raw: Option<f64>,
    #[cfg(feature = "npu-hw")]
    npu_class: Option<usize>,
}

fn classify(raw: f64) -> usize {
    if raw > 0.5 { 1 } else { 0 }
}

fn phase_label(class: usize) -> &'static str {
    if class == 1 { "DECONF" } else { "CONFND" }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  SU(N) NPU MetalForge — Comprehensive Phase Classification    ║");
    println!("║  Features: position-space + momentum-space (DFT) observables   ║");
    println!("║  Stats: confusion matrix, precision, recall, F1, MCC           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Phase 1: Rich observable extraction ────────────────────────────
    println!("═══ Phase 1: Rich Observable Extraction ═══");
    println!();

    let t0 = Instant::now();
    let su2_obs = scan_su2_configs();
    let su3_obs = scan_su3_configs();
    let scan_time = t0.elapsed();

    println!("  SU(2): {} configs across {} sizes", su2_obs.len(), {
        let mut sizes: Vec<usize> = su2_obs.iter().map(|o| o.lattice_l()).collect();
        sizes.dedup();
        sizes.len()
    });

    let mut size_groups: BTreeMap<usize, Vec<&RichObservable>> = BTreeMap::new();
    for obs in &su2_obs {
        size_groups.entry(obs.lattice_l()).or_default().push(obs);
    }
    for (l, group) in &size_groups {
        let betas: Vec<f64> = group.iter().map(|o| o.beta).collect();
        println!(
            "    {}⁴ ({} vol): {} configs, β ∈ [{:.2}, {:.2}]",
            l,
            l.pow(4),
            group.len(),
            betas.first().unwrap_or(&0.0),
            betas.last().unwrap_or(&0.0)
        );
    }
    println!();

    println!("  SU(3): {} configs", su3_obs.len());
    for obs in &su3_obs {
        println!(
            "    {}⁴ β={:.2}  ⟨P⟩={:.6}  |L|={:.6}  Re(L)={:+.6}  Im(L)={:+.6}",
            obs.dims[0], obs.beta, obs.plaquette, obs.polyakov_abs, obs.polyakov_re, obs.polyakov_im
        );
    }
    println!();

    println!("  Observable detail per SU(2) config:");
    for obs in &su2_obs {
        let creutz_str = obs
            .creutz_2_2
            .map(|c| format!("{c:.4}"))
            .unwrap_or_else(|| "—".to_string());
        let dft_str: Vec<String> = obs
            .polyakov_spatial_dft
            .iter()
            .skip(1)
            .take(3)
            .map(|v| format!("{:.2e}", v))
            .collect();
        println!(
            "    {}⁴ β={:.2}  ⟨P⟩={:.6}  |L|={:.6}  W(1,1)={:.6}  W(2,1)={:.6}  χ(2,2)={}  DFT[k=1..3]=[{}]",
            obs.dims[0], obs.beta, obs.plaquette, obs.polyakov_abs,
            obs.wilson_1_1, obs.wilson_2_1, creutz_str, dft_str.join(", ")
        );
    }
    println!();
    println!("  Scan time: {:.2}s", scan_time.as_secs_f64());
    println!();

    if su2_obs.len() < 4 {
        println!("  ⚠ Insufficient SU(2) configs for meaningful experiment.");
        return;
    }

    // ── Phase 2: Feature engineering ───────────────────────────────────
    println!("═══ Phase 2: Feature Engineering ({} features) ═══", N_FEATURES);
    println!();
    println!("  Base features ({}):", N_BASE_FEATURES);
    println!("    [0] β_norm = (β - β_c) / 2");
    println!("    [1] ⟨P⟩ (average plaquette)");
    println!("    [2] ⟨|L|⟩ (Polyakov loop magnitude)");
    println!("    [3] Re⟨L⟩ (Polyakov real part)");
    println!("    [4] Im⟨L⟩ (Polyakov imaginary part)");
    println!("    [5] W(1,1) (1×1 Wilson loop)");
    println!("    [6] W(2,1) (2×1 Wilson loop)");
    println!("    [7] ln(V)/10 (volume normalization)");
    println!();
    println!("  Momentum-space features ({}):", N_DFT_FEATURES);
    println!("    [8]  √|P̃(k=1)|² (first non-zero Fourier mode)");
    println!("    [9]  √|P̃(k=2)|² (second mode)");
    println!("    [10] √|P̃(k=3)|² (third mode)");
    println!();

    // ── Phase 3: Train ESN — baseline (3 features) vs expanded (11) ──
    println!("═══ Phase 3: ESN Training — Baseline vs Expanded ═══");
    println!();

    let beta_c_su2 = 2.30;

    let all_obs: Vec<&RichObservable> = su2_obs.iter().collect();
    let labels: Vec<usize> = all_obs.iter().map(|o| ground_truth(o.beta, beta_c_su2)).collect();

    // --- Baseline: 3 features (original sun_npu_monitor) ---
    let baseline_seqs: Vec<Vec<Vec<f64>>> = all_obs
        .iter()
        .map(|o| {
            let beta_norm = (o.beta - beta_c_su2) / 2.0;
            vec![vec![beta_norm, o.plaquette, o.polyakov_abs]]
        })
        .collect();
    let targets: Vec<Vec<f64>> = labels.iter().map(|&l| vec![l as f64]).collect();

    let baseline_config = EsnConfig {
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

    let t1 = Instant::now();
    let mut esn_baseline = EchoStateNetwork::new(baseline_config);
    esn_baseline.train(&baseline_seqs, &targets);
    let baseline_time = t1.elapsed();

    // --- Expanded: N_FEATURES features ---
    let expanded_seqs: Vec<Vec<Vec<f64>>> = all_obs
        .iter()
        .map(|o| vec![feature_vector(o, beta_c_su2)])
        .collect();

    let expanded_config = EsnConfig {
        input_size: N_FEATURES,
        reservoir_size: 50,
        output_size: 1,
        spectral_radius: 0.95,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-2,
        seed: 42,
        ..Default::default()
    };

    let t2 = Instant::now();
    let mut esn_expanded = EchoStateNetwork::new(expanded_config);
    esn_expanded.train(&expanded_seqs, &targets);
    let expanded_time = t2.elapsed();

    println!("  Baseline (3 features, 30 reservoir): trained in {baseline_time:.1?}");
    println!("  Expanded ({N_FEATURES} features, 50 reservoir): trained in {expanded_time:.1?}");
    println!();

    // ── Phase 4: Evaluate on all backends ──────────────────────────────
    println!("═══ Phase 4: Multi-Backend Evaluation ═══");
    println!();

    let mut cm_base_cpu = ConfusionMatrix::default();
    let mut cm_exp_cpu = ConfusionMatrix::default();
    let mut cm_base_sim = ConfusionMatrix::default();
    let mut cm_exp_sim = ConfusionMatrix::default();

    let base_weights = esn_baseline.export_weights().expect("export baseline");
    let exp_weights = esn_expanded.export_weights().expect("export expanded");
    let mut base_sim = NpuSimulator::from_exported(&base_weights);
    let mut exp_sim = NpuSimulator::from_exported(&exp_weights);

    let mut results: Vec<SampleResult> = Vec::new();

    for (i, obs) in all_obs.iter().enumerate() {
        let truth = labels[i];

        let base_cpu_raw = esn_baseline.predict(&baseline_seqs[i]).expect("predict")[0];
        let base_cpu_cls = classify(base_cpu_raw);
        cm_base_cpu.add(base_cpu_cls, truth);

        let exp_cpu_raw = esn_expanded.predict(&expanded_seqs[i]).expect("predict")[0];
        let exp_cpu_cls = classify(exp_cpu_raw);
        cm_exp_cpu.add(exp_cpu_cls, truth);

        let base_sim_raw = base_sim.predict(&baseline_seqs[i])[0];
        let base_sim_cls = classify(base_sim_raw as f64);
        cm_base_sim.add(base_sim_cls, truth);

        let exp_sim_raw = exp_sim.predict(&expanded_seqs[i])[0];
        let exp_sim_cls = classify(exp_sim_raw as f64);
        cm_exp_sim.add(exp_sim_cls, truth);

        results.push(SampleResult {
            obs: (*obs).clone(),
            truth,
            cpu_raw: exp_cpu_raw,
            cpu_class: exp_cpu_cls,
            sim_raw: exp_sim_raw as f64,
            sim_class: exp_sim_cls,
            #[cfg(feature = "npu-hw")]
            npu_raw: None,
            #[cfg(feature = "npu-hw")]
            npu_class: None,
        });
    }

    println!("  ── Baseline Model (3 features: β, ⟨P⟩, |L|) ──");
    println!();
    cm_base_cpu.print("CPU f64");
    println!();
    cm_base_sim.print("NpuSimulator f32");
    println!();

    println!("  ── Expanded Model ({N_FEATURES} features: position + momentum space) ──");
    println!();
    cm_exp_cpu.print("CPU f64");
    println!();
    cm_exp_sim.print("NpuSimulator f32");
    println!();

    // ── Phase 5: AKD1000 hardware inference ────────────────────────────
    #[cfg(feature = "npu-hw")]
    {
        println!("═══ Phase 5: AKD1000 NPU Hardware Inference ═══");
        println!();

        match NpuHardware::discover() {
            Some(hw_info) => {
                println!(
                    "  Device: {} @ {} — {} NPs, {} MB SRAM",
                    hw_info.chip_version, hw_info.pcie_address, hw_info.npu_count, hw_info.memory_mb
                );
                println!();

                let mut npu_exp = NpuHardware::from_exported(&exp_weights, hw_info);

                let hw_info2 = NpuHardware::discover().unwrap();
                let mut npu_base = NpuHardware::from_exported(&base_weights, hw_info2);

                let mut cm_base_npu = ConfusionMatrix::default();
                let mut cm_exp_npu = ConfusionMatrix::default();

                let t3 = Instant::now();
                for (i, _obs) in all_obs.iter().enumerate() {
                    let truth = labels[i];

                    let base_npu_raw = npu_base.predict(&baseline_seqs[i])[0];
                    cm_base_npu.add(classify(base_npu_raw as f64), truth);

                    let exp_npu_raw = npu_exp.predict(&expanded_seqs[i])[0];
                    let exp_npu_cls = classify(exp_npu_raw as f64);
                    cm_exp_npu.add(exp_npu_cls, truth);

                    results[i].npu_raw = Some(exp_npu_raw as f64);
                    results[i].npu_class = Some(exp_npu_cls);
                }
                let npu_time = t3.elapsed();

                println!("  ── Baseline (3 features) on AKD1000 ──");
                println!();
                cm_base_npu.print("AKD1000 NPU");
                println!();

                println!("  ── Expanded ({N_FEATURES} features) on AKD1000 ──");
                println!();
                cm_exp_npu.print("AKD1000 NPU");
                println!();

                let n_total = all_obs.len();
                let per_sample = npu_time.as_micros() as f64 / (2 * n_total) as f64;
                println!(
                    "  NPU timing: {} total inferences in {npu_time:.1?} ({per_sample:.0} µs/sample)",
                    2 * n_total
                );
                println!();
            }
            None => {
                println!("  ⚠ No AKD1000 detected — hardware results skipped.");
                println!();
            }
        }
    }

    #[cfg(not(feature = "npu-hw"))]
    {
        println!("═══ Phase 5: AKD1000 NPU (skipped — compile with --features npu-hw) ═══");
        println!();
    }

    // ── Phase 6: Size-stratified analysis ──────────────────────────────
    println!("═══ Phase 6: Size-Stratified Analysis ═══");
    println!();

    let mut size_cm: BTreeMap<usize, (ConfusionMatrix, ConfusionMatrix)> = BTreeMap::new();

    for r in &results {
        let l = r.obs.lattice_l();
        let entry = size_cm.entry(l).or_default();
        entry.0.add(r.cpu_class, r.truth);
        entry.1.add(r.sim_class, r.truth);
    }

    println!(
        "  {:>4}  {:>5}  {:>5}  {:>6}  {:>6}  {:>6}  {:>5}  {:>5}  {:>5}",
        "L", "N", "Vol", "Acc%", "Prec%", "Rec%", "F1", "MCC", "Spec%"
    );
    println!("  {}", "─".repeat(70));

    for (l, (cm, _)) in &size_cm {
        if cm.total() == 0 {
            continue;
        }
        println!(
            "  {:>4}  {:>5}  {:>5}  {:>5.1}  {:>5.1}  {:>5.1}  {:>.3}  {:>+.3}  {:>5.1}",
            format!("{}⁴", l),
            cm.total(),
            l.pow(4),
            cm.accuracy() * 100.0,
            cm.precision() * 100.0,
            cm.recall() * 100.0,
            cm.f1(),
            cm.mcc(),
            cm.specificity() * 100.0,
        );
    }
    println!();

    // ── Phase 7: Per-sample detail ─────────────────────────────────────
    println!("═══ Phase 7: Per-Sample Detail (Expanded Model) ═══");
    println!();
    println!(
        "  {:>4} {:>5}  {:>8}  {:>6}  {:>6}  {:>6}  {:>7}  {:>7}  {}",
        "L", "β", "Truth", "CPU", "SIM", "CPU_p", "SIM_p", "Δ(C-S)", "Status"
    );
    println!("  {}", "─".repeat(80));

    for r in &results {
        let truth_lbl = phase_label(r.truth);
        let cpu_lbl = phase_label(r.cpu_class);
        let sim_lbl = phase_label(r.sim_class);
        let delta = (r.cpu_raw - r.sim_raw).abs();
        let status = if r.cpu_class == r.truth && r.sim_class == r.truth {
            "  ✓"
        } else if r.cpu_class != r.truth {
            "CPU✗"
        } else {
            "SIM✗"
        };

        #[cfg(feature = "npu-hw")]
        let npu_info = r
            .npu_raw
            .map(|raw| {
                let cls = r.npu_class.unwrap_or(0);
                format!("  NPU:{} ({:.4})", phase_label(cls), raw)
            })
            .unwrap_or_default();
        #[cfg(not(feature = "npu-hw"))]
        let npu_info = String::new();

        println!(
            "  {:>4} {:>5.2}  {:>6}  {:>6}  {:>6}  {:.4}  {:.4}  {:.2e}  {}{npu_info}",
            format!("{}⁴", r.obs.lattice_l()),
            r.obs.beta,
            truth_lbl,
            cpu_lbl,
            sim_lbl,
            r.cpu_raw,
            r.sim_raw,
            delta,
            status,
        );
    }
    println!();

    // ── Phase 8: DFT power spectrum analysis ───────────────────────────
    println!("═══ Phase 8: Momentum-Space (DFT) Analysis ═══");
    println!();
    println!("  Polyakov loop spatial DFT power spectrum |P̃(k)|²:");
    println!("  (Deconfined phase: k=0 mode dominates; confined: power spreads)");
    println!();
    println!(
        "  {:>4} {:>5}  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>7}",
        "L", "β", "Phase", "|P̃(0)|²", "|P̃(1)|²", "|P̃(2)|²", "|P̃(3)|²", "k0/k1"
    );
    println!("  {}", "─".repeat(80));

    for obs in &su2_obs {
        let phase = phase_label(ground_truth(obs.beta, beta_c_su2));
        let k0 = obs.polyakov_spatial_dft.first().copied().unwrap_or(0.0);
        let k1 = obs.polyakov_spatial_dft.get(1).copied().unwrap_or(0.0);
        let k2 = obs.polyakov_spatial_dft.get(2).copied().unwrap_or(0.0);
        let k3 = obs.polyakov_spatial_dft.get(3).copied().unwrap_or(0.0);
        let ratio = if k1 > 1e-15 { k0 / k1 } else { f64::INFINITY };

        println!(
            "  {:>4} {:>5.2}  {:>6}  {:>10.4e}  {:>10.4e}  {:>10.4e}  {:>10.4e}  {:>7.1}",
            format!("{}⁴", obs.dims[0]),
            obs.beta,
            phase,
            k0,
            k1,
            k2,
            k3,
            ratio
        );
    }
    println!();

    // ── Summary ────────────────────────────────────────────────────────
    println!("═══ Summary ═══");
    println!();
    println!("  Configs:   {} SU(2) + {} SU(3) = {} total", su2_obs.len(), su3_obs.len(), su2_obs.len() + su3_obs.len());
    println!("  Features:  {} (position) + {} (momentum/DFT) = {} total", N_BASE_FEATURES, N_DFT_FEATURES, N_FEATURES);
    println!("  Scan time: {:.2}s", scan_time.as_secs_f64());
    println!();

    let conf_count = labels.iter().filter(|&&l| l == 0).count();
    let deconf_count = labels.iter().filter(|&&l| l == 1).count();
    println!("  Class balance: {} confined / {} deconfined", conf_count, deconf_count);
    println!();

    println!("  Baseline (3 feat)  → CPU accuracy: {:.1}%, F1: {:.4}, MCC: {:.4}",
        cm_base_cpu.accuracy() * 100.0, cm_base_cpu.f1(), cm_base_cpu.mcc());
    println!("  Expanded ({} feat) → CPU accuracy: {:.1}%, F1: {:.4}, MCC: {:.4}",
        N_FEATURES, cm_exp_cpu.accuracy() * 100.0, cm_exp_cpu.f1(), cm_exp_cpu.mcc());
    println!();
    println!("  Pipeline: CPU thermalizer → BLAKE3 cache → rich observables → DFT → ESN → NPU");
}
