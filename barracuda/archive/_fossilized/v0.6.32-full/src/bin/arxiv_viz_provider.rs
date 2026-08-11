// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
//! arXiv Visualization Provider — Grammar of Graphics figures via petalTongue
//!
//! Parses measurement battery JSONL + dual-GPU scan results, constructs
//! GrammarExpr scenes, and compiles them through petalTongue's scene engine
//! to produce publication-quality SVG figures for the arXiv preprint.
//!
//! Two modes:
//!   1. `--headless` (default): Direct SVG export to disk
//!   2. `--live`: JSON-RPC push to petalTongue UDS for interactive dashboard
//!
//! Usage:
//!   cargo run --release --features petaltongue-viz --bin arxiv_viz_provider
//!   cargo run --release --features petaltongue-viz --bin arxiv_viz_provider -- --live

#[cfg(not(feature = "petaltongue-viz"))]
fn main() {
    eprintln!("ERROR: This binary requires --features petaltongue-viz");
    std::process::exit(1);
}

#[cfg(feature = "petaltongue-viz")]
fn main() {
    viz::run();
}

#[cfg(feature = "petaltongue-viz")]
mod viz {
    use petal_tongue_scene::compiler::GrammarCompiler;
    use petal_tongue_scene::grammar::{CoordinateSystem, GeometryType, GrammarExpr, ScaleType};
    use petal_tongue_scene::modality::{ModalityCompiler, SvgCompiler};
    use petal_tongue_types::DataBinding;
    use petal_tongue_scene::data_binding::DataBindingCompiler;
    use serde::Deserialize;
    use std::fs;
    use std::io::{BufRead, BufReader};
    use std::path::{Path, PathBuf};

    const MEASURE_BATTERY_PATH: &str = "/tmp/measure_battery_output.txt";
    const DUAL_GPU_PATH: &str = "/tmp/dual_gpu_output.txt";
    const OUTPUT_DIR: &str =
        "/home/strandgate/Development/ecoPrimals/infra/whitePaper/subGen/figures";

    #[derive(Debug, Deserialize)]
    struct MeasurementRecord {
        group: String,
        nc: u32,
        dims: Vec<u32>,
        beta: f64,
        plaquette: f64,
        polyakov_abs: f64,
        #[serde(default)]
        wilson_loops: Vec<(u32, u32, f64)>,
        #[serde(default)]
        creutz_ratios: Vec<(u32, u32, f64)>,
        #[serde(default)]
        topo_q: Option<f64>,
        elapsed_s: f64,
        file: String,
    }

    #[derive(Debug)]
    struct DualGpuPoint {
        beta: f64,
        gpu_name: String,
        plaquette: f64,
        sigma: f64,
        ms_per_traj: f64,
        accept_pct: f64,
    }

    pub fn run() {
        let live_mode = std::env::args().any(|a| a == "--live");
        fs::create_dir_all(OUTPUT_DIR).expect("create output dir");

        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  arXiv Viz Provider — petalTongue Grammar of Graphics       ║");
        println!("║  Mode: {}                                          ║",
            if live_mode { "LIVE (JSON-RPC)" } else { "HEADLESS (SVG)" });
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        let records = load_measurement_battery();
        let dual_gpu = load_dual_gpu_results();

        println!("  Loaded {} measurement records", records.len());
        println!("  Loaded {} dual-GPU data points\n", dual_gpu.len());

        let compiler = GrammarCompiler::new();
        let svg_compiler = SvgCompiler::new();

        // Figure 1: Beta-scan (plaquette vs beta, colored by gauge group)
        println!("  [1/6] Compiling: β-scan (plaquette vs coupling)...");
        let (expr, data) = build_beta_scan_scene(&records);
        let scene = compiler.compile(&expr, &data);
        let output = svg_compiler.compile(&scene);
        write_svg(&output, "fig1_beta_scan.svg");

        // Figure 2: Cross-vendor comparison
        println!("  [2/6] Compiling: Cross-vendor GPU comparison...");
        let (expr, data) = build_cross_vendor_scene(&dual_gpu);
        let scene = compiler.compile(&expr, &data);
        let output = svg_compiler.compile(&scene);
        write_svg(&output, "fig2_cross_vendor.svg");

        // Figure 3: Volume scaling (plaquette convergence across volumes)
        println!("  [3/6] Compiling: Volume scaling convergence...");
        let (expr, data) = build_volume_scaling_scene(&records);
        let scene = compiler.compile(&expr, &data);
        let output = svg_compiler.compile(&scene);
        write_svg(&output, "fig3_volume_scaling.svg");

        // Figure 4: Creutz ratios heatmap
        println!("  [4/6] Compiling: Creutz ratios...");
        let (expr, data) = build_creutz_scene(&records);
        let scene = compiler.compile(&expr, &data);
        let output = svg_compiler.compile(&scene);
        write_svg(&output, "fig4_creutz_ratios.svg");

        // Figure 5: Topological charge distribution
        println!("  [5/6] Compiling: Topological charge sampling...");
        let (expr, data) = build_topology_scene(&records);
        let scene = compiler.compile(&expr, &data);
        let output = svg_compiler.compile(&scene);
        write_svg(&output, "fig5_topology.svg");

        // Figure 6: GPU performance scaling (ms/traj vs volume)
        println!("  [6/6] Compiling: GPU performance comparison...");
        let (expr, data) = build_performance_scene(&dual_gpu);
        let scene = compiler.compile(&expr, &data);
        let output = svg_compiler.compile(&scene);
        write_svg(&output, "fig6_gpu_performance.svg");

        println!("\n  ═══ All 6 figures exported to {OUTPUT_DIR}/ ═══");

        if live_mode {
            push_to_petaltongue(&records, &dual_gpu);
        }
    }

    fn load_measurement_battery() -> Vec<MeasurementRecord> {
        let path = Path::new(MEASURE_BATTERY_PATH);
        if !path.exists() {
            eprintln!("  WARNING: {MEASURE_BATTERY_PATH} not found, using empty dataset");
            return Vec::new();
        }
        let file = fs::File::open(path).expect("open battery file");
        let reader = BufReader::new(file);
        reader
            .lines()
            .filter_map(|line| {
                let line = line.ok()?;
                serde_json::from_str::<MeasurementRecord>(&line).ok()
            })
            .collect()
    }

    fn load_dual_gpu_results() -> Vec<DualGpuPoint> {
        let path = Path::new(DUAL_GPU_PATH);
        if !path.exists() {
            eprintln!("  WARNING: {DUAL_GPU_PATH} not found, using empty dataset");
            return Vec::new();
        }
        let content = fs::read_to_string(path).expect("read dual-GPU file");
        let mut points = Vec::new();

        for line in content.lines() {
            // Parse lines like: [GPU NAME] 16⁴ β=X.X seed=42: ⟨P⟩=Y.Y ± Z.Z, acc=N%, τ=T, Mms/traj
            if line.contains("⟨P⟩=") && line.contains("ms/traj") {
                let gpu_name = if line.contains("NVIDIA") {
                    "NVIDIA RTX 3090".to_string()
                } else if line.contains("AMD") {
                    "AMD RX 6950 XT".to_string()
                } else {
                    continue;
                };

                let beta = extract_after(line, "β=")
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                let plaquette = extract_after(line, "⟨P⟩=")
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                let sigma = extract_after(line, "± ")
                    .and_then(|s| s.split(',').next())
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .unwrap_or(0.0);

                let ms_per_traj = extract_after(line, "ms/traj")
                    .map(|_| {
                        // ms/traj is at the end, extract the number before it
                        line.split(',')
                            .last()
                            .and_then(|s| s.trim().strip_suffix("ms/traj"))
                            .and_then(|s| s.trim().parse::<f64>().ok())
                            .unwrap_or(0.0)
                    })
                    .unwrap_or(0.0);

                let accept_pct = extract_after(line, "acc=")
                    .and_then(|s| s.strip_suffix('%').or(s.split('%').next()))
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                if plaquette > 0.0 {
                    points.push(DualGpuPoint {
                        beta,
                        gpu_name,
                        plaquette,
                        sigma,
                        ms_per_traj,
                        accept_pct,
                    });
                }
            }
        }
        points
    }

    fn extract_after<'a>(s: &'a str, pattern: &str) -> Option<&'a str> {
        s.find(pattern).map(|i| &s[i + pattern.len()..])
    }

    // ─── Scene Builders ───────────────────────────────────────────

    fn build_beta_scan_scene(records: &[MeasurementRecord]) -> (GrammarExpr, Vec<serde_json::Value>) {
        let expr = GrammarExpr::new("beta_scan", GeometryType::Point)
            .with_x("beta")
            .with_y("plaquette")
            .with_title("SU(N) Plaquette vs Coupling Constant")
            .with_scale("x", ScaleType::Linear)
            .with_scale("y", ScaleType::Linear);

        let data: Vec<serde_json::Value> = records
            .iter()
            .filter(|r| r.dims.iter().all(|&d| d == r.dims[0]))
            .map(|r| {
                serde_json::json!({
                    "beta": r.beta,
                    "plaquette": r.plaquette,
                    "group": r.group,
                    "volume": format!("{}⁴", r.dims[0]),
                    "data_id": format!("{}_{:.2}", r.group, r.beta),
                })
            })
            .collect();

        (expr, data)
    }

    fn build_cross_vendor_scene(dual_gpu: &[DualGpuPoint]) -> (GrammarExpr, Vec<serde_json::Value>) {
        let expr = GrammarExpr::new("cross_vendor", GeometryType::Point)
            .with_x("beta")
            .with_y("plaquette")
            .with_title("Cross-Vendor Plaquette Agreement (16⁴ SU(3))")
            .with_scale("x", ScaleType::Linear)
            .with_scale("y", ScaleType::Linear);

        let data: Vec<serde_json::Value> = dual_gpu
            .iter()
            .map(|p| {
                serde_json::json!({
                    "beta": p.beta,
                    "plaquette": p.plaquette,
                    "gpu": p.gpu_name,
                    "sigma": p.sigma,
                    "ms_per_traj": p.ms_per_traj,
                    "data_id": format!("{}_{:.1}", p.gpu_name, p.beta),
                })
            })
            .collect();

        (expr, data)
    }

    fn build_volume_scaling_scene(records: &[MeasurementRecord]) -> (GrammarExpr, Vec<serde_json::Value>) {
        let expr = GrammarExpr::new("volume_scaling", GeometryType::Point)
            .with_x("volume_log")
            .with_y("plaquette")
            .with_title("Finite-Volume Convergence: ⟨P⟩ vs Lattice Size")
            .with_scale("x", ScaleType::Log)
            .with_scale("y", ScaleType::Linear);

        let data: Vec<serde_json::Value> = records
            .iter()
            .filter(|r| r.group == "su3" && r.dims.iter().all(|&d| d == r.dims[0]))
            .map(|r| {
                let vol: u64 = r.dims.iter().map(|&d| u64::from(d)).product();
                serde_json::json!({
                    "volume_log": vol as f64,
                    "plaquette": r.plaquette,
                    "beta": r.beta,
                    "label": format!("{}⁴ β={:.2}", r.dims[0], r.beta),
                    "data_id": format!("su3_{}_{:.2}", r.dims[0], r.beta),
                })
            })
            .collect();

        (expr, data)
    }

    fn build_creutz_scene(records: &[MeasurementRecord]) -> (GrammarExpr, Vec<serde_json::Value>) {
        let expr = GrammarExpr::new("creutz_ratios", GeometryType::Point)
            .with_x("r")
            .with_y("chi")
            .with_title("Creutz Ratios χ(R,R) — String Tension Approach")
            .with_scale("x", ScaleType::Linear)
            .with_scale("y", ScaleType::Linear);

        let data: Vec<serde_json::Value> = records
            .iter()
            .filter(|r| !r.creutz_ratios.is_empty())
            .flat_map(|r| {
                r.creutz_ratios
                    .iter()
                    .filter(|(ri, ti, _)| ri == ti)
                    .map(move |(ri, _, chi)| {
                        serde_json::json!({
                            "r": *ri as f64,
                            "chi": chi,
                            "beta": r.beta,
                            "group": r.group,
                            "volume": format!("{:?}", r.dims),
                            "data_id": format!("{}_b{:.2}_r{}", r.group, r.beta, ri),
                        })
                    })
            })
            .collect();

        (expr, data)
    }

    fn build_topology_scene(records: &[MeasurementRecord]) -> (GrammarExpr, Vec<serde_json::Value>) {
        let binding = DataBinding::Distribution {
            id: "topo_q".into(),
            label: "Topological Charge Q Distribution (SU(3))".into(),
            unit: "Q".into(),
            values: records
                .iter()
                .filter(|r| r.group == "su3")
                .filter_map(|r| r.topo_q)
                .collect(),
            mean: 0.0,
            std: 3.0,
            comparison_value: 0.0,
        };

        DataBindingCompiler::compile(&binding, Some("physics"))
    }

    fn build_performance_scene(dual_gpu: &[DualGpuPoint]) -> (GrammarExpr, Vec<serde_json::Value>) {
        let expr = GrammarExpr::new("gpu_perf", GeometryType::Bar)
            .with_x("label")
            .with_y("ms_per_traj")
            .with_title("DF64 HMC Performance: NVIDIA vs AMD at 16⁴")
            .with_scale("x", ScaleType::Categorical)
            .with_scale("y", ScaleType::Log);

        let data: Vec<serde_json::Value> = dual_gpu
            .iter()
            .map(|p| {
                serde_json::json!({
                    "label": format!("{} β={:.1}", p.gpu_name, p.beta),
                    "ms_per_traj": p.ms_per_traj,
                    "gpu": p.gpu_name,
                    "data_id": format!("{}_{:.1}", p.gpu_name, p.beta),
                })
            })
            .collect();

        (expr, data)
    }

    fn write_svg(output: &petal_tongue_scene::modality::ModalityOutput, filename: &str) {
        use petal_tongue_scene::modality::ModalityOutput;
        let path = PathBuf::from(OUTPUT_DIR).join(filename);
        match output {
            ModalityOutput::Svg(bytes) => {
                fs::write(&path, bytes.as_ref()).expect("write SVG");
                println!("    → {path:?} ({} bytes)", bytes.len());
            }
            _ => eprintln!("    ! Unexpected output modality for {filename}"),
        }
    }

    fn push_to_petaltongue(_records: &[MeasurementRecord], _dual_gpu: &[DualGpuPoint]) {
        println!("\n  ─── Live Mode: Connecting to petalTongue UDS ───");

        let socket_path = "/run/user/1000/petaltongue.sock";
        if !Path::new(socket_path).exists() {
            println!("    petalTongue socket not found at {socket_path}");
            println!("    Start petalTongue: `petaltongue server`");
            println!("    Figures exported as static SVG (headless mode fallback).");
            return;
        }

        println!("    TODO: JSON-RPC push to petalTongue for live dashboard");
        println!("    Socket found — wire visualization.render.grammar calls");
    }
}
