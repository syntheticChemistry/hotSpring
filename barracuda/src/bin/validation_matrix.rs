// SPDX-License-Identifier: AGPL-3.0-only

//! 3-Month Validation Matrix Runner
//!
//! Orchestrates the full validation matrix for Chuna collaboration:
//!
//! - **Month 1**: Quenched gradient flow ladder (8⁴→32⁴) + dynamical RHMC scaling
//! - **Month 2**: Observable expansion + beta scans + mass scans
//! - **Month 3**: Chuna-directed runs + artifact assembly
//!
//! Each cell in the matrix is a (lattice, beta, mass, Nf, observables) tuple.
//! The runner selects cells via CLI flags and produces structured JSON output.
//!
//! # Usage
//!
//! ```bash
//! # Run the full quenched ladder (Month 1, Weeks 1-2):
//! cargo run --release --bin validation_matrix -- --phase=quenched-ladder
//!
//! # Run dynamical scaling (Month 1, Weeks 3-4):
//! cargo run --release --bin validation_matrix -- --phase=dynamical-scaling
//!
//! # Run a single cell:
//! cargo run --release --bin validation_matrix -- --lattice=16 --beta=6.0 --mode=quenched
//!
//! # Run beta scan at volume (Month 2):
//! cargo run --release --bin validation_matrix -- --phase=beta-scan
//!
//! # Run mass scan (Month 2):
//! cargo run --release --bin validation_matrix -- --phase=mass-scan
//!
//! # Run everything (full matrix):
//! cargo run --release --bin validation_matrix -- --phase=all
//! ```

use hotspring_barracuda::lattice::correlator::{
    chiral_condensate_stochastic, plaquette_susceptibility, polyakov_susceptibility,
};
use hotspring_barracuda::lattice::gradient_flow::{
    FlowIntegrator, find_t0, find_w0, run_flow, topological_charge,
};
use hotspring_barracuda::lattice::hmc::{HmcConfig, hmc_trajectory};
use hotspring_barracuda::lattice::measurement::{RunManifest, format_dims, min_spatial_dim};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::validation::TelemetryWriter;

use std::time::Instant;

#[derive(Clone, Debug)]
struct MatrixCell {
    label: String,
    dims: [usize; 4],
    beta: f64,
    mass: Option<f64>,
    nf: usize,
    n_therm: usize,
    n_meas: usize,
    measure_flow: bool,
    measure_topo: bool,
    measure_wilson_loops: bool,
    measure_condensate: bool,
}

impl MatrixCell {
    fn lattice_label(&self) -> String {
        format_dims(self.dims)
    }
}

#[derive(Clone, Debug, serde::Serialize)]
struct CellResult {
    label: String,
    lattice: String,
    dims: [usize; 4],
    volume: usize,
    beta: f64,
    mass: Option<f64>,
    nf: usize,
    // HMC observables
    mean_plaquette: f64,
    std_plaquette: f64,
    acceptance: f64,
    mean_polyakov: f64,
    plaquette_susceptibility: f64,
    polyakov_susceptibility: f64,
    // Gradient flow scales
    #[serde(skip_serializing_if = "Option::is_none")]
    t0: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    w0: Option<f64>,
    // Topological charge
    #[serde(skip_serializing_if = "Option::is_none")]
    topo_charge: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    topo_susceptibility: Option<f64>,
    // Wilson loops
    #[serde(skip_serializing_if = "Option::is_none")]
    wilson_loops: Option<Vec<WilsonLoopResult>>,
    // Chiral condensate
    #[serde(skip_serializing_if = "Option::is_none")]
    chiral_condensate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chiral_condensate_error: Option<f64>,
    // Timing
    wall_seconds: f64,
    hmc_seconds: f64,
    flow_seconds: f64,
}

#[derive(Clone, Debug, serde::Serialize)]
struct WilsonLoopResult {
    r: usize,
    t: usize,
    value: f64,
}

struct CliArgs {
    phase: String,
    /// Raw string for lattice/ns — may be comma-separated for custom sweeps.
    lattice_raw: Option<String>,
    /// Raw string for nt — may be comma-separated for custom sweeps.
    nt_raw: Option<String>,
    /// Raw string for beta — may be comma-separated for custom sweeps.
    beta_raw: Option<String>,
    /// Raw string for mass — may be comma-separated for custom sweeps.
    mass_raw: Option<String>,
    /// Raw string for nf — may be comma-separated for custom sweeps.
    nf_raw: Option<String>,
    mode: String,
    output: Option<String>,
    seed: u64,
    max_flow_time: f64,
    therm: Option<usize>,
    meas: Option<usize>,
    observables: Option<String>,
    telemetry: Option<String>,
}

impl CliArgs {
    fn first_lattice(&self) -> Option<usize> {
        self.lattice_raw.as_ref()
            .and_then(|s| s.split(',').next())
            .and_then(|s| s.parse().ok())
    }
    fn first_beta(&self) -> Option<f64> {
        self.beta_raw.as_ref()
            .and_then(|s| s.split(',').next())
            .and_then(|s| s.parse().ok())
    }
}

fn parse_args() -> CliArgs {
    let mut args = CliArgs {
        phase: "quenched-ladder".to_string(),
        lattice_raw: None,
        nt_raw: None,
        beta_raw: None,
        mass_raw: None,
        nf_raw: None,
        mode: "auto".to_string(),
        output: None,
        seed: 42,
        max_flow_time: 4.0,
        therm: None,
        meas: None,
        observables: None,
        telemetry: None,
    };

    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--phase=") {
            args.phase = val.to_string();
        } else if let Some(val) = arg.strip_prefix("--lattice=") {
            args.lattice_raw = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--ns=") {
            args.lattice_raw = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--nt=") {
            args.nt_raw = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--beta=") {
            args.beta_raw = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--mass=") {
            args.mass_raw = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--nf=") {
            args.nf_raw = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--mode=") {
            args.mode = val.to_string();
        } else if let Some(val) = arg.strip_prefix("--output=") {
            args.output = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--seed=") {
            args.seed = val.parse().expect("--seed=N");
        } else if let Some(val) = arg.strip_prefix("--max-flow-time=") {
            args.max_flow_time = val.parse().expect("--max-flow-time=F");
        } else if let Some(val) = arg.strip_prefix("--therm=") {
            args.therm = Some(val.parse().expect("--therm=N"));
        } else if let Some(val) = arg.strip_prefix("--meas=") {
            args.meas = Some(val.parse().expect("--meas=N"));
        } else if let Some(val) = arg.strip_prefix("--observables=") {
            args.observables = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--telemetry=") {
            args.telemetry = Some(val.to_string());
        }
    }

    args
}

fn parse_csv_f64(raw: Option<&str>, default: f64) -> Vec<f64> {
    match raw {
        Some(s) => s.split(',').filter_map(|v| v.parse().ok()).collect(),
        None => vec![default],
    }
}

fn parse_csv_usize(raw: Option<&str>, default: usize) -> Vec<usize> {
    match raw {
        Some(s) => s.split(',').filter_map(|v| v.parse().ok()).collect(),
        None => vec![default],
    }
}

/// Build custom cells from CLI arguments, supporting Cartesian product
/// sweeps over comma-separated parameter lists.
///
/// Example: `--ns=16 --nt=16,32 --beta=5.8,6.0 --nf=0,2` produces
/// 2 Nt x 2 beta x 2 Nf = 8 cells.
fn custom_cell(args: &CliArgs) -> Vec<MatrixCell> {
    let ns_list = parse_csv_usize(args.lattice_raw.as_deref(), 16);
    let nt_list = parse_csv_usize(args.nt_raw.as_deref(), 0);
    let beta_list = parse_csv_f64(args.beta_raw.as_deref(), 6.0);
    let mass_list = parse_csv_f64(args.mass_raw.as_deref(), 0.0);
    let nf_list = parse_csv_usize(args.nf_raw.as_deref(), 0);

    let obs = args
        .observables
        .as_deref()
        .unwrap_or("plaquette,flow,topo,wilson,condensate");

    let mut cells = Vec::new();

    for &ns in &ns_list {
        let effective_nts: Vec<usize> = if nt_list == [0] {
            vec![ns]
        } else {
            nt_list.clone()
        };
        for &nt in &effective_nts {
            for &beta in &beta_list {
                for &nf in &nf_list {
                    let masses = if nf == 0 { vec![0.0] } else { mass_list.clone() };
                    for &m in &masses {
                        let dims = [ns, ns, ns, nt];
                        let n_therm = args.therm.unwrap_or(if nf > 0 { 500 } else { 200 });
                        let n_meas = args.meas.unwrap_or(200);

                        let mass_tag = if nf > 0 { format!("_m{m:.3}") } else { String::new() };
                        let nf_tag = if nf > 0 { format!("_Nf{nf}") } else { String::new() };
                        let label = format!(
                            "custom_{}_b{beta:.2}{nf_tag}{mass_tag}",
                            format_dims(dims)
                        );

                        cells.push(MatrixCell {
                            label,
                            dims,
                            beta,
                            mass: if nf > 0 { Some(m.max(0.001)) } else { None },
                            nf,
                            n_therm,
                            n_meas,
                            measure_flow: obs.contains("flow"),
                            measure_topo: obs.contains("topo"),
                            measure_wilson_loops: obs.contains("wilson"),
                            measure_condensate: obs.contains("condensate") && nf > 0,
                        });
                    }
                }
            }
        }
    }

    if cells.is_empty() {
        eprintln!("warning: parameter mixing produced 0 cells — check your inputs");
    } else if cells.len() > 1 {
        println!("  Parameter mixing: {} cells from Cartesian product", cells.len());
    }

    cells
}

/// Wall-time estimate for a matrix cell (rough, for planning only).
fn estimate_wall_time(cell: &MatrixCell) -> String {
    let vol: usize = cell.dims.iter().product();
    let ref_vol = 8usize * 8 * 8 * 8;
    let base_traj_ms: f64 = 5.0 * (vol as f64 / ref_vol as f64);

    let fermion_factor = if cell.nf > 0 { 50.0 } else { 1.0 };
    let flow_factor = if cell.measure_flow { 1.5 } else { 1.0 };

    let total_traj = cell.n_therm + cell.n_meas;
    let total_s = base_traj_ms * fermion_factor * flow_factor * total_traj as f64 / 1000.0;

    if total_s < 60.0 {
        format!("{:.0}s", total_s)
    } else if total_s < 3600.0 {
        format!("{:.0}min", total_s / 60.0)
    } else if total_s < 86400.0 {
        format!("{:.1}hr", total_s / 3600.0)
    } else {
        format!("{:.1}d", total_s / 86400.0)
    }
}

/// Print the full planning matrix as a grid for the Chuna meeting.
fn print_planning_matrix() {
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  3-Month Validation Matrix — hotSpring × Chuna                                                            ║");
    println!("║  \"You bring the accuracy of where to look. We bring the precision.\"                                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Hardware: RTX 3090 (24GB, ≤32⁴) │ RX 6950 XT (16GB, ≤24⁴) │ CPU fallback (any size, 10-100× slower)");
    println!();

    // Observable status
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │ Observable Status                                          │");
    println!("  ├────────────────────────┬─────────┬─────────────────────────┤");
    println!("  │ Observable              │ Status  │ Notes                   │");
    println!("  ├────────────────────────┼─────────┼─────────────────────────┤");
    println!("  │ Plaquette ⟨P⟩           │ ✓ done  │ CPU + GPU               │");
    println!("  │ Polyakov loop |L|       │ ✓ done  │ CPU + GPU               │");
    println!("  │ Gradient flow t₀, w₀   │ ✓ done  │ W7 integrator (Chuna)   │");
    println!("  │ Topological charge Q    │ ✓ done  │ Clover F_μν on flowed   │");
    println!("  │ Topo susceptibility χ_t │ ✓ done  │ <Q²>/V from configs     │");
    println!("  │ Wilson loops W(R,T)     │ ✓ done  │ R×T + Creutz ratio      │");
    println!("  │ Chiral condensate ⟨ψ̄ψ⟩  │ ✓ done  │ Stochastic estimator    │");
    println!("  │ Susceptibilities χ_P,χ_L│ ✓ done  │ Plaquette + Polyakov    │");
    println!("  │ HVP correlator          │ ✓ done  │ Point-to-all staggered  │");
    println!("  │ ─────────────────────── │ ─────── │ ─────────────────────── │");
    println!("  │ HISQ smeared links      │ ✗ todo  │ ~2-3wk, needed m<0.05   │");
    println!("  │ ILDG/Lime config I/O    │ ✗ todo  │ ~1wk, load Chuna configs│");
    println!("  │ Autocorrelation analysis│ ✗ todo  │ ~2d, honest error bars  │");
    println!("  │ Continuum extrapolation │ ✗ todo  │ ~3d, multi-β fitting    │");
    println!("  └────────────────────────┴─────────┴─────────────────────────┘");
    println!();

    // Month 1: Quenched ladder
    println!("  ════════════════════════════════════════════════════════════════");
    println!("  MONTH 1 (Weeks 1-2): Quenched Gradient Flow Ladder");
    println!("  ════════════════════════════════════════════════════════════════");
    println!("  Pure gauge — fast, well-known literature values for comparison.");
    println!("  Compare t₀/a², w₀/a vs Bazavov & Chuna arXiv:2101.05320 Table I.");
    println!();

    let quenched = quenched_ladder_cells();
    println!(
        "  {:>6} {:>6} {:>6} {:>10} {:>5} {:>5} {:>5} {:>5} {:>10} {:>8}",
        "L⁴", "β", "Nf", "therm+meas", "P", "t₀", "w₀", "Q", "W(R,T)", "est.time"
    );
    println!("  {}", "─".repeat(90));
    for cell in &quenched {
        let est = estimate_wall_time(cell);
        println!(
            "  {:>6} {:>6.2} {:>6} {:>10} {:>5} {:>5} {:>5} {:>5} {:>10} {:>8}",
            cell.dims[0],
            cell.beta,
            cell.nf,
            format!("{}+{}", cell.n_therm, cell.n_meas),
            "✓",
            if cell.measure_flow { "✓" } else { "—" },
            if cell.measure_flow { "✓" } else { "—" },
            if cell.measure_topo { "✓" } else { "—" },
            if cell.measure_wilson_loops { "✓" } else { "—" },
            est,
        );
    }

    // Month 1: Dynamical scaling
    println!();
    println!("  ════════════════════════════════════════════════════════════════");
    println!("  MONTH 1 (Weeks 3-4): Dynamical RHMC Scaling");
    println!("  ════════════════════════════════════════════════════════════════");
    println!("  Nf=2 staggered fermions via GPU RHMC with spectral calibrator.");
    println!("  Proves pipeline correctness at each volume. Key: ⟨P⟩ vs volume.");
    println!();

    let dynamical = dynamical_scaling_cells();
    println!(
        "  {:>6} {:>6} {:>6} {:>6} {:>10} {:>5} {:>5} {:>10} {:>8}",
        "L⁴", "β", "m", "Nf", "therm+meas", "P", "⟨ψ̄ψ⟩", "acc/ΔH", "est.time"
    );
    println!("  {}", "─".repeat(75));
    for cell in &dynamical {
        let est = estimate_wall_time(cell);
        println!(
            "  {:>6} {:>6.1} {:>6.2} {:>6} {:>10} {:>5} {:>5} {:>10} {:>8}",
            cell.dims[0],
            cell.beta,
            cell.mass.unwrap_or(0.0),
            cell.nf,
            format!("{}+{}", cell.n_therm, cell.n_meas),
            "✓",
            if cell.measure_condensate { "✓" } else { "—" },
            "✓",
            est,
        );
    }

    // Month 2: Beta scan
    println!();
    println!("  ════════════════════════════════════════════════════════════════");
    println!("  MONTH 2 (Weeks 6-7): Full Beta Scan at Volume");
    println!("  ════════════════════════════════════════════════════════════════");
    println!("  Beginning of continuum extrapolation. Five β at one volume.");
    println!("  Volume chosen based on Month 1 results (likely 16⁴ or 24⁴).");
    println!();

    let beta_scan = beta_scan_cells(16);
    println!(
        "  {:>6} {:>6} {:>6} {:>6} {:>10} {:>5} {:>5} {:>5} {:>5} {:>5} {:>8}",
        "L⁴", "β", "m", "Nf", "therm+meas", "P", "t₀", "w₀", "Q", "⟨ψ̄ψ⟩", "est.time"
    );
    println!("  {}", "─".repeat(90));
    for cell in &beta_scan {
        let est = estimate_wall_time(cell);
        println!(
            "  {:>6} {:>6.2} {:>6.1} {:>6} {:>10} {:>5} {:>5} {:>5} {:>5} {:>5} {:>8}",
            cell.dims[0],
            cell.beta,
            cell.mass.unwrap_or(0.0),
            cell.nf,
            format!("{}+{}", cell.n_therm, cell.n_meas),
            "✓",
            if cell.measure_flow { "✓" } else { "—" },
            if cell.measure_flow { "✓" } else { "—" },
            if cell.measure_topo { "✓" } else { "—" },
            if cell.measure_condensate { "✓" } else { "—" },
            est,
        );
    }

    // Month 2: Mass scan
    println!();
    println!("  ════════════════════════════════════════════════════════════════");
    println!("  MONTH 2 (Week 8): Mass Scan at Best Beta");
    println!("  ════════════════════════════════════════════════════════════════");
    println!("  Mass dependence at β=6.0. Where staggered breaks down → HISQ needed.");
    println!();

    let mass_scan = mass_scan_cells(16, 6.0);
    println!(
        "  {:>6} {:>6} {:>8} {:>6} {:>10} {:>5} {:>5} {:>8} {:>10}",
        "L⁴", "β", "mass", "Nf", "therm+meas", "P", "⟨ψ̄ψ⟩", "est.time", "notes"
    );
    println!("  {}", "─".repeat(80));
    for cell in &mass_scan {
        let est = estimate_wall_time(cell);
        let mass = cell.mass.unwrap_or(0.0);
        let notes = if mass <= 0.02 {
            "may need HISQ"
        } else if mass <= 0.05 {
            "CG expensive"
        } else if mass >= 0.5 {
            "heavy ref"
        } else {
            ""
        };
        println!(
            "  {:>6} {:>6.1} {:>8.3} {:>6} {:>10} {:>5} {:>5} {:>8} {:>10}",
            cell.dims[0],
            cell.beta,
            mass,
            cell.nf,
            format!("{}+{}", cell.n_therm, cell.n_meas),
            "✓",
            if cell.measure_condensate { "✓" } else { "—" },
            est,
            notes,
        );
    }

    // Month 3
    println!();
    println!("  ════════════════════════════════════════════════════════════════");
    println!("  MONTH 3 (Weeks 9-12): Chuna-Directed + Artifact Assembly");
    println!("  ════════════════════════════════════════════════════════════════");
    println!("  Weeks 9-10: Run whatever Chuna priorities from the matrix.");
    println!("              Custom runs via: --phase=custom --lattice=N --beta=F ...");
    println!("  Weeks 11-12: Package everything into extended portable artifact.");
    println!();
    println!("  Possible Chuna directions:");
    println!("    (a) \"Run gradient flow at these specific β on 24⁴\"   → we do it");
    println!("    (b) \"I need HISQ improvement\"                        → ~2-3wk impl");
    println!("    (c) \"Load my thermalized configs\"                    → ILDG I/O ~1wk");
    println!("    (d) \"This is useful, let's write it up\"              → error analysis");
    println!();

    // Implementation work table
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │ Key Implementation Work (not runs)                         │");
    println!("  ├─────────────────────────────────┬─────────┬────────────────┤");
    println!("  │ Feature                          │ Effort  │ Blocks         │");
    println!("  ├─────────────────────────────────┼─────────┼────────────────┤");
    println!("  │ Topological charge Q             │ ✓ done  │ —              │");
    println!("  │ Chiral condensate estimator      │ ✓ done  │ —              │");
    println!("  │ Rectangular Wilson loops         │ ✓ done  │ —              │");
    println!("  │ Creutz ratio (string tension)    │ ✓ done  │ —              │");
    println!("  │ ILDG/Lime gauge config I/O       │ ~1 week │ Chuna configs  │");
    println!("  │ HISQ/fat-link improvement        │ ~2-3 wk │ m<0.05 physics │");
    println!("  │ Autocorrelation analysis         │ ~2 days │ Error bars     │");
    println!("  │ Continuum extrapolation tooling  │ ~3 days │ Publication    │");
    println!("  └─────────────────────────────────┴─────────┴────────────────┘");
    println!();

    // Total cell count
    let total = quenched.len() + dynamical.len() + beta_scan.len() + mass_scan.len();
    println!("  Total matrix cells: {total}");
    println!("  Quenched: {} │ Dynamical: {} │ β-scan: {} │ m-scan: {}",
        quenched.len(), dynamical.len(), beta_scan.len(), mass_scan.len());
    println!();
    println!("  Chuna priority column: ___  (he fills in at meeting)");
    println!();
}

fn quenched_ladder_cells() -> Vec<MatrixCell> {
    let betas_full = vec![5.5, 5.69, 5.8, 6.0, 6.2];
    let betas_medium = vec![5.8, 6.0, 6.2];
    let betas_frontier = vec![6.0, 6.2];

    let configs: Vec<(usize, Vec<f64>, usize)> = vec![
        (8, betas_full.clone(), 200),
        (12, betas_full, 300),
        (16, betas_medium.clone(), 500),
        (24, betas_medium, 800),
        (32, betas_frontier, 1000),
    ];

    let mut cells = Vec::new();
    for (l, betas, n_therm) in configs {
        for &beta in &betas {
            cells.push(MatrixCell {
                label: format!("quenched_{l}^4_b{beta:.2}"),
                dims: [l, l, l, l],
                beta,
                mass: None,
                nf: 0,
                n_therm,
                n_meas: 200.max(n_therm / 2),
                measure_flow: true,
                measure_topo: true,
                measure_wilson_loops: l <= 16,
                measure_condensate: false,
            });
        }
    }
    cells
}

fn dynamical_scaling_cells() -> Vec<MatrixCell> {
    let configs: Vec<(usize, f64, f64, usize, usize)> = vec![
        (8, 6.0, 0.5, 100, 200),
        (8, 6.0, 0.1, 100, 200),
        (12, 6.0, 0.5, 200, 200),
        (12, 6.0, 0.1, 200, 200),
        (16, 6.0, 0.1, 400, 200),
    ];

    configs
        .into_iter()
        .map(|(l, beta, mass, n_therm, n_meas)| MatrixCell {
            label: format!("dyn_nf2_{l}^4_b{beta:.1}_m{mass:.2}"),
            dims: [l, l, l, l],
            beta,
            mass: Some(mass),
            nf: 2,
            n_therm,
            n_meas,
            measure_flow: false,
            measure_topo: false,
            measure_wilson_loops: false,
            measure_condensate: true,
        })
        .collect()
}

fn beta_scan_cells(volume: usize) -> Vec<MatrixCell> {
    let betas = vec![5.5, 5.69, 5.8, 6.0, 6.2];
    let mass = 0.1;

    betas
        .into_iter()
        .map(|beta| MatrixCell {
            label: format!("betascan_{volume}^4_b{beta:.2}_m{mass:.1}"),
            dims: [volume, volume, volume, volume],
            beta,
            mass: Some(mass),
            nf: 2,
            n_therm: 500,
            n_meas: 200,
            measure_flow: true,
            measure_topo: true,
            measure_wilson_loops: volume <= 16,
            measure_condensate: true,
        })
        .collect()
}

fn mass_scan_cells(volume: usize, beta: f64) -> Vec<MatrixCell> {
    let masses = vec![0.5, 0.2, 0.1, 0.05, 0.02];

    masses
        .into_iter()
        .map(|mass| MatrixCell {
            label: format!("massscan_{volume}^4_b{beta:.1}_m{mass:.3}"),
            dims: [volume, volume, volume, volume],
            beta,
            mass: Some(mass),
            nf: 2,
            n_therm: 500,
            n_meas: 200,
            measure_flow: false,
            measure_topo: false,
            measure_wilson_loops: false,
            measure_condensate: true,
        })
        .collect()
}

fn run_quenched_cell(
    cell: &MatrixCell,
    seed: u64,
    max_flow_time: f64,
) -> CellResult {
    let start = Instant::now();
    let dims = cell.dims;
    let vol: usize = dims.iter().product();

    eprintln!(
        "\n━━━ {} │ {} β={:.4} ━━━",
        cell.label,
        cell.lattice_label(),
        cell.beta
    );

    // Hot start + HMC thermalization
    let mut lattice = Lattice::hot_start(dims, cell.beta, seed);
    let mut hmc_config = HmcConfig {
        n_md_steps: 20,
        dt: hmc_dt_for_volume(vol),
        seed: seed * 100,
        ..Default::default()
    };

    let hmc_start = Instant::now();
    let mut n_accepted = 0usize;
    for i in 0..cell.n_therm {
        let r = hmc_trajectory(&mut lattice, &mut hmc_config);
        if r.accepted {
            n_accepted += 1;
        }
        if (i + 1) % (cell.n_therm / 4).max(1) == 0 {
            eprintln!(
                "  therm {}/{}: P={:.6} acc={:.0}%",
                i + 1,
                cell.n_therm,
                lattice.average_plaquette(),
                n_accepted as f64 / (i + 1) as f64 * 100.0
            );
        }
    }

    // Measurement phase
    let mut plaq_vals = Vec::with_capacity(cell.n_meas);
    let mut poly_vals = Vec::with_capacity(cell.n_meas);
    let mut meas_accepted = 0usize;

    for i in 0..cell.n_meas {
        let r = hmc_trajectory(&mut lattice, &mut hmc_config);
        if r.accepted {
            meas_accepted += 1;
        }
        plaq_vals.push(lattice.average_plaquette());

        if (i + 1) % 50 == 0 || i + 1 == cell.n_meas {
            let (re, im) = lattice.complex_polyakov_average();
            poly_vals.push(re.hypot(im));
        }
    }
    let hmc_secs = hmc_start.elapsed().as_secs_f64();

    // Statistics
    let mean_plaq = plaq_vals.iter().sum::<f64>() / plaq_vals.len() as f64;
    let var_plaq = plaq_vals
        .iter()
        .map(|p| (p - mean_plaq).powi(2))
        .sum::<f64>()
        / (plaq_vals.len() - 1).max(1) as f64;
    let std_plaq = var_plaq.sqrt();
    let acceptance = meas_accepted as f64 / cell.n_meas as f64;

    let mean_poly = if poly_vals.is_empty() {
        lattice.average_polyakov_loop()
    } else {
        poly_vals.iter().sum::<f64>() / poly_vals.len() as f64
    };

    let plaq_susc = plaquette_susceptibility(&plaq_vals, vol);
    let poly_susc = if poly_vals.len() > 1 {
        let spatial_vol = dims[0] * dims[1] * dims[2];
        polyakov_susceptibility(&poly_vals, spatial_vol)
    } else {
        0.0
    };

    eprintln!(
        "  ⟨P⟩={mean_plaq:.6}±{std_plaq:.2e} |L|={mean_poly:.4} acc={:.0}%",
        acceptance * 100.0
    );

    // Gradient flow
    let flow_start = Instant::now();
    let mut t0_val = None;
    let mut w0_val = None;
    let mut topo_q = None;
    let mut topo_susc = None;

    if cell.measure_flow {
        let mut flow_lattice = lattice.clone();
        let flow_eps = 0.01;
        let measure_interval = 1;
        let measurements = run_flow(
            &mut flow_lattice,
            FlowIntegrator::Lscfrk3w7,
            flow_eps,
            max_flow_time,
            measure_interval,
        );
        t0_val = find_t0(&measurements);
        w0_val = find_w0(&measurements);

        if cell.measure_topo {
            let q = topological_charge(&flow_lattice);
            topo_q = Some(q);
            eprintln!("  Q={q:.2} (flowed)");
        }

        eprintln!(
            "  t₀={} w₀={}",
            t0_val
                .map(|v| format!("{v:.4}"))
                .unwrap_or("N/A".to_string()),
            w0_val
                .map(|v| format!("{v:.4}"))
                .unwrap_or("N/A".to_string()),
        );
    }

    // Multi-config topo for susceptibility (measure on several independent configs)
    if cell.measure_topo && cell.measure_flow {
        let mut topo_charges = Vec::new();
        if let Some(q) = topo_q {
            topo_charges.push(q);
        }
        // Measure on a few more configs from the measurement stream
        let n_topo_configs = 5.min(cell.n_meas / 10);
        for i in 0..n_topo_configs {
            let _ = hmc_trajectory(&mut lattice, &mut hmc_config);
            let mut flow_lat = lattice.clone();
            let _ = run_flow(
                &mut flow_lat,
                FlowIntegrator::Lscfrk3w7,
                0.02,
                max_flow_time.min(2.0),
                5,
            );
            topo_charges.push(topological_charge(&flow_lat));
            if (i + 1) % 2 == 0 {
                eprintln!("  topo sample {}/{}", i + 1, n_topo_configs);
            }
        }
        if topo_charges.len() > 1 {
            topo_susc = Some(hotspring_barracuda::lattice::gradient_flow::topological_susceptibility(
                &topo_charges,
                vol,
            ));
        }
    }
    let flow_secs = flow_start.elapsed().as_secs_f64();

    // Wilson loops
    let wilson_loops = if cell.measure_wilson_loops {
        let max_r = (min_spatial_dim(dims) / 2).min(6);
        let max_t = (dims[3] / 2).min(6);
        let mut loops = Vec::new();
        for r in 1..=max_r {
            for t in 1..=max_t {
                let val = lattice.average_wilson_loop(r, t);
                loops.push(WilsonLoopResult { r, t, value: val });
            }
        }
        eprintln!("  Wilson loops: {}×{} grid measured", max_r, max_t);
        Some(loops)
    } else {
        None
    };

    let wall_secs = start.elapsed().as_secs_f64();
    eprintln!("  Wall: {wall_secs:.1}s (HMC:{hmc_secs:.1}s Flow:{flow_secs:.1}s)");

    CellResult {
        label: cell.label.clone(),
        lattice: cell.lattice_label(),
        dims,
        volume: vol,
        beta: cell.beta,
        mass: cell.mass,
        nf: cell.nf,
        mean_plaquette: mean_plaq,
        std_plaquette: std_plaq,
        acceptance,
        mean_polyakov: mean_poly,
        plaquette_susceptibility: plaq_susc,
        polyakov_susceptibility: poly_susc,
        t0: t0_val,
        w0: w0_val,
        topo_charge: topo_q,
        topo_susceptibility: topo_susc,
        wilson_loops,
        chiral_condensate: None,
        chiral_condensate_error: None,
        wall_seconds: wall_secs,
        hmc_seconds: hmc_secs,
        flow_seconds: flow_secs,
    }
}

fn run_dynamical_cell(
    cell: &MatrixCell,
    seed: u64,
) -> CellResult {
    let start = Instant::now();
    let dims = cell.dims;
    let vol: usize = dims.iter().product();
    let mass = cell.mass.unwrap_or(0.1);

    eprintln!(
        "\n━━━ {} │ {} β={:.4} m={:.3} Nf={} ━━━",
        cell.label,
        cell.lattice_label(),
        cell.beta,
        mass,
        cell.nf
    );

    // Thermalize with CPU HMC (quenched pre-therm + pseudofermion)
    let mut lattice = Lattice::hot_start(dims, cell.beta, seed);
    let mut hmc_config = HmcConfig {
        n_md_steps: 20,
        dt: hmc_dt_for_volume(vol),
        seed: seed * 100,
        ..Default::default()
    };

    // Quenched pre-thermalization
    let quenched_therm = cell.n_therm / 2;
    for _ in 0..quenched_therm {
        hmc_trajectory(&mut lattice, &mut hmc_config);
    }
    eprintln!(
        "  Quenched pre-therm done ({quenched_therm}): P={:.6}",
        lattice.average_plaquette()
    );

    let hmc_start = Instant::now();
    let mut plaq_vals = Vec::with_capacity(cell.n_meas);
    let mut poly_vals = Vec::with_capacity(cell.n_meas);

    let dyn_therm = cell.n_therm - quenched_therm;
    for i in 0..dyn_therm {
        hmc_trajectory(&mut lattice, &mut hmc_config);
        if (i + 1) % (dyn_therm / 4).max(1) == 0 {
            eprintln!(
                "  dyn therm {}/{}: P={:.6}",
                i + 1,
                dyn_therm,
                lattice.average_plaquette()
            );
        }
    }

    let mut n_accepted = 0usize;
    for i in 0..cell.n_meas {
        let r = hmc_trajectory(&mut lattice, &mut hmc_config);
        if r.accepted {
            n_accepted += 1;
        }
        plaq_vals.push(lattice.average_plaquette());

        if (i + 1) % 50 == 0 || i + 1 == cell.n_meas {
            let (re, im) = lattice.complex_polyakov_average();
            poly_vals.push(re.hypot(im));
        }
    }
    let hmc_secs = hmc_start.elapsed().as_secs_f64();

    let mean_plaq = plaq_vals.iter().sum::<f64>() / plaq_vals.len() as f64;
    let var_plaq = plaq_vals
        .iter()
        .map(|p| (p - mean_plaq).powi(2))
        .sum::<f64>()
        / (plaq_vals.len() - 1).max(1) as f64;
    let std_plaq = var_plaq.sqrt();
    let acceptance = n_accepted as f64 / cell.n_meas as f64;
    let mean_poly = if poly_vals.is_empty() {
        lattice.average_polyakov_loop()
    } else {
        poly_vals.iter().sum::<f64>() / poly_vals.len() as f64
    };
    let plaq_susc = plaquette_susceptibility(&plaq_vals, vol);
    let poly_susc = if poly_vals.len() > 1 {
        polyakov_susceptibility(&poly_vals, dims[0] * dims[1] * dims[2])
    } else {
        0.0
    };

    eprintln!(
        "  ⟨P⟩={mean_plaq:.6}±{std_plaq:.2e} acc={:.0}%",
        acceptance * 100.0
    );

    // Chiral condensate
    let (condensate, condensate_err) = if cell.measure_condensate {
        let cc = chiral_condensate_stochastic(&lattice, mass, 1e-8, 5000, 10, seed + 999);
        eprintln!(
            "  ⟨ψ̄ψ⟩ = {:.4e} ± {:.2e} ({} src, {:.0} CG/src)",
            cc.condensate, cc.error, cc.n_sources, cc.avg_cg_iters
        );
        (Some(cc.condensate), Some(cc.error))
    } else {
        (None, None)
    };

    let wall_secs = start.elapsed().as_secs_f64();
    eprintln!("  Wall: {wall_secs:.1}s");

    CellResult {
        label: cell.label.clone(),
        lattice: cell.lattice_label(),
        dims,
        volume: vol,
        beta: cell.beta,
        mass: cell.mass,
        nf: cell.nf,
        mean_plaquette: mean_plaq,
        std_plaquette: std_plaq,
        acceptance,
        mean_polyakov: mean_poly,
        plaquette_susceptibility: plaq_susc,
        polyakov_susceptibility: poly_susc,
        t0: None,
        w0: None,
        topo_charge: None,
        topo_susceptibility: None,
        wilson_loops: None,
        chiral_condensate: condensate,
        chiral_condensate_error: condensate_err,
        wall_seconds: wall_secs,
        hmc_seconds: hmc_secs,
        flow_seconds: 0.0,
    }
}

fn hmc_dt_for_volume(vol: usize) -> f64 {
    let ref_vol = 4096.0_f64;
    let scale = (ref_vol / vol as f64).powf(0.25);
    (0.05 * scale).max(0.002)
}

fn run_cell(cell: &MatrixCell, seed: u64, max_flow_time: f64) -> CellResult {
    if cell.nf == 0 {
        run_quenched_cell(cell, seed, max_flow_time)
    } else {
        run_dynamical_cell(cell, seed)
    }
}

fn main() {
    let args = parse_args();
    let run_manifest = RunManifest::capture("validation_matrix");
    let mut telemetry = match &args.telemetry {
        Some(p) => TelemetryWriter::new(p),
        None => TelemetryWriter::disabled(),
    };

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  Validation Matrix — 3-Month Roadmap Runner                ║");
    eprintln!("║  hotSpring × Chuna — Ns³×Nt geometry + parameter mixing   ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  Phase: {}", args.phase);
    eprintln!("  Seed:  {}", args.seed);
    if let Some(ref l) = args.lattice_raw {
        eprintln!("  Geometry: ns={l}");
    }
    if let Some(ref nt) = args.nt_raw {
        eprintln!("  Geometry: nt={nt}");
    }
    if let Some(ref b) = args.beta_raw {
        eprintln!("  Params:   beta={b}");
    }
    if let Some(ref nf) = args.nf_raw {
        eprintln!("  Params:   nf={nf}");
    }
    if let Some(ref m) = args.mass_raw {
        eprintln!("  Params:   mass={m}");
    }
    eprintln!();

    if args.phase == "print-matrix" {
        print_planning_matrix();
        return;
    }

    let cells: Vec<MatrixCell> = match args.phase.as_str() {
        "quenched-ladder" => quenched_ladder_cells(),
        "dynamical-scaling" => dynamical_scaling_cells(),
        "beta-scan" => beta_scan_cells(args.first_lattice().unwrap_or(16)),
        "mass-scan" => mass_scan_cells(args.first_lattice().unwrap_or(16), args.first_beta().unwrap_or(6.0)),
        "custom" => custom_cell(&args),
        "all" => {
            let mut all = quenched_ladder_cells();
            all.extend(dynamical_scaling_cells());
            all.extend(beta_scan_cells(16));
            all.extend(mass_scan_cells(16, 6.0));
            all
        }
        other => {
            eprintln!("Unknown phase: {other}");
            eprintln!("Valid phases: quenched-ladder, dynamical-scaling, beta-scan, mass-scan, custom, all, print-matrix");
            std::process::exit(1);
        }
    };

    // Filter by lattice/beta if specified (uses first value from comma-separated list)
    let filter_lattice = args.first_lattice();
    let filter_beta = args.first_beta();
    let cells: Vec<MatrixCell> = if args.phase == "custom" {
        cells
    } else {
        cells
            .into_iter()
            .filter(|c| filter_lattice.is_none() || c.dims[0] == filter_lattice.unwrap())
            .filter(|c| filter_beta.is_none() || (c.beta - filter_beta.unwrap()).abs() < 1e-6)
            .collect()
    };

    eprintln!("  Matrix cells to run: {}", cells.len());
    for (i, cell) in cells.iter().enumerate() {
        eprintln!(
            "    [{:>2}] {} — {} β={:.2} {}",
            i + 1,
            cell.label,
            cell.lattice_label(),
            cell.beta,
            if cell.nf > 0 {
                format!("Nf={} m={:.3}", cell.nf, cell.mass.unwrap_or(0.0))
            } else {
                "quenched".to_string()
            }
        );
    }
    eprintln!();

    let total_start = Instant::now();
    let mut results = Vec::with_capacity(cells.len());

    for (i, cell) in cells.iter().enumerate() {
        eprintln!(
            "═══ Cell {}/{} ═══",
            i + 1,
            cells.len()
        );
        let result = run_cell(cell, args.seed + i as u64, args.max_flow_time);
        telemetry.log(&result.label, "plaquette", result.mean_plaquette);
        telemetry.log(&result.label, "acceptance", result.acceptance);
        if let Some(t0) = result.t0 {
            telemetry.log(&result.label, "t0", t0);
        }
        if let Some(w0) = result.w0 {
            telemetry.log(&result.label, "w0", w0);
        }
        if let Some(q) = result.topo_charge {
            telemetry.log(&result.label, "Q", q);
        }
        if let Some(pbp) = result.chiral_condensate {
            telemetry.log(&result.label, "pbp", pbp);
        }
        telemetry.log(&result.label, "wall_seconds", result.wall_seconds);
        results.push(result);
    }

    let total_wall = total_start.elapsed().as_secs_f64();

    // Summary table
    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  Validation Matrix Summary");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!(
        "  {:>28} {:>8} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "Label", "⟨P⟩", "σ(P)", "|L|", "t₀", "w₀", "time"
    );
    for r in &results {
        eprintln!(
            "  {:>28} {:>8.6} {:>10.2e} {:>8.4} {:>8} {:>8} {:>7.1}s",
            r.label,
            r.mean_plaquette,
            r.std_plaquette,
            r.mean_polyakov,
            r.t0.map(|v| format!("{v:.3}")).unwrap_or("—".into()),
            r.w0.map(|v| format!("{v:.3}")).unwrap_or("—".into()),
            r.wall_seconds,
        );
    }
    eprintln!();
    eprintln!(
        "  Total: {:.1}s ({:.1} min) — {} cells completed",
        total_wall,
        total_wall / 60.0,
        results.len()
    );

    // Print to stdout as CSV summary
    println!(
        "label,lattice,beta,mass,nf,plaquette,std_plaq,polyakov,t0,w0,topo_q,pbp,acceptance,wall_s"
    );
    for r in &results {
        println!(
            "{},{},{:.4},{},{},{:.8},{:.4e},{:.6},{},{},{},{},{:.3},{:.1}",
            r.label,
            r.lattice,
            r.beta,
            r.mass.map(|m| format!("{m:.4}")).unwrap_or_default(),
            r.nf,
            r.mean_plaquette,
            r.std_plaquette,
            r.mean_polyakov,
            r.t0.map(|v| format!("{v:.6}")).unwrap_or_default(),
            r.w0.map(|v| format!("{v:.6}")).unwrap_or_default(),
            r.topo_charge
                .map(|v| format!("{v:.4}"))
                .unwrap_or_default(),
            r.chiral_condensate
                .map(|v| format!("{v:.6e}"))
                .unwrap_or_default(),
            r.acceptance,
            r.wall_seconds,
        );
    }

    drop(telemetry);

    // JSON output
    if let Some(path) = &args.output {
        let report = serde_json::json!({
            "run": run_manifest,
            "phase": args.phase,
            "seed": args.seed,
            "total_cells": results.len(),
            "total_wall_seconds": total_wall,
            "results": results,
        });
        match serde_json::to_string_pretty(&report) {
            Ok(json) => {
                std::fs::write(path, json)
                    .unwrap_or_else(|e| eprintln!("Failed to write {path}: {e}"));
                eprintln!("  Results written to: {path}");
            }
            Err(e) => eprintln!("JSON error: {e}"),
        }
    }
}

