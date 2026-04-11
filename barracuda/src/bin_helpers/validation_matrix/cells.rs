// SPDX-License-Identifier: AGPL-3.0-or-later

//! Matrix cell definitions, CLI parsing, and planning tables for the Chuna validation matrix.

use hotspring_barracuda::lattice::measurement::format_dims;

#[derive(Clone, Debug)]
pub struct MatrixCell {
    pub label: String,
    pub dims: [usize; 4],
    pub beta: f64,
    pub mass: Option<f64>,
    pub nf: usize,
    pub n_therm: usize,
    pub n_meas: usize,
    pub measure_flow: bool,
    pub measure_topo: bool,
    pub measure_wilson_loops: bool,
    pub measure_condensate: bool,
}

impl MatrixCell {
    pub fn lattice_label(&self) -> String {
        format_dims(self.dims)
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct CellResult {
    pub label: String,
    pub lattice: String,
    pub dims: [usize; 4],
    pub volume: usize,
    pub beta: f64,
    pub mass: Option<f64>,
    pub nf: usize,
    // HMC observables
    pub mean_plaquette: f64,
    pub std_plaquette: f64,
    pub acceptance: f64,
    pub mean_polyakov: f64,
    pub plaquette_susceptibility: f64,
    pub polyakov_susceptibility: f64,
    // Gradient flow scales
    #[serde(skip_serializing_if = "Option::is_none")]
    pub t0: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub w0: Option<f64>,
    // Topological charge
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topo_charge: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topo_susceptibility: Option<f64>,
    // Wilson loops
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wilson_loops: Option<Vec<WilsonLoopResult>>,
    // Chiral condensate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chiral_condensate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chiral_condensate_error: Option<f64>,
    // Timing
    pub wall_seconds: f64,
    pub hmc_seconds: f64,
    pub flow_seconds: f64,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct WilsonLoopResult {
    pub r: usize,
    pub t: usize,
    pub value: f64,
}

pub struct CliArgs {
    pub phase: String,
    /// Raw string for lattice/ns — may be comma-separated for custom sweeps.
    pub lattice_raw: Option<String>,
    /// Raw string for nt — may be comma-separated for custom sweeps.
    pub nt_raw: Option<String>,
    /// Raw string for beta — may be comma-separated for custom sweeps.
    pub beta_raw: Option<String>,
    /// Raw string for mass — may be comma-separated for custom sweeps.
    pub mass_raw: Option<String>,
    /// Raw string for nf — may be comma-separated for custom sweeps.
    pub nf_raw: Option<String>,
    pub mode: String,
    pub output: Option<String>,
    pub seed: u64,
    pub max_flow_time: f64,
    pub therm: Option<usize>,
    pub meas: Option<usize>,
    pub observables: Option<String>,
    pub telemetry: Option<String>,
}

impl CliArgs {
    pub fn first_lattice(&self) -> Option<usize> {
        self.lattice_raw
            .as_ref()
            .and_then(|s| s.split(',').next())
            .and_then(|s| s.parse().ok())
    }
    pub fn first_beta(&self) -> Option<f64> {
        self.beta_raw
            .as_ref()
            .and_then(|s| s.split(',').next())
            .and_then(|s| s.parse().ok())
    }
}

#[expect(
    clippy::expect_used,
    reason = "CLI argument parsing — invalid input should panic with diagnostic"
)]
pub fn parse_args() -> CliArgs {
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

pub fn parse_csv_f64(raw: Option<&str>, default: f64) -> Vec<f64> {
    match raw {
        Some(s) => s.split(',').filter_map(|v| v.parse().ok()).collect(),
        None => vec![default],
    }
}

pub fn parse_csv_usize(raw: Option<&str>, default: usize) -> Vec<usize> {
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
pub fn custom_cell(args: &CliArgs) -> Vec<MatrixCell> {
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
                    let masses = if nf == 0 {
                        vec![0.0]
                    } else {
                        mass_list.clone()
                    };
                    for &m in &masses {
                        let dims = [ns, ns, ns, nt];
                        let n_therm = args.therm.unwrap_or(if nf > 0 { 500 } else { 200 });
                        let n_meas = args.meas.unwrap_or(200);

                        let mass_tag = if nf > 0 {
                            format!("_m{m:.3}")
                        } else {
                            String::new()
                        };
                        let nf_tag = if nf > 0 {
                            format!("_Nf{nf}")
                        } else {
                            String::new()
                        };
                        let label =
                            format!("custom_{}_b{beta:.2}{nf_tag}{mass_tag}", format_dims(dims));

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
        println!(
            "  Parameter mixing: {} cells from Cartesian product",
            cells.len()
        );
    }

    cells
}

/// Wall-time estimate for a matrix cell (rough, for planning only).
pub fn estimate_wall_time(cell: &MatrixCell) -> String {
    let vol: usize = cell.dims.iter().product();
    let ref_vol = 8usize * 8 * 8 * 8;
    let base_traj_ms: f64 = 5.0 * (vol as f64 / ref_vol as f64);

    let fermion_factor = if cell.nf > 0 { 50.0 } else { 1.0 };
    let flow_factor = if cell.measure_flow { 1.5 } else { 1.0 };

    let total_traj = cell.n_therm + cell.n_meas;
    let total_s = base_traj_ms * fermion_factor * flow_factor * total_traj as f64 / 1000.0;

    if total_s < 60.0 {
        format!("{total_s:.0}s")
    } else if total_s < 3600.0 {
        format!("{:.0}min", total_s / 60.0)
    } else if total_s < 86400.0 {
        format!("{:.1}hr", total_s / 3600.0)
    } else {
        format!("{:.1}d", total_s / 86400.0)
    }
}

/// Print the full planning matrix as a grid for the Chuna meeting.
pub fn print_planning_matrix() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║  3-Month Validation Matrix — hotSpring × Chuna                                                            ║"
    );
    println!(
        "║  \"You bring the accuracy of where to look. We bring the precision.\"                                        ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!();
    println!(
        "  Hardware: RTX 3090 (24GB, ≤32⁴) │ RX 6950 XT (16GB, ≤24⁴) │ CPU fallback (any size, 10-100× slower)"
    );
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
            if cell.measure_wilson_loops {
                "✓"
            } else {
                "—"
            },
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
            if cell.measure_condensate {
                "✓"
            } else {
                "—"
            },
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
            if cell.measure_condensate {
                "✓"
            } else {
                "—"
            },
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
            if cell.measure_condensate {
                "✓"
            } else {
                "—"
            },
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
    println!(
        "  Quenched: {} │ Dynamical: {} │ β-scan: {} │ m-scan: {}",
        quenched.len(),
        dynamical.len(),
        beta_scan.len(),
        mass_scan.len()
    );
    println!();
    println!("  Chuna priority column: ___  (he fills in at meeting)");
    println!();
}

pub fn quenched_ladder_cells() -> Vec<MatrixCell> {
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

pub fn dynamical_scaling_cells() -> Vec<MatrixCell> {
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

pub fn beta_scan_cells(volume: usize) -> Vec<MatrixCell> {
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

pub fn mass_scan_cells(volume: usize, beta: f64) -> Vec<MatrixCell> {
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
