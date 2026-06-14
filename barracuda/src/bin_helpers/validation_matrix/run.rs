// SPDX-License-Identifier: AGPL-3.0-or-later

//! Execute matrix cells: quenched vs dynamical HMC, flow, and extended observables.

use hotspring_barracuda::lattice::correlator::{
    chiral_condensate_stochastic, plaquette_susceptibility, polyakov_susceptibility,
};
use hotspring_barracuda::lattice::gradient_flow::{
    FlowIntegrator, find_t0, find_w0, run_flow, topological_charge,
};
use hotspring_barracuda::lattice::hmc::{HmcConfig, hmc_trajectory};
use hotspring_barracuda::lattice::measurement::min_spatial_dim;
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

use super::cells::{CellResult, MatrixCell, WilsonLoopResult};

pub fn run_quenched_cell(cell: &MatrixCell, seed: u64, max_flow_time: f64) -> CellResult {
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
            t0_val.map_or_else(|| "N/A".to_string(), |v| format!("{v:.4}")),
            w0_val.map_or_else(|| "N/A".to_string(), |v| format!("{v:.4}")),
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
            topo_susc = Some(
                hotspring_barracuda::lattice::gradient_flow::topological_susceptibility(
                    &topo_charges,
                    vol,
                ),
            );
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
        eprintln!("  Wilson loops: {max_r}×{max_t} grid measured");
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

pub fn run_dynamical_cell(cell: &MatrixCell, seed: u64) -> CellResult {
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

pub fn hmc_dt_for_volume(vol: usize) -> f64 {
    let ref_vol = 4096.0_f64;
    let scale = (ref_vol / vol as f64).powf(0.25);
    (0.05 * scale).max(0.002)
}

pub fn run_cell(cell: &MatrixCell, seed: u64, max_flow_time: f64) -> CellResult {
    if cell.nf == 0 {
        run_quenched_cell(cell, seed, max_flow_time)
    } else {
        run_dynamical_cell(cell, seed)
    }
}
