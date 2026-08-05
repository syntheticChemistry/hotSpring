// SPDX-License-Identifier: AGPL-3.0-or-later

//! Observable measurement battery on cached SU(N) configurations.
//!
//! Scans the config cache for all gauge groups and runs the full measurement
//! pass on each: plaquette, Polyakov loop, Wilson loops, Creutz ratios.
//! For SU(3) configs, additionally runs gradient flow (t₀, w₀) and
//! topological charge via the validated production code.
//!
//! Results are output as JSON lines to stdout for downstream analysis.
//!
//! Usage:
//!   cargo run --release --features barracuda-local --bin arxiv_measure_battery
//!
//! Environment:
//!   MEASURE_GROUP=2   — only measure SU(2) configs
//!   MAX_WILSON_R=6    — max Wilson loop extent (default: 4)

use hotspring_barracuda::lattice::generic_lattice::GenericLattice;
use hotspring_barracuda::lattice::su2::Su2Matrix;
use std::path::PathBuf;
use std::time::Instant;

fn config_base_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hotspring")
        .join("configs")
}

struct ConfigEntry {
    path: PathBuf,
    gauge_group: String,
    nc: usize,
}

fn discover_configs(filter_nc: Option<usize>) -> Vec<ConfigEntry> {
    let base = config_base_dir();
    let mut entries = Vec::new();

    let groups: Vec<(&str, usize)> = vec![
        ("su2", 2), ("su3", 3), ("su4", 4), ("su5", 5), ("su6", 6), ("su8", 8),
    ];

    for &(group_name, nc) in &groups {
        if let Some(filter) = filter_nc {
            if nc != filter { continue; }
        }
        let dir = base.join(group_name);
        if !dir.exists() { continue; }

        if let Ok(rd) = std::fs::read_dir(&dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.extension().map_or(false, |e| e == "lat") {
                    entries.push(ConfigEntry {
                        path: p,
                        gauge_group: group_name.to_string(),
                        nc,
                    });
                }
            }
        }
    }

    entries.sort_by(|a, b| a.gauge_group.cmp(&b.gauge_group).then(a.path.cmp(&b.path)));
    entries
}

fn parse_header(path: &std::path::Path) -> Option<([usize; 4], f64, usize)> {
    let buf = std::fs::read(path).ok()?;
    if buf.len() < 48 { return None; }

    let dims = [
        u64::from_le_bytes(buf[0..8].try_into().ok()?) as usize,
        u64::from_le_bytes(buf[8..16].try_into().ok()?) as usize,
        u64::from_le_bytes(buf[16..24].try_into().ok()?) as usize,
        u64::from_le_bytes(buf[24..32].try_into().ok()?) as usize,
    ];
    let beta = f64::from_le_bytes(buf[32..40].try_into().ok()?);
    let nc = u64::from_le_bytes(buf[40..48].try_into().ok()?) as usize;

    Some((dims, beta, nc))
}

fn dims_label(dims: &[usize; 4]) -> String {
    if dims[0] == dims[3] {
        format!("{}⁴", dims[0])
    } else {
        format!("{}³×{}", dims[0], dims[3])
    }
}

fn measure_su2(path: &std::path::Path, max_r: usize) {
    let lat = match GenericLattice::<Su2Matrix>::load(path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("  ERROR loading SU(2) config: {e}");
            return;
        }
    };

    let dims = lat.dims;
    let beta = lat.beta;
    let t0 = Instant::now();

    let plaq = lat.average_plaquette();
    let poly_abs = lat.average_polyakov_loop();
    let (poly_re, poly_im) = lat.complex_polyakov_average();

    let mut wilson_data = Vec::new();
    let w_max = max_r.min(dims[0] / 2);
    for r in 1..=w_max {
        for t in 1..=w_max {
            let w = lat.spatial_temporal_wilson_loop(r, t);
            wilson_data.push(format!("[{r},{t},{w:.8e}]"));
        }
    }

    let creutz = lat.creutz_ratio_scan(w_max);
    let mut creutz_data = Vec::new();
    for &(r, t, chi) in &creutz {
        creutz_data.push(format!("[{r},{t},{chi:.8e}]"));
    }

    let elapsed = t0.elapsed().as_secs_f64();

    println!(
        r#"{{"group":"su2","nc":2,"dims":[{},{},{},{}],"beta":{beta:.6},"plaquette":{plaq:.10},"polyakov_abs":{poly_abs:.10},"polyakov_re":{poly_re:.10},"polyakov_im":{poly_im:.10},"wilson_loops":[{}],"creutz_ratios":[{}],"elapsed_s":{elapsed:.2},"file":"{}"}}"#,
        dims[0], dims[1], dims[2], dims[3],
        wilson_data.join(","),
        creutz_data.join(","),
        path.file_name().unwrap_or_default().to_string_lossy()
    );
}

fn measure_su3(path: &std::path::Path, max_r: usize) {
    use hotspring_barracuda::lattice::wilson::Lattice;

    // SU(3) configs from the thermalizer use Lattice::save (40-byte header).
    // Try GenericLattice first (48-byte), fall back to Lattice (40-byte).
    let (dims, beta, lat_su3, lat_generic);

    if let Ok(l) = Lattice::load(path) {
        dims = l.dims;
        beta = l.beta;
        lat_su3 = Some(l);
        lat_generic = None;
    } else {
        use hotspring_barracuda::lattice::su3::Su3Matrix;
        match GenericLattice::<Su3Matrix>::load(path) {
            Ok(l) => {
                dims = l.dims;
                beta = l.beta;
                lat_su3 = None;
                lat_generic = Some(l);
            }
            Err(e) => {
                eprintln!("  ERROR loading SU(3) config: {e}");
                return;
            }
        }
    }

    let t0 = Instant::now();

    let plaq;
    let poly_abs;
    let poly_re;
    let poly_im;
    let mut wilson_data = Vec::new();
    let mut creutz_data = Vec::new();
    let w_max = max_r.min(dims[0] / 2);

    if let Some(ref lat) = lat_su3 {
        plaq = lat.average_plaquette();
        poly_abs = lat.average_polyakov_loop();
        let pc = lat.complex_polyakov_average();
        poly_re = pc.0;
        poly_im = pc.1;

        for r in 1..=w_max {
            for t in 1..=w_max {
                let w = lat.spatial_temporal_wilson_loop(r, t);
                wilson_data.push(format!("[{r},{t},{w:.8e}]"));
            }
        }

        if let Some(cr) = lat.creutz_ratio(2, 2) {
            creutz_data.push(format!("[2,2,{cr:.8e}]"));
        }
        for r in 2..=w_max {
            for t in 2..=w_max {
                if let Some(cr) = lat.creutz_ratio(r, t) {
                    if r != 2 || t != 2 {
                        creutz_data.push(format!("[{r},{t},{cr:.8e}]"));
                    }
                }
            }
        }
    } else if let Some(ref lat) = lat_generic {
        plaq = lat.average_plaquette();
        poly_abs = lat.average_polyakov_loop();
        let pc = lat.complex_polyakov_average();
        poly_re = pc.0;
        poly_im = pc.1;

        for r in 1..=w_max {
            for t in 1..=w_max {
                let w = lat.spatial_temporal_wilson_loop(r, t);
                wilson_data.push(format!("[{r},{t},{w:.8e}]"));
            }
        }

        let creutz_scan = lat.creutz_ratio_scan(w_max);
        for &(r, t, chi) in &creutz_scan {
            creutz_data.push(format!("[{r},{t},{chi:.8e}]"));
        }
    } else {
        return;
    }

    // Gradient flow + topological charge (SU(3)-specific, via validated Lattice code)
    let mut flow_t0_val = None;
    let mut flow_w0_val = None;
    let mut topo_q = None;

    #[cfg(feature = "barracuda-local")]
    if let Some(ref lat) = lat_su3 {
        use hotspring_barracuda::lattice::gradient_flow::{
            FlowIntegrator, find_t0, find_w0, run_flow, topological_charge,
        };

        topo_q = Some(topological_charge(lat));

        let mut flow_lat = lat.clone();
        let measurements = run_flow(
            &mut flow_lat, FlowIntegrator::Rk3Luscher,
            0.01, 1.0, 5,
        );
        flow_t0_val = find_t0(&measurements);
        flow_w0_val = find_w0(&measurements);
    }

    let elapsed = t0.elapsed().as_secs_f64();

    let flow_str = match (flow_t0_val, flow_w0_val) {
        (Some(t0v), Some(w0v)) => format!(r#","flow_t0":{t0v:.8},"flow_w0":{w0v:.8}"#),
        (Some(t0v), None) => format!(r#","flow_t0":{t0v:.8}"#),
        _ => String::new(),
    };
    let topo_str = match topo_q {
        Some(q) => format!(r#","topo_q":{q:.8}"#),
        None => String::new(),
    };

    println!(
        r#"{{"group":"su3","nc":3,"dims":[{},{},{},{}],"beta":{beta:.6},"plaquette":{plaq:.10},"polyakov_abs":{poly_abs:.10},"polyakov_re":{poly_re:.10},"polyakov_im":{poly_im:.10},"wilson_loops":[{}],"creutz_ratios":[{}]{flow_str}{topo_str},"elapsed_s":{elapsed:.2},"file":"{}"}}"#,
        dims[0], dims[1], dims[2], dims[3],
        wilson_data.join(","),
        creutz_data.join(","),
        path.file_name().unwrap_or_default().to_string_lossy()
    );
}

fn measure_sun(path: &std::path::Path, nc: usize, group_name: &str, max_r: usize) {
    let lat = match GenericLattice::load_sun(path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("  ERROR loading SU({nc}) config: {e}");
            return;
        }
    };

    let dims = lat.dims;
    let beta = lat.beta;
    let t0 = Instant::now();

    let plaq = lat.average_plaquette();
    let poly_abs = lat.average_polyakov_loop();
    let (poly_re, poly_im) = lat.complex_polyakov_average();

    let mut wilson_data = Vec::new();
    let w_max = max_r.min(dims[0] / 2);
    for r in 1..=w_max {
        for t in 1..=w_max {
            let w = lat.spatial_temporal_wilson_loop(r, t);
            wilson_data.push(format!("[{r},{t},{w:.8e}]"));
        }
    }

    let creutz = lat.creutz_ratio_scan(w_max);
    let mut creutz_data = Vec::new();
    for &(r, t, chi) in &creutz {
        creutz_data.push(format!("[{r},{t},{chi:.8e}]"));
    }

    let elapsed = t0.elapsed().as_secs_f64();

    println!(
        r#"{{"group":"{group_name}","nc":{nc},"dims":[{},{},{},{}],"beta":{beta:.6},"plaquette":{plaq:.10},"polyakov_abs":{poly_abs:.10},"polyakov_re":{poly_re:.10},"polyakov_im":{poly_im:.10},"wilson_loops":[{}],"creutz_ratios":[{}],"elapsed_s":{elapsed:.2},"file":"{}"}}"#,
        dims[0], dims[1], dims[2], dims[3],
        wilson_data.join(","),
        creutz_data.join(","),
        path.file_name().unwrap_or_default().to_string_lossy()
    );
}

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  SU(N) Observable Battery — Measurement Pass on Cached Configs ║");
    eprintln!("║  plaquette · Polyakov · Wilson loops · Creutz · flow · Q_top    ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();

    let filter_nc: Option<usize> = std::env::var("MEASURE_GROUP")
        .ok()
        .and_then(|s| s.parse().ok());

    let max_r: usize = std::env::var("MAX_WILSON_R")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    if let Some(nc) = filter_nc {
        eprintln!("  Filter: SU({nc}) only");
    }
    eprintln!("  Max Wilson loop R: {max_r}");
    eprintln!("  Config base: {}", config_base_dir().display());
    eprintln!();

    let configs = discover_configs(filter_nc);

    if configs.is_empty() {
        eprintln!("  No cached configs found. Run arxiv_thermalize_sun first.");
        return;
    }

    eprintln!("  Found {} configs:", configs.len());
    for entry in &configs {
        if let Some((dims, beta, nc)) = parse_header(&entry.path) {
            eprintln!(
                "    SU({nc}) {} β={beta:.2} — {}",
                dims_label(&dims),
                entry.path.file_name().unwrap_or_default().to_string_lossy()
            );
        }
    }
    eprintln!();

    let total_start = Instant::now();

    for (i, entry) in configs.iter().enumerate() {
        if let Some((dims, beta, _)) = parse_header(&entry.path) {
            eprintln!(
                "  [{}/{}] Measuring SU({}) {} β={:.2}...",
                i + 1, configs.len(), entry.nc, dims_label(&dims), beta
            );
        }

        match entry.nc {
            2 => measure_su2(&entry.path, max_r),
            3 => measure_su3(&entry.path, max_r),
            _ => measure_sun(&entry.path, entry.nc, &entry.gauge_group, max_r),
        }
    }

    let total = total_start.elapsed().as_secs_f64();
    eprintln!();
    eprintln!("═══ Measurement Battery Complete ═══");
    eprintln!("  {} configs measured in {total:.1}s ({:.1} min)", configs.len(), total / 60.0);
}
