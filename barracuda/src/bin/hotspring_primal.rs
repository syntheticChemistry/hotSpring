// SPDX-License-Identifier: AGPL-3.0-or-later

//! hotspring_primal — primalSpring-compatible JSON-RPC server for hotSpring.
//!
//! Satisfies the `hotspring_validate.toml` graph contract:
//! - `health.check` / `health.liveness` → liveness probes
//! - `capabilities.list` → advertises physics + compute capabilities
//!
//! Capabilities are discovered at startup from metalForge substrate detection:
//! GPU model, FP64 rate, df64 support, precision strategy, available VRAM.
//!
//! Socket: resolved via `niche::resolve_server_socket()` — `hotspring-physics-{family_id}.sock`
//!
//! Usage:
//!   hotspring_primal server          # start JSON-RPC server
//!   hotspring_primal capabilities    # print capabilities and exit

use hotspring_barracuda::composition;
use hotspring_barracuda::lattice::cg::cg_solve;
use hotspring_barracuda::lattice::dirac::{FermionField, apply_dirac};
use hotspring_barracuda::lattice::gradient_flow::{self, FlowIntegrator, find_t0, find_w0};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::md::config::MdConfig;
use hotspring_barracuda::md::cpu_reference::run_simulation_cpu;
use hotspring_barracuda::physics;
use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::provenance::SLY4_PARAMS;
use hotspring_forge::probe;
use hotspring_forge::substrate::{Capability, Fp64Rate, Fp64Strategy, SubstrateKind};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::Arc;

fn socket_path(cli_override: Option<&str>) -> PathBuf {
    if let Some(p) = cli_override {
        return PathBuf::from(p);
    }
    hotspring_barracuda::niche::resolve_server_socket()
}

struct HotSpringState {
    capabilities: Vec<String>,
    gpu_info: Vec<GpuSummary>,
    nucleus: NucleusContext,
    version: &'static str,
    socket_path: PathBuf,
}

struct GpuSummary {
    name: String,
    fp64_rate: String,
    strategy: String,
    has_f64: bool,
    has_df64: bool,
    vram_bytes: u64,
}

fn discover_capabilities() -> (Vec<String>, Vec<GpuSummary>) {
    let gpus = probe::probe_gpus();
    let cpu = probe::probe_cpu();
    let npus = probe::probe_npus();

    let mut caps: Vec<String> = hotspring_barracuda::niche::all_capabilities()
        .into_iter()
        .map(String::from)
        .collect();

    let mut gpu_summaries = Vec::new();

    for gpu in &gpus {
        if gpu.kind != SubstrateKind::Gpu {
            continue;
        }

        let props = &gpu.properties;
        let rate = props.fp64_rate.unwrap_or(Fp64Rate::Narrow);
        let strategy = Fp64Strategy::for_properties(props);

        if props.has_f64 {
            caps.push(format!("compute.gpu.{}", sanitize_name(&gpu.identity.name)));
        }
        if props.has_df64 {
            caps.push("compute.df64".into());
        }
        if gpu.capabilities.contains(&Capability::ConjugateGradient) {
            caps.push("compute.cg_solver".into());
        }

        let rate_str = match rate {
            Fp64Rate::Full => "1:1 (datacenter)",
            Fp64Rate::Half => "1:2 (Volta/HBM2)",
            Fp64Rate::Narrow => "1:16+ (consumer)",
        };
        let strat_str = match strategy {
            Fp64Strategy::Native => "Native f64",
            Fp64Strategy::Hybrid => "DF64 (f32-pair)",
            Fp64Strategy::Concurrent => "Concurrent (f64 + df64)",
        };

        gpu_summaries.push(GpuSummary {
            name: gpu.identity.name.clone(),
            fp64_rate: rate_str.into(),
            strategy: strat_str.into(),
            has_f64: props.has_f64,
            has_df64: props.has_df64,
            vram_bytes: props.memory_bytes.unwrap_or(0),
        });
    }

    if !npus.is_empty() {
        caps.push("compute.npu".into());
    }

    if cpu
        .capabilities
        .iter()
        .any(|c| matches!(c, Capability::SimdVector))
    {
        caps.push("compute.cpu.avx2".into());
    }

    caps.sort();
    caps.dedup();
    (caps, gpu_summaries)
}

fn sanitize_name(name: &str) -> String {
    name.to_lowercase()
        .replace(' ', "_")
        .replace(['(', ')', '/'], "")
        .replace("__", "_")
}

/// Dispatch result: either a successful JSON value or a JSON-RPC 2.0 error object.
enum DispatchResult {
    Ok(Value),
    Err { code: i64, message: String },
}

const DEFAULT_LATTICE_DIMS: [usize; 4] = [4, 4, 4, 4];
const LATTICE_DIM_CAP: usize = 12;

fn params_map(params: &Value) -> Option<&serde_json::Map<String, Value>> {
    match params {
        Value::Object(m) => Some(m),
        Value::Array(a) => a.first().and_then(Value::as_object),
        _ => None,
    }
}

fn parse_usize(m: &serde_json::Map<String, Value>, key: &str, default: usize) -> usize {
    m.get(key)
        .and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)))
        .map_or(default, |u| u as usize)
}

fn parse_u64(m: &serde_json::Map<String, Value>, key: &str, default: u64) -> u64 {
    m.get(key)
        .and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)))
        .unwrap_or(default)
}

fn parse_f64(m: &serde_json::Map<String, Value>, key: &str, default: f64) -> f64 {
    m.get(key)
        .and_then(|v| v.as_f64().or_else(|| v.as_u64().map(|u| u as f64)))
        .unwrap_or(default)
}

fn parse_dims(m: &serde_json::Map<String, Value>) -> [usize; 4] {
    let mut out = DEFAULT_LATTICE_DIMS;
    if let Some(Value::Array(arr)) = m.get("dims")
        && arr.len() == 4
    {
        for (i, v) in arr.iter().enumerate() {
            let n = v
                .as_u64()
                .or_else(|| v.as_f64().map(|f| f as u64))
                .unwrap_or(out[i] as u64) as usize;
            out[i] = n.clamp(2, LATTICE_DIM_CAP);
        }
    }
    out
}

fn parse_skyrme_params(m: &serde_json::Map<String, Value>) -> Vec<f64> {
    if let Some(Value::Array(arr)) = m.get("params") {
        arr.iter()
            .filter_map(|v| v.as_f64().or_else(|| v.as_u64().map(|u| u as f64)))
            .collect()
    } else {
        Vec::new()
    }
}

/// Run lattice/physics work under `catch_unwind` so a buggy kernel never tears down the server.
fn catch_physics<F>(method: &str, f: F) -> DispatchResult
where
    F: FnOnce() -> Value + std::panic::UnwindSafe,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => DispatchResult::Ok(v),
        Err(_) => DispatchResult::Err {
            code: -32603,
            message: format!("Internal error: unwound panic in {method}"),
        },
    }
}

fn handle_request(state: &HotSpringState, method: &str, params: &Value) -> DispatchResult {
    let method = normalize_method(method);
    match method {
        "health.check" | "health.liveness" => DispatchResult::Ok(json!({
            "status": "ok",
            "primal": "hotspring",
            "version": state.version,
            "gpus": state.gpu_info.len(),
        })),
        "health.readiness" => {
            let gpu_ready = !state.gpu_info.is_empty();
            let status = if gpu_ready { "ready" } else { "degraded" };
            DispatchResult::Ok(json!({
                "status": status,
                "primal": "hotspring",
                "version": state.version,
                "gpu_ready": gpu_ready,
                "gpu_count": state.gpu_info.len(),
                "capabilities_count": state.capabilities.len(),
            }))
        }
        "capabilities.list" | "capability.list" => DispatchResult::Ok(json!({
            "capabilities": state.capabilities,
        })),
        "compute.status" => {
            let gpus: Vec<Value> = state
                .gpu_info
                .iter()
                .map(|g| {
                    json!({
                        "name": g.name,
                        "fp64_rate": g.fp64_rate,
                        "strategy": g.strategy,
                        "has_f64": g.has_f64,
                        "has_df64": g.has_df64,
                        "vram_bytes": g.vram_bytes,
                    })
                })
                .collect();
            DispatchResult::Ok(json!({ "gpus": gpus, "status": "ok" }))
        }
        "composition.health" | "composition.nucleus_health" => {
            DispatchResult::Ok(composition::nucleus_health(&state.nucleus))
        }
        "composition.tower_health" => DispatchResult::Ok(composition::tower_health(&state.nucleus)),
        "composition.node_health" => DispatchResult::Ok(composition::node_health(&state.nucleus)),
        "composition.nest_health" => DispatchResult::Ok(composition::nest_health(&state.nucleus)),
        "composition.science_health" => DispatchResult::Ok(state.nucleus.physics_health()),
        "mcp.tools.list" => DispatchResult::Ok(hotspring_barracuda::mcp_tools::tools_list_json()),
        "physics.lattice_qcd" | "physics.lattice_gauge_update" => {
            let Some(m) = params_map(params) else {
                return DispatchResult::Err {
                    code: -32602,
                    message: "Invalid params: expected object with optional dims, beta, seed"
                        .into(),
                };
            };
            catch_physics(method, || {
                let dims = parse_dims(m);
                let beta = parse_f64(m, "beta", 6.0);
                let seed = parse_u64(m, "seed", 42);
                let lat = Lattice::hot_start(dims, beta, seed);
                let v = lat.volume();
                json!({
                    "plaquette": lat.average_plaquette(),
                    "volume": v,
                })
            })
        }
        "physics.hmc_trajectory" => {
            let Some(m) = params_map(params) else {
                return DispatchResult::Err {
                    code: -32602,
                    message: "Invalid params: expected object with dims, beta, n_steps, dt, seed"
                        .into(),
                };
            };
            catch_physics(method, || {
                let dims = parse_dims(m);
                let beta = parse_f64(m, "beta", 6.0);
                let seed = parse_u64(m, "seed", 42);
                let n_md_steps = parse_usize(m, "n_steps", 10).clamp(1, 10_000);
                let dt = parse_f64(m, "dt", 0.05);
                let mut lat = Lattice::hot_start(dims, beta, seed);
                let mut cfg = HmcConfig {
                    n_md_steps,
                    dt,
                    seed,
                    integrator: IntegratorType::Leapfrog,
                };
                let r = hmc::hmc_trajectory(&mut lat, &mut cfg);
                json!({
                    "plaquette": r.plaquette,
                    "accepted": r.accepted,
                    "delta_h": r.delta_h,
                })
            })
        }
        "physics.wilson_dirac" => {
            let Some(m) = params_map(params) else {
                return DispatchResult::Err {
                    code: -32602,
                    message: "Invalid params: expected object with dims, beta, mass, seed".into(),
                };
            };
            catch_physics(method, || {
                let dims = parse_dims(m);
                let beta = parse_f64(m, "beta", 6.0);
                let mass = parse_f64(m, "mass", 0.1);
                let seed = parse_u64(m, "seed", 42);
                let lat = Lattice::hot_start(dims, beta, seed);
                let vol = lat.volume();
                let psi = FermionField::random(vol, seed);
                let dpsi = apply_dirac(&lat, &psi, mass);
                let norm = dpsi.norm_sq().sqrt();
                json!({ "norm": norm, "volume": vol })
            })
        }
        "physics.molecular_dynamics" => {
            let Some(m) = params_map(params) else {
                return DispatchResult::Err {
                    code: -32602,
                    message: "Invalid params: expected object with n_particles, gamma, kappa, n_steps, seed"
                        .into(),
                };
            };
            catch_physics(method, || {
                let n_particles = parse_usize(m, "n_particles", 32).clamp(4, 512);
                let gamma = parse_f64(m, "gamma", 72.0);
                let kappa = parse_f64(m, "kappa", 1.0);
                let prod_steps = parse_usize(m, "n_steps", 40).clamp(1, 50_000);
                let _seed = parse_u64(m, "seed", 42);
                let rc = match kappa {
                    x if x >= 2.5 => 6.0,
                    x if x >= 1.5 => 6.5,
                    _ => 8.0,
                };
                let config = MdConfig {
                    label: "jsonrpc_md".into(),
                    n_particles,
                    kappa,
                    gamma,
                    dt: 0.01,
                    rc,
                    equil_steps: 0,
                    prod_steps,
                    dump_step: 1,
                    berendsen_tau: 5.0,
                    rdf_bins: 8,
                    vel_snapshot_interval: 1000,
                };
                let sim = run_simulation_cpu(&config);
                let first = sim.energy_history.first();
                let last = sim.energy_history.last();
                let (final_energy, temperature, energy_drift) = match (first, last) {
                    (Some(a), Some(b)) => (b.total, b.temperature, b.total - a.total),
                    _ => (f64::NAN, f64::NAN, f64::NAN),
                };
                json!({
                    "final_energy": final_energy,
                    "temperature": temperature,
                    "energy_drift": energy_drift,
                })
            })
        }
        "physics.nuclear_eos" => {
            let Some(m) = params_map(params) else {
                return DispatchResult::Err {
                    code: -32602,
                    message: "Invalid params: expected object with Z and N (optional params array)"
                        .into(),
                };
            };
            catch_physics(method, || {
                let z = parse_usize(m, "Z", 8).max(0);
                let n = parse_usize(m, "N", 8).max(0);
                let pvec = parse_skyrme_params(m);
                let be = if pvec.len() >= 10 {
                    physics::semf_binding_energy(z, n, &pvec[..10])
                } else {
                    physics::semf_binding_energy(z, n, &SLY4_PARAMS)
                };
                let a = z + n;
                let bpa = if a > 0 { be / a as f64 } else { 0.0 };
                json!({
                    "binding_energy_mev": be,
                    "binding_energy_per_nucleon": bpa,
                })
            })
        }
        "physics.fluid" => DispatchResult::Ok(json!({
            "status": "available",
            "implementations": [
                "gpu_euler",
                "gpu_kinetic_fluid",
                "kinetic_fluid_coupling",
            ],
        })),
        "physics.thermal" => DispatchResult::Ok(json!({
            "status": "available",
            "implementations": [
                "md_observables_transport",
                "gpu_dielectric",
                "fpeos_tables",
            ],
        })),
        "physics.radiation" => DispatchResult::Ok(json!({
            "status": "available",
            "implementations": [
                "dielectric_plasma_dispersion",
                "gpu_dielectric_multicomponent",
                "average_atom_wdm",
            ],
        })),
        "compute.df64" => {
            let names: Vec<&str> = state
                .gpu_info
                .iter()
                .filter(|g| g.has_df64)
                .map(|g| g.name.as_str())
                .collect();
            DispatchResult::Ok(json!({
                "available": state.gpu_info.iter().any(|g| g.has_df64),
                "gpus": names,
            }))
        }
        "compute.f64" => {
            let names: Vec<&str> = state
                .gpu_info
                .iter()
                .filter(|g| g.has_f64)
                .map(|g| g.name.as_str())
                .collect();
            DispatchResult::Ok(json!({
                "available": state.gpu_info.iter().any(|g| g.has_f64),
                "gpus": names,
            }))
        }
        "compute.cg_solver" => {
            let Some(m) = params_map(params) else {
                return DispatchResult::Err {
                    code: -32602,
                    message: "Invalid params: expected object with dims, beta, mass, seed".into(),
                };
            };
            catch_physics(method, || {
                let dims = parse_dims(m);
                let beta = parse_f64(m, "beta", 6.0);
                let mass = parse_f64(m, "mass", 0.1);
                let seed = parse_u64(m, "seed", 42);
                let lat = Lattice::hot_start(dims, beta, seed);
                let vol = lat.volume();
                let b = FermionField::random(vol, seed ^ 0xA5A5_A5A5_A5A5_A5A5);
                let mut x = FermionField::zeros(vol);
                let res = cg_solve(&lat, &mut x, &b, mass, 1e-8, 500);
                json!({
                    "converged": res.converged,
                    "iterations": res.iterations,
                    "residual": res.final_residual,
                })
            })
        }
        "compute.gradient_flow" => {
            let Some(m) = params_map(params) else {
                return DispatchResult::Err {
                    code: -32602,
                    message: "Invalid params: expected object with dims, beta, flow_steps, eps, seed"
                        .into(),
                };
            };
            catch_physics(method, || {
                let dims = parse_dims(m);
                let beta = parse_f64(m, "beta", 6.0);
                let seed = parse_u64(m, "seed", 42);
                let flow_steps = parse_usize(m, "flow_steps", 10).clamp(1, 50_000);
                let eps = parse_f64(m, "eps", 0.01);
                let t_max = flow_steps as f64 * eps;
                let mut lat = Lattice::hot_start(dims, beta, seed);
                let measurements =
                    gradient_flow::run_flow(&mut lat, FlowIntegrator::Rk3Luscher, eps, t_max, 1);
                let t0 = find_t0(&measurements);
                let w0 = find_w0(&measurements);
                let final_energy = measurements
                    .last()
                    .map_or(f64::NAN, |x| x.energy_density);
                json!({
                    "t0": t0,
                    "w0": w0,
                    "final_energy": final_energy,
                })
            })
        }
        _ => DispatchResult::Err {
            code: -32601,
            message: format!("Method not found: {method}"),
        },
    }
}

fn normalize_method(method: &str) -> &str {
    const PREFIXES: &[&str] = &["hotspring.", "primalspring.", "barracuda.", "biomeos."];
    let stripped = PREFIXES
        .iter()
        .find_map(|p| method.strip_prefix(p))
        .unwrap_or(method);
    match stripped {
        "capability.list" => "capabilities.list",
        other => other,
    }
}

fn run_server(state: Arc<HotSpringState>) {
    let sock = &state.socket_path;
    if let Some(parent) = sock.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::remove_file(sock);
    let listener = match UnixListener::bind(sock) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[hotspring_primal] bind error: {e} at {}", sock.display());
            std::process::exit(1);
        }
    };

    hotspring_barracuda::niche::register_with_target(sock);

    eprintln!("[hotspring_primal] listening on {}", sock.display());
    eprintln!(
        "[hotspring_primal] capabilities: {}",
        state.capabilities.len()
    );
    for g in &state.gpu_info {
        eprintln!("  GPU: {} | {} | {}", g.name, g.fp64_rate, g.strategy);
    }

    for stream in listener.incoming().flatten() {
        let state = Arc::clone(&state);
        std::thread::spawn(move || {
            let reader = BufReader::new(&stream);
            for line in reader.lines().map_while(Result::ok) {
                let req: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let id = req.get("id").cloned().unwrap_or(Value::Null);
                let method = req.get("method").and_then(Value::as_str).unwrap_or("");
                let params = req.get("params").cloned().unwrap_or(Value::Null);

                let dispatch = handle_request(&state, method, &params);

                let response = match dispatch {
                    DispatchResult::Ok(result) => {
                        json!({ "jsonrpc": "2.0", "id": id, "result": result })
                    }
                    DispatchResult::Err { code, message } => {
                        json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
                    }
                };

                let mut out = response.to_string();
                out.push('\n');
                let _ = (&stream).write_all(out.as_bytes());
                let _ = (&stream).flush();
            }
        });
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map_or("server", String::as_str);

    let mut socket_override = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--socket" if i + 1 < args.len() => {
                socket_override = Some(args[i + 1].as_str());
                i += 2;
            }
            "--family-id" if i + 1 < args.len() => {
                // SAFETY: called before spawning any threads (single-threaded
                // CLI arg parsing at startup). Required unsafe in Rust 2024.
                unsafe { std::env::set_var("FAMILY_ID", &args[i + 1]) };
                i += 2;
            }
            _ => i += 1,
        }
    }

    let (capabilities, gpu_info) = discover_capabilities();
    let nucleus = NucleusContext::detect();

    let state = Arc::new(HotSpringState {
        capabilities: capabilities.clone(),
        gpu_info,
        nucleus,
        version: env!("CARGO_PKG_VERSION"),
        socket_path: socket_path(socket_override),
    });

    match cmd {
        "server" => run_server(state),
        "capabilities" => {
            println!("hotSpring primal capabilities ({}):", capabilities.len());
            for cap in &capabilities {
                println!("  {cap}");
            }
            println!("\nGPU substrates:");
            for g in &state.gpu_info {
                println!(
                    "  {} | {} | {} | VRAM: {} MB",
                    g.name,
                    g.fp64_rate,
                    g.strategy,
                    g.vram_bytes / (1024 * 1024)
                );
            }
        }
        _ => {
            eprintln!("Usage: hotspring_primal [server|capabilities]");
            eprintln!("  --socket PATH    override socket path");
            eprintln!("  --family-id ID   primalSpring family identifier");
            std::process::exit(1);
        }
    }
}
