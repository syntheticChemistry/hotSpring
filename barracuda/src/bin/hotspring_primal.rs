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
use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_forge::probe;
use hotspring_forge::substrate::{Capability, Fp64Rate, Fp64Strategy, SubstrateKind};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
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

fn handle_request(state: &HotSpringState, method: &str, _params: &Value) -> DispatchResult {
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
        m if is_registered_but_pending(m) => DispatchResult::Err {
            code: -32001,
            message: format!(
                "Method '{m}' is registered but dispatch is pending — \
                 use validation binaries directly or wait for full dispatch wiring"
            ),
        },
        _ => DispatchResult::Err {
            code: -32601,
            message: format!("Method not found: {method}"),
        },
    }
}

fn is_registered_but_pending(method: &str) -> bool {
    hotspring_barracuda::niche::LOCAL_CAPABILITIES
        .iter()
        .any(|cap| *cap == method)
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
