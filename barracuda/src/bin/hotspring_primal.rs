// SPDX-License-Identifier: AGPL-3.0-only

//! hotspring_primal — primalSpring-compatible JSON-RPC server for hotSpring.
//!
//! Satisfies the `hotspring_validate.toml` graph contract:
//! - `health.check` / `health.liveness` → liveness probes
//! - `capabilities.list` → advertises physics + compute capabilities
//!
//! Capabilities are discovered at startup from metalForge substrate detection:
//! GPU model, FP64 rate, df64 support, precision strategy, available VRAM.
//!
//! Socket: `$XDG_RUNTIME_DIR/biomeos/hotspring-physics.sock` (primalSpring convention)
//!
//! Usage:
//!   hotspring_primal server          # start JSON-RPC server
//!   hotspring_primal capabilities    # print capabilities and exit

use hotspring_forge::probe;
use hotspring_forge::substrate::{Capability, Fp64Rate, Fp64Strategy, SubstrateKind};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::sync::Arc;

fn socket_path(cli_override: Option<&str>) -> PathBuf {
    if let Some(p) = cli_override {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("HOTSPRING_SOCKET") {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("PRIMAL_SOCKET") {
        return PathBuf::from(p);
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let dir = PathBuf::from(&xdg).join("biomeos");
        let _ = std::fs::create_dir_all(&dir);
        return dir.join("hotspring-physics.sock");
    }
    PathBuf::from("/tmp/biomeos/hotspring-physics.sock")
}

struct HotSpringState {
    capabilities: Vec<String>,
    gpu_info: Vec<GpuSummary>,
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

    let mut caps = vec![
        "physics.thermal".into(),
        "physics.fluid".into(),
        "physics.radiation".into(),
        "physics.lattice_qcd".into(),
        "physics.nuclear_eos".into(),
        "physics.molecular_dynamics".into(),
        "compute.f64".into(),
        "compute.gradient_flow".into(),
        "health.check".into(),
        "health.liveness".into(),
        "capabilities.list".into(),
    ];

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

    if cpu.capabilities.iter().any(|c| matches!(c, Capability::SimdVector)) {
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

fn handle_request(state: &HotSpringState, method: &str, _params: &Value) -> Value {
    let method = normalize_method(method);
    match method {
        "health.check" | "health.liveness" => json!({
            "status": "ok",
            "primal": "hotspring",
            "version": state.version,
            "gpus": state.gpu_info.len(),
        }),
        "capabilities.list" | "capability.list" => json!({
            "capabilities": state.capabilities,
        }),
        "compute.status" => {
            let gpus: Vec<Value> = state.gpu_info.iter().map(|g| json!({
                "name": g.name,
                "fp64_rate": g.fp64_rate,
                "strategy": g.strategy,
                "has_f64": g.has_f64,
                "has_df64": g.has_df64,
                "vram_bytes": g.vram_bytes,
            })).collect();
            json!({ "gpus": gpus, "status": "ok" })
        }
        _ => json!({
            "error": { "code": -32601, "message": format!("Method not found: {method}") }
        }),
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

    eprintln!("[hotspring_primal] listening on {}", sock.display());
    eprintln!("[hotspring_primal] capabilities: {}", state.capabilities.len());
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

                let result = handle_request(&state, method, &params);

                let response = if result.get("error").is_some() {
                    json!({ "jsonrpc": "2.0", "id": id, "error": result["error"] })
                } else {
                    json!({ "jsonrpc": "2.0", "id": id, "result": result })
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
    let cmd = args.get(1).map(String::as_str).unwrap_or("server");

    let mut socket_override = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--socket" if i + 1 < args.len() => {
                socket_override = Some(args[i + 1].as_str());
                i += 2;
            }
            "--family-id" if i + 1 < args.len() => {
                i += 2; // accepted but unused — family baked into socket name
            }
            _ => i += 1,
        }
    }

    let (capabilities, gpu_info) = discover_capabilities();

    let state = Arc::new(HotSpringState {
        capabilities: capabilities.clone(),
        gpu_info,
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
                println!("  {} | {} | {} | VRAM: {} MB",
                    g.name, g.fp64_rate, g.strategy,
                    g.vram_bytes / (1024 * 1024));
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
