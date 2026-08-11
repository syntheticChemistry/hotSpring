// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSON-RPC server for NUCLEUS deploy graph integration.
//!
//! Exposes hotSpring physics capabilities, health endpoints, and MCP tools
//! over Unix domain sockets. Discovered by biomeOS via `capability.list`.

mod dispatch;
mod params;
mod transport;

use crate::error::HotSpringError;
use crate::niche;
use crate::primal_bridge::NucleusContext;
use hotspring_forge::probe;
use hotspring_forge::substrate::{Capability, Fp64Rate, Fp64Strategy, SubstrateKind};
use log::warn;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Dispatch result: either a successful JSON value or a JSON-RPC 2.0 error object.
pub(super) enum DispatchResult {
    Ok(serde_json::Value),
    Err { code: i64, message: String },
}

pub(super) struct HotSpringState {
    capabilities: Vec<String>,
    gpu_info: Vec<GpuSummary>,
    nucleus: NucleusContext,
    version: &'static str,
    start_time: Instant,
}

struct GpuSummary {
    name: String,
    fp64_rate: String,
    strategy: String,
    has_f64: bool,
    has_df64: bool,
    vram_bytes: u64,
}

pub(super) const DEFAULT_LATTICE_DIMS: [usize; 4] = [4, 4, 4, 4];
pub(super) const LATTICE_DIM_CAP: usize = 12;

/// Start the JSON-RPC server on a Unix domain socket.
///
/// Socket path is resolved via [`niche::resolve_server_socket`] unless overridden.
/// Registers with biomeOS once at startup (gracefully skipped if unreachable).
pub fn run_server(
    socket_override: Option<&str>,
    family_id: Option<&str>,
) -> Result<(), HotSpringError> {
    if let Some(id) = family_id {
        niche::set_family_id(id.to_owned());
    }

    let socket_path = resolve_socket_path(socket_override);
    let (capabilities, gpu_info) = discover_capabilities();
    let nucleus = NucleusContext::detect();

    let state = Arc::new(HotSpringState {
        capabilities,
        gpu_info,
        nucleus,
        version: env!("CARGO_PKG_VERSION"),
        start_time: Instant::now(),
    });

    let bind_mode = std::env::var("PRIMAL_BIND_MODE").unwrap_or_default();
    if bind_mode == "tcp_only" {
        let port: u16 = std::env::var("HOTSPRING_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(9800);
        transport::serve_tcp_listener(state, port)
    } else {
        transport::serve_listener(state, &socket_path)
    }
}

fn resolve_socket_path(cli_override: Option<&str>) -> PathBuf {
    if let Some(p) = cli_override {
        PathBuf::from(p)
    } else {
        niche::resolve_server_socket()
    }
}

fn discover_capabilities() -> (Vec<String>, Vec<GpuSummary>) {
    let gpus = safe_probe_gpus();
    let cpu = probe::probe_cpu();
    let npus = probe::probe_npus();

    let mut caps: Vec<String> = niche::all_capabilities()
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

fn safe_probe_gpus() -> Vec<hotspring_forge::substrate::Substrate> {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    if let Ok(gpus) = catch_unwind(AssertUnwindSafe(probe::probe_gpus)) {
        gpus
    } else {
        warn!(
            target: "serve",
            "GPU substrate probe failed — starting without GPU detection"
        );
        Vec::new()
    }
}

fn sanitize_name(name: &str) -> String {
    name.to_lowercase()
        .replace(' ', "_")
        .replace(['(', ')', '/'], "")
        .replace("__", "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::niche;
    use crate::primal_bridge::NucleusContext;
    use dispatch::handle_request;
    use serde_json::Value;
    use transport::handle_connection_generic;

    fn test_state() -> HotSpringState {
        HotSpringState {
            capabilities: vec!["compute.physics".into()],
            gpu_info: vec![],
            nucleus: NucleusContext::detect(),
            version: env!("CARGO_PKG_VERSION"),
            start_time: Instant::now(),
        }
    }

    fn dispatch_ok(state: &HotSpringState, method: &str) -> Value {
        match handle_request(state, method, &Value::Null) {
            DispatchResult::Ok(v) => v,
            DispatchResult::Err { code, message } => {
                panic!("dispatch error {code}: {message}");
            }
        }
    }

    #[test]
    fn health_bare_alias_returns_guidestone_schema() {
        let state = test_state();
        let v = dispatch_ok(&state, "health");
        assert_eq!(v["status"], "ok");
        assert_eq!(v["primal"], niche::NICHE_NAME);
        assert!(v["version"].is_string(), "version must be present");
        assert!(v["uptime_s"].is_number(), "uptime_s must be present");
    }

    #[test]
    fn health_check_returns_guidestone_schema() {
        let state = test_state();
        let v = dispatch_ok(&state, "health.check");
        assert_eq!(v["status"], "ok");
        assert!(v["uptime_s"].is_number());
    }

    #[test]
    fn health_liveness_returns_guidestone_schema() {
        let state = test_state();
        let v = dispatch_ok(&state, "health.liveness");
        assert_eq!(v["status"], "ok");
        assert!(v["uptime_s"].is_number());
    }

    #[test]
    fn health_readiness_returns_uptime() {
        let state = test_state();
        let v = dispatch_ok(&state, "health.readiness");
        assert!(v["uptime_s"].is_number());
        assert!(v["status"].is_string());
    }

    #[test]
    fn health_with_primal_prefix_normalizes() {
        let state = test_state();
        let v = dispatch_ok(&state, "hotspring.health");
        assert_eq!(v["status"], "ok");
        assert!(v["uptime_s"].is_number());
    }

    #[test]
    fn capabilities_list_succeeds() {
        let state = test_state();
        let v = dispatch_ok(&state, "capabilities.list");
        assert!(v["capabilities"].is_array());
    }

    #[test]
    fn ribocipher_clear_signal_routes_ndjson() {
        use std::io::Cursor;
        let state = test_state();
        let request = r#"{"jsonrpc":"2.0","method":"health","params":null,"id":1}"#;
        let mut input: Vec<u8> = vec![0xEC, 0x01];
        input.extend_from_slice(request.as_bytes());
        input.push(b'\n');
        let mut output: Vec<u8> = Vec::new();
        let mut stream = Cursor::new(input);
        let combined = TestDuplex { read: &mut stream, write: &mut output };
        handle_connection_generic(combined, &state);
        let resp: Value = serde_json::from_slice(&output.split(|&b| b == b'\n').next().unwrap()).unwrap();
        assert_eq!(resp["result"]["status"], "ok");
    }

    #[test]
    fn ribocipher_unsignalled_json_is_rejected() {
        use std::io::Cursor;
        let state = test_state();
        let request = r#"{"jsonrpc":"2.0","method":"health","params":null,"id":1}"#;
        let mut input = request.as_bytes().to_vec();
        input.push(b'\n');
        let mut output: Vec<u8> = Vec::new();
        let mut stream = Cursor::new(input);
        let combined = TestDuplex { read: &mut stream, write: &mut output };
        handle_connection_generic(combined, &state);
        let resp: Value = serde_json::from_slice(&output.split(|&b| b == b'\n').next().unwrap()).unwrap();
        assert_eq!(resp["error"]["code"], -32002);
        assert!(resp["error"]["message"].as_str().unwrap().contains("riboCipher"));
    }

    struct TestDuplex<'a> {
        read: &'a mut dyn std::io::Read,
        write: &'a mut dyn std::io::Write,
    }

    impl std::io::Read for TestDuplex<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.read.read(buf)
        }
    }

    impl std::io::Write for TestDuplex<'_> {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.write.write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.write.flush()
        }
    }
}
