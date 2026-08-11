// SPDX-License-Identifier: AGPL-3.0-or-later

use super::dispatch::handle_request;
use super::{DispatchResult, HotSpringState};
use crate::error::HotSpringError;
use crate::niche;
use log::{error, info, warn};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::Arc;
use std::thread;

fn log_startup(state: &HotSpringState) {
    info!(
        target: "serve",
        "capabilities: {} ({} GPUs detected)",
        state.capabilities.len(),
        state.gpu_info.len(),
    );
    for g in &state.gpu_info {
        info!(
            target: "serve",
            "  GPU: {} | {} | {}",
            g.name,
            g.fp64_rate,
            g.strategy
        );
    }
}

pub(super) fn serve_listener(state: Arc<HotSpringState>, sock: &Path) -> Result<(), HotSpringError> {
    if let Some(parent) = sock.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _ = std::fs::remove_file(sock);
    let listener = UnixListener::bind(sock)
        .map_err(|e| HotSpringError::Ipc(format!("bind error: {e} at {}", sock.display())))?;

    niche::register_with_target(sock);
    info!(target: "serve", "listening on {} (UDS)", sock.display());
    log_startup(&state);

    loop {
        let (stream, _addr) = match listener.accept() {
            Ok(pair) => pair,
            Err(e) => {
                warn!(target: "serve", "accept error: {e}");
                continue;
            }
        };

        let state = Arc::clone(&state);
        thread::spawn(move || handle_connection(stream, &state));
    }
}

pub(super) fn serve_tcp_listener(state: Arc<HotSpringState>, port: u16) -> Result<(), HotSpringError> {
    let bind_addr = std::env::var("HOTSPRING_BIND_ADDRESS").unwrap_or_else(|_| "0.0.0.0".into());
    let addr = format!("{bind_addr}:{port}");
    let listener = TcpListener::bind(&addr)
        .map_err(|e| HotSpringError::Ipc(format!("TCP bind error: {e} at {addr}")))?;

    info!(target: "serve", "listening on {addr} (TCP, PRIMAL_BIND_MODE=tcp_only)");
    log_startup(&state);

    loop {
        let (stream, peer) = match listener.accept() {
            Ok(pair) => pair,
            Err(e) => {
                warn!(target: "serve", "TCP accept error: {e}");
                continue;
            }
        };

        info!(target: "serve", "TCP connection from {peer}");
        let state = Arc::clone(&state);
        thread::spawn(move || handle_connection_generic(stream, &state));
    }
}

fn handle_connection(stream: UnixStream, state: &HotSpringState) {
    handle_connection_generic(stream, state);
}

pub(super) fn handle_connection_generic<S: std::io::Read + std::io::Write>(
    mut stream: S,
    state: &HotSpringState,
) {
    let mut first_byte = [0u8; 1];
    if stream.read_exact(&mut first_byte).is_err() {
        return;
    }

    if first_byte[0] == 0xEC {
        let mut protocol_type = [0u8; 1];
        if stream.read_exact(&mut protocol_type).is_err() {
            return;
        }
        if protocol_type[0] != 0x01 {
            warn!(target: "serve", "riboCipher: unsupported protocol type 0x{:02X}", protocol_type[0]);
            return;
        }
        handle_ndjson_loop(stream, state);
    } else {
        error!(target: "serve", "REJECTED: unsignalled connection (first byte 0x{:02X}). riboCipher signal required.", first_byte[0]);
        let reject = json!({
            "jsonrpc": "2.0",
            "error": { "code": -32002, "message": "riboCipher signal required. Prepend [0xEC, 0x01] for NDJSON JSON-RPC." },
            "id": null
        });
        let _ = write_reject(&mut stream, &reject);
    }
}

fn write_reject<W: std::io::Write>(stream: &mut W, response: &Value) -> std::io::Result<()> {
    let mut out = response.to_string();
    out.push('\n');
    stream.write_all(out.as_bytes())?;
    stream.flush()
}

fn handle_ndjson_loop<S: std::io::Read + std::io::Write>(stream: S, state: &HotSpringState) {
    let mut reader = BufReader::new(stream);
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                warn!(target: "serve", "read error: {e}");
                break;
            }
        }

        if line.trim().is_empty() {
            continue;
        }

        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                warn!(target: "serve", "invalid JSON: {e}");
                continue;
            }
        };

        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let params = req.get("params").cloned().unwrap_or(Value::Null);

        let dispatch = handle_request(state, method, &params);

        let response = match dispatch {
            DispatchResult::Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            DispatchResult::Err { code, message } => {
                json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
            }
        };

        if write_response(reader.get_mut(), &response).is_err() {
            break;
        }
    }
}

fn write_response<W: std::io::Write>(
    stream: &mut W,
    response: &Value,
) -> Result<(), HotSpringError> {
    let mut out = response.to_string();
    out.push('\n');
    stream.write_all(out.as_bytes())?;
    stream.flush()?;
    Ok(())
}
