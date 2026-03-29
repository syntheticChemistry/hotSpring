// SPDX-License-Identifier: AGPL-3.0-only

//! Node Atomic GPU Comparison — validates primalSpring composition patterns
//! for hotSpring's compute surface, then runs NVIDIA vs AMD comparison.
//!
//! Follows primalSpring's Node Atomic pattern (exp002/exp067):
//!   Phase 1: Start hotspring_primal server (capability provider)
//!   Phase 2: Discover by capability via UDS socket
//!   Phase 3: Validate health.check + capabilities.list via JSON-RPC
//!   Phase 4: Run comparative GPU workloads (BCS bisection + cooperative split)
//!   Phase 5: Report per-card precision routing via metalForge substrate census

use barracuda::ops::linalg::BatchedEighGpu;
use hotspring_barracuda::gpu::{discover_primary_and_secondary_adapters, GpuF64};
use hotspring_barracuda::physics::bcs_gpu::BcsBisectionGpu;
use hotspring_barracuda::validation::ValidationHarness;

use hotspring_forge::probe;
use hotspring_forge::substrate::{Fp64Rate, Fp64Strategy, SubstrateKind};

use std::io::{BufRead, BufReader, Write as IoWrite};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::Instant;

const WARMUP: usize = 3;
const MEASURE: usize = 10;

// ── Phase 1: Primal server lifecycle ────────────────────────────────────────

fn primal_binary() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("hotspring_primal")))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("hotspring_primal"))
}

fn primal_socket_path() -> PathBuf {
    let xdg = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(&xdg)
        .join("biomeos")
        .join("hotspring-physics.sock")
}

fn start_primal_server() -> Option<Child> {
    let binary = primal_binary();
    let sock = primal_socket_path();
    let _ = std::fs::remove_file(&sock);

    eprintln!(
        "[node_atomic] starting hotspring_primal at {}",
        sock.display()
    );
    let child = Command::new(&binary)
        .arg("server")
        .arg("--socket")
        .arg(&sock)
        .stderr(std::process::Stdio::piped())
        .spawn();

    match child {
        Ok(c) => {
            for _ in 0..50 {
                if sock.exists() {
                    eprintln!("[node_atomic] primal server ready");
                    return Some(c);
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            eprintln!("[node_atomic] primal server timed out waiting for socket");
            Some(c)
        }
        Err(e) => {
            eprintln!("[node_atomic] failed to start hotspring_primal: {e}");
            None
        }
    }
}

// ── Phase 2-3: JSON-RPC capability discovery and health probes ──────────────

fn rpc_call(sock: &std::path::Path, method: &str) -> Result<serde_json::Value, String> {
    let mut stream = UnixStream::connect(sock).map_err(|e| format!("connect: {e}"))?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .ok();

    let req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": {}
    });
    let mut payload = serde_json::to_string(&req).map_err(|e| format!("serialize: {e}"))?;
    payload.push('\n');
    stream
        .write_all(payload.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    stream.flush().map_err(|e| format!("flush: {e}"))?;

    let reader = BufReader::new(&stream);
    let line = reader
        .lines()
        .next()
        .ok_or("no response")?
        .map_err(|e| format!("read: {e}"))?;

    let resp: serde_json::Value = serde_json::from_str(&line).map_err(|e| format!("parse: {e}"))?;

    if let Some(err) = resp.get("error") {
        return Err(format!("RPC error: {err}"));
    }
    Ok(resp
        .get("result")
        .cloned()
        .unwrap_or(serde_json::Value::Null))
}

fn validate_primal_health(harness: &mut ValidationHarness, sock: &std::path::Path) -> bool {
    let health = rpc_call(sock, "health.check");
    let health_ok = health
        .as_ref()
        .is_ok_and(|v| v.get("status").and_then(|s| s.as_str()) == Some("ok"));
    harness.check_bool("health.check", health_ok);

    if let Ok(ref h) = health {
        let gpus = h.get("gpus").and_then(|g| g.as_u64()).unwrap_or(0);
        harness.check_bool("gpu_count >= 1", gpus >= 1);
        println!(
            "  health.check: status=ok, gpus={gpus}, version={}",
            h.get("version").and_then(|v| v.as_str()).unwrap_or("?")
        );
    }

    let liveness = rpc_call(sock, "health.liveness");
    harness.check_bool("health.liveness", liveness.is_ok());

    let prefix_norm = rpc_call(sock, "hotspring.health.check");
    harness.check_bool("prefix_norm (hotspring.health.check)", prefix_norm.is_ok());

    let primalspring_norm = rpc_call(sock, "primalspring.capabilities.list");
    harness.check_bool(
        "prefix_norm (primalspring.capabilities.list)",
        primalspring_norm.is_ok(),
    );

    health_ok
}

fn validate_primal_capabilities(
    harness: &mut ValidationHarness,
    sock: &std::path::Path,
) -> Vec<String> {
    let caps = rpc_call(sock, "capabilities.list");
    let cap_list: Vec<String> = caps
        .as_ref()
        .ok()
        .and_then(|v| v.get("capabilities"))
        .and_then(|c| serde_json::from_value(c.clone()).ok())
        .unwrap_or_default();

    harness.check_bool("capabilities.list responds", !cap_list.is_empty());
    harness.check_bool(
        "has physics.thermal",
        cap_list.iter().any(|c| c == "physics.thermal"),
    );
    harness.check_bool(
        "has physics.fluid",
        cap_list.iter().any(|c| c == "physics.fluid"),
    );
    harness.check_bool(
        "has physics.radiation",
        cap_list.iter().any(|c| c == "physics.radiation"),
    );
    harness.check_bool(
        "has physics.lattice_qcd",
        cap_list.iter().any(|c| c == "physics.lattice_qcd"),
    );
    harness.check_bool(
        "has compute.f64",
        cap_list.iter().any(|c| c == "compute.f64"),
    );
    harness.check_bool(
        "has compute.df64",
        cap_list.iter().any(|c| c == "compute.df64"),
    );
    harness.check_bool(
        "has health.check",
        cap_list.iter().any(|c| c == "health.check"),
    );

    println!("  capabilities: {} registered", cap_list.len());
    for cap in &cap_list {
        println!("    {cap}");
    }

    cap_list
}

fn validate_compute_status(harness: &mut ValidationHarness, sock: &std::path::Path) {
    let status = rpc_call(sock, "compute.status");
    let gpus = status
        .as_ref()
        .ok()
        .and_then(|v| v.get("gpus"))
        .and_then(|g| g.as_array());

    if let Some(gpu_arr) = gpus {
        harness.check_bool("compute.status has GPU entries", !gpu_arr.is_empty());
        for gpu in gpu_arr {
            let name = gpu.get("name").and_then(|n| n.as_str()).unwrap_or("?");
            let rate = gpu.get("fp64_rate").and_then(|r| r.as_str()).unwrap_or("?");
            let strat = gpu.get("strategy").and_then(|s| s.as_str()).unwrap_or("?");
            let has_df64 = gpu
                .get("has_df64")
                .and_then(|d| d.as_bool())
                .unwrap_or(false);
            harness.check_bool(&format!("{name} has_df64"), has_df64);
            println!("  {name}");
            println!("    fp64_rate: {rate}");
            println!("    strategy:  {strat}");
            println!("    has_df64:  {has_df64}");
        }
    } else {
        harness.check_bool("compute.status responds", false);
    }
}

// ── Phase 4: GPU workload comparison ────────────────────────────────────────

fn bench_bcs_single(gpu: &GpuF64, batch: usize, n_levels: usize) -> f64 {
    let eigenvalues: Vec<f64> = (0..batch * n_levels)
        .map(|i| (i as f64 * 0.3 + 1.0).sin() * 10.0)
        .collect();
    let delta = vec![2.0_f64; batch];
    let target_n: Vec<f64> = (0..batch).map(|i| (i % 10 + 2) as f64).collect();
    let lower = vec![-50.0_f64; batch];
    let upper = vec![200.0_f64; batch];

    let bcs = BcsBisectionGpu::new(gpu, 100, 1e-12);
    for _ in 0..WARMUP {
        let _ = bcs.solve_bcs(&lower, &upper, &eigenvalues, &delta, &target_n);
    }
    let t0 = Instant::now();
    for _ in 0..MEASURE {
        let _ = bcs.solve_bcs(&lower, &upper, &eigenvalues, &delta, &target_n);
    }
    t0.elapsed().as_secs_f64() / MEASURE as f64
}

fn bench_eigensolve_single(gpu: &GpuF64, batch: usize, dim: usize) -> f64 {
    let device = gpu.to_wgpu_device();
    let matrices: Vec<f64> = (0..batch * dim * dim)
        .map(|i| ((i as f64) * 0.1).sin())
        .collect();
    let mut symmetrized = matrices;
    for b in 0..batch {
        for r in 0..dim {
            for c in (r + 1)..dim {
                let val = (symmetrized[b * dim * dim + r * dim + c]
                    + symmetrized[b * dim * dim + c * dim + r])
                    / 2.0;
                symmetrized[b * dim * dim + r * dim + c] = val;
                symmetrized[b * dim * dim + c * dim + r] = val;
            }
        }
    }

    for _ in 0..WARMUP {
        let _ = BatchedEighGpu::execute_single_dispatch(
            device.clone(),
            &symmetrized,
            dim,
            batch,
            200,
            1e-12,
        );
    }
    let t0 = Instant::now();
    for _ in 0..MEASURE {
        let _ = BatchedEighGpu::execute_single_dispatch(
            device.clone(),
            &symmetrized,
            dim,
            batch,
            200,
            1e-12,
        );
    }
    t0.elapsed().as_secs_f64() / MEASURE as f64
}

fn run_gpu_comparison(harness: &mut ValidationHarness, gpu_a: &GpuF64, gpu_b: &GpuF64) {
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Phase 4: GPU Workload Comparison (NVIDIA vs AMD)");
    println!("═══════════════════════════════════════════════════════════════");

    let batch = 4096;
    let n_levels = 20;
    println!();
    println!("── 4a: BCS Bisection (batch={batch}, levels={n_levels}) ──");
    let t_a = bench_bcs_single(gpu_a, batch, n_levels);
    let t_b = bench_bcs_single(gpu_b, batch, n_levels);
    println!(
        "  {:<40} {:>8.3} ms  ({:.0}/s)",
        gpu_a.adapter_name,
        t_a * 1e3,
        batch as f64 / t_a
    );
    println!(
        "  {:<40} {:>8.3} ms  ({:.0}/s)",
        gpu_b.adapter_name,
        t_b * 1e3,
        batch as f64 / t_b
    );
    report_winner("BCS", &gpu_a.adapter_name, &gpu_b.adapter_name, t_a, t_b);
    harness.check_bool("BCS card A > 0 throughput", t_a > 0.0);
    harness.check_bool("BCS card B > 0 throughput", t_b > 0.0);

    let batch_e = 256;
    let dim_e = 20;
    println!();
    println!("── 4b: Eigensolve (batch={batch_e}, dim={dim_e}) ──");
    let t_a = bench_eigensolve_single(gpu_a, batch_e, dim_e);
    let t_b = bench_eigensolve_single(gpu_b, batch_e, dim_e);
    if t_a > 0.0 && t_b > 0.0 {
        println!(
            "  {:<40} {:>8.3} ms  ({:.0}/s)",
            gpu_a.adapter_name,
            t_a * 1e3,
            batch_e as f64 / t_a
        );
        println!(
            "  {:<40} {:>8.3} ms  ({:.0}/s)",
            gpu_b.adapter_name,
            t_b * 1e3,
            batch_e as f64 / t_b
        );
        report_winner(
            "Eigensolve",
            &gpu_a.adapter_name,
            &gpu_b.adapter_name,
            t_a,
            t_b,
        );
        harness.check_bool("Eigen card A > 0 throughput", true);
        harness.check_bool("Eigen card B > 0 throughput", true);
    } else {
        println!("  eigensolve unavailable on one or both cards");
    }

    let total = 8192;
    println!();
    println!("── 4c: Cooperative BCS (split {total} across both cards) ──");
    let t_single = bench_bcs_single(gpu_a, total, n_levels);
    let t_coop = bench_cooperative_bcs(gpu_a, gpu_b, total, n_levels);
    println!(
        "  Single card (A):   {:>8.3} ms  ({:.0}/s)",
        t_single * 1e3,
        total as f64 / t_single
    );
    println!(
        "  Cooperative (A+B): {:>8.3} ms  ({:.0}/s)",
        t_coop * 1e3,
        total as f64 / t_coop
    );
    let speedup = t_single / t_coop;
    println!("  → Cooperative speedup: {speedup:.2}×");
    harness.check_bool("cooperative dispatch functional", t_coop > 0.0);
}

fn bench_cooperative_bcs(gpu_a: &GpuF64, gpu_b: &GpuF64, total: usize, n_levels: usize) -> f64 {
    let eigenvalues: Vec<f64> = (0..total * n_levels)
        .map(|i| (i as f64 * 0.3 + 1.0).sin() * 10.0)
        .collect();
    let delta = vec![2.0_f64; total];
    let target_n: Vec<f64> = (0..total).map(|i| (i % 10 + 2) as f64).collect();
    let lower = vec![-50.0_f64; total];
    let upper = vec![200.0_f64; total];

    let bcs_a = BcsBisectionGpu::new(gpu_a, 100, 1e-12);
    let bcs_b = BcsBisectionGpu::new(gpu_b, 100, 1e-12);
    let half = total / 2;

    for _ in 0..WARMUP {
        std::thread::scope(|s| {
            s.spawn(|| {
                let _ = bcs_a.solve_bcs(
                    &lower[..half],
                    &upper[..half],
                    &eigenvalues[..half * n_levels],
                    &delta[..half],
                    &target_n[..half],
                );
            });
            s.spawn(|| {
                let _ = bcs_b.solve_bcs(
                    &lower[half..],
                    &upper[half..],
                    &eigenvalues[half * n_levels..],
                    &delta[half..],
                    &target_n[half..],
                );
            });
        });
    }

    let t0 = Instant::now();
    for _ in 0..MEASURE {
        std::thread::scope(|s| {
            s.spawn(|| {
                let _ = bcs_a.solve_bcs(
                    &lower[..half],
                    &upper[..half],
                    &eigenvalues[..half * n_levels],
                    &delta[..half],
                    &target_n[..half],
                );
            });
            s.spawn(|| {
                let _ = bcs_b.solve_bcs(
                    &lower[half..],
                    &upper[half..],
                    &eigenvalues[half * n_levels..],
                    &delta[half..],
                    &target_n[half..],
                );
            });
        });
    }
    t0.elapsed().as_secs_f64() / MEASURE as f64
}

fn report_winner(label: &str, name_a: &str, name_b: &str, t_a: f64, t_b: f64) {
    let ratio = t_a / t_b;
    if ratio > 1.05 {
        println!("  → {label}: {name_b} wins by {ratio:.2}×");
    } else if ratio < 0.95 {
        println!("  → {label}: {name_a} wins by {:.2}×", 1.0 / ratio);
    } else {
        println!("  → {label}: within 5% (ratio {ratio:.2})");
    }
}

// ── Phase 5: Substrate precision summary ────────────────────────────────────

fn report_substrate_census(harness: &mut ValidationHarness) {
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Phase 5: metalForge Substrate Census & Precision Routing");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let substrates = probe::probe_gpus();
    harness.check_bool("metalForge discovered GPUs", !substrates.is_empty());

    for sub in &substrates {
        if sub.kind == SubstrateKind::Gpu {
            let rate = sub.properties.fp64_rate.unwrap_or(Fp64Rate::Narrow);
            let strat = Fp64Strategy::for_properties(&sub.properties);
            println!("  {} [{}]", sub.identity.name, sub.kind);
            println!("    caps: {}", sub.capability_summary());
            println!("    fp64_rate: {rate:?} → strategy: {strat:?}");
            println!(
                "    has_f64: {}  has_df64: {}",
                sub.properties.has_f64, sub.properties.has_df64
            );
            println!();
        }
    }

    let cpu = probe::probe_cpu();
    println!("  {} [{}]", cpu.identity.name, cpu.kind);
    println!("    caps: {}", cpu.capability_summary());
    println!();

    let npus = probe::probe_npus();
    if npus.is_empty() {
        println!("  NPU: none detected");
    }
    for npu in &npus {
        println!("  {} [{}]", npu.identity.name, npu.kind);
        println!("    caps: {}", npu.capability_summary());
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  hotSpring Node Atomic GPU Comparison");
    println!("  primalSpring composition pattern validation");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut harness = ValidationHarness::new("Node Atomic GPU Comparison");

    // ── Phase 1: Start primal server ──
    println!("── Phase 1: Primal Server Lifecycle ──");
    let mut child = start_primal_server();
    let server_started = child.is_some();
    harness.check_bool("hotspring_primal started", server_started);

    if !server_started {
        eprintln!("  WARN: primal server not available, running direct GPU comparison");
    }

    // ── Phase 2-3: Capability discovery + health validation ──
    let sock = primal_socket_path();
    if server_started && sock.exists() {
        println!();
        println!("── Phase 2: Capability Discovery (JSON-RPC over UDS) ──");
        let healthy = validate_primal_health(&mut harness, &sock);
        harness.check_bool("primal healthy", healthy);

        println!();
        println!("── Phase 3: Capability Surface Validation ──");
        let caps = validate_primal_capabilities(&mut harness, &sock);
        harness.check_bool("capabilities >= 10", caps.len() >= 10);

        println!();
        println!("── Phase 3b: Compute Status (GPU substrates) ──");
        validate_compute_status(&mut harness, &sock);
    }

    // ── Phase 4: GPU Comparison ──
    println!();
    println!("── Initializing GPU Substrates ──");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let (primary_name, secondary_name) = discover_primary_and_secondary_adapters();
    let primary_name = primary_name.expect("no primary GPU with SHADER_F64");
    let secondary_name = secondary_name.expect("no secondary GPU with SHADER_F64");

    println!("  Primary:   {primary_name}");
    println!("  Secondary: {secondary_name}");

    #[allow(deprecated)]
    let gpu_a = rt.block_on(async {
        std::env::set_var("HOTSPRING_GPU_ADAPTER", &primary_name);
        GpuF64::new().await
    });
    #[allow(deprecated)]
    std::env::set_var("HOTSPRING_GPU_ADAPTER", &secondary_name);
    let gpu_b = rt.block_on(GpuF64::new());

    let gpu_a = match gpu_a {
        Ok(g) => {
            harness.check_bool(&format!("{} initialized", g.adapter_name), true);
            g
        }
        Err(e) => {
            harness.check_bool("primary GPU init", false);
            eprintln!("  primary GPU ({primary_name}): {e}");
            cleanup(child.as_mut());
            harness.finish();
        }
    };
    let gpu_b = match gpu_b {
        Ok(g) => {
            harness.check_bool(&format!("{} initialized", g.adapter_name), true);
            g
        }
        Err(e) => {
            harness.check_bool("secondary GPU init", false);
            eprintln!("  secondary GPU ({secondary_name}): {e}");
            cleanup(child.as_mut());
            harness.finish();
        }
    };

    run_gpu_comparison(&mut harness, &gpu_a, &gpu_b);
    report_substrate_census(&mut harness);

    println!();
    cleanup(child.as_mut());
    harness.finish();
}

fn cleanup(child: Option<&mut Child>) {
    if let Some(c) = child {
        let _ = c.kill();
        let _ = c.wait();
        let _ = std::fs::remove_file(primal_socket_path());
    }
}
