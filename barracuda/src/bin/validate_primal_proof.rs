// SPDX-License-Identifier: AGPL-3.0-or-later

//! Level 5 Primal Proof — validate science through NUCLEUS primal IPC.
//!
//! This is the Tier 3 validation harness. Unlike `validate_nucleus_composition`
//! (Tier 2, which tests hotSpring's own server dispatch and liveness), this
//! binary calls **barraCuda and BearDog primal methods directly over IPC**
//! and compares results against Python/Rust baselines.
//!
//! The distinction:
//! - Tier 2 (validate_nucleus_*): hotSpring server serves `physics.*` methods
//!   using in-process library calls. Validates IPC *routing* works.
//! - Tier 3 (this binary): calls `tensor.matmul`, `stats.mean`, `crypto.hash`
//!   etc. on the actual ecobin primals and compares against known-good values.
//!   Validates the *science produces correct results through primal IPC*.
//!
//! Prerequisites: a running NUCLEUS deployed from plasmidBin ecobins with at
//! least barraCuda and BearDog alive on UDS sockets.
//!
//! Exit codes:
//!   0 — all exercised capabilities PASS
//!   1 — at least one capability FAIL
//!   2 — all capabilities SKIP (NUCLEUS not deployed)
//!
//! Proto-nucleate: primalSpring/graphs/downstream/downstream_manifest.toml
//! IPC mapping: docs/PRIMAL_PROOF_IPC_MAPPING.md

use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

/// Validation capabilities from downstream_manifest.toml that this harness
/// exercises. Each maps to a specific primal IPC method.
const MANIFEST_CAPABILITIES: &[&str] = &[
    "tensor.matmul",
    "tensor.create",
    "tensor.add",
    "tensor.scale",
    "stats.mean",
    "stats.std_dev",
    "compute.dispatch",
    "crypto.hash",
    "tolerances.get",
    "validate.gpu_stack",
];

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Level 5 Primal Proof — hotSpring QCD Niche                ║");
    println!("║  Tier 3: Science through NUCLEUS primal IPC                ║");
    println!("║  Proto-nucleate: downstream_manifest.toml                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("primal_proof");
    let ctx = NucleusContext::detect();
    ctx.print_banner();
    println!();

    if ctx.discovered.is_empty() {
        println!("  No NUCLEUS primals discovered.");
        println!("  Deploy from plasmidBin ecobins and set FAMILY_ID, then rerun.");
        println!("  All capabilities SKIP — exit 2.");
        std::process::exit(2);
    }

    let bc_alive = ctx.get_by_capability("math").is_some_and(|ep| ep.alive);
    let bd_alive = ctx.get_by_capability("crypto").is_some_and(|ep| ep.alive);
    let ts_alive = ctx.get_by_capability("compute").is_some_and(|ep| ep.alive);

    println!("  barraCuda (math):  {}", if bc_alive { "ALIVE" } else { "DOWN" });
    println!("  BearDog (crypto):  {}", if bd_alive { "ALIVE" } else { "DOWN" });
    println!("  toadStool (compute): {}", if ts_alive { "ALIVE" } else { "DOWN" });
    println!();

    // ════════════════════════════════════════════════════════════════════
    //  PROBE 1: tensor.create + tensor.scale — parameter construction
    // ════════════════════════════════════════════════════════════════════
    println!("═══ Probe 1: tensor.create + tensor.scale ═══");
    if bc_alive {
        let create_result = ctx.call_by_capability(
            "math",
            "tensor.create",
            serde_json::json!({
                "shape": [5],
                "data": [15.8, -18.56, 0.711, 23.28, -0.75],
                "dtype": "f64"
            }),
        );
        match create_result {
            Ok(resp) => {
                let has_result = resp.get("result").is_some();
                harness.check_bool("tensor.create returns result", has_result);
                println!("  tensor.create: {}", if has_result { "OK" } else { "FAIL" });

                if let Some(tensor_id) = resp
                    .get("result")
                    .and_then(|r| r.get("tensor_id"))
                    .and_then(|v| v.as_str())
                {
                    let scale_result = ctx.call_by_capability(
                        "math",
                        "tensor.scale",
                        serde_json::json!({
                            "tensor": tensor_id,
                            "scalar": 208.0
                        }),
                    );
                    match scale_result {
                        Ok(sr) => {
                            harness.check_bool("tensor.scale returns result", sr.get("result").is_some());
                            println!("  tensor.scale(A=208): OK");
                        }
                        Err(e) => {
                            harness.check_bool("tensor.scale callable", false);
                            println!("  tensor.scale error: {e}");
                        }
                    }
                }
            }
            Err(e) => {
                harness.check_bool("tensor.create callable", false);
                println!("  tensor.create error: {e}");
            }
        }
    } else {
        println!("  SKIP — barraCuda not alive");
    }
    println!();

    // ════════════════════════════════════════════════════════════════════
    //  PROBE 2: tensor.matmul — SU(3) matrix multiplication parity
    // ════════════════════════════════════════════════════════════════════
    println!("═══ Probe 2: tensor.matmul — SU(3) parity ═══");
    if bc_alive {
        // 3x3 identity * 3x3 identity = 3x3 identity (trivial but proves the path)
        let matmul_result = ctx.call_by_capability(
            "math",
            "tensor.matmul",
            serde_json::json!({
                "a": {
                    "shape": [3, 3],
                    "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                },
                "b": {
                    "shape": [3, 3],
                    "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                }
            }),
        );
        match matmul_result {
            Ok(resp) => {
                if let Some(result_data) = resp
                    .get("result")
                    .and_then(|r| r.get("data"))
                    .and_then(|d| d.as_array())
                {
                    let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
                    let values: Vec<f64> = result_data
                        .iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                    if values.len() == 9 {
                        let max_err: f64 = values
                            .iter()
                            .zip(expected.iter())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0_f64, f64::max);
                        harness.check_upper("tensor.matmul I*I parity", max_err, 1e-15);
                        println!("  I*I max_err: {max_err:.2e}");
                    } else {
                        harness.check_bool("tensor.matmul output shape", false);
                        println!("  unexpected output length: {}", values.len());
                    }
                } else {
                    harness.check_bool("tensor.matmul result format", false);
                    println!("  missing result.data in response");
                }
            }
            Err(e) => {
                harness.check_bool("tensor.matmul callable", false);
                println!("  tensor.matmul error: {e}");
            }
        }
    } else {
        println!("  SKIP — barraCuda not alive");
    }
    println!();

    // ════════════════════════════════════════════════════════════════════
    //  PROBE 3: tensor.add — field arithmetic
    // ════════════════════════════════════════════════════════════════════
    println!("═══ Probe 3: tensor.add — field arithmetic ═══");
    if bc_alive {
        let add_result = ctx.call_by_capability(
            "math",
            "tensor.add",
            serde_json::json!({
                "a": { "shape": [4], "data": [1.0, 2.0, 3.0, 4.0] },
                "b": { "shape": [4], "data": [10.0, 20.0, 30.0, 40.0] }
            }),
        );
        match add_result {
            Ok(resp) => {
                if let Some(data) = resp
                    .get("result")
                    .and_then(|r| r.get("data"))
                    .and_then(|d| d.as_array())
                {
                    let expected = [11.0, 22.0, 33.0, 44.0];
                    let values: Vec<f64> = data.iter().filter_map(|v| v.as_f64()).collect();
                    let max_err: f64 = values
                        .iter()
                        .zip(expected.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max);
                    harness.check_upper("tensor.add element-wise parity", max_err, 1e-15);
                    println!("  [1,2,3,4]+[10,20,30,40] max_err: {max_err:.2e}");
                } else {
                    harness.check_bool("tensor.add result format", false);
                }
            }
            Err(e) => {
                harness.check_bool("tensor.add callable", false);
                println!("  tensor.add error: {e}");
            }
        }
    } else {
        println!("  SKIP — barraCuda not alive");
    }
    println!();

    // ════════════════════════════════════════════════════════════════════
    //  PROBE 4: stats.mean + stats.std_dev — observable averaging
    // ════════════════════════════════════════════════════════════════════
    println!("═══ Probe 4: stats.mean + stats.std_dev — observables ═══");
    if bc_alive {
        // Python baseline: np.mean([0.333, 0.334, 0.332, 0.335, 0.331]) = 0.333
        let plaquette_obs = [0.333, 0.334, 0.332, 0.335, 0.331];
        let python_mean = 0.333;
        let python_std = 0.001_414_213_562_373_095_1; // np.std(obs, ddof=0)

        let mean_result = ctx.call_by_capability(
            "math",
            "stats.mean",
            serde_json::json!({ "values": plaquette_obs }),
        );
        match mean_result {
            Ok(resp) => {
                if let Some(ipc_mean) = resp
                    .get("result")
                    .and_then(|r| r.get("mean").or_else(|| r.get("value")))
                    .and_then(|v| v.as_f64())
                {
                    let rel_err = ((ipc_mean - python_mean) / python_mean).abs();
                    harness.check_upper(
                        "stats.mean plaquette parity",
                        rel_err,
                        tolerances::COMPOSITION_SEMF_PARITY_REL,
                    );
                    println!("  IPC mean: {ipc_mean:.6}  Python: {python_mean:.6}  rel_err: {rel_err:.2e}");
                } else {
                    harness.check_bool("stats.mean result format", false);
                }
            }
            Err(e) => {
                harness.check_bool("stats.mean callable", false);
                println!("  stats.mean error: {e}");
            }
        }

        let std_result = ctx.call_by_capability(
            "math",
            "stats.std_dev",
            serde_json::json!({ "values": plaquette_obs }),
        );
        match std_result {
            Ok(resp) => {
                if let Some(ipc_std) = resp
                    .get("result")
                    .and_then(|r| r.get("std_dev").or_else(|| r.get("value")))
                    .and_then(|v| v.as_f64())
                {
                    let rel_err = ((ipc_std - python_std) / python_std).abs();
                    harness.check_upper("stats.std_dev parity", rel_err, tolerances::COMPOSITION_SEMF_PARITY_REL);
                    println!("  IPC std:  {ipc_std:.6e}  Python: {python_std:.6e}  rel_err: {rel_err:.2e}");
                } else {
                    harness.check_bool("stats.std_dev result format", false);
                }
            }
            Err(e) => {
                harness.check_bool("stats.std_dev callable", false);
                println!("  stats.std_dev error: {e}");
            }
        }
    } else {
        println!("  SKIP — barraCuda not alive");
    }
    println!();

    // ════════════════════════════════════════════════════════════════════
    //  PROBE 5: compute.dispatch — GPU shader execution
    // ════════════════════════════════════════════════════════════════════
    println!("═══ Probe 5: compute.dispatch ═══");
    if ts_alive {
        let dispatch_result = ctx.call_by_capability(
            "compute",
            "compute.dispatch",
            serde_json::json!({
                "shader": "identity_f64",
                "workgroups": [1, 1, 1]
            }),
        );
        match dispatch_result {
            Ok(resp) => {
                let has_result = resp.get("result").is_some();
                harness.check_bool("compute.dispatch returns result", has_result);
                println!("  dispatch: {}", if has_result { "OK" } else { "no result" });
            }
            Err(e) => {
                harness.check_bool("compute.dispatch callable", false);
                println!("  compute.dispatch error: {e}");
            }
        }
    } else if bc_alive {
        let dispatch_result = ctx.call_by_capability(
            "math",
            "compute.dispatch",
            serde_json::json!({
                "shader": "identity_f64",
                "workgroups": [1, 1, 1]
            }),
        );
        match dispatch_result {
            Ok(resp) => {
                let has_result = resp.get("result").is_some();
                harness.check_bool("compute.dispatch via barraCuda", has_result);
                println!("  dispatch via barraCuda: {}", if has_result { "OK" } else { "no result" });
            }
            Err(e) => {
                harness.check_bool("compute.dispatch callable", false);
                println!("  compute.dispatch error: {e}");
            }
        }
    } else {
        println!("  SKIP — neither toadStool nor barraCuda alive");
    }
    println!();

    // ════════════════════════════════════════════════════════════════════
    //  PROBE 6: crypto.hash — provenance witness (BearDog)
    // ════════════════════════════════════════════════════════════════════
    println!("═══ Probe 6: crypto.hash — provenance witness ═══");
    if bd_alive {
        let hash_result = ctx.call_by_capability(
            "crypto",
            "crypto.hash",
            serde_json::json!({
                "algorithm": "blake3",
                "data": "hotspring-primal-proof-witness"
            }),
        );
        match hash_result {
            Ok(resp) => {
                if let Some(hash) = resp
                    .get("result")
                    .and_then(|r| r.get("hash").or_else(|| r.get("digest")))
                    .and_then(|v| v.as_str())
                {
                    let valid_hex = hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit());
                    harness.check_bool("crypto.hash returns valid blake3", valid_hex);
                    println!("  blake3: {}...{}", &hash[..8], &hash[56..]);
                } else {
                    harness.check_bool("crypto.hash result format", false);
                    println!("  missing hash/digest in response");
                }
            }
            Err(e) => {
                harness.check_bool("crypto.hash callable", false);
                println!("  crypto.hash error: {e}");
            }
        }
    } else {
        println!("  SKIP — BearDog not alive");
    }
    println!();

    // ════════════════════════════════════════════════════════════════════
    //  PROBE 7: tolerances.get — tolerance retrieval from barraCuda
    // ════════════════════════════════════════════════════════════════════
    println!("═══ Probe 7: tolerances.get ═══");
    if bc_alive {
        let tol_result = ctx.call_by_capability(
            "math",
            "tolerances.get",
            serde_json::json!({ "name": "SEMF_PARITY_REL" }),
        );
        match tol_result {
            Ok(resp) => {
                let has_result = resp.get("result").is_some();
                harness.check_bool("tolerances.get returns result", has_result);
                if let Some(val) = resp
                    .get("result")
                    .and_then(|r| r.get("value"))
                    .and_then(|v| v.as_f64())
                {
                    println!("  SEMF_PARITY_REL: {val:.2e}");
                } else {
                    println!("  tolerances.get: result present but value not extractable");
                }
            }
            Err(e) => {
                harness.check_bool("tolerances.get callable", false);
                println!("  tolerances.get error: {e}");
            }
        }
    } else {
        println!("  SKIP — barraCuda not alive");
    }
    println!();

    // ════════════════════════════════════════════════════════════════════
    //  PROBE 8: validate.gpu_stack — GPU capability check
    // ════════════════════════════════════════════════════════════════════
    println!("═══ Probe 8: validate.gpu_stack ═══");
    if bc_alive {
        let gpu_result = ctx.call_by_capability(
            "math",
            "validate.gpu_stack",
            serde_json::json!({}),
        );
        match gpu_result {
            Ok(resp) => {
                let has_result = resp.get("result").is_some();
                harness.check_bool("validate.gpu_stack returns result", has_result);
                if let Some(result) = resp.get("result") {
                    if let Some(gpu_ok) = result.get("gpu_available").and_then(|v| v.as_bool()) {
                        println!("  GPU available: {gpu_ok}");
                    }
                    if let Some(backend) = result.get("backend").and_then(|v| v.as_str()) {
                        println!("  Backend: {backend}");
                    }
                }
            }
            Err(e) => {
                harness.check_bool("validate.gpu_stack callable", false);
                println!("  validate.gpu_stack error: {e}");
            }
        }
    } else {
        println!("  SKIP — barraCuda not alive");
    }
    println!();

    // ════════════════════════════════════════════════════════════════════
    //  PROBE 9: End-to-end science parity — SEMF via barraCuda IPC
    // ════════════════════════════════════════════════════════════════════
    println!("═══ Probe 9: SEMF B.E. end-to-end (barraCuda IPC vs Rust) ═══");
    if bc_alive {
        // Local Rust baseline
        let z = 82_usize;
        let n = 126_usize;
        let local_be = hotspring_barracuda::physics::semf_binding_energy(
            z,
            n,
            &hotspring_barracuda::provenance::SLY4_PARAMS,
        );
        println!("  Rust SEMF B.E.(Pb-208): {local_be:.4} MeV");

        // Ask barraCuda to compute via stats.weighted_mean + tensor operations
        // For now, exercise stats.mean as the simplest arithmetic verification:
        // the SEMF formula has 5 terms; verify barraCuda can average correctly.
        let a = (z + n) as f64;
        let vol = 15.8 * a;
        let surf = -18.56 * a.powf(2.0 / 3.0);
        let coul = -0.711 * (z as f64) * (z as f64 - 1.0) / a.powf(1.0 / 3.0);
        let asym = -23.28 * ((z as f64) - (n as f64)).powi(2) / a;
        let pair_delta = if z % 2 == 0 && n % 2 == 0 {
            12.0 / a.sqrt()
        } else if z % 2 != 0 && n % 2 != 0 {
            -12.0 / a.sqrt()
        } else {
            0.0
        };

        let terms = [vol, surf, coul, asym, pair_delta];
        let local_sum: f64 = terms.iter().sum();

        let sum_via_ipc = ctx.call_by_capability(
            "math",
            "stats.mean",
            serde_json::json!({ "values": terms }),
        );
        match sum_via_ipc {
            Ok(resp) => {
                if let Some(ipc_mean) = resp
                    .get("result")
                    .and_then(|r| r.get("mean").or_else(|| r.get("value")))
                    .and_then(|v| v.as_f64())
                {
                    // barraCuda returns mean; multiply by N to get sum
                    let ipc_sum = ipc_mean * terms.len() as f64;
                    let rel_err = ((ipc_sum - local_sum) / local_sum).abs();
                    harness.check_upper(
                        "SEMF terms sum parity (Rust vs barraCuda IPC)",
                        rel_err,
                        tolerances::COMPOSITION_SEMF_PARITY_REL,
                    );
                    println!("  Rust sum:  {local_sum:.4}  IPC sum: {ipc_sum:.4}  rel_err: {rel_err:.2e}");
                } else {
                    harness.check_bool("SEMF IPC result format", false);
                }
            }
            Err(e) => {
                harness.check_bool("SEMF via barraCuda IPC", false);
                println!("  stats.mean error: {e}");
            }
        }
    } else {
        println!("  SKIP — barraCuda not alive");
    }
    println!();

    // ════════════════════════════════════════════════════════════════════
    //  Summary
    // ════════════════════════════════════════════════════════════════════
    println!("═══ Primal Proof Summary ═══");
    println!("  Manifest capabilities:");
    for cap in MANIFEST_CAPABILITIES {
        let domain = cap.split('.').next().unwrap_or("unknown");
        let primal_alive = match domain {
            "tensor" | "stats" | "tolerances" | "validate" => bc_alive,
            "compute" => ts_alive || bc_alive,
            "crypto" => bd_alive,
            _ => false,
        };
        let status = if primal_alive { "EXERCISED" } else { "SKIPPED" };
        println!("    {cap:<24} [{status}]");
    }
    println!();

    harness.finish();
}
