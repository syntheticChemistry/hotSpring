// SPDX-License-Identifier: AGPL-3.0-or-later

use super::params::{
    params_map, parse_dims, parse_f64, parse_skyrme_params, parse_u64, parse_usize,
};
use super::{DispatchResult, HotSpringState};
use crate::composition;
use crate::lattice::cg::cg_solve;
use crate::lattice::dirac::{FermionField, apply_dirac};
#[cfg(feature = "barracuda-local")]
use crate::lattice::gradient_flow::{self, FlowIntegrator, find_t0, find_w0};
use crate::lattice::hmc::{self, HmcConfig, IntegratorType};
use crate::lattice::wilson::Lattice;
use crate::mcp_tools;
#[cfg(feature = "barracuda-local")]
use crate::md::config::MdConfig;
#[cfg(feature = "barracuda-local")]
use crate::md::cpu_reference::run_simulation_cpu;
use crate::niche;
use crate::physics;
use crate::primal_bridge;
use crate::provenance::SLY4_PARAMS;
use serde_json::{Value, json};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// Run lattice/physics work under `catch_unwind` so a buggy kernel never tears down the server.
fn catch_physics<F>(method: &str, f: F) -> DispatchResult
where
    F: FnOnce() -> Value + std::panic::UnwindSafe,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => DispatchResult::Ok(v),
        Err(_) => DispatchResult::Err {
            code: -32_603,
            message: format!("Internal error: unwound panic in {method}"),
        },
    }
}

fn domain_for_routed_method(method: &str) -> Option<&'static str> {
    let (_, provider) = niche::ROUTED_CAPABILITIES
        .iter()
        .find(|(m, _)| *m == method)?;

    niche::DEPENDENCIES
        .iter()
        .find(|d| d.name == *provider)
        .map(|d| d.capability_domain)
        .or_else(|| {
            method.split('.').next().and_then(|prefix| {
                niche::DEPENDENCIES
                    .iter()
                    .find(|d| d.capability_domain == prefix)
                    .map(|d| d.capability_domain)
            })
        })
}

fn is_routed_method(method: &str) -> bool {
    niche::ROUTED_CAPABILITIES.iter().any(|(m, _)| *m == method)
}

fn route_to_primal(state: &HotSpringState, method: &str, params: &Value) -> DispatchResult {
    let domain = domain_for_routed_method(method).unwrap_or("discovery");
    match state
        .nucleus
        .call_by_capability(domain, method, params.clone())
    {
        Ok(resp) => match primal_bridge::parse_jsonrpc_response(&resp, method) {
            Ok(result) => DispatchResult::Ok(result),
            Err(e) => DispatchResult::Err {
                code: -32_603,
                message: e.to_string(),
            },
        },
        Err(e) => DispatchResult::Err {
            code: -32_603,
            message: e.to_string(),
        },
    }
}

pub(super) fn handle_request(state: &HotSpringState, method: &str, params: &Value) -> DispatchResult {
    let method = normalize_method(method);

    if is_routed_method(method) {
        return route_to_primal(state, method, params);
    }

    match method {
        "health" | "health.check" | "health.liveness" => DispatchResult::Ok(json!({
            "status": "ok",
            "primal": niche::NICHE_NAME,
            "version": state.version,
            "uptime_s": state.start_time.elapsed().as_secs(),
            "gpus": state.gpu_info.len(),
        })),
        "health.readiness" => {
            let gpu_ready = !state.gpu_info.is_empty();
            let status = if gpu_ready { "ready" } else { "degraded" };
            DispatchResult::Ok(json!({
                "status": status,
                "primal": niche::NICHE_NAME,
                "version": state.version,
                "uptime_s": state.start_time.elapsed().as_secs(),
                "gpu_ready": gpu_ready,
                "gpu_count": state.gpu_info.len(),
                "capabilities_count": state.capabilities.len(),
            }))
        }
        "capabilities.list" | "capability.list" => DispatchResult::Ok(json!({
            "capabilities": state.capabilities,
            "count": state.capabilities.len(),
            "primal": niche::NICHE_NAME,
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
        "mcp.tools.list" => DispatchResult::Ok(mcp_tools::tools_list_json()),
        "physics.lattice_qcd" | "physics.lattice_gauge_update" => {
            let Some(m) = params_map(params) else {
                return DispatchResult::Err {
                    code: -32_602,
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
                    code: -32_602,
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
                    code: -32_602,
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
            #[cfg(not(feature = "barracuda-local"))]
            {
                DispatchResult::Err {
                    code: -32_603,
                    message: "physics.molecular_dynamics requires barracuda-local build".into(),
                }
            }
            #[cfg(feature = "barracuda-local")]
            {
                let Some(m) = params_map(params) else {
                    return DispatchResult::Err {
                        code: -32_602,
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
        }
        "physics.nuclear_eos" => {
            let Some(m) = params_map(params) else {
                return DispatchResult::Err {
                    code: -32_602,
                    message: "Invalid params: expected object with Z and N (optional params array)"
                        .into(),
                };
            };
            catch_physics(method, || {
                let z = parse_usize(m, "Z", 8);
                let n = parse_usize(m, "N", 8);
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
                    code: -32_602,
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
                let res = cg_solve(
                    &lat,
                    &mut x,
                    &b,
                    mass,
                    crate::tolerances::DYNAMICAL_CG_TOLERANCE,
                    500,
                );
                json!({
                    "converged": res.converged,
                    "iterations": res.iterations,
                    "residual": res.final_residual,
                })
            })
        }
        "compute.gradient_flow" => {
            #[cfg(not(feature = "barracuda-local"))]
            {
                DispatchResult::Err {
                    code: -32_603,
                    message: "compute.gradient_flow requires barracuda-local build".into(),
                }
            }
            #[cfg(feature = "barracuda-local")]
            {
                let Some(m) = params_map(params) else {
                    return DispatchResult::Err {
                        code: -32_602,
                        message:
                            "Invalid params: expected object with dims, beta, flow_steps, eps, seed"
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
                    let measurements = gradient_flow::run_flow(
                        &mut lat,
                        FlowIntegrator::Rk3Luscher,
                        eps,
                        t_max,
                        1,
                    );
                    let t0 = find_t0(&measurements);
                    let w0 = find_w0(&measurements);
                    let final_energy = measurements.last().map_or(f64::NAN, |x| x.energy_density);
                    json!({
                        "t0": t0,
                        "w0": w0,
                        "final_energy": final_energy,
                    })
                })
            }
        }
        _ => DispatchResult::Err {
            code: -32_601,
            message: format!("Method not found: {method}"),
        },
    }
}

fn normalize_method(method: &str) -> &str {
    // Protocol-level method namespace prefixes (JSON-RPC convention, not runtime discovery).
    // "hotspring." is self-knowledge; others are standard ecosystem namespaces.
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
