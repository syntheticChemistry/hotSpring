// SPDX-License-Identifier: AGPL-3.0-only

//! Sovereign Dispatch Benchmark — wgpu/Vulkan vs coralReef → DRM
//!
//! Runs the same Yukawa OCP simulation through both dispatch backends and
//! compares wall time, steps/s, and energy conservation.
//!
//! Also reports cross-spring shader evolution: which precision shaders
//! originated in which spring, and how they've propagated through the
//! barraCuda ecosystem (hotSpring → barraCuda ← wetSpring ← neuralSpring).
//!
//! Usage:
//!   cargo run --release --bin bench_sovereign_dispatch
//!   cargo run --release --features sovereign-dispatch --bin bench_sovereign_dispatch

use barracuda::device::backend::GpuBackend;
use barracuda::device::WgpuDevice;
use hotspring_barracuda::md::config::MdConfig;
use hotspring_barracuda::md::sovereign_engine::run_simulation_generic;
use std::sync::Arc;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Sovereign Dispatch Benchmark — Yukawa OCP MD              ║");
    println!("║  wgpu/Vulkan vs coralReef → DRM (backend-agnostic engine)  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let config = MdConfig {
        label: "sovereign_bench".to_string(),
        n_particles: 2000,
        kappa: 2.0,
        gamma: 160.0,
        rc: 6.0,
        dt: 0.005,
        equil_steps: 2000,
        prod_steps: 5000,
        dump_step: 500,
        vel_snapshot_interval: 5,
        berendsen_tau: 0.5,
        rdf_bins: 200,
    };

    println!("  Config: N={}, κ={}, Γ={}, equil={}, prod={}\n",
        config.n_particles, config.kappa, config.gamma,
        config.equil_steps, config.prod_steps);

    // ── Tier 1: wgpu/Vulkan backend ──
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  TIER 1: wgpu/Vulkan (GpuBackend = WgpuDevice)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let wgpu_result = {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            match WgpuDevice::new().await {
                Ok(dev) => {
                    let dev = Arc::new(dev);
                    println!("  Adapter: {}", dev.name());
                    println!("  f64 shaders: {}\n", dev.has_f64_shaders());
                    Some(run_simulation_generic(dev.as_ref(), &config))
                }
                Err(e) => {
                    println!("  wgpu unavailable: {e}\n");
                    None
                }
            }
        })
    };

    // ── Tier 2: Sovereign/coralReef backend ──
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  TIER 2: Sovereign (GpuBackend = CoralReefDevice)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let sovereign_result = try_sovereign(&config);

    // ── Comparison ──
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Results Comparison                                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("  {:<20} {:>12} {:>12} {:>12} {:>12}",
        "Backend", "Wall (s)", "Steps/s", "Final KE", "Final PE");
    println!("  {}", "-".repeat(72));

    if let Some(ref r) = wgpu_result {
        match r {
            Ok(sim) => {
                let (ke, pe) = last_energy(sim);
                println!("  {:<20} {:>12.2} {:>12.1} {:>12.4} {:>12.4}",
                    "wgpu/Vulkan", sim.wall_time_s, sim.steps_per_sec, ke, pe);
            }
            Err(e) => println!("  {:<20} FAILED: {e}", "wgpu/Vulkan"),
        }
    } else {
        println!("  {:<20} UNAVAILABLE", "wgpu/Vulkan");
    }

    if let Some(ref r) = sovereign_result {
        match r {
            Ok(sim) => {
                let (ke, pe) = last_energy(sim);
                println!("  {:<20} {:>12.2} {:>12.1} {:>12.4} {:>12.4}",
                    "Sovereign/DRM", sim.wall_time_s, sim.steps_per_sec, ke, pe);
            }
            Err(e) => println!("  {:<20} FAILED: {e}", "Sovereign/DRM"),
        }
    } else {
        println!("  {:<20} UNAVAILABLE", "Sovereign/DRM");
    }

    // ── Speedup ──
    if let (Some(Ok(wgpu_sim)), Some(Ok(sov_sim))) = (&wgpu_result, &sovereign_result) {
        let speedup = sov_sim.steps_per_sec / wgpu_sim.steps_per_sec;
        let (wke, wpe) = last_energy(wgpu_sim);
        let (ske, spe) = last_energy(sov_sim);
        let ke_diff = ((ske - wke) / wke.abs().max(1e-30)).abs();
        let pe_diff = ((spe - wpe) / wpe.abs().max(1e-30)).abs();

        println!("\n  Speedup: {speedup:.2}x (sovereign / wgpu)");
        println!("  Energy agreement: KE δ={ke_diff:.2e}, PE δ={pe_diff:.2e}");

        if ke_diff < 0.01 && pe_diff < 0.01 {
            println!("  ✓ Physics validated: <1% energy divergence between backends");
        } else {
            println!("  ⚠ Physics divergence detected — investigate precision routing");
        }
    }

    // ── Cross-spring shader evolution ──
    print_cross_spring_evolution();
}

fn last_energy(sim: &hotspring_barracuda::md::simulation::MdSimulation) -> (f64, f64) {
    sim.energy_history
        .last()
        .map_or((0.0, 0.0), |e| (e.ke, e.pe))
}

#[cfg(feature = "sovereign-dispatch")]
fn try_sovereign(
    config: &MdConfig,
) -> Option<Result<hotspring_barracuda::md::simulation::MdSimulation, String>> {
    use barracuda::device::CoralReefDevice;

    // Try multiple driver backends in priority order:
    // 1. Auto-detect (prefers whatever coral-gpu finds first)
    // 2. Nouveau on Titan V (SM70) — DRM dispatch implemented
    // 3. nvidia-drm on RTX 3090 (SM86) — pending UVM integration
    // 4. AMD (amdgpu) — E2E verified by coralReef
    let strategies: &[(&str, Box<dyn Fn() -> barracuda::error::Result<CoralReefDevice>>)] = &[
        ("auto", Box::new(|| CoralReefDevice::with_auto_device())),
        ("nouveau (SM70/Titan V)", Box::new(|| {
            CoralReefDevice::from_descriptor("nvidia", Some("sm70"), Some("nouveau"))
        })),
        ("nouveau (SM86/Ampere)", Box::new(|| {
            CoralReefDevice::from_descriptor("nvidia", Some("sm86"), Some("nouveau"))
        })),
        ("nvidia-drm (SM86)", Box::new(|| {
            CoralReefDevice::from_descriptor("nvidia", Some("sm86"), None)
        })),
        ("amdgpu", Box::new(|| {
            CoralReefDevice::from_descriptor("amd", None, None)
        })),
    ];

    for (label, init_fn) in strategies {
        print!("  Trying {label}... ");
        match init_fn() {
            Ok(dev) => {
                if !dev.has_dispatch() {
                    println!("no dispatch capability");
                    continue;
                }
                println!("OK");
                println!("  Adapter: {}", GpuBackend::name(&dev));
                println!("  f64 shaders: {}\n", GpuBackend::has_f64_shaders(&dev));
                return Some(run_simulation_generic(&dev, config));
            }
            Err(e) => {
                println!("failed ({e})");
            }
        }
    }

    println!("  No sovereign dispatch backend available\n");
    None
}

#[cfg(not(feature = "sovereign-dispatch"))]
fn try_sovereign(
    _config: &MdConfig,
) -> Option<Result<hotspring_barracuda::md::simulation::MdSimulation, String>> {
    println!("  sovereign-dispatch feature not enabled");
    println!("  Re-run with: cargo run --release --features sovereign-dispatch --bin bench_sovereign_dispatch\n");
    None
}

fn print_cross_spring_evolution() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Cross-Spring Shader Evolution                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("  Shader provenance across the ecoPrimals ecosystem:\n");
    println!("  {:<35} {:<15} {:<20}", "Shader", "Origin", "Absorbed by");
    println!("  {}", "-".repeat(72));
    println!("  {:<35} {:<15} {:<20}",
        "df64_core.wgsl", "hotSpring", "barraCuda → all springs");
    println!("  {:<35} {:<15} {:<20}",
        "df64_transcendentals.wgsl", "hotSpring", "barraCuda, coralReef");
    println!("  {:<35} {:<15} {:<20}",
        "yukawa_force_f64.wgsl", "hotSpring", "barraCuda md/");
    println!("  {:<35} {:<15} {:<20}",
        "smith_waterman_f64.wgsl", "wetSpring", "barraCuda bio/");
    println!("  {:<35} {:<15} {:<20}",
        "hmm_viterbi_f64.wgsl", "neuralSpring", "barraCuda bio/");
    println!("  {:<35} {:<15} {:<20}",
        "matrix_correlation_f64.wgsl", "neuralSpring", "barraCuda stats/");
    println!("  {:<35} {:<15} {:<20}",
        "perlin_2d_f64.wgsl", "ludoSpring", "barraCuda procedural/");
    println!("  {:<35} {:<15} {:<20}",
        "PrecisionBrain", "hotSpring", "barraCuda a012076");
    println!("  {:<35} {:<15} {:<20}",
        "HardwareCalibration", "hotSpring", "barraCuda a012076");
    println!("\n  Pipeline: hotSpring precision shaders validated sovereign");
    println!("  compilation via coralReef Iter 33 (WGSL → SASS, no naga).");
    println!("  wetSpring bio shaders + neuralSpring stats shaders benefit");
    println!("  from the same sovereign bypass for DF64 transcendentals.");
}
