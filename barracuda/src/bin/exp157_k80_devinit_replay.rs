// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 157: K80 DEVINIT Replay
//!
//! Replays the VBIOS DEVINIT initialization sequence on a cold Tesla K80 (GK210)
//! through the ember MMIO gateway. This is the hardware POST sequence that trains
//! GDDR5, enables engines, and brings the GPU to a compute-ready state.
//!
//! The DEVINIT recipe was captured from nvidia-470 VM captures (exp124b) and contains
//! the exact register writes the VBIOS executes during cold boot.
//!
//! ## Operations
//!
//! - `ZmReg` / `ZM_REG`: Direct register write
//! - `NvReg` / `NV_REG`: Read-modify-write (read, AND mask, OR value)
//! - `ZmMaskAdd`: Read, AND mask, ADD value, write
//! - `Time`: Microsecond delay
//!
//! ## Usage
//!
//! ```text
//! sudo ./target/release/exp157_k80_devinit_replay --bdf 0000:4c:00.0 \
//!     --recipe data/k80/gk210_devinit_recipe.json
//! ```

use std::path::Path;
use std::time::Duration;

use hotspring_barracuda::fleet_client::{EmberClient, FleetDiscovery};
use hotspring_barracuda::validation::ValidationHarness;

const PRAMIN_WINDOW: u32 = 0x700000;
const PGRAPH_STATUS: u32 = 0x400700;
const PMC_ENABLE: u32 = 0x200;
const PMC_BOOT0: u32 = 0x0;
const FECS_CPUCTL: u32 = 0x409100;

const VRAM_DEAD_PATTERNS: [u32; 5] = [
    0xBAD0_AC00, 0xBAD0_AC01, 0xBAD0_AC02, 0xBADF_3000, 0xFFFF_FFFF,
];

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  K80 DEVINIT Replay — Experiment 157                       ║");
    println!("║  Pipeline: VBIOS DEVINIT → GDDR5 train → engine enable     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut harness = ValidationHarness::new("exp157_k80_devinit_replay");

    let args: Vec<String> = std::env::args().collect();
    let bdf = extract_arg(&args, "--bdf").unwrap_or_else(|| "0000:4c:00.0".to_string());
    let recipe_path = extract_arg(&args, "--recipe").unwrap_or_else(|| {
        "data/k80/gk210_devinit_recipe.json".to_string()
    });
    let dry_run = args.iter().any(|a| a == "--dry-run");

    println!("  Target BDF: {bdf}");
    println!("  Recipe: {recipe_path}");
    if dry_run {
        println!("  Mode: DRY RUN (no writes)");
    }

    let ember = match connect_ember(&bdf) {
        Some(e) => e,
        None => {
            eprintln!("ERROR: cannot connect to ember for {bdf}");
            harness.check_bool("ember reachable", false);
            harness.finish();
        }
    };
    harness.check_bool("ember reachable", true);

    let recipe = match load_recipe(&recipe_path) {
        Some(r) => r,
        None => {
            eprintln!("ERROR: cannot load recipe from {recipe_path}");
            harness.check_bool("recipe loaded", false);
            harness.finish();
        }
    };
    harness.check_bool("recipe loaded", true);

    let total_ops: usize = recipe.iter().map(|s| s.ops.len()).sum();
    println!("  Loaded {total_ops} ops across {} scripts\n", recipe.len());

    // Phase 1: Pre-DEVINIT baseline
    println!("━━━ Phase 1: Pre-DEVINIT Baseline ━━━\n");
    let pre_state = read_key_registers(&ember, &bdf);
    print_state("  PRE", &pre_state);
    let pre_vram_dead = is_vram_dead(pre_state.pramin);
    println!("  VRAM dead: {pre_vram_dead}");
    harness.check_bool("pre-DEVINIT baseline readable", pre_state.boot0 != 0);

    // Phase 2: DEVINIT replay
    println!("\n━━━ Phase 2: DEVINIT Replay ━━━\n");
    let mut ops_executed = 0;
    let mut ops_failed = 0;
    let mut breaker_resets = 0;
    let mut last_pmc_enable = pre_state.pmc_enable;

    for (script_idx, script) in recipe.iter().enumerate() {
        println!("  Script {script_idx} ({} ops, addr={}):", script.ops.len(), script.addr);

        for (op_idx, raw) in script.ops.iter().enumerate() {
            let op = DevinitOp::from(raw);

            // After read failures, try resetting the circuit breaker and waiting
            // for the GPU to stabilize (PLL changes can cause transient BAR0 blackouts)
            let try_read = |e: &EmberClient, b: &str, reg: u32| -> Result<u32, String> {
                match e.mmio_read(b, reg) {
                    Ok(r) => Ok(r.value),
                    Err(err) => {
                        if err.contains("circuit breaker") || err.contains("non-responsive") {
                            let _ = e.mmio_circuit_breaker(b, Some("reset"));
                            std::thread::sleep(Duration::from_millis(50));
                            e.mmio_read(b, reg).map(|r| r.value)
                        } else {
                            Err(err)
                        }
                    }
                }
            };

            match &op {
                DevinitOp::ZmReg { reg, val } => {
                    if dry_run {
                        if *reg == PMC_ENABLE {
                            println!("    [{op_idx:3}] DRY: PMC_ENABLE = {val:#010x}");
                        }
                        ops_executed += 1;
                        continue;
                    }
                    match ember.mmio_write(&bdf, *reg, *val) {
                        Ok(_) => {
                            ops_executed += 1;
                            if *reg == PMC_ENABLE {
                                last_pmc_enable = *val;
                                println!("    [{op_idx:3}] PMC_ENABLE = {val:#010x}");
                                // PLL/engine changes need settle time
                                std::thread::sleep(Duration::from_millis(10));
                                let _ = ember.mmio_circuit_breaker(&bdf, Some("reset"));
                                breaker_resets += 1;
                            }
                        }
                        Err(e) => {
                            // For writes, try resetting breaker and retrying
                            if e.contains("circuit breaker") || e.contains("non-responsive") {
                                let _ = ember.mmio_circuit_breaker(&bdf, Some("reset"));
                                breaker_resets += 1;
                                std::thread::sleep(Duration::from_millis(50));
                                match ember.mmio_write(&bdf, *reg, *val) {
                                    Ok(_) => { ops_executed += 1; continue; }
                                    Err(_) => {}
                                }
                            }
                            ops_failed += 1;
                            if *reg == PMC_ENABLE || ops_failed <= 10 {
                                eprintln!("    [{op_idx:3}] WRITE FAILED: reg={reg:#010x} err={e}");
                            }
                        }
                    }
                }
                DevinitOp::NvReg { reg, mask, or_val } => {
                    if dry_run {
                        ops_executed += 1;
                        continue;
                    }
                    match try_read(&ember, &bdf, *reg) {
                        Ok(val) => {
                            let new_val = (val & mask) | or_val;
                            match ember.mmio_write(&bdf, *reg, new_val) {
                                Ok(_) => ops_executed += 1,
                                Err(e) => {
                                    ops_failed += 1;
                                    if ops_failed <= 10 {
                                        eprintln!("    [{op_idx:3}] RMW WRITE FAILED: reg={reg:#010x} err={e}");
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            ops_failed += 1;
                            if ops_failed <= 10 {
                                eprintln!("    [{op_idx:3}] RMW READ FAILED: reg={reg:#010x} err={e}");
                            }
                        }
                    }
                }
                DevinitOp::ZmMaskAdd { reg, inv_mask, add_val } => {
                    if dry_run {
                        ops_executed += 1;
                        continue;
                    }
                    match try_read(&ember, &bdf, *reg) {
                        Ok(val) => {
                            let new_val = (val & !inv_mask).wrapping_add(*add_val);
                            match ember.mmio_write(&bdf, *reg, new_val) {
                                Ok(_) => ops_executed += 1,
                                Err(e) => {
                                    ops_failed += 1;
                                    if ops_failed <= 10 {
                                        eprintln!("    [{op_idx:3}] MADD WRITE FAILED: err={e}");
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            ops_failed += 1;
                            if ops_failed <= 10 {
                                eprintln!("    [{op_idx:3}] MADD READ FAILED: reg={reg:#010x} err={e}");
                            }
                        }
                    }
                }
                DevinitOp::Time { usec } => {
                    std::thread::sleep(Duration::from_micros(*usec));
                    // After delays, reset circuit breaker in case a PLL change
                    // caused transient BOOT0 faults during the wait period
                    if *usec >= 10 {
                        let _ = ember.mmio_circuit_breaker(&bdf, Some("reset"));
                        breaker_resets += 1;
                    }
                    ops_executed += 1;
                }
            }
        }
        println!("    → {ops_executed} ok, {ops_failed} failed (breaker resets: {breaker_resets})");
    }

    let replay_ok = ops_failed == 0;
    println!("\n  DEVINIT replay: {ops_executed} executed, {ops_failed} failed");
    harness.check_bool("DEVINIT replay completes", ops_executed > 0);
    harness.check_bool("DEVINIT zero write failures", replay_ok);

    // Phase 3: Post-DEVINIT state check
    println!("\n━━━ Phase 3: Post-DEVINIT State ━━━\n");
    std::thread::sleep(Duration::from_millis(100));

    let post_state = read_key_registers(&ember, &bdf);
    print_state("  POST", &post_state);

    let pmc_changed = post_state.pmc_enable != pre_state.pmc_enable;
    println!("  PMC_ENABLE changed: {pmc_changed} ({:#010x} → {:#010x})",
        pre_state.pmc_enable, post_state.pmc_enable);
    harness.check_bool("PMC_ENABLE updated by DEVINIT", pmc_changed);

    // Phase 4: VRAM liveness
    println!("\n━━━ Phase 4: VRAM Liveness ━━━\n");
    let post_vram_dead = is_vram_dead(post_state.pramin);
    let vram_alive = !post_vram_dead;
    println!("  PRAMIN[{PRAMIN_WINDOW:#x}] = {:#010x}", post_state.pramin);
    println!("  VRAM alive: {vram_alive} (was dead: {pre_vram_dead})");
    harness.check_bool("VRAM alive after DEVINIT", vram_alive);

    if vram_alive {
        match ember.pramin_read(&bdf, PRAMIN_WINDOW as u64, 16) {
            Ok(data) => {
                println!("  PRAMIN bulk read: {:02x?}", &data[..data.len().min(16)]);
                harness.check_bool("PRAMIN bulk read succeeds", true);
            }
            Err(e) => {
                println!("  PRAMIN bulk read failed: {e}");
                harness.check_bool("PRAMIN bulk read succeeds", false);
            }
        }
    }

    // Phase 5: PGRAPH / FECS check
    println!("\n━━━ Phase 5: PGRAPH / FECS State ━━━\n");
    let pgraph_ok = post_state.pgraph != 0 && !is_badf(post_state.pgraph);
    println!("  PGRAPH_STATUS: {:#010x} (alive: {pgraph_ok})", post_state.pgraph);
    harness.check_bool("PGRAPH alive after DEVINIT", pgraph_ok);

    let fecs_ok = post_state.fecs_cpuctl != 0 && !is_badf(post_state.fecs_cpuctl);
    println!("  FECS CPUCTL: {:#010x} (responsive: {fecs_ok})", post_state.fecs_cpuctl);
    harness.check_bool("FECS responsive after DEVINIT", fecs_ok);

    // Phase 6: Additional engine checks
    println!("\n━━━ Phase 6: Engine Status ━━━\n");
    let engines = [
        (0x001000, "PBUS_INTR"),
        (0x002100, "PFIFO_INTR"),
        (0x020000, "PTHERM"),
        (0x022004, "PTIMER_TIME_0"),
        (0x100c80, "MMU_PRI_CTRL"),
        (0x409800, "FECS_OS"),
        (0x10a100, "PMU_CPUCTL"),
    ];
    for (reg, name) in engines {
        match ember.mmio_read(&bdf, reg) {
            Ok(r) => {
                let alive = !is_badf(r.value);
                println!("  {name} ({reg:#08x}) = {:#010x} {}", r.value,
                    if alive { "✓" } else { "×" });
            }
            Err(e) => println!("  {name}: read failed: {e}"),
        }
    }

    // Cleanup
    println!("\n━━━ Cleanup ━━━\n");
    println!("  PMC_ENABLE final: {last_pmc_enable:#010x}");

    harness.finish();
}

#[derive(Debug)]
struct GpuState {
    boot0: u32,
    pmc_enable: u32,
    pgraph: u32,
    pramin: u32,
    fecs_cpuctl: u32,
}

fn read_key_registers(ember: &EmberClient, bdf: &str) -> GpuState {
    let read = |reg: u32| -> u32 {
        ember.mmio_read(bdf, reg).map(|r| r.value).unwrap_or(0xDEAD_DEAD)
    };
    GpuState {
        boot0: read(PMC_BOOT0),
        pmc_enable: read(PMC_ENABLE),
        pgraph: read(PGRAPH_STATUS),
        pramin: read(PRAMIN_WINDOW),
        fecs_cpuctl: read(FECS_CPUCTL),
    }
}

fn print_state(prefix: &str, s: &GpuState) {
    println!("{prefix} BOOT0:      {:#010x}", s.boot0);
    println!("{prefix} PMC_ENABLE: {:#010x}", s.pmc_enable);
    println!("{prefix} PGRAPH:     {:#010x}", s.pgraph);
    println!("{prefix} PRAMIN:     {:#010x}", s.pramin);
    println!("{prefix} FECS_CPUCTL:{:#010x}", s.fecs_cpuctl);
}

fn is_vram_dead(val: u32) -> bool {
    VRAM_DEAD_PATTERNS
        .iter()
        .any(|&p| val & 0xFFFF_FF00 == p & 0xFFFF_FF00)
}

fn is_badf(val: u32) -> bool {
    val & 0xBADF_0000 == 0xBADF_0000
}

// --- Recipe loading ---

#[derive(Debug, serde::Deserialize)]
struct DevinitScript {
    #[allow(dead_code)]
    id: Option<u32>,
    #[allow(dead_code)]
    addr: String,
    ops: Vec<RawDevinitOp>,
}

#[derive(Debug, serde::Deserialize)]
struct RawDevinitOp {
    #[serde(rename = "type")]
    op_type: String,
    #[serde(default)]
    reg: u32,
    #[serde(default)]
    val: u32,
    #[serde(default)]
    mask: u32,
    #[serde(default)]
    or_val: u32,
    #[serde(default, alias = "add")]
    add_val: u32,
    #[serde(default, alias = "mask")]
    inv_mask: u32,
    #[serde(default)]
    usec: u64,
}

enum DevinitOp {
    ZmReg { reg: u32, val: u32 },
    NvReg { reg: u32, mask: u32, or_val: u32 },
    ZmMaskAdd { reg: u32, inv_mask: u32, add_val: u32 },
    Time { usec: u64 },
}

impl From<&RawDevinitOp> for DevinitOp {
    fn from(raw: &RawDevinitOp) -> Self {
        match raw.op_type.as_str() {
            "ZM_REG" | "ZmReg" => DevinitOp::ZmReg { reg: raw.reg, val: raw.val },
            "NV_REG" | "NvReg" => DevinitOp::NvReg { reg: raw.reg, mask: raw.mask, or_val: raw.or_val },
            "ZM_MASK_ADD" | "ZmMaskAdd" => DevinitOp::ZmMaskAdd {
                reg: raw.reg,
                inv_mask: if raw.inv_mask != 0 { raw.inv_mask } else { raw.mask },
                add_val: if raw.add_val != 0 { raw.add_val } else { raw.val },
            },
            "Time" => DevinitOp::Time { usec: raw.usec },
            other => {
                eprintln!("WARNING: unknown op type: {other}");
                DevinitOp::Time { usec: 0 }
            }
        }
    }
}

fn load_recipe(path: &str) -> Option<Vec<DevinitScript>> {
    let data = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

fn connect_ember(bdf: &str) -> Option<EmberClient> {
    let slug = bdf.replace(':', "-");
    let candidates = [
        format!("/run/coralreef/fleet/ember-{slug}.sock"),
        "/run/coralreef/ember.sock".to_string(),
    ];
    for c in &candidates {
        if Path::new(c).exists() {
            let client = EmberClient::connect(c);
            if client.mmio_read(bdf, 0).is_ok() {
                return Some(client);
            }
        }
    }

    let fleet_path = FleetDiscovery::resolve_path();
    if let Ok(fleet) = FleetDiscovery::load(&fleet_path) {
        if let Some(r) = fleet.file().devices.iter().find(|d| d.bdf == bdf) {
            if let Some(sock) = &r.socket {
                let client = EmberClient::connect(sock);
                if client.mmio_read(bdf, 0).is_ok() {
                    return Some(client);
                }
            }
        }
    }
    None
}

fn extract_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
