// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 154: Titan V SEC2/ACR Pipeline (PMU-First Hypothesis)
//!
//! Routes the full SEC2/ACR boot pipeline through ember IPC, testing
//! hypothesis A from Experiment 151: the BL's HS authentication needs
//! keys/signatures that depend on prior PMU initialization.
//!
//! ## Pipeline (all through ember IPC)
//!
//! 1. `prepare_dma` — quiesce + AER mask + bus master
//! 2. `sec2.prepare_physical` — PMC reset, instance bind, PHYS_VID
//! 3. PMU-first: upload PMU firmware → start → poll
//! 4. SEC2 ACR: upload SEC2 BL → start → poll
//! 5. DMEM readback for forensics (hypothesis D)
//! 6. BROM register comparison
//! 7. `cleanup_dma` — decontaminate
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --bin exp154_sec2_acr_pipeline -- [--bdf 0000:03:00.0]
//! ```

use std::path::Path;

use hotspring_barracuda::ember_types::MmioBatchOp;
use hotspring_barracuda::fleet_client::{EmberClient, FleetDiscovery};
use hotspring_barracuda::glowplug_client::GlowplugClient;
use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::validation::ValidationHarness;

// GV100 falcon engine bases (BAR0 offsets)
const SEC2_BASE: u32 = 0x087000;
const PMU_BASE: u32 = 0x10A000;
// Key BROM / PMU registers
const BROM_MODSEL: u32 = 0x300200;
const BROM_UCODEID: u32 = 0x300204;
const BROM_ENGID_MASK: u32 = 0x300208;
const BROM_PARAADDR0: u32 = 0x300400;

// SEC2 registers (falcon at 0x087000)
const SEC2_CPUCTL: u32 = 0x087100;
const SEC2_SCTL: u32 = 0x087240;
const SEC2_PC: u32 = 0x087110;

// PMU registers
const PMU_CPUCTL: u32 = 0x10A100;
const PMU_SCTL: u32 = 0x10A240;
const PMU_MAILBOX0: u32 = 0x10A040;
const PMU_PC: u32 = 0x10A110;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  SEC2/ACR Pipeline — Experiment 154 (PMU-First)            ║");
    println!("║  Titan V (GV100) sovereign boot via ember IPC              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut harness = ValidationHarness::new("exp154_sec2_acr_pipeline");

    let args: Vec<String> = std::env::args().collect();
    let bdf = extract_arg(&args, "--bdf").unwrap_or_else(|| "0000:03:00.0".to_string());
    println!("  Target BDF: {bdf}\n");

    let glowplug = connect_glowplug();
    let ember = connect_ember(&bdf);

    let Some(glowplug) = glowplug else {
        harness.check_bool("glowplug reachable", false);
        harness.finish();
    };
    harness.check_bool("glowplug reachable", true);

    let Some(ember) = ember else {
        harness.check_bool("ember reachable for Titan V", false);
        harness.finish();
    };
    harness.check_bool("ember reachable for Titan V", true);

    // Mark experiment start
    if let Err(e) = glowplug.experiment_start(&bdf) {
        println!("  experiment_start warning: {e}");
    }

    // ── Phase 1: BROM baseline (before any boot attempts) ──
    println!("━━━ Phase 1: BROM + Engine Baseline ━━━\n");
    phase1_baseline(&mut harness, &ember, &bdf);

    // ── Phase 2: DMA prepare + SEC2 physical prepare ──
    println!("\n━━━ Phase 2: DMA + SEC2 Prepare Physical ━━━\n");
    let prepared = phase2_prepare(&mut harness, &ember, &bdf);

    // ── Phase 3: PMU-first boot (hypothesis A) ──
    // IMEM upload is pure PIO to falcon registers — no VRAM needed.
    // Proceed even when SEC2 prepare's VRAM check fails (cold HBM2).
    println!("\n━━━ Phase 3: PMU-First Boot (Hypothesis A) ━━━\n");
    if !prepared {
        println!("  NOTE: SEC2 prepare failed (VRAM dead) — proceeding with IMEM-only path");
    }
    phase3_pmu_first(&mut harness, &ember, &bdf);

    // ── Phase 4: SEC2 ACR boot ──
    println!("\n━━━ Phase 4: SEC2 ACR Boot ━━━\n");
    if !prepared {
        println!(
            "  NOTE: SEC2 prepare failed — IMEM-only (no DMA/PRAMIN), results are best-effort"
        );
    }
    phase4_sec2_acr(&mut harness, &ember, &bdf);

    // ── Phase 5: Post-boot BROM comparison ──
    println!("\n━━━ Phase 5: Post-Boot BROM Comparison ━━━\n");
    phase5_brom_compare(&mut harness, &ember, &bdf);

    // ── Phase 6: DMEM forensics (hypothesis D) ──
    println!("\n━━━ Phase 6: DMEM Forensics ━━━\n");
    phase6_dmem_forensics(&mut harness, &ember, &bdf);

    // ── Cleanup: halt falcons + quiesce before warm cycle ──
    println!("\n━━━ Cleanup ━━━\n");

    // Halt any falcons we started to prevent runaway execution during warm cycle.
    // Write CPUCTL.HALT (bit 4) to PMU and SEC2 to force halt before unbinding.
    let halt_ops = vec![
        MmioBatchOp::write(PMU_CPUCTL, 0x10),  // PMU: force halt
        MmioBatchOp::write(SEC2_CPUCTL, 0x10), // SEC2: force halt
        MmioBatchOp::read(PMU_CPUCTL),         // readback verify
        MmioBatchOp::read(SEC2_CPUCTL),        // readback verify
    ];
    match ember.mmio_batch(&bdf, &halt_ops) {
        Ok(r) => {
            let pmu = r.read_value(2).unwrap_or(0);
            let sec2 = r.read_value(3).unwrap_or(0);
            println!("  Falcon halt: PMU_CPUCTL={pmu:#010x} SEC2_CPUCTL={sec2:#010x}");
        }
        Err(e) => println!("  Falcon halt batch failed: {e}"),
    }

    match ember.cleanup_dma(&bdf) {
        Ok(r) => println!(
            "  DMA cleanup: ok={}, decontaminated={:?}",
            r.ok, r.decontaminated
        ),
        Err(e) => println!("  DMA cleanup error: {e}"),
    }
    if glowplug.experiment_end(&bdf).is_ok() {
        println!("  Experiment ended for {bdf}");
    }

    println!();
    harness.finish();
}

fn phase1_baseline(harness: &mut ValidationHarness, ember: &EmberClient, bdf: &str) {
    let ops = vec![
        MmioBatchOp::read(0x000000), // BOOT0
        MmioBatchOp::read(0x000200), // PMC_ENABLE
        MmioBatchOp::read(BROM_MODSEL),
        MmioBatchOp::read(BROM_UCODEID),
        MmioBatchOp::read(BROM_ENGID_MASK),
        MmioBatchOp::read(BROM_PARAADDR0),
        MmioBatchOp::read(SEC2_CPUCTL),
        MmioBatchOp::read(SEC2_SCTL),
        MmioBatchOp::read(SEC2_PC),
        MmioBatchOp::read(PMU_CPUCTL),
        MmioBatchOp::read(PMU_PC),
    ];

    match ember.mmio_batch(bdf, &ops) {
        Ok(result) => {
            println!("  Engine baseline (pre-boot):");
            for (i, op) in ops.iter().enumerate() {
                let val = result.read_value(i).unwrap_or(0xDEAD_DEAD);
                let name = offset_label(op.offset);
                println!("    [{:#08x}] {:<20} = {val:#010x}", op.offset, name,);
            }
            harness.check_bool("pre-boot MMIO baseline readable", true);
        }
        Err(e) => {
            println!("  MMIO batch failed: {e}");
            harness.check_bool("pre-boot MMIO baseline readable", false);
        }
    }
}

fn phase2_prepare(harness: &mut ValidationHarness, ember: &EmberClient, bdf: &str) -> bool {
    // DMA prepare (bus_master=true for DMA-capable ACR path)
    let dma_ok = match ember.prepare_dma(bdf, true) {
        Ok(r) => {
            println!(
                "  DMA prepare: pmc_before={:?}, pmc_after={:?}",
                r.pmc_before, r.pmc_after
            );
            harness.check_bool("DMA prepare", r.ok);
            r.ok
        }
        Err(e) => {
            println!("  DMA prepare failed: {e}");
            harness.check_bool("DMA prepare", false);
            false
        }
    };

    if !dma_ok {
        return false;
    }

    // SEC2 physical prepare (PMC reset, instance bind, PHYS_VID)
    match ember.sec2_prepare_physical(bdf) {
        Ok(r) => {
            println!("  SEC2 prepare: ok={}", r.ok);
            for note in &r.notes {
                println!("    {note}");
            }
            harness.check_bool("SEC2 prepare_physical", r.ok);
            r.ok
        }
        Err(e) => {
            println!("  SEC2 prepare failed: {e}");
            harness.check_bool("SEC2 prepare_physical", false);
            false
        }
    }
}

fn phase3_pmu_first(harness: &mut ValidationHarness, ember: &EmberClient, bdf: &str) {
    // Minimal PMU micro-firmware: NOP sled + mailbox write + halt
    let pmu_fw = build_probe_firmware();
    println!(
        "  Uploading PMU firmware ({} bytes) at base {PMU_BASE:#x}...",
        pmu_fw.len()
    );

    let upload_ok = match ember.falcon_upload_imem(bdf, PMU_BASE, 0, &pmu_fw, 0, false) {
        Ok(r) => {
            println!("  PMU IMEM upload: ok={}, bytes={:?}", r.ok, r.bytes);
            harness.check_bool("PMU IMEM upload", r.ok);
            r.ok
        }
        Err(e) => {
            println!("  PMU IMEM upload failed: {e}");
            harness.check_bool("PMU IMEM upload", false);
            false
        }
    };

    if !upload_ok {
        return;
    }

    // Start PMU falcon
    match ember.falcon_start_cpu(bdf, PMU_BASE) {
        Ok(r) => {
            println!(
                "  PMU start: ok={}, pc={:?}, exci={:?}",
                r.ok,
                r.pc.map(|v| format!("{v:#010x}")),
                r.exci.map(|v| format!("{v:#010x}")),
            );
            harness.check_bool("PMU falcon start_cpu", r.ok);
        }
        Err(e) => {
            println!("  PMU start failed: {e}");
            harness.check_bool("PMU falcon start_cpu", false);
            return;
        }
    }

    // Poll PMU
    match ember.falcon_poll(bdf, PMU_BASE, 5000, 0xDEAD_A5A5) {
        Ok(r) => {
            println!("  PMU poll: {} snapshots", r.snapshots.len());
            if let Some(f) = &r.final_state {
                println!(
                    "  PMU final: pc={:?}, mb0={:?}",
                    f.pc.map(|v| format!("{v:#010x}")),
                    f.mailbox0.map(|v| format!("{v:#010x}")),
                );
            }
            harness.check_bool("PMU falcon poll completed", true);
        }
        Err(e) => {
            println!("  PMU poll failed: {e}");
            harness.check_bool("PMU falcon poll completed", false);
        }
    }

    // Read PMU state after boot
    let ops = vec![
        MmioBatchOp::read(PMU_CPUCTL),
        MmioBatchOp::read(PMU_SCTL),
        MmioBatchOp::read(PMU_MAILBOX0),
        MmioBatchOp::read(PMU_PC),
    ];
    if let Ok(r) = ember.mmio_batch(bdf, &ops) {
        println!("  PMU post-boot state:");
        for (i, op) in ops.iter().enumerate() {
            let val = r.read_value(i).unwrap_or(0xDEAD_DEAD);
            println!("    [{:#08x}] = {val:#010x}", op.offset);
        }
    }
}

fn phase4_sec2_acr(harness: &mut ValidationHarness, ember: &EmberClient, bdf: &str) {
    let sec2_fw = build_probe_firmware();
    println!(
        "  Uploading SEC2 BL ({} bytes) at base {SEC2_BASE:#x}...",
        sec2_fw.len()
    );

    let upload_ok = match ember.falcon_upload_imem(bdf, SEC2_BASE, 0, &sec2_fw, 0, true) {
        Ok(r) => {
            println!("  SEC2 IMEM upload: ok={}, bytes={:?}", r.ok, r.bytes);
            harness.check_bool("SEC2 IMEM upload (secure)", r.ok);
            r.ok
        }
        Err(e) => {
            println!("  SEC2 IMEM upload failed: {e}");
            harness.check_bool("SEC2 IMEM upload (secure)", false);
            false
        }
    };

    if !upload_ok {
        return;
    }

    match ember.falcon_start_cpu(bdf, SEC2_BASE) {
        Ok(r) => {
            println!(
                "  SEC2 start: ok={}, pc={:?}, exci={:?}, cpuctl={:?}",
                r.ok,
                r.pc.map(|v| format!("{v:#010x}")),
                r.exci.map(|v| format!("{v:#010x}")),
                r.cpuctl.map(|v| format!("{v:#010x}")),
            );
            harness.check_bool("SEC2 falcon start_cpu", r.ok);
        }
        Err(e) => {
            println!("  SEC2 start failed: {e}");
            harness.check_bool("SEC2 falcon start_cpu", false);
            return;
        }
    }

    // Poll SEC2 with longer timeout (ACR may take time)
    match ember.falcon_poll(bdf, SEC2_BASE, 10000, 0xDEAD_A5A5) {
        Ok(r) => {
            println!(
                "  SEC2 poll: {} snapshots, pc_trace len={}",
                r.snapshots.len(),
                r.pc_trace.len()
            );
            if let Some(f) = &r.final_state {
                println!(
                    "  SEC2 final: pc={:?}, sctl={:?}, mb0={:?}",
                    f.pc.map(|v| format!("{v:#010x}")),
                    f.sctl.map(|v| format!("{v:#010x}")),
                    f.mailbox0.map(|v| format!("{v:#010x}")),
                );
                let hs_reached = f.sctl.map_or(false, |s| s & 0x4000 != 0);
                println!("  HS mode reached: {hs_reached}");
                harness.check_bool("SEC2 HS mode reached", hs_reached);
            }
            harness.check_bool("SEC2 falcon poll completed", true);
        }
        Err(e) => {
            println!("  SEC2 poll failed: {e}");
            harness.check_bool("SEC2 falcon poll completed", false);
        }
    }
}

fn phase5_brom_compare(harness: &mut ValidationHarness, ember: &EmberClient, bdf: &str) {
    let ops = vec![
        MmioBatchOp::read(BROM_MODSEL),
        MmioBatchOp::read(BROM_UCODEID),
        MmioBatchOp::read(BROM_ENGID_MASK),
        MmioBatchOp::read(BROM_PARAADDR0),
    ];

    match ember.mmio_batch(bdf, &ops) {
        Ok(r) => {
            println!("  BROM state (post-boot):");
            let mut all_badf = true;
            for (i, op) in ops.iter().enumerate() {
                let val = r.read_value(i).unwrap_or(0xDEAD_DEAD);
                let name = offset_label(op.offset);
                println!("    [{:#08x}] {:<20} = {val:#010x}", op.offset, name);
                if val != 0xBADF_5040 {
                    all_badf = false;
                }
            }
            println!(
                "  All BROM registers = 0xBADF5040 (uninitialized): {}",
                all_badf
            );
            harness.check_bool("BROM post-boot state captured", true);
        }
        Err(e) => {
            println!("  BROM read failed: {e}");
            harness.check_bool("BROM post-boot state captured", false);
        }
    }
}

fn phase6_dmem_forensics(harness: &mut ValidationHarness, ember: &EmberClient, bdf: &str) {
    // Read SEC2 DMEM via PRAMIN to inspect what the BL had access to
    // SEC2 DMEM is accessed through the falcon's dedicated DMEM window
    let sec2_dmem_ops = vec![
        MmioBatchOp::read(SEC2_BASE + 0x1C0), // DMEMC (DMEM control)
        MmioBatchOp::read(SEC2_BASE + 0x1C4), // DMEMD (DMEM data)
    ];

    match ember.mmio_batch(bdf, &sec2_dmem_ops) {
        Ok(r) => {
            println!("  SEC2 DMEM registers:");
            for (i, op) in sec2_dmem_ops.iter().enumerate() {
                let val = r.read_value(i).unwrap_or(0xDEAD_DEAD);
                println!("    [{:#08x}] = {val:#010x}", op.offset);
            }
            harness.check_bool("SEC2 DMEM forensics captured", true);
        }
        Err(e) => {
            println!("  SEC2 DMEM read failed: {e}");
            harness.check_bool("SEC2 DMEM forensics captured", false);
        }
    }
}

/// Minimal probe firmware: 16 NOPs + mailbox sentinel write + halt.
fn build_probe_firmware() -> Vec<u8> {
    let mut fw = Vec::new();
    for _ in 0..16 {
        fw.extend_from_slice(&0x0000_0000u32.to_le_bytes());
    }
    fw.extend_from_slice(&0xF800_0000u32.to_le_bytes());
    fw
}

fn offset_label(offset: u32) -> &'static str {
    match offset {
        0x000000 => "BOOT0",
        0x000200 => "PMC_ENABLE",
        0x300200 => "BROM_MODSEL",
        0x300204 => "BROM_UCODEID",
        0x300208 => "BROM_ENGID_MASK",
        0x300400 => "BROM_PARAADDR0",
        0x840100 => "SEC2_CPUCTL",
        0x840240 => "SEC2_SCTL",
        0x840040 => "SEC2_MB0",
        0x840110 => "SEC2_PC",
        0x10A100 => "PMU_CPUCTL",
        0x10A240 => "PMU_SCTL",
        0x10A040 => "PMU_MB0",
        0x10A110 => "PMU_PC",
        _ => "?",
    }
}

fn connect_glowplug() -> Option<GlowplugClient> {
    let nucleus = NucleusContext::detect();
    match GlowplugClient::from_nucleus(&nucleus) {
        Ok(g) => Some(g),
        Err(_) => {
            let sock = Path::new("/run/coralreef/glowplug.sock");
            if sock.exists() {
                Some(GlowplugClient::from_socket(sock))
            } else {
                None
            }
        }
    }
}

fn connect_ember(bdf: &str) -> Option<EmberClient> {
    if let Ok(disc) = FleetDiscovery::load_default() {
        if let Some(sock) = disc.file().routes.get(bdf) {
            return Some(EmberClient::connect(sock));
        }
    }
    let slug = bdf.replace(':', "-");
    let fleet_sock = format!("/run/coralreef/fleet/ember-{slug}.sock");
    if Path::new(&fleet_sock).exists() {
        return Some(EmberClient::connect(&fleet_sock));
    }
    let per_device = format!("/run/coralreef/ember-{slug}.sock");
    if Path::new(&per_device).exists() {
        return Some(EmberClient::connect(&per_device));
    }
    let legacy = Path::new("/run/coralreef/ember.sock");
    if legacy.exists() {
        return Some(EmberClient::connect(legacy));
    }
    None
}

fn extract_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}
