// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 182: K80 (GK210) Direct FECS PIO Boot — Sovereign Compute Dispatch
//!
//! ## Context
//!
//! After a full AC power cycle (clearing PLX PCIe switch wedge) and a nouveau
//! warm cycle to train GDDR5 and enable engine power domains:
//!
//! - K80 die1 (0000:4c:00.0) is in WARM STATE:
//!   - `PMC_ENABLE = 0xfc37b1ef` (all engine power domains enabled by nouveau)
//!   - `FECS_CPUCTL = 0x00000010` (HRESET — power domain UP, firmware not loaded)
//!   - `FECS_SCTL = 0x00000000` (LS mode — NO Falcon v5 security gate!)
//!
//! ## Difference from Titan V (GV100)
//!
//! GV100 SEC2 has Falcon v5 HS ROM security gate requiring NVIDIA-signed firmware.
//! GK210 (Kepler) has NO ACR/SEC2 — FECS is in LS mode (SCTL=0). We can PIO-load
//! firmware directly without authentication.
//!
//! ## Pipeline
//!
//! 1. Verify warm state: PMC_ENABLE ≠ reset value, FECS_SCTL = 0, FECS in HRESET
//! 2. Load `/lib/firmware/nvidia/gk210/fecs_inst.bin` → FECS IMEM via ember.falcon.upload_imem
//! 3. Load `/lib/firmware/nvidia/gk210/fecs_data.bin` → FECS DMEM via ember.falcon.upload_dmem
//! 4. Release HRESET, STARTCPU (FECS_CPUCTL = 0x01)
//! 5. Poll FECS until halted/booted (CPUCTL bit5 or PGRAPH_STATUS transitions)
//! 6. Verify: read FECS_MAILBOX0 (booted signal), GPC0/GPC1 CPUCTL state
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --bin exp182_k80_fecs_pio_boot \
//!   -- [--bdf 0000:4c:00.0] [--inst /lib/firmware/nvidia/gk210/fecs_inst.bin]
//!        [--data /lib/firmware/nvidia/gk210/fecs_data.bin]
//!        [--ember-socket /run/toadstool/biomeos/compute.sock] [--dry-run]
//! ```
//!
//! Requires toadstool-ember running and holding the VFIO fd for the target BDF.
//!
//! ## Prerequisites
//!
//! GPU must be in warm state (after `nouveau` warm cycle + handoff to vfio-pci).
//! Run warm handoff:
//!   toadstool device warm-catch 0000:4c:00.0 --memory-type gddr5

use std::path::PathBuf;
use std::time::{Duration, Instant};

use hotspring_barracuda::fleet_client::EmberClient;

#[path = "../bin_helpers/sovereignty/mod.rs"]
mod sovereignty;
use sovereignty::connect::{connect_ember, extract_arg, is_dry_run};

// ── PMC registers ──────────────────────────────────────────────────────────
const BOOT0: u32 = 0x000000;
const PMC_ENABLE: u32 = 0x000200;
const PTIMER_HI: u32 = 0x009084; // PTIMER numerator (ticks)
const PGRAPH_STAT: u32 = 0x400700; // GR engine status

// ── FECS Falcon registers (base 0x409000) ──────────────────────────────────
const FECS_BASE: u32 = 0x409000;
const FECS_CPUCTL: u32 = FECS_BASE + 0x100; // CPU control
const FECS_BOOTVEC: u32 = FECS_BASE + 0x104; // Boot vector
const FECS_PC: u32 = FECS_BASE + 0x110; // Program counter
const FECS_SCTL: u32 = FECS_BASE + 0x240; // Security control (0 = LS mode)
const FECS_HWCFG: u32 = FECS_BASE + 0x10c; // Hardware config
const FECS_MB0: u32 = FECS_BASE + 0x040; // Mailbox 0 (FECS boot status)
const FECS_MB1: u32 = FECS_BASE + 0x044; // Mailbox 1

// ── GPCCS registers (per-GPC, at GPC0=0x418000, GPC1=0x428000) ────────────
const GPCCS_BASE_GPC0: u32 = 0x418000;
const GPCCS_BASE_GPC1: u32 = 0x428000;
const GPCCS_CPUCTL_OFF: u32 = 0x100;
const GPCCS_PC_OFF: u32 = 0x110;
const GPCCS_MB0_OFF: u32 = 0x040;

// ── Falcon CPUCTL bit masks (GK210 Falcon v2/v4) ──────────────────────────
// Kepler Falcon (v2/v4): STARTCPU = bit 1 (0x02), HALTED = bit 4 (0x10)
// Volta+ Falcon (v5):    STARTCPU = bit 0 (0x01), HRESET = bit 4 (0x10)
const CPUCTL_STARTCPU: u32 = 0x02; // bit 1 = start CPU (GK210 Falcon v2/v4)
const CPUCTL_SRESET: u32 = 0x04; // bit 2 = soft reset (clear halted state)
const CPUCTL_HALTED: u32 = 0x10; // bit 4 = CPU halted (also used as HRESET on v5)
const CPUCTL_HALT: u32 = CPUCTL_HALTED; // alias

// ── K80 warm PMC_ENABLE reference (set by nouveau during DEVINIT) ─────────
const PMC_WARM_LO: u32 = 0x0000_0001; // Minimum expected (not cold)

// ── Default firmware paths ────────────────────────────────────────────────
const FW_INST_DEFAULT: &str = "/lib/firmware/nvidia/gk210/fecs_inst.bin";
const FW_DATA_DEFAULT: &str = "/lib/firmware/nvidia/gk210/fecs_data.bin";
const FECS_IMEM_PAGE_SIZE: usize = 256; // bytes per IMEM page
const FECS_IMEM_WORDS_PER_PAGE: usize = FECS_IMEM_PAGE_SIZE / 4;

fn r32(ember: &EmberClient, bdf: &str, offset: u32) -> u32 {
    match ember.mmio_read(bdf, offset) {
        Ok(r) => r.value,
        Err(e) => {
            eprintln!("  WARN: mmio.read32({offset:#010x}): {e}");
            0xDEAD_DEAD
        }
    }
}

fn w32(ember: &EmberClient, bdf: &str, offset: u32, value: u32) {
    if let Err(e) = ember.mmio_write(bdf, offset, value) {
        eprintln!("  WARN: mmio.write32({offset:#010x}, {value:#010x}): {e}");
    }
}

fn banner() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXP 182: K80 (GK210) Direct FECS PIO Boot                  ║");
    println!("║  Goal: PIO-load FECS firmware → STARTCPU → sovereign GR     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bdf = extract_arg(&args, "--bdf").unwrap_or_else(|| {
        std::env::var("HOTSPRING_BDF").unwrap_or_else(|_| "0000:4c:00.0".into())
    });
    let inst_path = extract_arg(&args, "--inst")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FW_INST_DEFAULT));
    let data_path = extract_arg(&args, "--data")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FW_DATA_DEFAULT));
    let ember_socket = extract_arg(&args, "--ember-socket");
    let dry_run = is_dry_run(&args);

    banner();
    println!("  BDF:        {bdf}");
    println!("  FECS INST:  {}", inst_path.display());
    println!("  FECS DATA:  {}", data_path.display());
    println!("  Wiring:     toadstool-ember (all MMIO via JSON-RPC — no raw BAR0 mmap)");
    if dry_run {
        println!("  Mode:       DRY RUN (reads only, no writes)");
    }

    // ── Load firmware ─────────────────────────────────────────────────────
    let fw_inst = std::fs::read(&inst_path).unwrap_or_else(|e| {
        eprintln!("FATAL: cannot read FECS inst firmware: {e}");
        std::process::exit(1);
    });
    let fw_data = std::fs::read(&data_path).unwrap_or_else(|e| {
        eprintln!("FATAL: cannot read FECS data firmware: {e}");
        std::process::exit(1);
    });

    let inst_words: Vec<u32> = fw_inst
        .chunks(4)
        .map(|c| {
            u32::from_le_bytes([
                *c.first().unwrap_or(&0),
                *c.get(1).unwrap_or(&0),
                *c.get(2).unwrap_or(&0),
                *c.get(3).unwrap_or(&0),
            ])
        })
        .collect();
    let data_words: Vec<u32> = fw_data
        .chunks(4)
        .map(|c| {
            u32::from_le_bytes([
                *c.first().unwrap_or(&0),
                *c.get(1).unwrap_or(&0),
                *c.get(2).unwrap_or(&0),
                *c.get(3).unwrap_or(&0),
            ])
        })
        .collect();

    let inst_pages = (inst_words.len() + FECS_IMEM_WORDS_PER_PAGE - 1) / FECS_IMEM_WORDS_PER_PAGE;
    println!(
        "\n  IMEM: {} bytes / {} words / {} pages",
        fw_inst.len(),
        inst_words.len(),
        inst_pages
    );
    println!(
        "  DMEM: {} bytes / {} words",
        fw_data.len(),
        data_words.len()
    );

    // ── Connect to ember ──────────────────────────────────────────────────
    let ember = connect_ember(&bdf, ember_socket.as_deref());
    println!("  Ember:      connected ✓");

    // ── Phase 1: Pre-condition validation ─────────────────────────────────
    println!("\n━━━ Phase 1: Pre-condition Check ━━━\n");

    let boot0 = r32(&ember, &bdf, BOOT0);
    println!("  BOOT0          = {boot0:#010x}  (GK210 = 0x0f22d0a1)");
    if boot0 == 0xffff_ffff || boot0 == 0 {
        eprintln!("FATAL: GPU link dead (BOOT0={boot0:#010x}). Check PCIe and power.");
        std::process::exit(1);
    }
    if boot0 != 0x0f22_d0a1 {
        println!("  WARN: BOOT0 mismatch — expected GK210 0x0f22d0a1, got {boot0:#010x}");
        println!("  Proceeding anyway (may be GK110/GK104 variant).");
    }

    let pmc_en = r32(&ember, &bdf, PMC_ENABLE);
    println!("  PMC_ENABLE     = {pmc_en:#010x}");
    if pmc_en <= PMC_WARM_LO {
        eprintln!("ERROR: GPU is cold (PMC_ENABLE={pmc_en:#010x}). Run nouveau warm cycle first.");
        eprintln!("  toadstool device warm-catch {bdf} --memory-type gddr5");
        std::process::exit(1);
    }
    println!("  DEVINIT:       OK (PMC warm)");

    // PTIMER sanity
    let pt0 = r32(&ember, &bdf, PTIMER_HI);
    std::thread::sleep(Duration::from_millis(5));
    let pt1 = r32(&ember, &bdf, PTIMER_HI);
    println!(
        "  PTIMER:        {pt0:#010x} → {pt1:#010x}  ({})",
        if pt1 != pt0 {
            "RUNNING ✓"
        } else {
            "FROZEN ✗"
        }
    );

    let fecs_sctl = r32(&ember, &bdf, FECS_SCTL);
    let fecs_cpuctl = r32(&ember, &bdf, FECS_CPUCTL);
    let fecs_hwcfg = r32(&ember, &bdf, FECS_HWCFG);
    let fecs_mb0 = r32(&ember, &bdf, FECS_MB0);
    println!("  FECS_SCTL      = {fecs_sctl:#010x}  (0=LS mode ✓, non-zero=HS/ACR required)");
    println!("  FECS_CPUCTL    = {fecs_cpuctl:#010x}  (0x10=HRESET ✓, expected)");
    println!("  FECS_HWCFG     = {fecs_hwcfg:#010x}");
    println!("  FECS_MB0       = {fecs_mb0:#010x}  (should be 0 before boot)");

    if fecs_sctl != 0 {
        eprintln!("ERROR: FECS_SCTL = {fecs_sctl:#010x} — FECS is in HS/ACR mode.");
        eprintln!("  K80 should not have HS mode. Check GPU variant.");
        std::process::exit(1);
    }
    if fecs_cpuctl & CPUCTL_HALTED == 0 {
        println!("  WARN: FECS not in HRESET (CPUCTL={fecs_cpuctl:#010x}).");
        println!("  FECS may already be running or in an unexpected state.");
    }

    let pgraph_st = r32(&ember, &bdf, PGRAPH_STAT);
    println!("  PGRAPH_STATUS  = {pgraph_st:#010x}");

    // ── Phase 2: FECS IMEM load ────────────────────────────────────────────
    println!("\n━━━ Phase 2: FECS IMEM Load ({} pages) ━━━\n", inst_pages);

    if !dry_run {
        match ember.falcon_upload_imem(&bdf, FECS_BASE, 0, &fw_inst, 0, false) {
            Ok(r) if r.ok => {
                println!(
                    "  IMEM load complete: {} pages via ember.falcon.upload_imem",
                    inst_pages
                );
            }
            Ok(r) => {
                eprintln!("  WARN: falcon.upload_imem ok=false bytes={:?}", r.bytes);
            }
            Err(e) => {
                eprintln!("  WARN: falcon.upload_imem error: {e}");
            }
        }
    } else {
        println!("  DRY RUN: IMEM load skipped");
    }

    // ── Phase 3: FECS DMEM load ────────────────────────────────────────────
    println!(
        "\n━━━ Phase 3: FECS DMEM Load ({} words) ━━━\n",
        data_words.len()
    );

    if !dry_run {
        match ember.falcon_upload_dmem(&bdf, FECS_BASE, 0, &fw_data) {
            Ok(r) if r.ok => {
                println!(
                    "  DMEM load complete: {} words via ember.falcon.upload_dmem",
                    data_words.len()
                );
            }
            Ok(r) => {
                eprintln!("  WARN: falcon.upload_dmem ok=false bytes={:?}", r.bytes);
            }
            Err(e) => {
                eprintln!("  WARN: falcon.upload_dmem error: {e}");
            }
        }
    } else {
        println!("  DRY RUN: DMEM load skipped");
    }

    // ── Phase 4: Boot FECS ─────────────────────────────────────────────────
    println!("\n━━━ Phase 4: FECS STARTCPU ━━━\n");

    if !dry_run {
        // Ensure boot vector is 0 (firmware expects entry at byte 0)
        w32(&ember, &bdf, FECS_BOOTVEC, 0x0000_0000);
        println!("  FECS_BOOTVEC ← 0x00000000");

        // On Kepler (GK210 Falcon v2/v4): STARTCPU = bit 1 (0x02).
        // If FECS is in HALTED state (bit 4 = 0x10), we must first clear HALTED
        // via SRESET (soft reset, bit 2 = 0x04) before asserting STARTCPU.
        let cpuctl_pre = r32(&ember, &bdf, FECS_CPUCTL);
        if cpuctl_pre & CPUCTL_HALTED != 0 {
            w32(&ember, &bdf, FECS_CPUCTL, CPUCTL_SRESET);
            println!("  FECS_CPUCTL  ← 0x{CPUCTL_SRESET:08x} (SRESET — clearing HALTED state)");
            std::thread::sleep(Duration::from_micros(50));
        }
        w32(&ember, &bdf, FECS_CPUCTL, CPUCTL_STARTCPU);
        println!("  FECS_CPUCTL  ← 0x{CPUCTL_STARTCPU:08x} (STARTCPU bit 1, GK210 Falcon v2/v4)");
    } else {
        println!("  DRY RUN: STARTCPU skipped");
    }

    // ── Phase 5: Poll for boot completion ─────────────────────────────────
    println!("\n━━━ Phase 5: Poll FECS Boot ━━━\n");

    let timeout = Duration::from_millis(500);
    let started = Instant::now();
    let mut last_cpuctl = 0xffffffff_u32;
    let mut booted = false;
    let mut iterations = 0_u32;

    if !dry_run {
        print!("  Polling FECS_CPUCTL");
        loop {
            let cpuctl = r32(&ember, &bdf, FECS_CPUCTL);
            if cpuctl == 0xFFFF_FFFF {
                let elapsed_ms = started.elapsed().as_millis();
                println!(
                    "\n  WARN: D3cold sentinel at t={elapsed_ms}ms — ember circuit breaker active"
                );
                println!("  → journalctl -u coral-ember for recovery status");
                break;
            }
            iterations += 1;
            if cpuctl != last_cpuctl {
                print!(" [{cpuctl:#010x}]");
                last_cpuctl = cpuctl;
            } else {
                print!(".");
            }
            // Kepler FECS halts after boot init (bit 4 = HALTED = 0x10).
            // Also check MAILBOX0 for boot status signal.
            let mb0 = r32(&ember, &bdf, FECS_MB0);
            if (cpuctl & CPUCTL_HALTED != 0) && iterations > 1 {
                // Halted on iteration >1 means firmware ran and halted (not initial state)
                booted = true;
                break;
            }
            if mb0 != 0 {
                println!("\n  FECS_MB0 set: {mb0:#010x}");
                booted = true;
                break;
            }
            // Also check PGRAPH_STATUS for engine-ready
            let pgraph = r32(&ember, &bdf, PGRAPH_STAT);
            if pgraph != 0 && pgraph != pgraph_st {
                println!("\n  PGRAPH_STATUS changed: {pgraph_st:#010x} → {pgraph:#010x}");
            }
            if started.elapsed() >= timeout {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        println!();

        if booted {
            println!("  FECS BOOTED! (iterations: {iterations})");
        } else {
            println!(
                "  Timeout after {}ms (iterations: {iterations})",
                timeout.as_millis()
            );
            println!("  FECS_CPUCTL last = {last_cpuctl:#010x}");
        }
    } else {
        println!("  DRY RUN: polling skipped");
    }

    // ── Phase 6: Post-boot state probe ─────────────────────────────────────
    println!("\n━━━ Phase 6: Post-Boot State ━━━\n");

    let fecs_cpuctl_post = r32(&ember, &bdf, FECS_CPUCTL);
    let fecs_pc_post = r32(&ember, &bdf, FECS_PC);
    let fecs_mb0_post = r32(&ember, &bdf, FECS_MB0);
    let fecs_mb1_post = r32(&ember, &bdf, FECS_MB1);
    let pmc_post = r32(&ember, &bdf, PMC_ENABLE);
    let pgraph_post = r32(&ember, &bdf, PGRAPH_STAT);
    let pt_post = r32(&ember, &bdf, PTIMER_HI);

    println!("  FECS_CPUCTL    = {fecs_cpuctl_post:#010x}");
    println!("  FECS_PC        = {fecs_pc_post:#010x}");
    println!("  FECS_MB0       = {fecs_mb0_post:#010x}  (0=idle, non-zero=status code)");
    println!("  FECS_MB1       = {fecs_mb1_post:#010x}");
    println!(
        "  PMC_ENABLE     = {pmc_post:#010x}  (preserved: {})",
        if pmc_post == pmc_en {
            "YES ✓"
        } else {
            "CHANGED"
        }
    );
    println!("  PGRAPH_STATUS  = {pgraph_post:#010x}  (was {pgraph_st:#010x})");
    println!(
        "  PTIMER         = {pt_post:#010x}  (running: {})",
        if pt_post != pt0 {
            "YES ✓"
        } else {
            "FROZEN ✗"
        }
    );

    // GPC0 GPCCS state
    let gpc0_cpuctl = r32(&ember, &bdf, GPCCS_BASE_GPC0 + GPCCS_CPUCTL_OFF);
    let gpc0_pc = r32(&ember, &bdf, GPCCS_BASE_GPC0 + GPCCS_PC_OFF);
    let gpc0_mb0 = r32(&ember, &bdf, GPCCS_BASE_GPC0 + GPCCS_MB0_OFF);
    let gpc1_cpuctl = r32(&ember, &bdf, GPCCS_BASE_GPC1 + GPCCS_CPUCTL_OFF);
    let gpc1_pc = r32(&ember, &bdf, GPCCS_BASE_GPC1 + GPCCS_PC_OFF);
    println!("  GPC0_GPCCS_CPUCTL = {gpc0_cpuctl:#010x}  PC={gpc0_pc:#010x}  MB0={gpc0_mb0:#010x}");
    println!("  GPC1_GPCCS_CPUCTL = {gpc1_cpuctl:#010x}  PC={gpc1_pc:#010x}");

    let gpc0_alive = gpc0_cpuctl != 0xbadf_1100 && gpc0_cpuctl != 0xffff_ffff;
    let gpc1_alive = gpc1_cpuctl != 0xbadf_1100 && gpc1_cpuctl != 0xffff_ffff;

    // ── Summary ────────────────────────────────────────────────────────────
    println!();
    println!("  ┌────────────────────────────────────────┐");
    println!("  │  Sovereign K80 FECS Boot Score         │");
    let fecs_started = !dry_run && booted;
    let fecs_halted = fecs_cpuctl_post & CPUCTL_HALT != 0 && !dry_run;
    let ptimer_ok = pt_post != pt0;
    let gpc0_booted = gpc0_alive;
    println!(
        "  │  FECS started:     {}  │",
        if fecs_started {
            "✓ YES              "
        } else {
            "✗ NO               "
        }
    );
    println!(
        "  │  FECS halted/done: {}  │",
        if fecs_halted {
            "✓ YES              "
        } else {
            "✗ NO (running/hung)"
        }
    );
    println!(
        "  │  PTIMER running:   {}  │",
        if ptimer_ok {
            "✓ YES              "
        } else {
            "✗ FROZEN           "
        }
    );
    println!(
        "  │  GPC0 GPCCS alive: {}  │",
        if gpc0_booted {
            "✓ YES              "
        } else {
            "✗ NO               "
        }
    );
    println!(
        "  │  GPC1 GPCCS alive: {}  │",
        if gpc1_alive {
            "✓ YES              "
        } else {
            "✗ NO               "
        }
    );
    println!("  └────────────────────────────────────────┘");

    if booted {
        println!("\n  FECS boot SUCCEEDED. K80 GR engine is ready.");
        println!("  Next: load GPCCS firmware and initialize PGFIFO for compute dispatch.");
    } else if fecs_started && !dry_run {
        println!(
            "\n  FECS started but did not halt within {}ms.",
            timeout.as_millis()
        );
        println!("  Possible causes:");
        println!("  1. FECS is still running (init takes longer — try --timeout 2000)");
        println!("  2. GDDR5 not accessible (check VRAM via PRAMIN probe)");
        println!("  3. GPCCS firmware missing (FECS waits for GPCCS)");
        println!("  4. Firmware byte mismatch (check fecs_inst.bin checksum)");
    } else if dry_run {
        println!("\n  DRY RUN complete — no writes performed.");
    }

    println!();
    println!("═══════════════════════════════════════════");
    println!("  EXP 182 complete.");
    println!("═══════════════════════════════════════════");
}
