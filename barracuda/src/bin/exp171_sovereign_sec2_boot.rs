// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 171: Sovereign GV100 SEC2 boot — ACR/WPR configuration.
//!
//! ## Context
//!
//! exp170 successfully:
//! - Loaded PMU HS firmware to the Boot Falcon at 0x084000 via PIO
//! - Triggered DEVINIT (PMC_ENABLE = 0xffffffff → 0x5fecdff1)
//! - Confirmed PTIMER running and SEC2 became accessible
//!
//! **Key correction from trace analysis**: PRIV_RING registers are at the
//! `0x12xxxx` range (e.g. 0x12004c), NOT `0x012xxx`. The previous "bad00100"
//! was simply reading an unmapped BAR0 offset at 0x012004. PRIV_RING is
//! ALREADY functional after DEVINIT.
//!
//! ## SEC2 Initialization Sequence (from mmiotrace lines 281174-283100)
//!
//! After DEVINIT (PMC_ENABLE = 0x5fecdff1):
//! 1. Write WPR config registers (copy from hardware-detected 0x100cxx → 0x1FACxx)
//! 2. Perform PRIV_RING device enumeration (W 0x12004c = 0x00000002)
//! 3. Pre-SEC2 setup: toggle 0x0873c0 (soft-reset SEC2), PMC toggle
//! 4. Configure SEC2 ACR mode (write 0x087058 = 0x2)
//! 5. Load SEC2 IMEM: 128 words at physical page 0xfe, virtual tag 0xfd/0xfe
//! 6. Load SEC2 DMEM descriptor: 22 words (ACR bootstrap arguments)
//! 7. Set BOOTVEC = 0xfd00, write sentinel 0xcafebeef, STARTCPU
//! 8. Poll SEC2 CPUCTL until HRESET bit set (halted = done)
//! 9. Probe correct PRIV_RING (0x12004c), WPR (0x1fac80), ACR status
//!
//! ## SEC2 IMEM layout
//!
//! The 128-word firmware from `titanv_sec2_fw_from_trace.bin` is loaded at:
//! - Physical IMEM address 0xfe00 (via IMEMC = 0x0100fe00)
//! - Virtual page tag 0xfd (IMETTAG = 0xfd) for first 64 words
//! - Virtual page tag 0xfe (IMETTAG = 0xfe) for second 64 words
//! - BOOTVEC = 0x0000fd00 (entry at virtual 0xfd00)
//!
//! ## Usage
//!
//! ```text
//! sudo rustc --edition 2021 exp171_sovereign_sec2_boot.rs \
//!     --extern rustix=... -L ...  -o /tmp/exp171 && sudo /tmp/exp171
//!
//! # Or if cargo is available:
//! sudo cargo run --release --bin exp171_sovereign_sec2_boot -- \
//!     --bdf 0000:02:00.0 \
//!     --firmware wateringHole/titanv_sec2_fw_from_trace.bin
//! ```

use std::io::{self, Write as IoWrite};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use hotspring_barracuda::ember_types::MmioBatchOp;
use hotspring_barracuda::fleet_client::EmberClient;

#[path = "../bin_helpers/sovereignty/mod.rs"]
mod sovereignty;
use sovereignty::connect::{connect_ember, extract_arg, is_dry_run, resolve_target_bdf};

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

fn batch(ember: &EmberClient, bdf: &str, ops: &[MmioBatchOp]) {
    if let Err(e) = ember.mmio_batch(bdf, ops) {
        eprintln!("  WARN: mmio.batch ({} ops): {e}", ops.len());
    }
}

// ── Register Map ──────────────────────────────────────────────────────────────

// GPU identity
const BOOT0: u32 = 0x000000;

// Power management
const PMC_ENABLE: u32 = 0x000200;
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const PMC_INTR_0: u32 = 0x000160;
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const PMC_INTR_1: u32 = 0x000164;

// PTIMER
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const PTIMER_NUM: u32 = 0x009410;
const PTIMER_DEN: u32 = 0x009400;
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const PTIMER_CTRL: u32 = 0x009140;

// PRIV_RING (CORRECT addresses from trace analysis)
const PRIV_RING_INFO_0: u32 = 0x120058; // ring size
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const PRIV_RING_INFO_1: u32 = 0x12005c;
const PRIV_RING_MASTER0: u32 = 0x120070; // master control
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const PRIV_RING_MASTER1: u32 = 0x120074;
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const PRIV_RING_MASTER2: u32 = 0x120078;
const PRIV_RING_DEV_INFO: u32 = 0x122120; // device info base
const PRIV_RING_CMD: u32 = 0x12004c; // CMD: write 2 = ENUM_DEVICES
const PRIV_RING_STATUS: u32 = 0x12006c; // status (0xc = OK, 12 devices)

// WPR hardware-detected bounds (read from PMU/hardware)
const WPR_HW_CFG: u32 = 0x100c80; // detected WPR config
const WPR_HW_LO: u32 = 0x100cc4; // detected WPR low VRAM address
const WPR_HW_HI: u32 = 0x100cc8; // detected WPR high VRAM address
const WPR_HW_END: u32 = 0x100ccc; // detected WPR end VRAM address

// WPR controller write-back registers
const WPR_CFG: u32 = 0x1fac80;
const WPR_LO: u32 = 0x1facc4;
const WPR_HI: u32 = 0x1facc8;
const WPR_END: u32 = 0x1faccc;

// SEC2 Falcon (0x087000 base)
const SEC2_BASE: u32 = 0x087000;
const SEC2_IRQMASK: u32 = SEC2_BASE + 0x014; // IRQ mask
const SEC2_IRQSTAT: u32 = SEC2_BASE + 0x008; // IRQ status
const SEC2_IRQCLR: u32 = SEC2_BASE + 0x004; // IRQ clear (write)
const SEC2_MAILBOX0: u32 = SEC2_BASE + 0x040; // Mailbox 0 (cafebeef sentinel)
const SEC2_IRQMSKSET: u32 = SEC2_BASE + 0x0a4; // IRQ mask set
const SEC2_UNK1C: u32 = SEC2_BASE + 0x01c; // status mirror
const SEC2_HWCFG: u32 = SEC2_BASE + 0x10c; // HW config (bit0=idle, bit2=secured)
const SEC2_CPUCTL: u32 = SEC2_BASE + 0x100; // CPU control (bit1=STARTCPU)
const SEC2_BOOTVEC: u32 = SEC2_BASE + 0x104; // Boot vector (entry offset)
const SEC2_BOOTVEC2: u32 = SEC2_BASE + 0x084; // Secondary boot arg (GPU chip ID)
const SEC2_CONFIG2: u32 = SEC2_BASE + 0x048; // Falcon config 2
const SEC2_CONFIG3: u32 = SEC2_BASE + 0x054; // ACR alias selector
const SEC2_ACR_PRIV: u32 = SEC2_BASE + 0x090; // ACR privilege config
const SEC2_ACR_DC: u32 = SEC2_BASE + 0x0dc; // ACR data-cache config
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const SEC2_IMEMC: u32 = SEC2_BASE + 0x180; // IMEM control
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const SEC2_IMEMD: u32 = SEC2_BASE + 0x184; // IMEM data (auto-increment)
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const SEC2_IMETTAG: u32 = SEC2_BASE + 0x188; // IMEM virtual page tag
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const SEC2_DMEMC: u32 = SEC2_BASE + 0x1c0; // DMEM control
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const SEC2_DMEMD: u32 = SEC2_BASE + 0x1c4; // DMEM data
const SEC2_RESET: u32 = SEC2_BASE + 0x3c0; // Soft-reset toggle
const SEC2_MODE: u32 = SEC2_BASE + 0x058; // Mode (bit1 = ACR mode)
const SEC2_FBIF: u32 = SEC2_BASE + 0x604; // FBIF / context config
const SEC2_ACR_STATUS: u32 = SEC2_BASE + 0xa34; // ACR queue status
const SEC2_ACR_CTL: u32 = SEC2_BASE + 0xa30; // ACR queue control
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const SEC2_ACR_CMD: u32 = SEC2_BASE + 0xac0; // ACR command register
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const SEC2_ACR_DATA: u32 = SEC2_BASE + 0xac4; // ACR data register

// IMEMC fields
#[expect(dead_code, reason = "reserved GPU register — retained for sovereign boot documentation")]
const SEC2_IMEMC_AUTOINCR: u32 = 0x0100_0000; // bit24 = autoincrement
const SEC2_IMEMC_PAGE_FE00: u32 = 0x0000_fe00; // physical IMEM byte address 0xfe00
const SEC2_IMEM_WORDS_PER_PAGE: usize = 64; // 256 bytes per page

// SEC2 boot constants (from trace)
const SEC2_BOOTVEC_ENTRY: u32 = 0x0000_fd00; // virtual entry point at 0xfd00
const SEC2_BOOT_ARG: u32 = 0x1400_00a1; // GV100 chip ID as boot argument
const SEC2_MAGIC: u32 = 0xcafe_beef; // sentinel for boot-complete handshake

// PMC_ENABLE: disable SEC2 engine (bit14 = SEC2 clock)
const PMC_ENABLE_NO_SEC2: u32 = 0x5fec_9ff1; // SEC2 disabled
const PMC_ENABLE_WITH_SEC2: u32 = 0x5fec_dff1; // SEC2 enabled

// SEC2 DMEM descriptor (22 words from trace, ACR bootstrap args)
// These are the ACR parameters telling SEC2 what firmware to authenticate.
// Fields: [0..7] = reserved zeros, [8] = 1 (ACR mode), [12..13] = 0x100 (size?),
// [14] = 0x2e00, [16] = 0x2f00 (WPR page references), [18] = 0x1000 (granule)
const SEC2_DMEM_DESC: [u32; 22] = [
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0001,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0100,
    0x0000_0100,
    0x0000_2e00,
    0x0000_0000,
    0x0000_2f00,
    0x0000_0000,
    0x0000_1000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bdf = resolve_target_bdf(&args, 0);
    if bdf.is_empty() {
        eprintln!("FATAL: no target BDF — pass --bdf or set HOTSPRING_BARRACUDA_TARGET_BDF");
        std::process::exit(1);
    }
    let fw_path = extract_arg(&args, "--firmware")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("wateringHole/titanv_sec2_fw_from_trace.bin"));
    let dry_run = is_dry_run(&args);
    let ember_socket = extract_arg(&args, "--ember-socket");

    banner();
    println!("  BDF:       {bdf}");
    println!("  Firmware:  {}", fw_path.display());
    println!("  Dry-run:   {dry_run}\n");

    // ── Load SEC2 firmware ────────────────────────────────────────────────────
    let fw_bytes = std::fs::read(&fw_path).unwrap_or_else(|e| {
        eprintln!("FATAL: cannot read SEC2 firmware: {e}");
        std::process::exit(1);
    });
    let fw_words: Vec<u32> = fw_bytes
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
    println!(
        "  SEC2 firmware: {} bytes / {} words (expect 128)",
        fw_bytes.len(),
        fw_words.len()
    );
    if fw_words.len() != 128 {
        eprintln!(
            "  WARN: expected 128 words from trace, got {}",
            fw_words.len()
        );
    }

    // ── Connect to ember ───────────────────────────────────────────────────
    let ember = connect_ember(&bdf,  ember_socket.as_deref());
    println!("  Ember:     {}\n", ember.socket_path().display());

    // ── Phase 1: Verify pre-conditions ───────────────────────────────────────
    println!("\n━━━ Phase 1: Pre-condition Check ━━━\n");

    let boot0 = r32(&ember, &bdf, BOOT0);
    println!("  BOOT0          = {boot0:#010x}  (GV100 = 0x140000a1)");
    if boot0 == 0xffff_ffff || boot0 == 0 {
        eprintln!("FATAL: GPU link dead. Run exp170 first.");
        std::process::exit(1);
    }

    let pmc_en = r32(&ember, &bdf, PMC_ENABLE);
    println!("  PMC_ENABLE     = {pmc_en:#010x}");
    let devinit_done = pmc_en == PMC_ENABLE_WITH_SEC2 || pmc_en == PMC_ENABLE_NO_SEC2;
    if !devinit_done {
        eprintln!("  ERROR: DEVINIT has not completed (expected 0x5fecdff1 or 0x5fec9ff1).");
        eprintln!("  Run exp170 first to complete DEVINIT.");
        std::process::exit(1);
    }
    println!("  DEVINIT: OK ({pmc_en:#010x})");

    let pt_hi0 = r32(&ember, &bdf, PTIMER_DEN);
    std::thread::sleep(Duration::from_millis(5));
    let pt_hi1 = r32(&ember, &bdf, PTIMER_DEN);
    println!(
        "  PTIMER_DEN     = {pt_hi0:#010x} → {pt_hi1:#010x}  ({})",
        if pt_hi1 != pt_hi0 {
            "RUNNING ✓"
        } else {
            "FROZEN !"
        }
    );

    // Read PRIV_RING state at CORRECT addresses
    let pr_cmd = r32(&ember, &bdf, PRIV_RING_CMD);
    let pr_status = r32(&ember, &bdf, PRIV_RING_STATUS);
    let pr_info0 = r32(&ember, &bdf, PRIV_RING_INFO_0);
    let pr_master = r32(&ember, &bdf, PRIV_RING_MASTER0);
    println!("  PRIV_RING_CMD    [0x12004c] = {pr_cmd:#010x}  (0=idle, 2=pending)");
    println!("  PRIV_RING_STATUS [0x12006c] = {pr_status:#010x}");
    println!("  PRIV_RING_INFO   [0x120058] = {pr_info0:#010x}");
    println!("  PRIV_RING_MASTER [0x120070] = {pr_master:#010x}");

    // Read hardware-detected WPR bounds
    let wpr_hw_cfg = r32(&ember, &bdf, WPR_HW_CFG);
    let wpr_hw_lo = r32(&ember, &bdf, WPR_HW_LO);
    let wpr_hw_hi = r32(&ember, &bdf, WPR_HW_HI);
    let wpr_hw_end = r32(&ember, &bdf, WPR_HW_END);
    println!("  WPR_HW_CFG [0x100c80] = {wpr_hw_cfg:#010x}");
    println!("  WPR_HW_LO  [0x100cc4] = {wpr_hw_lo:#010x}");
    println!("  WPR_HW_HI  [0x100cc8] = {wpr_hw_hi:#010x}");
    println!("  WPR_HW_END [0x100ccc] = {wpr_hw_end:#010x}");

    // Read current WPR controller state
    let wpr_cfg_now = r32(&ember, &bdf, WPR_CFG);
    println!("  WPR_CFG    [0x1fac80] = {wpr_cfg_now:#010x}  (current)");

    // Read SEC2 baseline
    let sec2_hwcfg = r32(&ember, &bdf, SEC2_HWCFG);
    let sec2_cpuctl = r32(&ember, &bdf, SEC2_CPUCTL);
    let sec2_mb0 = r32(&ember, &bdf, SEC2_MAILBOX0);
    println!("  SEC2_HWCFG [0x08710c] = {sec2_hwcfg:#010x}  (bit0=idle)");
    println!("  SEC2_CPUCTL[0x087100] = {sec2_cpuctl:#010x}  (0x10=HRESET, 0=running)");
    println!("  SEC2_MB0   [0x087040] = {sec2_mb0:#010x}");

    if dry_run {
        println!("\n  (dry-run mode: no writes will occur)");
        return;
    }

    // ── Phase 2: WPR Configuration ────────────────────────────────────────────
    println!("\n━━━ Phase 2: WPR Configuration ━━━\n");

    // Copy hardware-detected WPR bounds to WPR controller
    // (as seen in mmiotrace lines 99082-99094, immediately after DEVINIT trigger)
    println!("  Writing WPR bounds from hardware detection...");
    batch(
        &ember,
        &bdf,
        &[
            MmioBatchOp::write(WPR_CFG, wpr_hw_cfg),
            MmioBatchOp::write(WPR_LO, wpr_hw_lo),
            MmioBatchOp::write(WPR_HI, wpr_hw_hi),
            MmioBatchOp::write(WPR_END, wpr_hw_end),
            MmioBatchOp::write(0x100c10, 0x00ff_fff0),
        ],
    );

    // Verify write-back
    let wpr_cfg_new = r32(&ember, &bdf, WPR_CFG);
    println!("  WPR_CFG [0x1fac80] wrote {wpr_hw_cfg:#010x}, reads back {wpr_cfg_new:#010x}");
    if wpr_cfg_new != wpr_hw_cfg {
        println!("  WARN: WPR_CFG readback mismatch (register may be WO or protected)");
    }

    // ── Phase 3: PRIV_RING Enumeration ────────────────────────────────────────
    println!("\n━━━ Phase 3: PRIV_RING Device Enumeration ━━━\n");

    // PRIV_RING_CMD = 2 triggers device enumeration (as in mmiotrace lines 99018-99020)
    let pr_before = r32(&ember, &bdf, PRIV_RING_CMD);
    println!("  PRIV_RING_CMD before = {pr_before:#010x}");
    w32(&ember, &bdf, PRIV_RING_CMD, 0x0000_0002);
    std::thread::sleep(Duration::from_millis(1));
    let pr_after = r32(&ember, &bdf, PRIV_RING_CMD);
    println!("  PRIV_RING_CMD after  = {pr_after:#010x}  (0=enum done)");

    // Read updated ring status
    let pr_status2 = r32(&ember, &bdf, PRIV_RING_STATUS);
    let pr_devinfo = r32(&ember, &bdf, PRIV_RING_DEV_INFO);
    println!("  PRIV_RING_STATUS  = {pr_status2:#010x}");
    println!("  PRIV_RING_DEVINFO = {pr_devinfo:#010x}");

    // ── Phase 4: SEC2 Pre-load Setup ──────────────────────────────────────────
    println!("\n━━━ Phase 4: SEC2 Pre-load Setup ━━━\n");

    // Set PMC_ENABLE to SEC2-enabled state
    w32(&ember, &bdf, PMC_ENABLE, PMC_ENABLE_WITH_SEC2);
    std::thread::sleep(Duration::from_millis(1));
    let pmc_now = r32(&ember, &bdf, PMC_ENABLE);
    println!("  PMC_ENABLE = {pmc_now:#010x}");

    // Clear SEC2 mailbox and interrupts
    w32(&ember, &bdf, SEC2_MAILBOX0, 0x0000_0000);
    w32(&ember, &bdf, SEC2_IRQMASK, 0xffff_ffff);

    // Set boot vector argument (GPU chip ID, as in trace line 281180)
    w32(&ember, &bdf, SEC2_BOOTVEC2, SEC2_BOOT_ARG);
    println!("  SEC2_BOOTVEC2 ← {SEC2_BOOT_ARG:#010x} (GV100 chip ID)");

    // Initial config2
    w32(&ember, &bdf, SEC2_CONFIG2, 0x0000_0004);

    // Wait for SEC2 idle (HWCFG bit0 = 1)
    print!("  Waiting for SEC2 idle: ");
    let _ = io::stdout().flush();
    for _ in 0..100 {
        let hwcfg = r32(&ember, &bdf, SEC2_HWCFG);
        if hwcfg == 0xFFFF_FFFF {
            eprintln!("WARN: D3cold sentinel");
            break;
        }
        if hwcfg & 0x1 != 0 {
            println!("OK ({hwcfg:#010x})");
            break;
        }
        print!(".");
        let _ = io::stdout().flush();
        std::thread::sleep(Duration::from_millis(2));
    }

    // PMC toggle: disable SEC2 engine, do reset cycle, re-enable
    // (replicating mmiotrace lines 281186-281226)
    println!("  PMC toggle: SEC2 disable → reset → re-enable");

    w32(&ember, &bdf, PMC_ENABLE, PMC_ENABLE_NO_SEC2);

    // First reset pulse
    w32(&ember, &bdf, SEC2_RESET, 0x0000_0001);
    std::thread::sleep(Duration::from_millis(1));
    let rst = r32(&ember, &bdf, SEC2_RESET);
    println!("  SEC2_RESET asserted = {rst:#010x}");
    w32(&ember, &bdf, SEC2_RESET, 0x0000_0000);

    // Poll SEC2_HWCFG until it settles (0x7 during reset, 0x1 when done)
    let t_rst = Instant::now();
    loop {
        let hwcfg = r32(&ember, &bdf, SEC2_HWCFG);
        if hwcfg == 0x0000_0001 || t_rst.elapsed() > Duration::from_millis(50) {
            println!("  SEC2_HWCFG after reset1 = {hwcfg:#010x}");
            break;
        }
        std::thread::sleep(Duration::from_micros(500));
    }

    // Second reset pulse (as in trace)
    w32(&ember, &bdf, SEC2_RESET, 0x0000_0001);
    std::thread::sleep(Duration::from_millis(1));
    w32(&ember, &bdf, SEC2_RESET, 0x0000_0000);

    let t_rst2 = Instant::now();
    loop {
        let hwcfg = r32(&ember, &bdf, SEC2_HWCFG);
        if hwcfg == 0x0000_0001 || t_rst2.elapsed() > Duration::from_millis(50) {
            println!("  SEC2_HWCFG after reset2 = {hwcfg:#010x}");
            break;
        }
        std::thread::sleep(Duration::from_micros(500));
    }

    // Re-enable SEC2
    w32(&ember, &bdf, PMC_ENABLE, PMC_ENABLE_WITH_SEC2);
    std::thread::sleep(Duration::from_millis(1));
    w32(&ember, &bdf, SEC2_MAILBOX0, 0x0000_0000);

    // Final idle check
    let hwcfg_final = r32(&ember, &bdf, SEC2_HWCFG);
    println!("  SEC2_HWCFG post-reset = {hwcfg_final:#010x}  (expect 0x1 = idle)");

    // Set boot vector argument again after reset
    w32(&ember, &bdf, SEC2_BOOTVEC2, SEC2_BOOT_ARG);

    // Config2 set to 0x5 (ACR variant, as in trace line 281234)
    w32(&ember, &bdf, SEC2_CONFIG2, 0x0000_0005);

    // FBIF config (read-modify-write, trace lines 281235-281236)
    let fbif = r32(&ember, &bdf, SEC2_FBIF);
    w32(&ember, &bdf, SEC2_FBIF, fbif);
    println!("  SEC2_FBIF = {fbif:#010x}");

    // ACR alias selector (trace line 281237: W 0x087054 = 0x402ffe57)
    w32(&ember, &bdf, SEC2_CONFIG3, 0x402f_fe57);
    println!("  SEC2_CONFIG3 ← 0x402ffe57 (ACR alias selector)");

    // ACR privilege mode (trace line 281238-281239)
    let acr_priv = r32(&ember, &bdf, SEC2_ACR_PRIV);
    w32(&ember, &bdf, SEC2_ACR_PRIV, acr_priv | 0x0001_0000);
    println!(
        "  SEC2_ACR_PRIV = {acr_priv:#010x} → {:#010x}",
        acr_priv | 0x0001_0000
    );

    // Enable ACR interrupt (trace lines 281240-281245)
    w32(&ember, &bdf, SEC2_IRQMSKSET, 0x0000_0008);
    let irqstat = r32(&ember, &bdf, SEC2_IRQSTAT);
    println!("  SEC2_IRQSTAT = {irqstat:#010x}");
    w32(&ember, &bdf, SEC2_IRQCLR, 0x0000_0008);

    // Set ACR mode (trace line 281247: W 0x087058 = 0x2)
    w32(&ember, &bdf, SEC2_MODE, 0x0000_0002);
    let dc = r32(&ember, &bdf, SEC2_ACR_DC);
    println!("  SEC2_MODE ← 0x2 (ACR mode)");
    println!("  SEC2_ACR_DC = {dc:#010x}");

    // ── Phase 5: SEC2 IMEM Load ───────────────────────────────────────────────
    println!("\n━━━ Phase 5: SEC2 IMEM Firmware Load ━━━\n");

    let page0_bytes_len = (SEC2_IMEM_WORDS_PER_PAGE * 4).min(fw_bytes.len());
    println!("  IMEMC ← 0x0100fe00 (autoincr, page 0xfe00)");
    println!(
        "  IMETTAG ← 0xfd  writing {} bytes (page 0)...",
        page0_bytes_len
    );
    match ember.falcon_upload_imem(
        &bdf,
        SEC2_BASE,
        SEC2_IMEMC_PAGE_FE00,
        &fw_bytes[..page0_bytes_len],
        0x0000_00fd,
        false,
    ) {
        Ok(r) if r.ok => println!("  Page 0 upload: {:?} bytes", r.bytes),
        Ok(r) => eprintln!("  WARN: page 0 upload ok=false bytes={:?}", r.bytes),
        Err(e) => eprintln!("  WARN: page 0 upload failed: {e}"),
    }

    if fw_bytes.len() > page0_bytes_len {
        let page1_len = fw_bytes.len() - page0_bytes_len;
        println!(
            "  IMETTAG ← 0xfe  writing {} bytes (page 1)...",
            page1_len
        );
        match ember.falcon_upload_imem(
            &bdf,
            SEC2_BASE,
            SEC2_IMEMC_PAGE_FE00 + page0_bytes_len as u32,
            &fw_bytes[page0_bytes_len..],
            0x0000_00fe,
            false,
        ) {
            Ok(r) if r.ok => println!("  Page 1 upload: {:?} bytes", r.bytes),
            Ok(r) => eprintln!("  WARN: page 1 upload ok=false bytes={:?}", r.bytes),
            Err(e) => eprintln!("  WARN: page 1 upload failed: {e}"),
        }
    }

    println!("  IMEM load complete ({} total words)", fw_words.len());

    // ── Phase 6: SEC2 DMEM Descriptor Load ────────────────────────────────────
    println!("\n━━━ Phase 6: SEC2 DMEM Descriptor ━━━\n");

    // DMEMC = autoincrement from offset 0
    let dmem_bytes: Vec<u8> = SEC2_DMEM_DESC
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    println!("  DMEMC ← 0x01000000 (autoincr, offset 0)");
    println!(
        "  Writing {} DMEM descriptor words...",
        SEC2_DMEM_DESC.len()
    );
    match ember.falcon_upload_dmem(&bdf, SEC2_BASE, 0, &dmem_bytes) {
        Ok(r) if r.ok => println!("  DMEM descriptor written ({:?} bytes)", r.bytes),
        Ok(r) => eprintln!("  WARN: DMEM upload ok=false bytes={:?}", r.bytes),
        Err(e) => eprintln!("  WARN: DMEM upload failed: {e}"),
    }

    // ── Phase 7: SEC2 STARTCPU ────────────────────────────────────────────────
    println!("\n━━━ Phase 7: SEC2 STARTCPU ━━━\n");

    // Write cafebeef sentinel to mailbox0 (Boot Falcon clears it when done)
    w32(&ember, &bdf, SEC2_MAILBOX0, SEC2_MAGIC);
    println!("  SEC2_MAILBOX0 ← 0xcafebeef (sentinel)");

    // Set boot vector (entry at virtual 0xfd00)
    w32(&ember, &bdf, SEC2_BOOTVEC, SEC2_BOOTVEC_ENTRY);
    println!("  SEC2_BOOTVEC  ← {SEC2_BOOTVEC_ENTRY:#010x}");

    // STARTCPU
    w32(&ember, &bdf, SEC2_CPUCTL, 0x0000_0002);
    let t_start = Instant::now();
    println!("  SEC2_CPUCTL   ← 0x2 (STARTCPU) — SEC2 is running");

    // ── Phase 8: Wait for SEC2 completion ─────────────────────────────────────
    println!("\n━━━ Phase 8: Polling SEC2 for completion ━━━\n");

    print!("  CPUCTL poll: ");
    let _ = io::stdout().flush();
    let mut sec2_done = false;

    for _ in 0..500 {
        std::thread::sleep(Duration::from_millis(2));
        let cpuctl = r32(&ember, &bdf, SEC2_CPUCTL);
        if cpuctl == 0xFFFF_FFFF {
            eprintln!("WARN: D3cold sentinel");
            break;
        }
        // Bit 4 = HRESET, bit 6 = halted; expect 0x10 or 0x50
        if cpuctl & 0x10 != 0 {
            let elapsed = t_start.elapsed();
            println!(
                "\n  SEC2 halted!  CPUCTL = {cpuctl:#010x}  in {}ms",
                elapsed.as_millis()
            );
            sec2_done = true;
            break;
        }
        print!(".");
        let _ = io::stdout().flush();
    }

    if !sec2_done {
        let cpuctl = r32(&ember, &bdf, SEC2_CPUCTL);
        println!("\n  TIMEOUT after 1s! SEC2_CPUCTL = {cpuctl:#010x}");
    }

    // ── Phase 9: Post-SEC2 State Probe ────────────────────────────────────────
    println!("\n━━━ Phase 9: Post-SEC2 State Probe ━━━\n");

    let sec2_cpuctl = r32(&ember, &bdf, SEC2_CPUCTL);
    let sec2_mb0 = r32(&ember, &bdf, SEC2_MAILBOX0);
    let sec2_irqstat = r32(&ember, &bdf, SEC2_IRQSTAT);
    let sec2_unk1c = r32(&ember, &bdf, SEC2_UNK1C);
    let sec2_irqmsk = r32(&ember, &bdf, SEC2_IRQMASK);
    let pmc_final = r32(&ember, &bdf, PMC_ENABLE);

    println!("  SEC2_CPUCTL  [0x087100] = {sec2_cpuctl:#010x}  (0x10=HRESET, 0x50=halt+bit6)");
    println!("  SEC2_MB0     [0x087040] = {sec2_mb0:#010x}  (0=sentinel cleared by SEC2)");
    println!("  SEC2_IRQSTAT [0x087008] = {sec2_irqstat:#010x}");
    println!("  SEC2_UNKIC   [0x08701c] = {sec2_unk1c:#010x}");
    println!("  SEC2_IRQMASK [0x087004] = {sec2_irqmsk:#010x}");
    println!("  PMC_ENABLE              = {pmc_final:#010x}");

    // Read ACR registers
    let acr_status = r32(&ember, &bdf, SEC2_ACR_STATUS);
    let acr_ctl = r32(&ember, &bdf, SEC2_ACR_CTL);
    println!("  SEC2_ACR_STATUS [0x087a34] = {acr_status:#010x}");
    println!("  SEC2_ACR_CTL    [0x087a30] = {acr_ctl:#010x}");

    // PRIV_RING final state (CORRECT addresses)
    println!();
    let pr_cmd_f = r32(&ember, &bdf, PRIV_RING_CMD);
    let pr_status_f = r32(&ember, &bdf, PRIV_RING_STATUS);
    let pr_info_f = r32(&ember, &bdf, PRIV_RING_DEV_INFO);
    let pr_cnt = r32(&ember, &bdf, PRIV_RING_DEV_INFO + 4);
    let pr_flags = r32(&ember, &bdf, PRIV_RING_DEV_INFO + 8);
    println!("  PRIV_RING_CMD    [0x12004c] = {pr_cmd_f:#010x}  (0=idle ✓)");
    println!("  PRIV_RING_STATUS [0x12006c] = {pr_status_f:#010x}");
    println!("  PRIV_RING_DEVINFO[0x122120] = {pr_info_f:#010x}");
    println!("  PRIV_RING_DEVCNT [0x122124] = {pr_cnt:#010x}");
    println!("  PRIV_RING_FLAGS  [0x122128] = {pr_flags:#010x}");

    // WPR final state
    println!();
    let wpr_cfg_f = r32(&ember, &bdf, WPR_CFG);
    let wpr_lo_f = r32(&ember, &bdf, WPR_LO);
    let wpr_hi_f = r32(&ember, &bdf, WPR_HI);
    println!("  WPR_CFG [0x1fac80] = {wpr_cfg_f:#010x}");
    println!("  WPR_LO  [0x1facc4] = {wpr_lo_f:#010x}");
    println!("  WPR_HI  [0x1facc8] = {wpr_hi_f:#010x}");

    // PTIMER sanity
    let pt0 = r32(&ember, &bdf, PTIMER_DEN);
    std::thread::sleep(Duration::from_millis(10));
    let pt1 = r32(&ember, &bdf, PTIMER_DEN);
    println!();
    println!(
        "  PTIMER_DEN = {pt0:#010x} → {pt1:#010x}  ({})",
        if pt1 != pt0 {
            "RUNNING ✓"
        } else {
            "FROZEN !"
        }
    );

    // ── Score ─────────────────────────────────────────────────────────────────
    println!();
    let sec2_ok = sec2_done && (sec2_mb0 == 0 || sec2_mb0 != SEC2_MAGIC);
    let priv_ok = pr_cmd_f == 0;
    let ptimer_ok = pt1 != pt0;
    let wpr_ok = wpr_cfg_f != 0 && wpr_cfg_f != 0xffff_ffff;

    println!("  ┌──────────────────────────────────────┐");
    println!("  │  Sovereign SEC2 Boot Score            │");
    println!("  │  SEC2 halted cleanly: {}              │", yn(sec2_done));
    println!("  │  SEC2 sentinel clear: {}              │", yn(sec2_ok));
    println!("  │  PRIV_RING idle:      {}              │", yn(priv_ok));
    println!("  │  WPR configured:      {}              │", yn(wpr_ok));
    println!("  │  PTIMER running:      {}              │", yn(ptimer_ok));
    println!("  └──────────────────────────────────────┘");

    if sec2_done && priv_ok && ptimer_ok {
        println!("\n  SOVEREIGN PATH ADVANCING: SEC2 ACR bootstrap complete.");
        println!("  PRIV_RING is idle and healthy.");
        println!("  Next: exp172 — load LS PMU firmware and FECS compute scheduler.");
    } else if sec2_done {
        println!("\n  SEC2 halted. Check ACR_STATUS for authentication result.");
        println!("  PRIV_RING_CMD = {pr_cmd_f:#010x} (0 = healthy, non-0 = pending)");
    } else {
        println!("\n  SEC2 did not halt. Possible causes:");
        println!("  1. Firmware bytes wrong (check titanv_sec2_fw_from_trace.bin)");
        println!("  2. DMEM descriptor invalid for this hardware state");
        println!("  3. ACR_PRIV config needs adjustment");
        println!("  Check SEC2_CPUCTL, SEC2_IRQSTAT, and ACR registers above.");
    }

    println!("\n═══════════════════════════════════════");
    println!("  EXP 171 complete.");
    println!("═══════════════════════════════════════");
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn yn(b: bool) -> &'static str {
    if b { "✓ YES" } else { "✗ NO " }
}

fn banner() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXP 171: Sovereign GV100 SEC2 Boot — ACR/WPR Init          ║");
    println!("║  Goal: SEC2 ACR → PRIV_RING idle → WPR configured           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}
