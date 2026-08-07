// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 170: Sovereign GV100 cold-boot init from mmiotrace replay.
//!
//! ## Context
//!
//! Previous experiments revealed:
//! - exp169: PMU blocked, PRIV_RING faulted (bad00100), FECS in HRESET
//! - Warm handoff (livepatch): PMU halted-idle (cpuctl=0x01), DEVINIT ran,
//!   PMC_ENABLE=0x5fecdff1, but PTIMER frozen and PRIV_RING HS-locked
//!
//! The mmiotrace of nouveau init revealed the COMPLETE init sequence:
//! 1. Write ROM control, INTR enables, PBDMA enables
//! 2. PMC_ENABLE gradual (0x40000021 → 0x41ecdff1)
//! 3. Load PMU HS firmware to "boot Falcon" at 0x084000 via IMEMC/IMEMD PIO
//! 4. STARTCPU on boot Falcon (0x084100 = 0x02) → triggers clock init
//! 5. PMC_ENABLE = 0xffffffff → hardware-driven DEVINIT (~1.27s)
//! 6. PMC_ENABLE → 0x5fecdff1 (post-DEVINIT)
//! 7. Load SEC2 firmware (post-DEVINIT, not covered here)
//!
//! ## Registers discovered from mmiotrace
//!
//! Boot Falcon (0x084000 base):
//! - 0x084040 = DMACTL
//! - 0x084084 = DMEMD[0] (boot argument: chip ID 0x140000a1)
//! - 0x084048 = ???
//! - 0x084014 = IRQMASK
//! - 0x084180 = IMEMC (IMEM ctrl: secure|autoincr|page)
//! - 0x084184 = IMEMD (IMEM data streaming port, autoincrement)
//! - 0x084188 = IMETTAG (virtual page tag)
//! - 0x0841c0 = signature control
//! - 0x0841c4 = signature data
//! - 0x084100 = CPUCTL (bit 1 = STARTCPU)
//! - 0x084104 = ALIAS_CFG
//!
//! ## PCIe D3hot→D0 Reset
//!
//! With --reset flag: writes PM_CSR[1:0]=3 (D3hot) then 0 (D0) via PCI
//! config space at offset 0x54, giving clean hardware state.
//! Note: PCI config space writes cannot go through ember (BAR0 only).
//! For a full power cycle, use `glowplug.device.reset()` RPC instead.
//!
//! ## Usage
//!
//! ```text
//! sudo cargo run --release --bin exp170_sovereign_cold_boot -- \
//!     --bdf 0000:02:00.0 \
//!     --firmware wateringHole/titanv_pmu_hs_fw_from_trace.bin \
//!     [--reset]
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

// ── Registers ────────────────────────────────────────────────────────────────

// Power management
const PMC_INTR_EN_SET_0: u32 = 0x000180;
const PMC_INTR_EN_SET_1: u32 = 0x000184;
const PMC_INTR_0: u32 = 0x000160;
const PMC_INTR_1: u32 = 0x000164;
const PMC_ENABLE: u32 = 0x000200;

// Boot Falcon (0x084000 base — runs PMU HS firmware, triggers DEVINIT)
const BF_BASE: u32 = 0x084000;
const BF_IRQMASK: u32 = BF_BASE + 0x014;
const BF_DMACTL: u32 = BF_BASE + 0x040;
const BF_DMATRFMOFFS: u32 = BF_BASE + 0x048;
const BF_DMEMD0: u32 = BF_BASE + 0x084; // DMEM[0] boot arg
const BF_CPUCTL: u32 = BF_BASE + 0x100;
const BF_ALIAS: u32 = BF_BASE + 0x104;
const BF_SCTL: u32 = BF_BASE + 0x10c;
#[expect(
    dead_code,
    reason = "reserved GPU register — retained for sovereign boot documentation"
)]
const BF_IMEMC: u32 = BF_BASE + 0x180; // IMEM control (page address, mode)
#[expect(
    dead_code,
    reason = "reserved GPU register — retained for sovereign boot documentation"
)]
const BF_IMEMD: u32 = BF_BASE + 0x184; // IMEM data (auto-increment)
#[expect(
    dead_code,
    reason = "reserved GPU register — retained for sovereign boot documentation"
)]
const BF_IMETTAG: u32 = BF_BASE + 0x188; // virtual page tag
const BF_SIG_CTL: u32 = BF_BASE + 0x1c0; // signature block control
const BF_SIG_DATA: u32 = BF_BASE + 0x1c4; // signature block data
const BF_FBIF: u32 = BF_BASE + 0x624; // FBIF config

// Main PMU (post-DEVINIT)
const PMU_BASE: u32 = 0x10a000;
const PMU_CPUCTL: u32 = PMU_BASE + 0x10c;
#[expect(
    dead_code,
    reason = "reserved GPU register — retained for sovereign boot documentation"
)]
const PMU_IRQSTAT: u32 = PMU_BASE + 0x008;
const PMU_MB0: u32 = PMU_BASE + 0x450;

// SEC2 (post-DEVINIT)
const SEC2_BASE: u32 = 0x087000;
const SEC2_CPUCTL: u32 = SEC2_BASE + 0x10c;

// PRIV_RING
const PRIV_RING_INTR: u32 = 0x012004;
const PRIV_RING_MASTER_CONFIG: u32 = 0x012004c;

// PTIMER
const PTIMER_TIME_HI: u32 = 0x009400;
#[expect(
    dead_code,
    reason = "reserved GPU register — retained for sovereign boot documentation"
)]
const PTIMER_TIME_LO: u32 = 0x009410;

// FB / PRAMIN
const FB_NISO: u32 = 0x100c10;
#[expect(
    dead_code,
    reason = "reserved GPU register — retained for sovereign boot documentation"
)]
const PRAMIN_WIN: u32 = 0x001700;

// ROM access
const ROM_ACCESS_CTL: u32 = 0x088050;

// IMEMC block structure
#[expect(
    dead_code,
    reason = "reserved GPU register — retained for sovereign boot documentation"
)]
const IMEMC_BLOCK0: u32 = 0x01000000; // non-secure, autoincr, page 0
#[expect(
    dead_code,
    reason = "reserved GPU register — retained for sovereign boot documentation"
)]
const IMEMC_BLOCK1: u32 = 0x11000100; // secure, autoincr, page 1
const IMEMC_WORDS_PER_BLOCK: usize = 64;
const DMACTL_MAGIC: u32 = 0xcafe_beef;
const BOOT_ARG_CHIPID: u32 = 0x1400_00a1; // GV100 chip ID

// PMU HS firmware signature (4 words from trace)
const FW_SIG: [u32; 4] = [0x614c_30ee, 0x854d_057c, 0x2679_2cd8, 0x33b4_42f8];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bdf = resolve_target_bdf(&args, 0);
    if bdf.is_empty() {
        eprintln!("FATAL: no target BDF — pass --bdf or set HOTSPRING_BARRACUDA_TARGET_BDF");
        std::process::exit(1);
    }
    let fw_path = extract_arg(&args, "--firmware")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("wateringHole/titanv_pmu_hs_fw_from_trace.bin"));
    let do_reset = args.iter().any(|a| a == "--reset");
    let dry_run = is_dry_run(&args);
    let ember_socket = extract_arg(&args, "--ember-socket");

    banner();
    println!("  BDF:       {bdf}");
    println!("  Firmware:  {}", fw_path.display());
    println!("  PCIe reset: {do_reset} (PCI config — not ember; use glowplug for full reset)");
    println!("  Dry-run:   {dry_run}\n");

    // ── Load PMU HS firmware ─────────────────────────────────────────────────
    let fw_bytes = std::fs::read(&fw_path).unwrap_or_else(|e| {
        eprintln!("FATAL: cannot read firmware: {e}");
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
        "  Firmware:  {} bytes / {} words ({} blocks × 64)",
        fw_bytes.len(),
        fw_words.len(),
        (fw_words.len() + 63) / 64
    );
    if fw_words.len() != 896 {
        eprintln!(
            "  WARN: expected 896 words from mmiotrace, got {}",
            fw_words.len()
        );
    }

    // ── PCIe D3hot→D0 Reset (optional) ──────────────────────────────────────
    if do_reset && !dry_run {
        println!("\n━━━ Phase 0: PCIe D3hot→D0 Reset ━━━\n");
        let cfg_path = format!("/sys/bus/pci/devices/{bdf}/config");
        match pcie_d3hot_d0_reset(&cfg_path) {
            Ok(()) => println!("  D3hot→D0 reset OK — GPU in hardware reset"),
            Err(e) => {
                eprintln!("  WARN: D3hot→D0 reset failed: {e}");
                eprintln!("  Continuing with current state...");
            }
        }
        println!("  Waiting 200ms for hardware to settle...");
        std::thread::sleep(Duration::from_millis(200));
    }

    // ── Connect to ember ───────────────────────────────────────────────────
    let ember = connect_ember(&bdf, ember_socket.as_deref());
    println!("  Ember:     {}\n", ember.socket_path().display());

    // ── Phase 1: Identity + State Baseline ──────────────────────────────────
    println!("\n━━━ Phase 1: GPU Identity + Baseline ━━━\n");

    let boot0 = r32(&ember, &bdf, 0);
    println!("  BOOT0         = {boot0:#010x}  (GV100 = 0x140000a1)");
    if boot0 == 0xffff_ffff || boot0 == 0 {
        eprintln!("FATAL: GPU link dead (BOOT0={boot0:#010x})");
        std::process::exit(1);
    }

    let pmc_en = r32(&ember, &bdf, PMC_ENABLE);
    println!("  PMC_ENABLE    = {pmc_en:#010x}");
    println!(
        "  PRIV_RING     = {:#010x}  (bad00100=HS-locked)",
        r32(&ember, &bdf, PRIV_RING_INTR)
    );
    println!(
        "  PMU cpuctl    = {:#010x}  (0x01=idle-halted)",
        r32(&ember, &bdf, PMU_CPUCTL)
    );
    println!("  BF cpuctl     = {:#010x}", r32(&ember, &bdf, BF_CPUCTL));

    let pt0 = r32(&ember, &bdf, PTIMER_TIME_HI);
    std::thread::sleep(Duration::from_millis(10));
    let pt1 = r32(&ember, &bdf, PTIMER_TIME_HI);
    println!(
        "  PTIMER_HI     = {pt0:#010x} → {pt1:#010x}  ({})",
        if pt1 != pt0 { "RUNNING" } else { "FROZEN" }
    );

    // If DEVINIT already ran and PMC_ENABLE = 0x5fecdff1, warn user
    if pmc_en == 0x5fec_dff1 && !do_reset {
        eprintln!("  NOTE: DEVINIT already ran (PMC=0x5fecdff1). Use --reset for cold boot.");
        eprintln!("  Attempting warm re-trigger of PMU HS load...");
    }

    if dry_run {
        println!("\n  (dry-run mode: no writes will occur)");
        return;
    }

    // ── Phase 2: Pre-PMU Setup (interrupt enables, early engines) ────────────
    println!("\n━━━ Phase 2: Pre-PMU Hardware Setup ━━━\n");

    // ROM access control (enable host read)
    w32(&ember, &bdf, ROM_ACCESS_CTL, 0x0000_0001);
    println!("  ROM_ACCESS_CTL ← 0x1 (enable)");

    // PMC interrupt enables
    w32(&ember, &bdf, PMC_INTR_EN_SET_0, 0xffff_ffff);
    w32(&ember, &bdf, PMC_INTR_EN_SET_1, 0xffff_ffff);
    println!("  PMC_INTR_EN_SET 0/1 ← 0xffffffff");

    // Clock gating (from mmiotrace lines 64922-64925)
    w32(&ember, &bdf, 0x6013d4, 0x0000_001f);
    w32(&ember, &bdf, 0x6013d4, 0x0000_003f);
    w32(&ember, &bdf, 0x6013d5, 0x0000_0057);
    println!("  Clock gating regs set");

    // PBDMA interrupt enables (6 PBDMAs, from lines 64927-64937)
    for base in [
        0x00d97c_u32,
        0x00d9cc,
        0x00da1c,
        0x00da6c,
        0x00dabc,
        0x00db0c,
    ] {
        w32(&ember, &bdf, base, 0x0000_0001);
    }
    println!("  PBDMA INTR_EN (6 units) ← 0x1");

    // FIFO class enables (9 classes, from lines 64938-64946)
    for base in [
        0x00d014_u32,
        0x00d034,
        0x00d054,
        0x00d074,
        0x00d094,
        0x00d0b4,
        0x00d0d4,
        0x00d0f4,
        0x00d114,
    ] {
        w32(&ember, &bdf, base, 0x0000_0007);
    }
    println!("  FIFO class enables (9) ← 0x7");

    // FB NISO mask
    w32(&ember, &bdf, FB_NISO, 0x00ff_fff0);
    println!("  FB_NISO ← 0x00fffff0");

    // PMC_ENABLE step 1+2
    w32(&ember, &bdf, PMC_ENABLE, 0x4000_0021);
    std::thread::sleep(Duration::from_millis(1));
    w32(&ember, &bdf, PMC_ENABLE, 0x4000_0121);
    std::thread::sleep(Duration::from_millis(5));
    println!("  PMC_ENABLE ← 0x40000021 → 0x40000121");

    // PMC_ENABLE step 3 (enable more engines before PMU FW load)
    w32(&ember, &bdf, PMC_ENABLE, 0x4000_8121);
    std::thread::sleep(Duration::from_millis(2));
    println!("  PMC_ENABLE ← 0x40008121");

    // Verify BOOT0 still responsive
    let b0 = r32(&ember, &bdf, 0);
    println!("  BOOT0 re-check = {b0:#010x}");
    if b0 != boot0 {
        eprintln!("  WARN: BOOT0 changed! GPU may have reset ({boot0:#010x} → {b0:#010x})");
    }

    // ── Phase 3: Boot Falcon firmware load (IMEMC/IMEMD PIO) ────────────────
    println!("\n━━━ Phase 3: Boot Falcon Firmware Load (IMEMC/IMEMD PIO) ━━━\n");

    // Initial boot Falcon setup
    batch(
        &ember,
        &bdf,
        &[
            MmioBatchOp::write(BF_DMACTL, 0x0000_0000),
            MmioBatchOp::write(BF_DMEMD0, BOOT_ARG_CHIPID),
            MmioBatchOp::write(BF_DMATRFMOFFS, 0x0000_0004),
            MmioBatchOp::write(BF_IRQMASK, 0xffff_ffff),
            MmioBatchOp::write(BF_DMACTL, 0x0000_0000),
            MmioBatchOp::write(BF_DMEMD0, BOOT_ARG_CHIPID),
            MmioBatchOp::write(BF_FBIF, 0x0000_0190),
            MmioBatchOp::write(BF_SCTL, 0x0000_0000),
        ],
    );
    println!("  Boot Falcon control regs initialized");

    // Load firmware blocks via ember.falcon.upload_imem
    let n_words = fw_words.len();
    let n_blocks = (n_words + IMEMC_WORDS_PER_BLOCK - 1) / IMEMC_WORDS_PER_BLOCK;
    println!("  Loading {n_words} words in {n_blocks} blocks...");

    let block0_end = (IMEMC_WORDS_PER_BLOCK * 4).min(fw_bytes.len());
    match ember.falcon_upload_imem(&bdf, BF_BASE, 0, &fw_bytes[..block0_end], 0, false) {
        Ok(r) if r.ok => println!("  Block 0 (non-secure): {block0_end} bytes uploaded"),
        Ok(r) => eprintln!("  WARN: block 0 upload ok=false bytes={:?}", r.bytes),
        Err(e) => eprintln!("  WARN: block 0 upload failed: {e}"),
    }
    if fw_bytes.len() > block0_end {
        match ember.falcon_upload_imem(
            &bdf,
            BF_BASE,
            block0_end as u32,
            &fw_bytes[block0_end..],
            1,
            true,
        ) {
            Ok(r) if r.ok => {
                println!(
                    "  Blocks 1..{} (secure): {} bytes uploaded",
                    n_blocks - 1,
                    fw_bytes.len() - block0_end
                );
            }
            Ok(r) => eprintln!("  WARN: secure blocks upload ok=false bytes={:?}", r.bytes),
            Err(e) => eprintln!("  WARN: secure blocks upload failed: {e}"),
        }
    }
    println!("  {n_blocks} blocks loaded to IMEM");

    // Write signature block (64 words: 4 sig + 60 padding zeros)
    let mut sig_ops = vec![MmioBatchOp::write(BF_SIG_CTL, 0x0100_0000)];
    for &sw in &FW_SIG {
        sig_ops.push(MmioBatchOp::write(BF_SIG_DATA, sw));
    }
    for _ in 4..64 {
        sig_ops.push(MmioBatchOp::write(BF_SIG_DATA, 0x0000_0000));
    }
    batch(&ember, &bdf, &sig_ops);
    println!("  Signature block written (4 words + 60 zeros)");

    // Unlock and start
    w32(&ember, &bdf, BF_DMACTL, DMACTL_MAGIC); // 0xcafebeef unlock
    w32(&ember, &bdf, BF_ALIAS, 0x0000_0000); // clear alias
    w32(&ember, &bdf, BF_CPUCTL, 0x0000_0002); // STARTCPU
    println!("  STARTCPU issued (BF_CPUCTL ← 0x02)");

    // ── Phase 4: Wait for PMU clock init ─────────────────────────────────────
    println!("\n━━━ Phase 4: Wait for PTIMER to start (clock init) ━━━\n");

    let t_start = Instant::now();
    let mut ptimer_started = false;
    let pt_base = r32(&ember, &bdf, PTIMER_TIME_HI);
    print!("  Polling PTIMER_HI (base={pt_base:#010x}): ");
    let _ = io::stdout().flush();

    for _ in 0..100 {
        std::thread::sleep(Duration::from_millis(10));
        let pt = r32(&ember, &bdf, PTIMER_TIME_HI);
        if pt == 0xFFFF_FFFF {
            eprintln!("WARN: D3cold sentinel");
            break;
        }
        if pt != pt_base && pt != 0 && pt != 0xffff_ffff {
            println!(
                "\n  PTIMER running! {pt_base:#010x} → {pt:#010x} in {}ms",
                t_start.elapsed().as_millis()
            );
            ptimer_started = true;
            break;
        }
        print!(".");
        let _ = io::stdout().flush();
    }

    if !ptimer_started {
        let pt = r32(&ember, &bdf, PTIMER_TIME_HI);
        println!("\n  PTIMER still frozen after 1s (={pt:#010x})");
        println!("  Boot Falcon may not have started. Checking state...");
        println!("  BF_CPUCTL = {:#010x}", r32(&ember, &bdf, BF_CPUCTL));
    }

    // ── Phase 5: Continue PMC_ENABLE ramp + PRIV_RING config ─────────────────
    println!("\n━━━ Phase 5: PMC_ENABLE ramp to pre-DEVINIT value ━━━\n");

    // Replicate nouveau's gradual PMC_ENABLE progression from mmiotrace
    let pmc_steps: &[u32] = &[
        0x4000_c121,
        0x4000_c131,
        0x4008_c131,
        0x400c_c131,
        0x400c_4131,
        0x400c_d131,
        0x400c_d031,
        0x400c_d131,
        0x400c_d931,
        0x400c_dd31,
        0x400c_df31,
        0x410c_df31,
        0x418c_df31,
        0x41cc_df31,
        0x41ec_df31,
        0x41ec_dfb1,
        0x41ec_dff1,
    ];
    for &val in pmc_steps {
        w32(&ember, &bdf, PMC_ENABLE, val);
        std::thread::sleep(Duration::from_millis(2));
    }
    let pmc_now = r32(&ember, &bdf, PMC_ENABLE);
    println!("  PMC_ENABLE reached: {pmc_now:#010x} (target 0x41ecdff1)");

    // Additional pre-DEVINIT setup from mmiotrace
    w32(&ember, &bdf, 0x100a34, 0x8000_0000); // (from line 98943)
    w32(&ember, &bdf, 0x009140, 0x0000_0000); // (from line 98944)
    w32(&ember, &bdf, 0x00dc68, 0x0000_0000); // (from line 98946)
    w32(&ember, &bdf, 0x00dc60, 0x0300_0000); // (from line 98949)
    w32(&ember, &bdf, 0x00dc08, 0x0000_0000); // (from line 98964)
    w32(&ember, &bdf, 0x00dc88, 0x0000_0000); // (from line 98965)
    w32(&ember, &bdf, 0x00dc00, 0x2001_2001); // (from line 98970)
    w32(&ember, &bdf, 0x00dc80, 0x0000_0000); // (from line 98971)
    println!("  FIFO/PBDMA final setup done");

    // PMC_INTR_0 final mask (sequence of individual bit-sets from mmiotrace)
    // Replicated as a single write of the final aggregate value
    w32(&ember, &bdf, PMC_INTR_0, 0x5f37_6eff);
    w32(&ember, &bdf, PMC_INTR_1, 0x0000_0000);
    println!("  PMC_INTR_0 ← 0x5f376eff");

    // PRIV_RING master config (line 99019 — accessible before DEVINIT)
    w32(&ember, &bdf, PRIV_RING_MASTER_CONFIG, 0x0000_0002);
    println!("  PRIV_RING_MASTER_CONFIG ← 0x00000002");

    // ── Phase 6: DEVINIT trigger ──────────────────────────────────────────────
    println!("\n━━━ Phase 6: DEVINIT Trigger (PMC_ENABLE = 0xffffffff) ━━━\n");
    println!("  Writing PMC_ENABLE = 0xffffffff...");
    println!("  Hardware will execute DEVINIT from VBIOS (~1.3s pause)...");

    w32(&ember, &bdf, PMC_ENABLE, 0xffff_ffff);
    let devinit_start = Instant::now();

    // Poll PMC_ENABLE until it changes from 0xffffffff (DEVINIT complete)
    print!("  Waiting for DEVINIT: ");
    let _ = io::stdout().flush();
    let mut devinit_done = false;

    for _ in 0..300 {
        std::thread::sleep(Duration::from_millis(10));
        let pmc = r32(&ember, &bdf, PMC_ENABLE);
        if pmc == 0xFFFF_FFFF {
            eprintln!("WARN: D3cold sentinel");
            break;
        }
        print!(".");
        let _ = io::stdout().flush();
        if pmc != 0xffff_ffff && pmc != 0 {
            println!(
                "\n  DEVINIT complete in {}ms! PMC_ENABLE = {pmc:#010x}",
                devinit_start.elapsed().as_millis()
            );
            devinit_done = true;
            break;
        }
    }

    if !devinit_done {
        let pmc = r32(&ember, &bdf, PMC_ENABLE);
        println!("\n  Timeout after 3s. PMC_ENABLE = {pmc:#010x}");
    }

    // ── Phase 7: Post-DEVINIT state probe ────────────────────────────────────
    println!("\n━━━ Phase 7: Post-DEVINIT State Probe ━━━\n");

    let pmc_final = r32(&ember, &bdf, PMC_ENABLE);
    let priv_ring = r32(&ember, &bdf, PRIV_RING_INTR);
    let pmu_ctrl = r32(&ember, &bdf, PMU_CPUCTL);
    let pmu_mb0 = r32(&ember, &bdf, PMU_MB0);
    let sec2_ctrl = r32(&ember, &bdf, SEC2_CPUCTL);
    let pt_hi0 = r32(&ember, &bdf, PTIMER_TIME_HI);
    std::thread::sleep(Duration::from_millis(20));
    let pt_hi1 = r32(&ember, &bdf, PTIMER_TIME_HI);

    println!("  PMC_ENABLE    = {pmc_final:#010x}  (expected 0x5fecdff1)");
    println!("  PRIV_RING     = {priv_ring:#010x}  (bad00100=HS-locked, 0=OK)");
    println!("  PMU cpuctl    = {pmu_ctrl:#010x}  (0x01=idle-halted)");
    println!("  PMU mb0       = {pmu_mb0:#010x}");
    println!("  SEC2 cpuctl   = {sec2_ctrl:#010x}");
    println!(
        "  PTIMER_HI     = {pt_hi0:#010x} → {pt_hi1:#010x}  ({})",
        if pt_hi1 != pt_hi0 {
            "RUNNING"
        } else {
            "FROZEN"
        }
    );

    // Score
    let devinit_ok = pmc_final == 0x5fec_dff1 || pmc_final == 0x5fec_9ff1;
    let priv_ok = priv_ring != 0xbad0_0100;
    let ptimer_ok = pt_hi1 != pt_hi0;

    println!("\n  ┌─────────────────────────────────┐");
    println!("  │  Sovereign Boot Score           │");
    println!(
        "  │  DEVINIT complete:   {}         │",
        if devinit_ok { "✓ YES" } else { "✗ NO " }
    );
    println!(
        "  │  PRIV_RING clear:    {}         │",
        if priv_ok { "✓ YES" } else { "✗ NO " }
    );
    println!(
        "  │  PTIMER running:     {}         │",
        if ptimer_ok { "✓ YES" } else { "✗ NO " }
    );
    println!("  └─────────────────────────────────┘");

    if devinit_ok && priv_ok && ptimer_ok {
        println!("\n  SOVEREIGN BOOT PATH CLEAR: GPU ready for SEC2+FECS init");
        println!("  Next: exp171 — load SEC2 firmware and configure FECS");
    } else if devinit_ok {
        println!("\n  DEVINIT ran. PRIV_RING still HS-locked → SEC2 must fix it.");
        println!("  Next: exp171 — load SEC2 firmware to clear PRIV_RING");
    } else {
        println!("\n  DEVINIT did not complete. Check:");
        println!("  1. Boot Falcon firmware bytes correct (896 words)?");
        println!("  2. PCIe D3hot→D0 needed? (re-run with --reset)");
        println!("  3. PRAMIN/VRAM accessible for firmware load?");
    }

    println!("\n═══════════════════════════════════════");
    println!("  EXP 170 complete.");
    println!("═══════════════════════════════════════");
}

// ── PCIe D3hot→D0 reset via config space ─────────────────────────────────────

fn pcie_d3hot_d0_reset(cfg_path: &str) -> io::Result<()> {
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};

    let mut f = OpenOptions::new().read(true).write(true).open(cfg_path)?;

    // Read PM_CSR at config offset 0x54
    f.seek(SeekFrom::Start(0x54))?;
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    let pm_csr = u32::from_le_bytes(buf);
    println!("  PM_CSR before = {pm_csr:#010x}");

    // Write D3hot (bits [1:0] = 0b11)
    let d3 = (pm_csr & !0x3) | 0x3;
    f.seek(SeekFrom::Start(0x54))?;
    f.write_all(&d3.to_le_bytes())?;
    println!("  PM_CSR ← D3hot ({d3:#010x})");
    std::thread::sleep(Duration::from_millis(50));

    // Write D0 (bits [1:0] = 0b00)
    let d0 = pm_csr & !0x3;
    f.seek(SeekFrom::Start(0x54))?;
    f.write_all(&d0.to_le_bytes())?;
    println!("  PM_CSR ← D0 ({d0:#010x})");

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn banner() {
    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║  EXP 170: Sovereign GV100 Cold Boot from mmiotrace Recipe  ║");
    println!("║  Goal: DEVINIT → PRIV_RING clear → PTIMER running          ║");
    println!("╚═════════════════════════════════════════════════════════════╝\n");
}
