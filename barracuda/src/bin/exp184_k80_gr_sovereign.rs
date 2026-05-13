// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 184: K80 Sovereign GR Engine Boot — ember-wired edition
//!
//! ## Architecture (rewired from raw Bar0Map to EmberClient RPCs)
//!
//! All GPU interaction routes through coral-ember's fork-isolated MMIO gateway.
//! This eliminates the BAR0 mmap bypass and places exp184 under ember's crash
//! interception: if the K80 triggers a PLX D3cold or BAR0 hang, ember's circuit
//! breaker catches it and its immortal VFIO fd keeps the group alive for recovery
//! without a power cycle.
//!
//! Experiment binaries are the research arm of the coral stack:
//! - They discover new register sequences via ember RPCs (not raw MMIO)
//! - Successful sequences are annotated and promoted to `sovereign_stages.rs`
//! - The pipeline evolves in cycles: experiment → discover → promote → ship
//!
//! ## Wiring map (exp184 → sovereign_stages.rs)
//!
//! | exp184 phase              | sovereign_stages.rs function  |
//! |---------------------------|-------------------------------|
//! | Phase 0: pre-conditions   | `bar0_probe`                  |
//! | Phase 1: PMC_ENABLE toggle| `kepler_falcon_boot` Phase 1  |
//! | Phase 2: MMIO init table  | `kepler_falcon_boot` Phase 2  |
//! | Phase 2b: FECS 0x409614   | `kepler_falcon_boot` Phase 2b |
//! | Phase 3: MC_UNK260=0      | `kepler_falcon_boot` Phase 3  |
//! | Phases 4-7: firmware PIO  | `kepler_falcon_boot` Phase 4-7|
//! | Phase 8: MC_UNK260=1      | `kepler_falcon_boot` Phase 8  |
//! | Phases 9-10: CSDATA       | `kepler_falcon_boot` Phase 9-10|
//! | Phase 11: scrub + STARTCPU| `kepler_falcon_boot` Phase 11 |
//! | Phase 12: GR_READY poll   | `kepler_falcon_boot` Phase 12 |
//!
//! ## Usage
//!
//! ```text
//! sudo ./target/release/exp184_k80_gr_sovereign \
//!      [--bdf 0000:4b:00.0]                         \
//!      [--fw-dir ../wateringHole/gk110]              \
//!      [--ember-socket /run/coralreef/ember.sock]    \
//!      [--dry-run]
//! ```
//!
//! Requires coral-ember running and holding the VFIO fd for the target BDF.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use hotspring_barracuda::ember_types::MmioBatchOp;
use hotspring_barracuda::fleet_client::{
    EmberClient, FleetDiscovery, discover_diesel_ember_socket,
};

// ── PMC / MC registers ─────────────────────────────────────────────────────
const BOOT0: u32 = 0x000000;
const PMC_ENABLE: u32 = 0x000200;
const MC_UNK260: u32 = 0x000260;
const PTIMER_LO: u32 = 0x009400; // TIME_0: increments ~32M/sec, readable while running

// ── GR engine PMC_ENABLE bitmask ──────────────────────────────────────────
const PMC_GR_BIT: u32 = 1 << 12; // bit 12 = GR engine (PGRAPH)

// ── PGRAPH / GR engine registers ──────────────────────────────────────────
const GR_READY: u32 = 0x409800; // FECS boot-ready flag (bit 31)
const GR_CTX_SIZE: u32 = 0x409804; // context buffer size (set by FECS at boot)

// ── FECS Falcon registers (base 0x409000) ─────────────────────────────────
const FECS_BASE: u32 = 0x409000;
const FECS_CPUCTL: u32 = FECS_BASE + 0x100;
const FECS_BOOTVEC: u32 = FECS_BASE + 0x104;
const FECS_PC: u32 = FECS_BASE + 0x110;
const FECS_SCTL: u32 = FECS_BASE + 0x240;
// GK110B: HWCFG at 0x10C (DMACTL in newer Falcon, repurposed in Kepler)
const FECS_HWCFG: u32 = FECS_BASE + 0x10C; // 0x40910C
// IMEMC/IMEMD/IMETTAG are handled internally by ember.falcon.upload_imem
const FECS_DMEMC: u32 = FECS_BASE + 0x1C0;
const FECS_DMEMD: u32 = FECS_BASE + 0x1C4;
const FECS_MB0: u32 = FECS_BASE + 0x040;
const FECS_MB1: u32 = FECS_BASE + 0x044;

// ── GPCCS Falcon registers (broadcast PIO at 0x41A000) ────────────────────
const GPCCS_BASE: u32 = 0x41A000;
const GPCCS_CPUCTL: u32 = GPCCS_BASE + 0x100;
const GPCCS_BOOTVEC: u32 = GPCCS_BASE + 0x104;
const GPCCS_PC: u32 = GPCCS_BASE + 0x110;
const GPCCS_HWCFG: u32 = GPCCS_BASE + 0x10C; // 0x41A10C
// IMEMC/IMEMD/IMETTAG are handled internally by ember.falcon.upload_imem
const GPCCS_DMEMC: u32 = GPCCS_BASE + 0x1C0;
const GPCCS_DMEMD: u32 = GPCCS_BASE + 0x1C4;
const GPCCS_MB0: u32 = GPCCS_BASE + 0x040;
const GPCCS_MB1: u32 = GPCCS_BASE + 0x044;

// ── GPC status ────────────────────────────────────────────────────────────
const GPC0_BOOT0: u32 = 0x418000;
const GPC1_BOOT0: u32 = 0x428000;

// ── FECS reset handshake (GR wrapper, not Falcon CPUCTL) ──────────────────
// nouveau's gf100_gr_fecs_reset: arms FECS's internal PRIV ring master unit
// BEFORE MC_UNK260=0. Sequence: 0x70 → 0x30 → poll bit4=0 → 0x10.
const FECS_RESET: u32 = FECS_BASE + 0x614; // 0x409614 — promoted to sovereign_stages.rs Phase 2b

// ── FECS init command ─────────────────────────────────────────────────────
// FECS reads MB0 exactly once at startup. MB0=4 → INIT_CTXSW:
//   train PRIV ring, establish GPC links, set GR_READY.
// Promoted from exp184 discovery → sovereign_stages.rs `FECS_CMD_INIT`.
const FECS_CMD_INIT: u32 = 0x0000_0004;

// ── DMEMC auto-increment flags ─────────────────────────────────────────────
const DMEMC_AINCW: u32 = 0x0100_0000; // bit 24: auto-increment write pointer

// ── FECS DMEM layout (r_hub = 0x600 from gk110b_grctx) ───────────────────
const FECS_DMEM_FW_OFFSET: u32 = 0x600;

// ── GPCCS DMEM layout (r_gpc ≈ 0x7ff → aligned to 0x800) ────────────────
const GPCCS_DMEM_FW_OFFSET: u32 = 0x800;

// ── IMEM page geometry (for display — PIO is handled internally by ember) ─
const IMEM_WORDS_PER_PAGE: usize = 64; // 256 bytes / 4 bytes per word

fn extract_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn load_fw_bytes(dir: &Path, name: &str) -> Vec<u8> {
    let path = dir.join(name);
    std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("FATAL: cannot read {}: {e}", path.display());
        std::process::exit(1);
    })
}

fn load_fw_words(dir: &Path, name: &str) -> Vec<u32> {
    let bytes = load_fw_bytes(dir, name);
    bytes
        .chunks(4)
        .map(|c| {
            u32::from_le_bytes([
                *c.first().unwrap_or(&0),
                *c.get(1).unwrap_or(&0),
                *c.get(2).unwrap_or(&0),
                *c.get(3).unwrap_or(&0),
            ])
        })
        .collect()
}

fn load_mmio_init(dir: &Path, name: &str) -> Vec<(u32, u32)> {
    let bytes = load_fw_bytes(dir, name);
    bytes
        .chunks(8)
        .filter(|c| c.len() == 8)
        .map(|c| {
            let addr = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let val = u32::from_le_bytes([c[4], c[5], c[6], c[7]]);
            (addr, val)
        })
        .collect()
}

// ── Ember RPC helpers ──────────────────────────────────────────────────────

/// Read a BAR0 register via ember, returning 0xDEAD_DEAD on error.
fn r32(ember: &EmberClient, bdf: &str, offset: u32) -> u32 {
    match ember.mmio_read(bdf, offset) {
        Ok(r) => r.value,
        Err(e) => {
            eprintln!("  WARN: ember.mmio.read({offset:#010x}): {e}");
            0xDEAD_DEAD
        }
    }
}

/// Write a BAR0 register via ember, ignoring errors (GPU may be in D3cold).
fn w32(ember: &EmberClient, bdf: &str, offset: u32, value: u32) {
    if let Err(e) = ember.mmio_write(bdf, offset, value) {
        eprintln!("  WARN: ember.mmio.write({offset:#010x}, {value:#010x}): {e}");
    }
}

/// Build a CSDATA batch: one DMEMC setup write followed by N DMEMD word writes.
/// The AINCW flag auto-increments the DMEM address on each DMEMD write.
fn csdata_batch_ops(dmemc: u32, dmemd: u32, words: &[u32], byte_offset: u32) -> Vec<MmioBatchOp> {
    let mut ops = Vec::with_capacity(1 + words.len());
    ops.push(MmioBatchOp::write(dmemc, DMEMC_AINCW | byte_offset));
    for &w in words {
        ops.push(MmioBatchOp::write(dmemd, w));
    }
    ops
}

/// PIO-upload firmware code to Falcon IMEM via ember.falcon.upload_imem.
///
/// Note: ember handles the page-by-page IMEMC/IMEMD/IMETTAG protocol internally.
fn upload_imem(ember: &EmberClient, bdf: &str, base: u32, imem_words: &[u32]) {
    // Convert words to bytes for the RPC (ember does the page-by-page PIO)
    let bytes: Vec<u8> = imem_words.iter().flat_map(|w| w.to_le_bytes()).collect();
    match ember.falcon_upload_imem(bdf, base, 0, &bytes, 0, false) {
        Ok(r) if r.ok => {}
        Ok(r) => eprintln!("  WARN: falcon.upload_imem ok=false bytes={:?}", r.bytes),
        Err(e) => eprintln!("  WARN: falcon.upload_imem error: {e}"),
    }
}

/// PIO-upload firmware data to Falcon DMEM at byte_offset via ember.falcon.upload_dmem.
fn upload_dmem(ember: &EmberClient, bdf: &str, base: u32, byte_offset: u32, data_words: &[u32]) {
    let bytes: Vec<u8> = data_words.iter().flat_map(|w| w.to_le_bytes()).collect();
    match ember.falcon_upload_dmem(bdf, base, byte_offset, &bytes) {
        Ok(r) if r.ok => {}
        Ok(r) => eprintln!("  WARN: falcon.upload_dmem ok=false bytes={:?}", r.bytes),
        Err(e) => eprintln!("  WARN: falcon.upload_dmem error: {e}"),
    }
}

/// Discover and connect to the coral-ember instance for a given BDF.
///
/// Search order: diesel engine scan → fleet discovery → slug-based socket → legacy.
/// Accepts `--ember-socket` CLI override to skip discovery.
fn connect_ember(bdf: &str, override_socket: Option<&str>) -> EmberClient {
    if let Some(sock) = override_socket {
        return EmberClient::connect(sock);
    }
    if let Some(sock) = discover_diesel_ember_socket(bdf) {
        eprintln!("  diesel engine: found ember at {}", sock.display());
        return EmberClient::connect(sock.to_string_lossy().as_ref());
    }
    if let Ok(disc) = FleetDiscovery::load_default() {
        if let Some(sock) = disc.file().routes.get(bdf) {
            return EmberClient::connect(sock);
        }
    }
    let slug = bdf.replace(':', "-");
    let candidates = [
        format!("/run/coralreef/fleet/ember-{slug}.sock"),
        format!("/run/coralreef/ember-{slug}.sock"),
        "/run/coralreef/ember.sock".to_string(),
    ];
    for sock in &candidates {
        if Path::new(sock).exists() {
            return EmberClient::connect(sock);
        }
    }
    eprintln!(
        "FATAL: no ember socket found for BDF {bdf}.\n\
         Start coral-glowplug (diesel engine) and verify: coralctl cylinder list\n\
         Override with: --ember-socket /path/to/ember.sock"
    );
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bdf = extract_arg(&args, "--bdf").unwrap_or_else(|| "0000:4b:00.0".into());
    let fw_dir = extract_arg(&args, "--fw-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../wateringHole/gk110"));
    let ember_socket = extract_arg(&args, "--ember-socket");
    let dry_run = args.iter().any(|a| a == "--dry-run");

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  EXP 184 — K80 Sovereign GR Engine Boot (ember-wired)          ║");
    println!("║  All GPU access via coral-ember RPCs — crash protected          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!("  BDF:    {bdf}");
    println!("  FW dir: {}", fw_dir.display());
    println!("  Wiring: coral-ember (all MMIO via JSON-RPC — no raw BAR0 mmap)");
    if dry_run {
        println!("  Mode:   DRY RUN");
    }

    // ── Connect to ember ───────────────────────────────────────────────────
    let ember = connect_ember(&bdf, ember_socket.as_deref());
    println!("  Ember:  connected ✓");

    // ── Load firmware and tables ───────────────────────────────────────────
    let fecs_code = load_fw_words(&fw_dir, "fecs_code.bin");
    let fecs_data = load_fw_words(&fw_dir, "fecs_data.bin");
    let gpccs_code = load_fw_words(&fw_dir, "gpccs_code.bin");
    let gpccs_data = load_fw_words(&fw_dir, "gpccs_data.bin");

    // CSDATA: context-switch register descriptor lists
    let hub_csdata = load_fw_words(&fw_dir, "hub_csdata.bin"); // → FECS DMEM[0x000]
    let gpc0_csdata = load_fw_words(&fw_dir, "gpc0_csdata.bin"); // → GPCCS DMEM[0x000]
    let gpc1_csdata = load_fw_words(&fw_dir, "gpc1_csdata.bin");
    let tpc0_csdata = load_fw_words(&fw_dir, "tpc0_csdata.bin");
    let ppc_csdata = load_fw_words(&fw_dir, "ppc_csdata.bin");

    let mut gpccs_csdata: Vec<u32> = Vec::with_capacity(
        gpc0_csdata.len() + gpc1_csdata.len() + tpc0_csdata.len() + ppc_csdata.len(),
    );
    gpccs_csdata.extend_from_slice(&gpc0_csdata);
    gpccs_csdata.extend_from_slice(&gpc1_csdata);
    gpccs_csdata.extend_from_slice(&tpc0_csdata);
    gpccs_csdata.extend_from_slice(&ppc_csdata);

    let mmio_init = load_mmio_init(&fw_dir, "gr_mmio_init.bin");

    println!("\n  Firmware/tables loaded:");
    println!(
        "    fecs_code:    {} words ({} bytes)",
        fecs_code.len(),
        fecs_code.len() * 4
    );
    println!(
        "    fecs_data:    {} words, DMEM offset=0x{FECS_DMEM_FW_OFFSET:04x}",
        fecs_data.len()
    );
    println!(
        "    gpccs_code:   {} words ({} bytes)",
        gpccs_code.len(),
        gpccs_code.len() * 4
    );
    println!(
        "    gpccs_data:   {} words, DMEM offset=0x{GPCCS_DMEM_FW_OFFSET:04x}",
        gpccs_data.len()
    );
    println!(
        "    hub_csdata:   {} words → FECS DMEM[0x000]",
        hub_csdata.len()
    );
    println!(
        "    gpccs_csdata: {} words → GPCCS DMEM[0x000]  (gpc0={} gpc1={} tpc={} ppc={})",
        gpccs_csdata.len(),
        gpc0_csdata.len(),
        gpc1_csdata.len(),
        tpc0_csdata.len(),
        ppc_csdata.len()
    );
    println!("    mmio_init:    {} register writes", mmio_init.len());

    let hub_bytes = (hub_csdata.len() as u32) * 4;
    let gpccs_bytes = (gpccs_csdata.len() as u32) * 4;
    if hub_bytes > FECS_DMEM_FW_OFFSET {
        eprintln!("FATAL: hub_csdata ({hub_bytes}B) overflows FECS DMEM fw offset");
        std::process::exit(1);
    }
    if gpccs_bytes > GPCCS_DMEM_FW_OFFSET {
        eprintln!("FATAL: gpccs_csdata ({gpccs_bytes}B) overflows GPCCS DMEM fw offset");
        std::process::exit(1);
    }
    println!(
        "    Layout OK: hub ends {hub_bytes:#06x} < fw@{FECS_DMEM_FW_OFFSET:#06x}  \
              gpccs ends {gpccs_bytes:#06x} < fw@{GPCCS_DMEM_FW_OFFSET:#06x}  ✓"
    );

    // ── Phase 0: Pre-condition check ──────────────────────────────────────
    println!("\n━━━ Phase 0: Pre-conditions (via ember.mmio.read) ━━━\n");

    let boot0 = r32(&ember, &bdf, BOOT0);
    let pmc_en = r32(&ember, &bdf, PMC_ENABLE);
    let fecs_sctl = r32(&ember, &bdf, FECS_SCTL);
    let gpc0_pre = r32(&ember, &bdf, GPC0_BOOT0);
    let gpc1_pre = r32(&ember, &bdf, GPC1_BOOT0);
    let pt0 = r32(&ember, &bdf, PTIMER_LO);
    std::thread::sleep(Duration::from_millis(5));
    let pt1 = r32(&ember, &bdf, PTIMER_LO);

    println!("  BOOT0        = {boot0:#010x}  (GK210 expect 0x0f22d0a1)");
    if boot0 == 0xFFFF_FFFF || boot0 == 0 {
        eprintln!(
            "FATAL: GPU link dead (BOOT0=0x{boot0:08x}). Check PLX D0 and ember circuit breaker."
        );
        std::process::exit(1);
    }
    println!(
        "  PMC_ENABLE   = {pmc_en:#010x}  (bit12 GR = {})",
        if pmc_en & PMC_GR_BIT != 0 {
            "ON ✓"
        } else {
            "OFF ✗"
        }
    );
    println!("  FECS_SCTL    = {fecs_sctl:#010x}");
    println!(
        "  GPC0_BOOT0   = {gpc0_pre:#010x}  ({})",
        if gpc0_pre == 0xBADF_1100 {
            "PRIV ring corrupt ✗"
        } else if gpc0_pre == 0xFFFF_FFFF {
            "dead ✗"
        } else {
            "accessible ✓"
        }
    );
    println!("  GPC1_BOOT0   = {gpc1_pre:#010x}");
    println!(
        "  PTIMER:      {} ({} → {})",
        if pt1 != pt0 {
            "RUNNING ✓"
        } else {
            "FROZEN ✗"
        },
        pt0,
        pt1
    );

    if dry_run {
        println!("\n  DRY RUN — no writes. Exiting.");
        return;
    }

    // ── Phase 1: PMC_ENABLE GR OFF→ON toggle (PRIV ring reset) ───────────
    // Promoted to sovereign_stages.rs kepler_falcon_boot Phase 1.
    // Always toggle: a 1→1 write does NOT cause a PRIV ring hardware reset;
    // only a 0→1 transition on bit 12 resets the PGRAPH domain.
    println!("\n━━━ Phase 1: PMC_ENABLE GR OFF→ON toggle (PRIV ring reset) ━━━\n");
    if pmc_en & PMC_GR_BIT != 0 {
        println!("  GR ON — toggling OFF first to ensure clean PRIV ring reset...");
        w32(&ember, &bdf, PMC_ENABLE, pmc_en & !PMC_GR_BIT);
        std::thread::sleep(Duration::from_millis(5));
    }
    w32(&ember, &bdf, PMC_ENABLE, pmc_en | PMC_GR_BIT);
    std::thread::sleep(Duration::from_millis(5));
    let pmc_after = r32(&ember, &bdf, PMC_ENABLE);
    println!(
        "  PMC_ENABLE after = {pmc_after:#010x}  ({})",
        if pmc_after & PMC_GR_BIT != 0 {
            "GR ON ✓"
        } else {
            "GR still OFF ✗"
        }
    );

    // ── Phase 2: GR MMIO init table (153 BAR0 writes) ─────────────────────
    // Promoted to sovereign_stages.rs kepler_falcon_boot Phase 2.
    // Applied as one mmio.batch RPC (single round-trip for all 153 writes).
    println!(
        "\n━━━ Phase 2: GR MMIO init ({} writes via mmio.batch) ━━━\n",
        mmio_init.len()
    );
    {
        let ops: Vec<MmioBatchOp> = mmio_init
            .iter()
            .map(|&(addr, val)| MmioBatchOp::write(addr, val))
            .collect();
        match ember.mmio_batch(&bdf, &ops) {
            Ok(_) => println!("  Applied {} register writes ✓", mmio_init.len()),
            Err(e) => println!("  WARN: mmio.batch failed: {e}"),
        }
    }
    std::thread::sleep(Duration::from_micros(100));

    // ── Phase 2b: FECS reset handshake (0x409614) ─────────────────────────
    // Promoted to sovereign_stages.rs kepler_falcon_boot Phase 2b.
    // Arms FECS's internal PRIV ring master unit BEFORE MC_UNK260=0.
    // Without this, FECS's private hub init path stays broken and INIT_CTXSW
    // hub entries (0x404xxx) fail with error codes 5/6.
    println!("\n━━━ Phase 2b: FECS reset handshake (0x409614 → arm PRIV ring master) ━━━\n");
    w32(&ember, &bdf, FECS_RESET, 0x70); // phase 1: assert
    w32(&ember, &bdf, FECS_RESET, 0x30); // phase 2: clear bit 6
    let hs_start = Instant::now();
    let mut fecs_reset_ok = false;
    while hs_start.elapsed() < Duration::from_millis(2000) {
        let rst = r32(&ember, &bdf, FECS_RESET);
        if rst & 0x10 == 0 {
            fecs_reset_ok = true;
            break;
        }
        std::thread::sleep(Duration::from_micros(10));
    }
    let rst_final = r32(&ember, &bdf, FECS_RESET);
    println!(
        "  0x409614 = {rst_final:#010x}  ({})",
        if fecs_reset_ok {
            "handshake complete ✓"
        } else {
            "TIMEOUT ✗"
        }
    );
    w32(&ember, &bdf, FECS_RESET, 0x10); // release
    w32(&ember, &bdf, FECS_HWCFG, 0x0); // clear HWCFG (disable scrub)
    w32(&ember, &bdf, FECS_CPUCTL, 0x0); // clear CPUCTL
    println!("  0x409614 ← 0x10 (release)  FECS_HWCFG ← 0  FECS_CPUCTL ← 0");

    // ── Phase 3: Assert HRESET (MC_UNK260 = 0) ────────────────────────────
    // Promoted to sovereign_stages.rs kepler_falcon_boot Phase 3.
    println!("\n━━━ Phase 3: MC_UNK260 = 0 (assert HRESET) ━━━\n");
    w32(&ember, &bdf, MC_UNK260, 0x0);
    std::thread::sleep(Duration::from_micros(20));
    println!("  MC_UNK260 ← 0x0  (Falcons in HRESET) ✓");

    // ── Phase 4: FECS firmware data → FECS DMEM[0x600] (while in HRESET) ─
    // Promoted to sovereign_stages.rs kepler_falcon_boot Phase 4.
    // ember.falcon.upload_dmem handles the DMEMC/DMEMD PIO internally.
    println!(
        "\n━━━ Phase 4: FECS firmware data → DMEM[0x{FECS_DMEM_FW_OFFSET:04x}] \
              ({} words) ━━━\n",
        fecs_data.len()
    );
    upload_dmem(&ember, &bdf, FECS_BASE, FECS_DMEM_FW_OFFSET, &fecs_data);
    // Readback verification via mmio.batch (DMEMC→address, then read DMEMD)
    {
        let ops = vec![
            MmioBatchOp::write(FECS_DMEMC, FECS_DMEM_FW_OFFSET),
            MmioBatchOp::read(FECS_DMEMD),
        ];
        if let Ok(result) = ember.mmio_batch(&bdf, &ops) {
            let fw_v0 = result.read_value(1).unwrap_or(0xDEAD);
            println!(
                "  FECS DMEM[0x{FECS_DMEM_FW_OFFSET:04x}] readback: {fw_v0:#010x} \
                      (expected {:#010x}) {}",
                fecs_data[0],
                if fw_v0 == fecs_data[0] {
                    "✓"
                } else {
                    "✗ mismatch"
                }
            );
        }
    }

    // ── Phase 5: FECS firmware code → FECS IMEM ───────────────────────────
    // Promoted to sovereign_stages.rs kepler_falcon_boot Phase 5.
    // ember.falcon.upload_imem handles page-by-page IMEMC/IMEMD/IMETTAG.
    println!(
        "\n━━━ Phase 5: FECS IMEM ({} words, {} pages) ━━━\n",
        fecs_code.len(),
        fecs_code.len().div_ceil(IMEM_WORDS_PER_PAGE)
    );
    upload_imem(&ember, &bdf, FECS_BASE, &fecs_code);
    println!("  FECS IMEM loaded via ember.falcon.upload_imem ✓");

    // ── Phase 6: GPCCS firmware data → GPCCS DMEM[0x800] (in HRESET) ─────
    println!(
        "\n━━━ Phase 6: GPCCS firmware data → DMEM[0x{GPCCS_DMEM_FW_OFFSET:04x}] \
              ({} words) ━━━\n",
        gpccs_data.len()
    );
    upload_dmem(&ember, &bdf, GPCCS_BASE, GPCCS_DMEM_FW_OFFSET, &gpccs_data);
    println!("  GPCCS DMEM[0x{GPCCS_DMEM_FW_OFFSET:04x}] loaded ✓");

    // ── Phase 7: GPCCS firmware code → GPCCS IMEM ────────────────────────
    println!(
        "\n━━━ Phase 7: GPCCS IMEM ({} words, {} pages) ━━━\n",
        gpccs_code.len(),
        gpccs_code.len().div_ceil(IMEM_WORDS_PER_PAGE)
    );
    upload_imem(&ember, &bdf, GPCCS_BASE, &gpccs_code);
    println!("  GPCCS IMEM loaded ✓");

    // ── Phase 8: Release HRESET (MC_UNK260 = 1) ───────────────────────────
    // CSDATA MUST be loaded after this release (matches nouveau sequence).
    // Falcon hardware scrub runs after HRESET release; loading CSDATA before
    // would risk the scrub zeroing DMEM[0x000..] where CSDATA lives.
    println!("\n━━━ Phase 8: MC_UNK260 = 1 (release HRESET) ━━━\n");
    w32(&ember, &bdf, MC_UNK260, 0x1);
    std::thread::sleep(Duration::from_millis(2));
    println!("  MC_UNK260 ← 0x1  (Falcons released from HRESET) ✓");

    // ── Phase 9: Hub CSDATA → FECS DMEM[0x000] (POST-HRESET) ────────────
    // Promoted to sovereign_stages.rs kepler_falcon_boot Phase 9.
    // Applied as one mmio.batch: DMEMC setup + all CSDATA words (AINCW auto-increment).
    println!(
        "\n━━━ Phase 9: Hub CSDATA → FECS DMEM[0x000] ({} words) — POST-HRESET ━━━\n",
        hub_csdata.len()
    );
    {
        let ops = csdata_batch_ops(FECS_DMEMC, FECS_DMEMD, &hub_csdata, 0);
        match ember.mmio_batch(&bdf, &ops) {
            Ok(_) => {}
            Err(e) => println!("  WARN: hub CSDATA batch failed: {e}"),
        }
        // Readback first two words
        let rb_ops = vec![
            MmioBatchOp::write(FECS_DMEMC, 0x0),
            MmioBatchOp::read(FECS_DMEMD),
            MmioBatchOp::write(FECS_DMEMC, 0x4),
            MmioBatchOp::read(FECS_DMEMD),
        ];
        if let Ok(r) = ember.mmio_batch(&bdf, &rb_ops) {
            let c0 = r.read_value(1).unwrap_or(0xDEAD);
            let c1 = r.read_value(3).unwrap_or(0xDEAD);
            println!(
                "  FECS DMEM[0x000]=0x{c0:08x} (expected 0x{:08x}) {}",
                hub_csdata[0],
                if c0 == hub_csdata[0] { "✓" } else { "✗" }
            );
            println!(
                "  FECS DMEM[0x004]=0x{c1:08x} (expected 0x{:08x}) {}",
                hub_csdata.get(1).copied().unwrap_or(0),
                if c1 == hub_csdata.get(1).copied().unwrap_or(0) {
                    "✓"
                } else {
                    "✗"
                }
            );
        }
    }

    // ── Phase 10: GPCCS CSDATA → GPCCS DMEM[0x000] (POST-HRESET) ─────────
    println!(
        "\n━━━ Phase 10: GPCCS CSDATA → GPCCS DMEM[0x000] ({} words) — POST-HRESET ━━━\n",
        gpccs_csdata.len()
    );
    {
        let ops = csdata_batch_ops(GPCCS_DMEMC, GPCCS_DMEMD, &gpccs_csdata, 0);
        match ember.mmio_batch(&bdf, &ops) {
            Ok(_) => {}
            Err(e) => println!("  WARN: GPCCS CSDATA batch failed: {e}"),
        }
        let rb_ops = vec![
            MmioBatchOp::write(GPCCS_DMEMC, 0x0),
            MmioBatchOp::read(GPCCS_DMEMD),
            MmioBatchOp::write(GPCCS_DMEMC, 0x4),
            MmioBatchOp::read(GPCCS_DMEMD),
        ];
        if let Ok(r) = ember.mmio_batch(&bdf, &rb_ops) {
            let c0 = r.read_value(1).unwrap_or(0xDEAD);
            let c1 = r.read_value(3).unwrap_or(0xDEAD);
            println!(
                "  GPCCS DMEM[0x000]=0x{c0:08x} (expected 0x{:08x}) {}",
                gpccs_csdata[0],
                if c0 == gpccs_csdata[0] { "✓" } else { "✗" }
            );
            println!(
                "  GPCCS DMEM[0x004]=0x{c1:08x} (expected 0x{:08x}) {}",
                gpccs_csdata.get(1).copied().unwrap_or(0),
                if c1 == gpccs_csdata.get(1).copied().unwrap_or(0) {
                    "✓"
                } else {
                    "✗"
                }
            );
        }
    }

    // ── Phase 11: HWCFG scrub wait + GPCCS boot + FECS boot ──────────────
    // Promoted to sovereign_stages.rs kepler_falcon_boot Phase 11.
    println!("\n━━━ Phase 11: Scrub wait + GPCCS boot + FECS boot ━━━\n");

    // Wait for FECS HWCFG scrub (bits 1-2 must clear before STARTCPU)
    let scrub_start = Instant::now();
    loop {
        let hwcfg = r32(&ember, &bdf, FECS_HWCFG);
        if hwcfg & 0x6 == 0 {
            break;
        }
        if scrub_start.elapsed() >= Duration::from_millis(500) {
            println!("  WARN: FECS scrub timeout (HWCFG={hwcfg:#010x})");
            break;
        }
        std::thread::sleep(Duration::from_micros(100));
    }
    let gscrub_start = Instant::now();
    loop {
        let hwcfg = r32(&ember, &bdf, GPCCS_HWCFG);
        if hwcfg & 0x6 == 0 {
            break;
        }
        if gscrub_start.elapsed() >= Duration::from_millis(500) {
            println!("  WARN: GPCCS scrub timeout (HWCFG={hwcfg:#010x})");
            break;
        }
        std::thread::sleep(Duration::from_micros(100));
    }

    let fecs_hwcfg = r32(&ember, &bdf, FECS_HWCFG);
    let gpccs_hwcfg = r32(&ember, &bdf, GPCCS_HWCFG);
    println!("  FECS_HWCFG   = {fecs_hwcfg:#010x}  (bits 1-2 should be 0)");
    println!("  GPCCS_HWCFG  = {gpccs_hwcfg:#010x}");

    // Boot GPCCS first (PRIV ring slave must be listening before FECS master starts).
    // Pre-load MB0=4 (INIT_CTXSW): GPCCS reads MB0 once at startup.
    // Promoted from exp184 discovery → sovereign_stages.rs: GPCCS-first boot.
    {
        let ops = vec![
            MmioBatchOp::write(GPCCS_BOOTVEC, 0x0),
            MmioBatchOp::write(GPCCS_HWCFG, 0x0), // clear scrub enable
            MmioBatchOp::write(GPCCS_MB1, 0x0),
            MmioBatchOp::write(GPCCS_MB0, FECS_CMD_INIT), // MB0=4 BEFORE STARTCPU
            MmioBatchOp::write(GPCCS_CPUCTL, 0x02),       // STARTCPU
        ];
        match ember.mmio_batch(&bdf, &ops) {
            Ok(_) => {}
            Err(e) => println!("  WARN: GPCCS boot batch: {e}"),
        }
        println!("  GPCCS: BOOTVEC←0 HWCFG←0 MB0←0x{FECS_CMD_INIT:08x} STARTCPU ✓");
    }
    std::thread::sleep(Duration::from_millis(20));

    let gpccs_cpuctl_running = r32(&ember, &bdf, GPCCS_CPUCTL);
    let gpccs_pc_running = r32(&ember, &bdf, GPCCS_PC);
    println!(
        "  GPCCS_CPUCTL after start = {gpccs_cpuctl_running:#010x}  \
              (0x00=running ✓, 0x10=still halted ✗)"
    );
    println!("  GPCCS_PC     after start = {gpccs_pc_running:#010x}");

    // Boot FECS: pre-load MB0=4 (INIT_CTXSW) before STARTCPU.
    // Promoted from exp184 discovery → sovereign_stages.rs: MB0=4 pre-load.
    {
        let ops = vec![
            MmioBatchOp::write(FECS_BOOTVEC, 0x0),
            MmioBatchOp::write(FECS_HWCFG, 0x0), // clear scrub enable
            MmioBatchOp::write(FECS_MB1, 0x0),
            MmioBatchOp::write(FECS_MB0, FECS_CMD_INIT), // MB0=4 BEFORE STARTCPU
            MmioBatchOp::write(FECS_CPUCTL, 0x02),       // STARTCPU
        ];
        match ember.mmio_batch(&bdf, &ops) {
            Ok(_) => {}
            Err(e) => println!("  WARN: FECS boot batch: {e}"),
        }
        println!("  FECS:  BOOTVEC←0 HWCFG←0 MB0←0x{FECS_CMD_INIT:08x} STARTCPU ✓");
    }

    // ── Phase 12: Poll GR_READY (0x409800 bit 31) ─────────────────────────
    println!("\n━━━ Phase 12: Poll GR_READY + MB0 completion (0x409800 bit 31) ━━━\n");
    println!(
        "  MB0 pre-poll = {:#010x}  (expect 0x00000004 = INIT_CTXSW in progress)",
        r32(&ember, &bdf, FECS_MB0)
    );

    let poll_timeout = Duration::from_millis(8000);
    let poll_start = Instant::now();
    let mut fecs_ready = false;
    let mut last_ready = 0xDEAD_BEEF_u32;
    let mut last_fecs_pc = 0xDEAD_BEEF_u32;
    let mut last_gpccs_pc = 0xDEAD_BEEF_u32;
    let mut last_sample_ms = 0u64;

    loop {
        let elapsed_ms = poll_start.elapsed().as_millis() as u64;
        let ready = r32(&ember, &bdf, GR_READY);
        let fpc = r32(&ember, &bdf, FECS_PC);
        let gpc_pc = r32(&ember, &bdf, GPCCS_PC);

        if ready != last_ready {
            println!("  t={elapsed_ms}ms GR_READY changed → {ready:#010x}");
            last_ready = ready;
        }
        if fpc != last_fecs_pc {
            println!("  t={elapsed_ms}ms FECS_PC moved    → {fpc:#010x}");
            last_fecs_pc = fpc;
        }
        if gpc_pc != last_gpccs_pc {
            println!("  t={elapsed_ms}ms GPCCS_PC moved   → {gpc_pc:#010x}");
            last_gpccs_pc = gpc_pc;
        }

        if elapsed_ms >= last_sample_ms + 500 {
            let mb0 = r32(&ember, &bdf, FECS_MB0);
            let mb1 = r32(&ember, &bdf, FECS_MB1);
            let fc = r32(&ember, &bdf, FECS_CPUCTL);
            let gc = r32(&ember, &bdf, GPCCS_CPUCTL);
            let gpc0 = r32(&ember, &bdf, GPC0_BOOT0);
            println!(
                "  t={elapsed_ms}ms  MB0={mb0:#010x} MB1={mb1:#010x} \
                      F_CTL={fc:#010x} G_CTL={gc:#010x} GPC0={gpc0:#010x}"
            );
            last_sample_ms = elapsed_ms;
        }

        if ready & 0x8000_0000 != 0 {
            fecs_ready = true;
            break;
        }

        // D3cold sentinel: ember's circuit breaker should catch this, but log here
        // for immediate diagnosis. If we see 0xffffffff, ember has already cut the
        // MMIO path; the next mmio_read call will return an error.
        if ready == 0xFFFF_FFFF {
            println!("  WARN: GR_READY=0xffffffff at t={elapsed_ms}ms — possible D3cold");
            println!("  → ember circuit breaker should intercept; check journalctl -u coral-ember");
            break;
        }

        if poll_start.elapsed() >= poll_timeout {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let poll_elapsed = poll_start.elapsed();

    // ── Phase 13: Post-boot diagnostic ────────────────────────────────────
    println!("\n━━━ Phase 13: Post-Boot Diagnostic ━━━\n");

    let fecs_cpuctl_post = r32(&ember, &bdf, FECS_CPUCTL);
    let fecs_pc_post = r32(&ember, &bdf, FECS_PC);
    let fecs_mb0_post = r32(&ember, &bdf, FECS_MB0);
    let fecs_mb1_post = r32(&ember, &bdf, FECS_MB1);
    let gr_ready_post = r32(&ember, &bdf, GR_READY);
    let gr_ctx_size = r32(&ember, &bdf, GR_CTX_SIZE);
    let gpc0_post = r32(&ember, &bdf, GPC0_BOOT0);
    let gpc1_post = r32(&ember, &bdf, GPC1_BOOT0);
    let gpccs_cpuctl_post = r32(&ember, &bdf, GPCCS_CPUCTL);
    let gpccs_pc_post = r32(&ember, &bdf, GPCCS_PC);
    let pmc_post = r32(&ember, &bdf, PMC_ENABLE);
    let pt_post = r32(&ember, &bdf, PTIMER_LO);

    println!(
        "  GR_READY     = {gr_ready_post:#010x}  (bit31={})",
        (gr_ready_post >> 31) & 1
    );
    println!("  GR_CTX_SIZE  = {gr_ctx_size:#010x}  ({gr_ctx_size} bytes)");
    println!("  FECS_CPUCTL  = {fecs_cpuctl_post:#010x}");
    println!("  FECS_PC      = {fecs_pc_post:#010x}");
    println!("  FECS_MB0     = {fecs_mb0_post:#010x}");
    println!("  FECS_MB1     = {fecs_mb1_post:#010x}");
    println!("  GPCCS_CPUCTL = {gpccs_cpuctl_post:#010x}  (0x00=running ✓)");
    println!("  GPCCS_PC     = {gpccs_pc_post:#010x}");
    println!(
        "  GPC0_BOOT0   = {gpc0_post:#010x}  ({})",
        if gpc0_post == 0xBADF_1100 {
            "power-gated ✗"
        } else if gpc0_post == 0xFFFF_FFFF {
            "dead ✗"
        } else {
            "alive ✓"
        }
    );
    println!("  GPC1_BOOT0   = {gpc1_post:#010x}");
    println!("  PMC_ENABLE   = {pmc_post:#010x}");
    println!(
        "  PTIMER       = {pt_post:#010x}  ({})",
        if pt_post != pt0 {
            "RUNNING ✓"
        } else {
            "FROZEN ✗"
        }
    );

    // DMEM dump via mmio.batch (AINCR: bit 25 = auto-increment on read)
    let dmemc_aincr: u32 = 0x0200_0000;
    print!("  FECS DMEM[0..15]: ");
    {
        let mut ops = vec![MmioBatchOp::write(FECS_DMEMC, dmemc_aincr)];
        for _ in 0..16 {
            ops.push(MmioBatchOp::read(FECS_DMEMD));
        }
        if let Ok(r) = ember.mmio_batch(&bdf, &ops) {
            for i in 1..=16 {
                print!("{:#010x} ", r.read_value(i).unwrap_or(0xDEAD));
            }
        }
    }
    println!();
    print!("  GPCCS DMEM[0..15]: ");
    {
        let mut ops = vec![MmioBatchOp::write(GPCCS_DMEMC, dmemc_aincr)];
        for _ in 0..16 {
            ops.push(MmioBatchOp::read(GPCCS_DMEMD));
        }
        if let Ok(r) = ember.mmio_batch(&bdf, &ops) {
            for i in 1..=16 {
                print!("{:#010x} ", r.read_value(i).unwrap_or(0xDEAD));
            }
        }
    }
    println!();

    // ── Scorecard ──────────────────────────────────────────────────────────
    println!();
    let gpc0_alive = gpc0_post != 0xBADF_1100 && gpc0_post != 0xFFFF_FFFF;
    let ctx_valid = gr_ctx_size > 0 && gr_ctx_size < 0x100_0000;
    let ptimer_ok = pt_post != pt0;

    println!("  ┌──────────────────────────────────────────────────────┐");
    println!("  │  EXP 184 — K80 Sovereign GR Boot Scorecard           │");
    println!("  │  Wiring: coral-ember RPCs (crash-protected MMIO)     │");
    println!("  ├──────────────────────────────────────────────────────┤");
    println!(
        "  │  PRIV ring restored:  {}  │",
        if gpc0_alive {
            "✓ YES — GPC accessible               "
        } else {
            "✗ NO  — 0xbadf1100 persists          "
        }
    );
    println!(
        "  │  GR_READY bit31:      {}  │",
        if fecs_ready {
            "✓ SET — FECS booted!                 "
        } else {
            "✗ CLEAR — FECS did not complete boot  "
        }
    );
    println!(
        "  │  GR ctx size valid:   {}  │",
        if ctx_valid {
            format!("✓ {gr_ctx_size:#x} bytes                       ")
        } else {
            "✗ 0 — FECS did not compute size      ".into()
        }
    );
    println!(
        "  │  GPC0 alive:          {}  │",
        if gpc0_alive {
            "✓ YES                                "
        } else {
            "✗ power-gated / dead                 "
        }
    );
    println!(
        "  │  PTIMER running:      {}  │",
        if ptimer_ok {
            "✓ YES                                "
        } else {
            "✗ FROZEN                             "
        }
    );
    println!(
        "  │  Poll time:           {:6}ms                          │",
        poll_elapsed.as_millis()
    );
    println!("  └──────────────────────────────────────────────────────┘");

    if fecs_ready {
        println!(
            "\n  FECS boot SUCCEEDED after {}ms!",
            poll_elapsed.as_millis()
        );
        println!("  GR context size = {gr_ctx_size} bytes ({gr_ctx_size:#010x})");
        println!("  Sequence validated → promoted to toadStool sovereign_stages.rs");
        println!(
            "  Next: toadstoolctl sovereign-boot should now succeed via the toadStool pipeline"
        );
    } else {
        println!(
            "\n  FECS did not set GR_READY within {}ms. Diagnosis:",
            poll_timeout.as_millis()
        );
        if !gpc0_alive {
            println!("  → PRIV ring STILL corrupt. Check Phase 2b (FECS reset handshake).");
        } else if fecs_cpuctl_post == 0x0 {
            println!("  → FECS RUNNING (CPUCTL=0) but stuck at PC={fecs_pc_post:#010x}");
            println!("    GPCCS may not have responded — check CSDATA format.");
        } else if fecs_cpuctl_post == 0x10 {
            println!("  → FECS HALTED. ctx_size={gr_ctx_size:#010x}");
        } else {
            println!("  → FECS_CPUCTL={fecs_cpuctl_post:#010x} FECS_PC={fecs_pc_post:#010x}");
        }
        println!("\n  Check ember circuit breaker: verify_ember_alive via coralctl status");
    }

    println!();
    println!("═══════════════════════════════════════════════");
    println!("  EXP 184 complete. All GPU access was via coral-ember.");
    println!("═══════════════════════════════════════════════");
}
