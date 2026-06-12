// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 183: K80 (GK210) Sovereign FECS Boot — Internal Firmware Path
//!
//! ## Key Discovery from exp182 post-mortem
//!
//! K80 (GK210 = GK110B) uses nouveau's **compiled-in** Falcon firmware
//! (`gk110_gr_fecs_ucode` / `gk110_gr_gpccs_ucode`), NOT the proprietary
//! `/lib/firmware/nvidia/gk210/fecs_inst.bin`. The nvidia firmware is for the
//! nvidia-470 driver only. Loading the wrong firmware caused FECS to crash.
//!
//! ## Correct Boot Sequence (gf100_gr_init_ctxctl_int)
//!
//! From Linux 6.12 nouveau source `nvkm/engine/gr/gf100.c`:
//! ```
//! 1. nvkm_mc_unk260(device, 0)       → write 0x0 to BAR0[0x000260]
//! 2. PIO-load FECS DMEM (fecs_data)  → via FECS_DMEMC/FECS_DMEMD
//! 3. PIO-load FECS IMEM (fecs_code)  → via FECS_IMEMC/FECS_IMEMD
//! 4. PIO-load GPCCS DMEM (gpccs_data) → via GPCCS_DMEMC/GPCCS_DMEMD (0x41a000 base)
//! 5. PIO-load GPCCS IMEM (gpccs_code) → via GPCCS_IMEMC/GPCCS_IMEMD
//! 6. nvkm_mc_unk260(device, 1)       → write 0x1 to BAR0[0x000260]
//! 7. write 0 to BAR0[0x40910c]       → FECS_HWCFG = 0
//! 8. write 0x2 to BAR0[0x409100]     → FECS_CPUCTL = STARTCPU (bit 1)
//! 9. poll BAR0[0x409800] & 0x80000000 → FECS ready (bit 31)
//! ```
//!
//! FECS internally initialises GPCs via PRIV ring; GPCCS is NOT started
//! directly via CPUCTL in the _int path.
//!
//! ## Firmware Source
//!
//! Extracted from nouveau.ko (`gk110_gr_fecs_ucode` / `gk110_gr_gpccs_ucode`
//! .data symbols) and saved to `wateringHole/gk110/`.
//! NOTE: extract_nouveau_gk110_fw.py has been superseded by toadstool device
//! firmware extraction.
//!
//! ## Prerequisites
//!
//! 1. nouveau warm cycle to train GDDR5 and set PMC_ENABLE:
//!    ```sh
//!    toadstool device warm-catch 0000:4b:00.0 --memory-type gddr5
//!    ```
//! 2. `sudo setpci -s 0000:4b:00.0 COMMAND=0x06`
//!
//! ## Usage
//!
//! ```text
//! sudo ./target/release/exp183_k80_fecs_int_boot [--bdf 0000:4b:00.0] \
//!      [--fw-dir ../wateringHole/gk110] \
//!      [--ember-socket /run/toadstool/biomeos/compute.sock] \
//!      [--dry-run]
//! ```
//!
//! Requires toadstool-ember running and holding the VFIO fd for the target BDF.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use hotspring_barracuda::ember_types::MmioBatchOp;
use hotspring_barracuda::fleet_client::EmberClient;

#[path = "../bin_helpers/sovereignty/mod.rs"]
mod sovereignty;
use sovereignty::connect::{connect_ember, extract_arg, is_dry_run};

// ── PMC / MC registers ─────────────────────────────────────────────────────
const BOOT0: u32 = 0x000000;
const PMC_ENABLE: u32 = 0x000200;
const MC_UNK260: u32 = 0x000260; // nvkm_mc_unk260 target register
const PTIMER_HI: u32 = 0x009084;

// ── PGRAPH / GR engine registers ───────────────────────────────────────────
const GR_READY: u32 = 0x409800; // FECS boot-ready flag (bit 31 in _int path)
const GR_CTX_SIZE: u32 = 0x409804; // FECS reports context size here after boot

// ── FECS Falcon registers (base 0x409000) ──────────────────────────────────
const FECS_BASE: u32 = 0x409000;
const FECS_CPUCTL: u32 = FECS_BASE + 0x100;
const FECS_PC: u32 = FECS_BASE + 0x110;
const FECS_SCTL: u32 = FECS_BASE + 0x240;
const FECS_HWCFG: u32 = FECS_BASE + 0x10c; // also: BAR0[0x40910c]
const FECS_DMEMC: u32 = FECS_BASE + 0x1c0;
const FECS_DMEMD: u32 = FECS_BASE + 0x1c4;
const FECS_MB0: u32 = FECS_BASE + 0x040;
const FECS_MB1: u32 = FECS_BASE + 0x044;

// ── GPCCS broadcast Falcon registers (PIO access base 0x41a000) ────────────
// NOTE: GPC status/BOOT0 is at 0x418000 (GPC_BCAST). The 0x41a000 address is
// the GPCCS falcon PIO access register window used by nouveau's init_fw.
const GPCCS_BASE: u32 = 0x41a000;
const GPCCS_HWCFG: u32 = GPCCS_BASE + 0x10c; // also: BAR0[0x41a10c]

// ── GPC status (BAR0 broadcast, read-only from CPU) ────────────────────────
const GPC0_BOOT0: u32 = 0x418000;
const GPC1_BOOT0: u32 = 0x428000;

// ── FECS CPUCTL bits (Falcon v2/v4 — GK110/GK210 Kepler) ──────────────────
const FECS_STARTCPU: u32 = 0x02; // bit 1: start CPU
const FECS_HALTED: u32 = 0x10; // bit 4: CPU halted

const IMEM_PAGE: usize = 256; // bytes per IMEM page (64 words)
const IMEM_WORDS_PER_PAGE: usize = IMEM_PAGE / 4;

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

fn load_fw(dir: &Path, name: &str) -> Vec<u32> {
    let path = dir.join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("FATAL: cannot read {}: {e}", path.display());
        std::process::exit(1);
    });
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

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn upload_dmem(ember: &EmberClient, bdf: &str, base: u32, words: &[u32]) {
    let bytes = words_to_bytes(words);
    match ember.falcon_upload_dmem(bdf, base, 0, &bytes) {
        Ok(r) if r.ok => {}
        Ok(r) => eprintln!("  WARN: falcon.upload_dmem ok=false bytes={:?}", r.bytes),
        Err(e) => eprintln!("  WARN: falcon.upload_dmem error: {e}"),
    }
}

fn upload_imem(ember: &EmberClient, bdf: &str, base: u32, words: &[u32]) {
    let bytes = words_to_bytes(words);
    match ember.falcon_upload_imem(bdf, base, 0, &bytes, 0, false) {
        Ok(r) if r.ok => {}
        Ok(r) => eprintln!("  WARN: falcon.upload_imem ok=false bytes={:?}", r.bytes),
        Err(e) => eprintln!("  WARN: falcon.upload_imem error: {e}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bdf = extract_arg(&args, "--bdf").unwrap_or_else(|| "0000:4b:00.0".into());
    let fw_dir = extract_arg(&args, "--fw-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../wateringHole/gk110"));
    let ember_socket = extract_arg(&args, "--ember-socket");
    let dry_run = is_dry_run(&args);

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  EXP 183 — K80 Sovereign FECS Boot (gf100_gr_init_ctxctl_int)  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!("  BDF:    {bdf}");
    println!("  FW dir: {}", fw_dir.display());
    println!("  Wiring: toadstool-ember (all MMIO via JSON-RPC — no raw BAR0 mmap)");
    if dry_run {
        println!("  Mode:   DRY RUN");
    }

    // ── Load firmware ─────────────────────────────────────────────────────
    let fecs_code = load_fw(&fw_dir, "fecs_code.bin");
    let fecs_data = load_fw(&fw_dir, "fecs_data.bin");
    let gpccs_code = load_fw(&fw_dir, "gpccs_code.bin");
    let gpccs_data = load_fw(&fw_dir, "gpccs_data.bin");

    println!("\n  Firmware loaded:");
    println!(
        "    fecs_code:  {} words ({} bytes)",
        fecs_code.len(),
        fecs_code.len() * 4
    );
    println!(
        "    fecs_data:  {} words ({} bytes)",
        fecs_data.len(),
        fecs_data.len() * 4
    );
    println!(
        "    gpccs_code: {} words ({} bytes)",
        gpccs_code.len(),
        gpccs_code.len() * 4
    );
    println!(
        "    gpccs_data: {} words ({} bytes)",
        gpccs_data.len(),
        gpccs_data.len() * 4
    );

    // ── Connect to ember ──────────────────────────────────────────────────
    let ember = connect_ember(&bdf, ember_socket.as_deref());
    println!("  Ember:  connected ✓");

    // ── Phase 1: Pre-condition validation ─────────────────────────────────
    println!("\n━━━ Phase 1: Pre-conditions ━━━\n");

    let boot0 = r32(&ember, &bdf, BOOT0);
    println!("  BOOT0        = {boot0:#010x}  (GK210 = 0x0f22d0a1)");
    if boot0 == 0xffff_ffff || boot0 == 0 {
        eprintln!("FATAL: GPU link dead. Check PCIe and power.");
        std::process::exit(1);
    }

    let pmc_en = r32(&ember, &bdf, PMC_ENABLE);
    let fecs_sctl = r32(&ember, &bdf, FECS_SCTL);
    let fecs_cpuctl = r32(&ember, &bdf, FECS_CPUCTL);
    let fecs_hwcfg = r32(&ember, &bdf, FECS_HWCFG);
    let gpc0_boot0 = r32(&ember, &bdf, GPC0_BOOT0);
    let gpc1_boot0 = r32(&ember, &bdf, GPC1_BOOT0);
    let gr_ready_pre = r32(&ember, &bdf, GR_READY);

    println!("  PMC_ENABLE   = {pmc_en:#010x}");
    println!("  FECS_SCTL    = {fecs_sctl:#010x}  (0=LS ✓)");
    println!("  FECS_CPUCTL  = {fecs_cpuctl:#010x}  (0x10=HRESET ✓)");
    println!("  FECS_HWCFG   = {fecs_hwcfg:#010x}");
    println!("  GPC0_BOOT0   = {gpc0_boot0:#010x}");
    println!("  GPC1_BOOT0   = {gpc1_boot0:#010x}");
    println!("  GR_READY     = {gr_ready_pre:#010x}  (expecting 0 before boot)");

    // Check warm state — PMC_ENABLE must have GR bit (0x1000) and more
    let pmc_warm = pmc_en > 0x0000_3000; // rough threshold: must have GR+FIFO bits
    if !pmc_warm {
        eprintln!("ERROR: GPU appears cold (PMC_ENABLE={pmc_en:#010x}).");
        eprintln!("  Run warm cycle first: toadstool device warm-catch {bdf} --memory-type gddr5");
        std::process::exit(1);
    }
    if fecs_sctl != 0 {
        eprintln!("ERROR: FECS_SCTL={fecs_sctl:#010x} — unexpected HS mode on GK210.");
        std::process::exit(1);
    }
    if fecs_cpuctl == 0xbadf_1200 || fecs_cpuctl == 0xbadf_1201 {
        eprintln!("ERROR: FECS power domain is OFF ({fecs_cpuctl:#010x}).");
        eprintln!("  GPU is in cold state — run nouveau warm cycle first.");
        std::process::exit(1);
    }

    let pt0 = r32(&ember, &bdf, PTIMER_HI);
    std::thread::sleep(Duration::from_millis(5));
    let pt1 = r32(&ember, &bdf, PTIMER_HI);
    println!(
        "  PTIMER:      {} ({})",
        if pt1 != pt0 {
            "RUNNING ✓"
        } else {
            "FROZEN ✗"
        },
        if pt1 != pt0 {
            format!("{pt0:#010x}→{pt1:#010x}")
        } else {
            format!("{pt0:#010x}")
        }
    );
    println!("  Pre-conditions OK ✓");

    if dry_run {
        println!("\n  DRY RUN — no writes. Exiting.");
        return;
    }

    // ── Phase 2: nvkm_mc_unk260(device, 0) ────────────────────────────────
    // From gf100_mc.c: gf100_mc_unk260 → nvkm_wr32(device, 0x000260, data)
    println!("\n━━━ Phase 2: MC UNK260 = 0 (reset falcons) ━━━\n");
    w32(&ember, &bdf, MC_UNK260, 0x0);
    println!("  BAR0[0x000260] ← 0x00000000");
    std::thread::sleep(Duration::from_micros(10));

    // ── Phase 3: Load FECS firmware ───────────────────────────────────────
    println!("\n━━━ Phase 3: FECS Firmware Load ━━━\n");
    println!("  Loading FECS DMEM ({} words)...", fecs_data.len());
    upload_dmem(&ember, &bdf, FECS_BASE, &fecs_data);
    println!(
        "  Loading FECS IMEM ({} words, {} pages)...",
        fecs_code.len(),
        (fecs_code.len() + IMEM_WORDS_PER_PAGE - 1) / IMEM_WORDS_PER_PAGE
    );
    upload_imem(&ember, &bdf, FECS_BASE, &fecs_code);

    // Readback verification via mmio.batch
    {
        let ops = vec![
            MmioBatchOp::write(FECS_DMEMC, 0x0),
            MmioBatchOp::read(FECS_DMEMD),
            MmioBatchOp::read(FECS_DMEMD),
        ];
        if let Ok(result) = ember.mmio_batch(&bdf, &ops) {
            let dmem_v0 = result.read_value(1).unwrap_or(0xDEAD);
            let dmem_v1 = result.read_value(2).unwrap_or(0xDEAD);
            println!(
                "  FECS DMEM readback: [{:#010x}, {:#010x}] (wrote [{:#010x}, {:#010x}])",
                dmem_v0,
                dmem_v1,
                fecs_data[0],
                fecs_data.get(1).copied().unwrap_or(0)
            );
            if dmem_v0 != fecs_data[0] {
                println!("  WARN: DMEM readback mismatch! FECS DMEM write may have failed.");
            }
        }
    }
    println!("  FECS firmware loaded ✓");

    // ── Phase 4: Load GPCCS firmware ──────────────────────────────────────
    println!("\n━━━ Phase 4: GPCCS Firmware Load ━━━\n");
    println!(
        "  Loading GPCCS DMEM ({} words) via broadcast 0x{GPCCS_BASE:06x}...",
        gpccs_data.len()
    );
    upload_dmem(&ember, &bdf, GPCCS_BASE, &gpccs_data);
    println!(
        "  Loading GPCCS IMEM ({} words, {} pages)...",
        gpccs_code.len(),
        (gpccs_code.len() + IMEM_WORDS_PER_PAGE - 1) / IMEM_WORDS_PER_PAGE
    );
    upload_imem(&ember, &bdf, GPCCS_BASE, &gpccs_code);
    println!("  GPCCS firmware loaded ✓");

    // ── Phase 5: nvkm_mc_unk260(device, 1) ────────────────────────────────
    println!("\n━━━ Phase 5: MC UNK260 = 1 (enable falcons) ━━━\n");
    w32(&ember, &bdf, MC_UNK260, 0x1);
    println!("  BAR0[0x000260] ← 0x00000001");
    std::thread::sleep(Duration::from_micros(10));

    // ── Phase 6: Boot FECS ─────────────────────────────────────────────────
    println!("\n━━━ Phase 6: FECS Boot ━━━\n");

    // Wait for FECS memory scrubbing complete (bits 1-2 of FECS_HWCFG must clear)
    println!("  Waiting for FECS mem scrubbing...");
    let t0 = Instant::now();
    let scrub_timeout = Duration::from_millis(2000);
    loop {
        let hwcfg = r32(&ember, &bdf, FECS_HWCFG);
        if hwcfg & 0x6 == 0 {
            break;
        }
        if t0.elapsed() >= scrub_timeout {
            println!("  WARN: FECS mem scrubbing timeout (HWCFG={hwcfg:#010x})");
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let fecs_hwcfg_after_unk260 = r32(&ember, &bdf, FECS_HWCFG);
    let gpccs_hwcfg_after = r32(&ember, &bdf, GPCCS_HWCFG);
    println!("  FECS_HWCFG after unk260(1): {fecs_hwcfg_after_unk260:#010x}");
    println!("  GPCCS_HWCFG after unk260(1): {gpccs_hwcfg_after:#010x}");

    // Per gf100_gr_init_ctxctl_int: write 0 to FECS_HWCFG, then STARTCPU
    w32(&ember, &bdf, FECS_HWCFG, 0x0);
    println!("  FECS_HWCFG (0x40910c) ← 0x00000000");

    // STARTCPU = 0x2 on GK110/GK210 (Falcon v2/v4)
    w32(&ember, &bdf, FECS_CPUCTL, FECS_STARTCPU);
    println!("  FECS_CPUCTL ← 0x{FECS_STARTCPU:08x} (STARTCPU bit 1)");

    // ── Phase 7: Poll GR_READY bit 31 ──────────────────────────────────────
    println!("\n━━━ Phase 7: Poll GR_READY (0x409800 bit 31) ━━━\n");

    let poll_timeout = Duration::from_millis(2000);
    let poll_start = Instant::now();
    let mut fecs_ready = false;
    let mut iters = 0_u32;
    let mut last_ready = 0_u32;

    print!("  Polling");
    loop {
        let ready = r32(&ember, &bdf, GR_READY);
        iters += 1;
        if ready == 0xFFFF_FFFF {
            let elapsed_ms = poll_start.elapsed().as_millis();
            println!(
                "\n  WARN: D3cold sentinel at t={elapsed_ms}ms — ember circuit breaker active"
            );
            println!("  → journalctl -u coral-ember for recovery status");
            break;
        }
        if ready != last_ready {
            print!(" [{ready:#010x}]");
            last_ready = ready;
        } else if iters % 100 == 0 {
            print!(".");
        }
        if ready & 0x8000_0000 != 0 {
            fecs_ready = true;
            break;
        }
        if poll_start.elapsed() >= poll_timeout {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    println!();

    let elapsed = poll_start.elapsed();

    // ── Phase 8: Post-boot state ────────────────────────────────────────────
    println!("\n━━━ Phase 8: Post-Boot State ━━━\n");

    let fecs_cpuctl_post = r32(&ember, &bdf, FECS_CPUCTL);
    let fecs_pc_post = r32(&ember, &bdf, FECS_PC);
    let fecs_mb0_post = r32(&ember, &bdf, FECS_MB0);
    let fecs_mb1_post = r32(&ember, &bdf, FECS_MB1);
    let gr_ready_post = r32(&ember, &bdf, GR_READY);
    let gr_ctx_size = r32(&ember, &bdf, GR_CTX_SIZE);
    let pmc_post = r32(&ember, &bdf, PMC_ENABLE);
    let gpc0_post = r32(&ember, &bdf, GPC0_BOOT0);
    let gpc1_post = r32(&ember, &bdf, GPC1_BOOT0);
    let gpccs_hwcfg_post = r32(&ember, &bdf, GPCCS_HWCFG);
    let pt_post = r32(&ember, &bdf, PTIMER_HI);

    println!(
        "  FECS_CPUCTL  = {fecs_cpuctl_post:#010x}  \
              ({})",
        match fecs_cpuctl_post {
            0x10 => "HALTED (normal after boot)",
            0x00 => "RUNNING",
            v if v == FECS_HALTED => "HALTED",
            _ => "unknown",
        }
    );
    println!("  FECS_PC      = {fecs_pc_post:#010x}");
    println!("  FECS_MB0     = {fecs_mb0_post:#010x}");
    println!("  FECS_MB1     = {fecs_mb1_post:#010x}");
    println!(
        "  GR_READY     = {gr_ready_post:#010x}  \
              (bit31={}, bit0={})",
        (gr_ready_post >> 31) & 1,
        gr_ready_post & 1
    );
    println!("  GR_CTX_SIZE  = {gr_ctx_size:#010x}  ({gr_ctx_size} bytes context)");
    println!(
        "  PMC_ENABLE   = {pmc_post:#010x}  (was {pmc_en:#010x}, {})",
        if pmc_post == pmc_en {
            "preserved ✓"
        } else {
            "CHANGED"
        }
    );
    println!(
        "  GPC0_BOOT0   = {gpc0_post:#010x}  \
              ({})",
        if gpc0_post == 0xbadf_1100 {
            "power-gated ✗"
        } else {
            "accessible ✓"
        }
    );
    println!(
        "  GPC1_BOOT0   = {gpc1_post:#010x}  \
              ({})",
        if gpc1_post == 0xbadf_1100 {
            "power-gated ✗"
        } else {
            "accessible ✓"
        }
    );
    println!("  GPCCS_HWCFG  = {gpccs_hwcfg_post:#010x}");
    println!(
        "  PTIMER       = {pt_post:#010x}  (running: {})",
        pt_post != pt0
    );

    // ── Summary ────────────────────────────────────────────────────────────
    println!();
    let succeeded = fecs_ready;
    let fecs_booted = fecs_cpuctl_post == 0x10 || fecs_cpuctl_post == FECS_HALTED;
    let ctx_valid = gr_ctx_size > 0 && gr_ctx_size < 0x10_0000; // sanity: < 1MB
    let gpc0_alive = gpc0_post != 0xbadf_1100 && gpc0_post != 0xffff_ffff;
    let gpc1_alive = gpc1_post != 0xbadf_1100 && gpc1_post != 0xffff_ffff;
    let ptimer_ok = pt_post != pt0;

    println!("  ┌────────────────────────────────────────────────────┐");
    println!("  │  EXP 183 — Sovereign K80 FECS Boot Scorecard       │");
    println!("  ├────────────────────────────────────────────────────┤");
    println!(
        "  │  GR_READY bit31:    {}  │",
        if succeeded {
            "✓ SET   — FECS booted!         "
        } else {
            "✗ CLEAR — FECS did not boot     "
        }
    );
    println!(
        "  │  FECS halted/idle:  {}  │",
        if fecs_booted {
            "✓ YES                          "
        } else {
            "? NO   (check FECS_CPUCTL)      "
        }
    );
    println!(
        "  │  GR ctx size valid: {}  │",
        if ctx_valid {
            "✓ YES                          "
        } else {
            "✗ NO   (FECS did not report)    "
        }
    );
    println!(
        "  │  GPC0 alive:        {}  │",
        if gpc0_alive {
            "✓ YES                          "
        } else {
            "✗ POWER-GATED                  "
        }
    );
    println!(
        "  │  GPC1 alive:        {}  │",
        if gpc1_alive {
            "✓ YES                          "
        } else {
            "✗ POWER-GATED                  "
        }
    );
    println!(
        "  │  PTIMER running:    {}  │",
        if ptimer_ok {
            "✓ YES                          "
        } else {
            "✗ FROZEN                       "
        }
    );
    println!(
        "  │  Poll time:         {:6}ms                        │",
        elapsed.as_millis()
    );
    println!("  └────────────────────────────────────────────────────┘");

    if succeeded {
        println!(
            "\n  🏆 FECS boot SUCCEEDED after {}ms!",
            elapsed.as_millis()
        );
        println!("  GR context size = {gr_ctx_size} bytes");
        if gpc0_alive {
            println!("  GPC0 came online — GR engine is LIVE!");
        }
        println!("\n  Next steps:");
        println!("  1. Apply CSDATA register lists (gk110b_grctx hub/gpc tables)");
        println!("  2. Initialize PGFIFO and CE for compute dispatch");
        println!("  3. Submit a test workload via FECS mailbox interface");
    } else {
        println!("\n  FECS did not set GR_READY bit 31 within 2s.");
        println!("  Diagnosis:");
        println!("    FECS_CPUCTL={fecs_cpuctl_post:#010x} FECS_PC={fecs_pc_post:#010x}");
        println!("    GR_READY={gr_ready_post:#010x} GR_CTX_SIZE={gr_ctx_size:#010x}");
        if fecs_cpuctl_post == 0x0 {
            println!("    → FECS is still executing (CPUCTL=0 = RUNNING).");
            println!("      Try increasing poll timeout or check FECS_PC for progress.");
        } else if fecs_cpuctl_post & FECS_HALTED != 0 {
            println!("    → FECS halted (CPUCTL has HALTED bit). Check if GR_CTX_SIZE was set.");
            println!("      The CSDATA register lists may be needed.");
        } else if fecs_cpuctl_post == 0xbadf_1200 {
            println!("    → GR power domain COLLAPSED after boot attempt.");
            println!("      FECS crashed — check firmware blob integrity.");
        }
    }

    println!();
    println!("═══════════════════════════════════════════");
    println!("  EXP 183 complete.");
    println!("═══════════════════════════════════════════");
}
