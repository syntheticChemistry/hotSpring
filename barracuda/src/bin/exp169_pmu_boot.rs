// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 169: PMU firmware boot attempt for NVIDIA GV100 (Titan V).
//!
//! ## Context
//!
//! The Titan V is blocked from compute dispatch because:
//!
//! 1. PMU is in HRESET (`cpuctl bit 4 = 1`, `PMC_ENABLE bit 13 = 0`)
//! 2. SEC2 ACR bootloader started (`mb0 = 1`) but never completed
//! 3. WPR never configured → FECS ROM security gate at `pc=0x15f1`
//!
//! EXP168 identified GV100 PMU firmware candidates in `nv-kernel.o_binary`
//! (nvidia-470). The 70KB blobs at file offsets 0x01f4cba0 and 0x01f5e620
//! match the PMU HWCFG IMEM size: `(0x400e0100 >> 9) & 0x1FF = 256 pages = 64KB`.
//!
//! ## Strategy
//!
//! 1. **Phase 0**: Read PMU HWCFG to confirm IMEM/DMEM sizes.
//! 2. **Phase 1**: Load extracted PMU firmware blob into VRAM via PRAMIN.
//! 3. **Phase 2**: Clear PMU HRESET by setting `PMC_ENABLE bit 13`.
//! 4. **Phase 3**: DMATRF from VRAM to PMU IMEM (same mechanism as FECS DMATRF).
//! 5. **Phase 4**: Load data section to PMU DMEM via PIO.
//! 6. **Phase 5**: Issue PMU STARTCPU, poll for boot mailbox signal.
//! 7. **Phase 6**: If PMU booted — re-probe SEC2 ACR state.
//!
//! ## Firmware blob structure (recovered from nv-kernel.o_binary)
//!
//! ```text
//! +0x00  u32  total_size         (= size of code + data + this header)
//! +0x04  u32  reserved[7]        (zeros)
//! +0x20  u8   code[imem_size]    (raw Falcon v4 IMEM code, begins with d0 xx xx 00)
//! +0x20+imem_size  u8  data[...]  (DMEM initializer data)
//! ```
//!
//! ## Usage
//!
//! ```text
//! sudo cargo run --release --bin exp169_pmu_boot -- \
//!     --bdf 0000:02:00.0 \
//!     --firmware /tmp/gv100_pmu_70k_B.bin
//! ```
//!
//! Requires `sudo` for `/sys/bus/pci/devices/.../resource0` write access.

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[allow(unsafe_code)]
#[path = "../low_level/bar0.rs"]
mod bar0_mmio;
use bar0_mmio::Bar0Map;

// ── BAR0 / PCI register offsets ──────────────────────────────────────────────

const PMC_ENABLE: u32 = 0x0000_0200;
const PMC_ENABLE_PMU_BIT: u32 = 1 << 13;

const PMU_BASE: u32 = 0x010A_000;
const PMU_CPUCTL: u32 = PMU_BASE + 0x100;
const PMU_BOOTVEC: u32 = PMU_BASE + 0x104;
const PMU_HWCFG: u32 = PMU_BASE + 0x108;
#[allow(dead_code)]
const PMU_HWCFG1: u32 = PMU_BASE + 0x10C;
#[allow(dead_code)]
const PMU_ITFEN: u32 = PMU_BASE + 0x048;
#[allow(dead_code)]
const PMU_DMACTL: u32 = PMU_BASE + 0x10C;
const PMU_SCTL: u32 = PMU_BASE + 0x240;
const PMU_MB0: u32 = PMU_BASE + 0x040;
const PMU_MB1: u32 = PMU_BASE + 0x044;
const PMU_PC: u32 = PMU_BASE + 0x030;
const PMU_EXCI: u32 = PMU_BASE + 0x148;
// DMEM PIO: write (1<<24)|addr to 0x1C0, then words to 0x1C4
const PMU_DMEM_PORT: u32 = PMU_BASE + 0x1C0;
const PMU_DMEM_DATA: u32 = PMU_BASE + 0x1C4;

// Falcon DMATRF (proven path: same offsets as FECS DMATRF in volta_warm_pipeline)
//   +0x054 : FBIF DMA VRAM base (physical address >> 8)
//   +0x1C0 : per-block IMEM tag descriptor  ((tag << 8) | 0x02)
//   +0x1C4 : per-block VRAM source block index
//   +0x1C8 : DMATRF CMD  (write 0x11 to trigger, poll bit 1 for busy)
const PMU_DMATRFBASE: u32 = PMU_BASE + 0x054;
const PMU_IMEM_TAG: u32 = PMU_BASE + 0x1C0;
const PMU_VRAM_BLOCK: u32 = PMU_BASE + 0x1C4;
const PMU_DMATRFCMD: u32 = PMU_BASE + 0x1C8;

// PRAMIN (VRAM staging window)
const NV_PRAMIN_WIN_BASE: u32 = 0x001700;
const NV_PRAMIN_DATA: u32 = 0x0070_0000;

const SEC2_BASE: u32 = 0x087000;
const SEC2_MB0: u32 = SEC2_BASE + 0x040;
const SEC2_CPUCTL: u32 = SEC2_BASE + 0x100;
const SEC2_PC: u32 = SEC2_BASE + 0x030;
const SEC2_SCTL: u32 = SEC2_BASE + 0x240;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let bdf = extract_arg(&args, "--bdf").unwrap_or_else(|| {
        std::env::var("HOTSPRING_BDF").unwrap_or_else(|_| "0000:02:00.0".into())
    });
    let fw_path = extract_arg(&args, "--firmware")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/gv100_pmu_70k_B.bin"));
    let dry_run = args.iter().any(|a| a == "--dry-run");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXP 169: GV100 PMU Boot — Titan V ACR unblock              ║");
    println!("║  Goal: PMU running → SEC2 ACR completes → FECS boots        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("  BDF:      {bdf}");
    println!("  Firmware: {}", fw_path.display());
    println!("  Dry-run:  {dry_run}\n");

    // ── Load firmware blob ───────────────────────────────────────────────────
    let fw_data = match std::fs::read(&fw_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FATAL: cannot read firmware: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "  Firmware:  {} bytes ({:.1} KB)",
        fw_data.len(),
        fw_data.len() as f64 / 1024.0
    );

    // ── Open BAR0 ────────────────────────────────────────────────────────────
    let resource0 = format!("/sys/bus/pci/devices/{bdf}/resource0");
    let bar0 = match Bar0Map::open(&resource0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FATAL: cannot open resource0 (need sudo): {e}");
            std::process::exit(1);
        }
    };
    println!("  resource0: {} bytes\n", bar0.len());

    // ── Phase 0: GPU identity + PMU baseline ─────────────────────────────────
    println!("━━━ Phase 0: GPU Identity + PMU Baseline ━━━\n");

    let boot0 = bar0.r32(0);
    println!("  BOOT0         = {boot0:#010x}  (SM70 = 0x140000a1)");
    if boot0 == 0xFFFF_FFFF || boot0 == 0 {
        eprintln!("FATAL: GPU link dead (BOOT0 = {boot0:#010x})");
        std::process::exit(1);
    }

    let pmc_en = bar0.r32(PMC_ENABLE);
    let pmu_enabled = pmc_en & PMC_ENABLE_PMU_BIT != 0;
    println!("  PMC_ENABLE    = {pmc_en:#010x}  (PMU bit 13 = {pmu_enabled})");

    let pmu_ctrl = bar0.r32(PMU_CPUCTL);
    let pmu_pc = bar0.r32(PMU_PC);
    let pmu_sctl = bar0.r32(PMU_SCTL);
    let pmu_mb0 = bar0.r32(PMU_MB0);
    let pmu_hwcfg = bar0.r32(PMU_HWCFG);
    let pmu_hreset = pmu_ctrl & 0x10 != 0;
    let pmu_halted = pmu_ctrl & 0x20 != 0;
    println!("  PMU cpuctl    = {pmu_ctrl:#010x}  (HRESET={pmu_hreset} HALTED={pmu_halted})");
    println!("  PMU pc        = {pmu_pc:#010x}");
    println!(
        "  PMU sctl      = {pmu_sctl:#010x}  (HS mode {})",
        (pmu_sctl >> 12) & 3
    );
    println!("  PMU mb0       = {pmu_mb0:#010x}");
    println!("  PMU hwcfg     = {pmu_hwcfg:#010x}");

    // Parse IMEM/DMEM sizes from HWCFG
    let imem_pages = (pmu_hwcfg >> 9) & 0x1FF;
    let dmem_pages = pmu_hwcfg & 0x1FF;
    let imem_bytes = imem_pages * 256;
    let dmem_bytes = dmem_pages * 256;
    println!(
        "  PMU IMEM      = {imem_pages} pages × 256 = {imem_bytes}B ({} KB)",
        imem_bytes / 1024
    );
    println!(
        "  PMU DMEM      = {dmem_pages} pages × 256 = {dmem_bytes}B ({} KB)\n",
        dmem_bytes / 1024
    );

    // Determine code vs data split in firmware blob
    // Heuristic: code = first imem_bytes, data = remainder
    let fw_code_size = imem_bytes as usize;
    let fw_data_size = fw_data.len().saturating_sub(fw_code_size);
    if fw_data.len() < 1024 {
        eprintln!("FATAL: firmware blob too small ({} bytes)", fw_data.len());
        std::process::exit(1);
    }
    let actual_code_size = fw_code_size.min(fw_data.len());
    println!(
        "  FW code slice: [0..{actual_code_size}] ({} KB)",
        actual_code_size / 1024
    );
    println!(
        "  FW data slice: [{actual_code_size}..{}] ({fw_data_size}B)\n",
        fw_data.len()
    );

    // ── Phase 1: Stage firmware to VRAM via PRAMIN ───────────────────────────
    println!("━━━ Phase 1: Stage PMU Firmware to VRAM via PRAMIN ━━━\n");

    let vram_fw_page: u32 = 0x0006_0000; // VRAM page (same as FECS DMATRF staging area)

    // Set PRAMIN window
    bar0.w32(NV_PRAMIN_WIN_BASE, vram_fw_page >> 16);
    let window_readback = bar0.r32(NV_PRAMIN_WIN_BASE);
    println!("  PRAMIN window → page {vram_fw_page:#010x} (readback={window_readback:#010x})");

    if !dry_run {
        // Write code section to VRAM via PRAMIN
        let pramin_data_base = NV_PRAMIN_DATA;
        let mut pramin_offset = pramin_data_base;

        let code_slice = &fw_data[..actual_code_size];
        for chunk in code_slice.chunks(4) {
            let word = u32::from_le_bytes([
                chunk.first().copied().unwrap_or(0),
                chunk.get(1).copied().unwrap_or(0),
                chunk.get(2).copied().unwrap_or(0),
                chunk.get(3).copied().unwrap_or(0),
            ]);
            bar0.w32(pramin_offset, word);
            pramin_offset += 4;
        }

        // Verify first word
        bar0.w32(NV_PRAMIN_WIN_BASE, vram_fw_page >> 16);
        let first_word = bar0.r32(NV_PRAMIN_DATA);
        let expected = u32::from_le_bytes([fw_data[0], fw_data[1], fw_data[2], fw_data[3]]);
        let ok = first_word == expected;
        println!("  PRAMIN verify: got={first_word:#010x} want={expected:#010x} ok={ok}");

        if !ok {
            eprintln!("WARN: PRAMIN verify failed — VRAM staging may be broken");
        }
    } else {
        println!("  (dry-run: skipping VRAM write)");
    }
    println!(
        "  Staged {actual_code_size}B ({} KB) to VRAM@{vram_fw_page:#010x}\n",
        actual_code_size / 1024
    );

    // ── Phase 2: Clear PMU HRESET via PMC_ENABLE ─────────────────────────────
    println!("━━━ Phase 2: Clear PMU HRESET (PMC_ENABLE bit 13) ━━━\n");

    if !dry_run {
        if !pmu_enabled {
            let new_pmc_en = pmc_en | PMC_ENABLE_PMU_BIT;
            println!("  PMC_ENABLE: {pmc_en:#010x} → {new_pmc_en:#010x} (setting bit 13)");
            bar0.w32(PMC_ENABLE, new_pmc_en);
            std::thread::sleep(Duration::from_millis(5));
        } else {
            println!("  PMC_ENABLE: PMU bit 13 already set — skipping");
        }

        let pmc_en_post = bar0.r32(PMC_ENABLE);
        let pmu_ctrl_post = bar0.r32(PMU_CPUCTL);
        let hreset_post = pmu_ctrl_post & 0x10 != 0;
        println!("  PMC_ENABLE post = {pmc_en_post:#010x}");
        println!("  PMU cpuctl post = {pmu_ctrl_post:#010x}  (HRESET={hreset_post})");
    } else {
        println!("  (dry-run: skipping PMC_ENABLE write)");
    }
    println!();

    // ── Phase 3: DMATRF from VRAM to PMU IMEM ────────────────────────────────
    println!("━━━ Phase 3: DMATRF VRAM→PMU IMEM ({actual_code_size}B) ━━━\n");

    // Configure DMATRF engine
    // DMATRFBASE = VRAM physical base address (in 256B blocks)
    // DMATRFMOFFS = IMEM destination offset
    // DMATRFCMD = command register
    // DMATRFFBOFFS = VRAM source offset within the block

    let block_size: u32 = 256;
    let n_blocks = (actual_code_size as u32 + block_size - 1) / block_size;

    println!("  Transferring {n_blocks} blocks × 256B via DMATRF...");

    if !dry_run {
        // Set VRAM DMA base: physical address >> 8 (256-byte granularity)
        let dma_base = vram_fw_page >> 8;
        bar0.w32(PMU_DMATRFBASE, dma_base);
        println!("  DMATRFBASE = {dma_base:#010x} (VRAM {vram_fw_page:#010x} >> 8)");

        let start = Instant::now();
        let mut ok_count = 0u32;

        for block_idx in 0..n_blocks {
            let tag = block_idx;
            // IMEM tag descriptor: (tag << 8) | 0x02 — same encoding as FECS DMATRF
            bar0.w32(PMU_IMEM_TAG, (tag << 8) | 0x02);
            // VRAM source block index
            bar0.w32(PMU_VRAM_BLOCK, block_idx);
            // Trigger DMATRF: 0x11 = bit 0 (trigger) + bit 4 (?)
            bar0.w32(PMU_DMATRFCMD, 0x0000_0011);

            // Poll: bit 1 of CMD clears when DMA engine is no longer busy
            let block_start = Instant::now();
            loop {
                let cmd = bar0.r32(PMU_DMATRFCMD);
                if cmd & 0x02 == 0 {
                    ok_count += 1;
                    break;
                }
                if block_start.elapsed() > Duration::from_millis(10) {
                    eprintln!("  WARN: DMATRF block {block_idx} timeout (cmd={cmd:#010x})");
                    break;
                }
                std::thread::sleep(Duration::from_micros(1));
            }
        }

        let elapsed = start.elapsed();
        println!(
            "  DMATRF: {ok_count}/{n_blocks} blocks in {:.0}µs",
            elapsed.as_micros()
        );

        // Attempt IMEM readback (may not work on all Falcons in this state)
        let imem_read_ctrl: u32 = 0x0200_0000; // read mode BIT(25)
        bar0.w32(PMU_DMEM_PORT, imem_read_ctrl | 0);
        let imem_first = bar0.r32(PMU_DMEM_DATA);
        println!("  DMEM_PORT[0] after DMATRF: {imem_first:#010x}  (informational)");
    } else {
        println!("  (dry-run: skipping DMATRF)");
    }
    println!();

    // ── Phase 4: Load data section to PMU DMEM via PIO ───────────────────────
    if fw_data_size > 0 {
        println!("━━━ Phase 4: Load PMU DMEM data ({fw_data_size}B) ━━━\n");

        if !dry_run {
            let data_slice = &fw_data[actual_code_size..];
            // Load data section to DMEM starting at address 0
            // Write mode: BIT(24) | addr — same encoding used for FECS DMEM PIO
            bar0.w32(PMU_DMEM_PORT, 0x0100_0000 | 0);

            for chunk in data_slice.chunks(4) {
                let word = u32::from_le_bytes([
                    chunk.first().copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                    chunk.get(2).copied().unwrap_or(0),
                    chunk.get(3).copied().unwrap_or(0),
                ]);
                bar0.w32(PMU_DMEM_DATA, word);
            }

            // Verify first DMEM word
            bar0.w32(PMU_DMEM_PORT, 0x0200_0000 | 0); // read mode BIT(25)
            let dmem_first = bar0.r32(PMU_DMEM_DATA);
            let exp_word = if data_slice.len() >= 4 {
                u32::from_le_bytes([data_slice[0], data_slice[1], data_slice[2], data_slice[3]])
            } else {
                0
            };
            println!("  DMEM[0] verify: got={dmem_first:#010x} want={exp_word:#010x}");
        } else {
            println!("  (dry-run: skipping DMEM load)");
        }
        println!();
    }

    // ── Phase 5: Start PMU ───────────────────────────────────────────────────
    println!("━━━ Phase 5: Set BOOTVEC + STARTCPU ━━━\n");

    if !dry_run {
        // Set BOOTVEC to 0 (start at address 0 in IMEM)
        bar0.w32(PMU_BOOTVEC, 0);
        println!("  BOOTVEC = 0x0");

        // Issue STARTCPU (CPUCTL bit 1)
        bar0.w32(PMU_CPUCTL, 0x02);
        println!("  CPUCTL ← 0x02 (STARTCPU issued)\n");
    } else {
        println!("  (dry-run: skipping STARTCPU)");
    }

    // ── Phase 6: Poll for PMU boot signal ────────────────────────────────────
    println!("━━━ Phase 6: Poll PMU Boot Signal ━━━\n");

    if !dry_run {
        let poll_start = Instant::now();
        let timeout = Duration::from_secs(3);
        let mut last_mb0 = u32::MAX;
        let mut last_pc = u32::MAX;
        let mut booted = false;

        print!("  Polling (3s): ");
        let _ = io::stdout().flush();

        while poll_start.elapsed() < timeout {
            let mb0 = bar0.r32(PMU_MB0);
            let pc = bar0.r32(PMU_PC);
            let ctrl = bar0.r32(PMU_CPUCTL);

            if mb0 != last_mb0 || pc != last_pc {
                println!(
                    "\n  t={:.0}ms: mb0={mb0:#010x} pc={pc:#06x} ctrl={ctrl:#010x}",
                    poll_start.elapsed().as_millis()
                );
                last_mb0 = mb0;
                last_pc = pc;
            }

            // Success: mb0 has a boot-complete signal (not 0, not 0xCAFE0000)
            if mb0 != 0 && mb0 != 0xCAFE_0000 && mb0 & 0xFF00_0000 == 0 {
                println!("\n  POSSIBLE BOOT SIGNAL: mb0={mb0:#010x}");
                booted = true;
                break;
            }

            // Failed: EXCI security trap
            let exci = bar0.r32(PMU_EXCI);
            if exci == 0x0407_0000 {
                println!("\n  FAIL: PMU hit security trap (exci=0x04070000) — HS ROM gate");
                break;
            }

            // Check if PMU is halted/crashed
            if ctrl & 0x20 != 0 {
                println!("\n  FAIL: PMU halted (ctrl={ctrl:#010x})");
                break;
            }

            std::thread::sleep(Duration::from_millis(20));
            print!(".");
            let _ = io::stdout().flush();
        }

        if !booted {
            println!("\n  Timeout: PMU did not complete boot in 3s");
        }

        println!();
        println!("━━━ Phase 6 Final PMU State ━━━\n");
        let ctrl = bar0.r32(PMU_CPUCTL);
        let pc = bar0.r32(PMU_PC);
        let sctl = bar0.r32(PMU_SCTL);
        let mb0 = bar0.r32(PMU_MB0);
        let mb1 = bar0.r32(PMU_MB1);
        let exci = bar0.r32(PMU_EXCI);
        println!(
            "  PMU cpuctl = {ctrl:#010x}  (HRESET={} HALTED={})",
            ctrl & 0x10 != 0,
            ctrl & 0x20 != 0
        );
        println!("  PMU pc     = {pc:#010x}");
        println!(
            "  PMU sctl   = {sctl:#010x}  (HS mode {})",
            (sctl >> 12) & 3
        );
        println!("  PMU mb0    = {mb0:#010x}");
        println!("  PMU mb1    = {mb1:#010x}");
        println!("  PMU exci   = {exci:#010x}");
    } else {
        println!("  (dry-run: skipping PMU poll)");
    }

    // ── Phase 7: Re-probe SEC2 ACR state ─────────────────────────────────────
    println!();
    println!("━━━ Phase 7: SEC2 ACR Re-Probe ━━━\n");

    {
        let sec2_ctrl = bar0.r32(SEC2_CPUCTL);
        let sec2_pc = bar0.r32(SEC2_PC);
        let sec2_sctl = bar0.r32(SEC2_SCTL);
        let sec2_mb0 = bar0.r32(SEC2_MB0);

        println!("  SEC2 cpuctl = {sec2_ctrl:#010x}");
        println!("  SEC2 pc     = {sec2_pc:#010x}");
        println!(
            "  SEC2 sctl   = {sec2_sctl:#010x}  (HS mode {})",
            (sec2_sctl >> 12) & 3
        );
        println!("  SEC2 mb0    = {sec2_mb0:#010x}");

        if sec2_mb0 > 1 && sec2_mb0 != 0xCAFE_0000 && sec2_mb0 != 0xCAFE_BEEF {
            println!("\n  SEC2 ACR ADVANCE: mb0 changed from 1 to {sec2_mb0:#010x}");
            println!("  → SEC2 ACR may have processed a message from PMU");
        } else if sec2_mb0 == 0x0000_0001 {
            println!("\n  SEC2 ACR still at mb0=1 (ACR BL started but waiting)");
            println!("  → PMU boot may not have sent the required IPC signal yet");
        }

        // Check WPR2 — if PMU ACR completed, WPR2 should be configured
        let wpr2_lo = bar0.r32(0x1FA824);
        let wpr2_hi = bar0.r32(0x1FA828);
        println!("\n  WPR2 lo = {wpr2_lo:#010x}");
        println!("  WPR2 hi = {wpr2_hi:#010x}");

        if wpr2_lo != 0 && wpr2_lo != 0xBADF_1100 {
            println!(
                "  WPR2 configured! addr = {:#012x}",
                wpr2_lo as u64 | ((wpr2_hi as u64 & 0xFF) << 32)
            );
            println!("  → ACR completed — FECS can now boot authenticated firmware");
        } else {
            println!("  WPR2 not configured (still BADF11xx pattern)");
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  EXP 169 complete. Check results above.");
    println!("  If PMU booted → run volta_warm_pipeline to attempt FECS dispatch.");
    println!("═══════════════════════════════════════════════════════════════");
}

// ── CLI argument extraction ───────────────────────────────────────────────────

fn extract_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}
