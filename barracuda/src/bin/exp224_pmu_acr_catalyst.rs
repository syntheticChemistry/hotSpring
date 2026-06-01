// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(dead_code, reason = "PMU ACR catalyst registers retained for sovereign boot replay")]

//! Experiment 224: PMU ACR Catalyst — Sovereign Falcon Boot via CPUCTL_ALIAS
//!
//! ## Context
//!
//! Exp 223 proved that the HS "unlock" via ENGCTL (0x3C0) is actually a
//! destructive **engine reset** that permanently transitions SEC_MODE from
//! HS (2) to NS (0), irreversibly killing CPU execution. CPUCTL_ALIAS
//! (0x130) becomes unresponsive in NS mode.
//!
//! Exp 206 validated FECS/GPCCS boot via CPUCTL_ALIAS + DMA on warm GPUs.
//! toadStool's `falcon_pio.rs` confirms PIO works in ALL security modes.
//!
//! This experiment uses the **correct** path:
//!   1. Stay in HS mode 2 (DO NOT touch ENGCTL)
//!   2. Use CPUCTL_ALIAS (0x130) for CPU control
//!   3. Load firmware via PIO (IMEMC/IMEMD/IMEMT)
//!   4. IINVAL → STARTCPU via ALIAS
//!
//! ## Usage
//!
//! ```text
//! sudo cargo run --release --features low-level \
//!     --bin exp224_pmu_acr_catalyst -- \
//!     --target 0000:49:00.0 --control 0000:02:00.0
//! ```

use std::io::{self, Write as IoWrite};
use std::time::{Duration, Instant};

use hotspring_barracuda::low_level::bar0::{Bar0Domain, Bar0Map, DenyEntry, SafeBar0};
use hotspring_barracuda::low_level::falcon::{self, FalconSnapshot};

// ── PMU FBIF base ────────────────────────────────────────────────────────────

const PMU_FBIF: u32 = 0xE00;

// ── mmiotrace-extracted ACR bootloader (128 words, IMEM offset 0xFE00) ───────

const ACR_BL_IMEM_OFFSET: u32 = 0xFE00;

#[rustfmt::skip]
const ACR_BL_IMEM: [u32; 128] = [
    0x00A000D0, 0x0004FE00, 0x107EA4BD, 0x02F80100,
    0x00000089, 0x98099E98, 0x12F90A9D, 0xB6129B98,
    0x0EFD049C, 0x00BD11FE, 0x010F26F0, 0xD000B604,
    0x0004FEA0, 0x00000089, 0x98099E98, 0x12F90A9D,
    0xB6129B98, 0x0EFD049C, 0x00BD11FE, 0x0627F001,
    0x04B60410, 0x00A0D000, 0xA4BD0002, 0x99E49898,
    0x9D129809, 0x98B6120A, 0x9C0EF909, 0xFE00BD04,
    0x01010F11, 0xD004B604, 0x0004FEA0, 0x00000089,
    0x98099E98, 0x12F90A9D, 0xB6129B98, 0x0EFD049C,
    0x00BD11FE, 0x010F26F0, 0xD000B604, 0x0002A0A0,
    0x9898A4BD, 0x0999E498, 0x0A9D1298, 0x0998B612,
    0x049C0EF9, 0x11FE00BD, 0x0401010F, 0xA0D004B6,
    0x89000200, 0x98000000, 0x98099E98, 0x9D12F90A,
    0x98B6129B, 0xFD049C0E, 0xBD11FE00, 0x2EF00100,
    0x00B60410, 0x03A0D000, 0xA4BD0002, 0x99E49898,
    0x9D129809, 0x98B6120A, 0x9C0EF909, 0xFE00BD04,
    // second 256-byte block (nvidia tag 0x101)
    0x01010F11, 0xD004B604, 0x0002A0A0, 0x00000089,
    0x98099E98, 0x12F90A9D, 0xB6129B98, 0x0EFD049C,
    0x00BD11FE, 0x010F36F0, 0xD000B604, 0x000200A0,
    0x9898A4BD, 0x0999E498, 0x0A9D1298, 0x0998B612,
    0x049C0EF9, 0x11FE00BD, 0x0401010F, 0xA0D004B6,
    0x89000400, 0x98000000, 0x98099E98, 0x9D12F90A,
    0x98B6129B, 0xFD049C0E, 0xBD11FE00, 0x3EF00100,
    0x00B60410, 0xA0D00000, 0xBD000200, 0xA4BD00A4,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
];

#[rustfmt::skip]
const ACR_DMEM_DESC: [u32; 21] = [
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000004, // flags/mode
    0xDD990000, // VRAM addr low: ACR ucode
    0x00000001, // VRAM addr high → 0x1_DD990000
    0x00000000,
    0x00000600, // ACR ucode size (1536 bytes)
    0x00000600, // aligned size
    0x00006900, // total payload (26880 bytes)
    0x00000000,
    0xDD998000, // VRAM addr low: load table
    0x00000001, // VRAM addr high → 0x1_DD998000
    0x000042F0, // load table size (17136 bytes)
    0x00000034, // 52 entries
    0x0000002F, // 47 falcons
];

// ── Boot sequences ───────────────────────────────────────────────────────────

/// Try booting PMU via CPUCTL_ALIAS (preferred path for HS falcons).
fn try_boot_alias(bar: &Bar0Map, base: u32, bootvec_val: u32) -> bool {
    println!("  CPUCTL_ALIAS ← {:#x} (HRESET)", falcon::CPUCTL_HRESET);
    bar.w32(base + falcon::CPUCTL_ALIAS, falcon::CPUCTL_HRESET);
    std::thread::sleep(Duration::from_millis(10));

    let alias_rb = bar.r32(base + falcon::CPUCTL_ALIAS);
    let cpuctl_rb = bar.r32(base + falcon::CPUCTL);
    println!("    readback: ALIAS={alias_rb:#010x} CPUCTL={cpuctl_rb:#010x}");

    bar.w32(base + falcon::BOOTVEC, bootvec_val);
    let bv = bar.r32(base + falcon::BOOTVEC);
    println!("  BOOTVEC ← {bootvec_val:#x} (readback={bv:#010x})");

    bar.w32(base + falcon::MAILBOX0, 0);
    bar.w32(base + falcon::MAILBOX1, 0);

    println!("  CPUCTL_ALIAS ← {:#x} (IINVAL)", falcon::CPUCTL_IINVAL);
    bar.w32(base + falcon::CPUCTL_ALIAS, falcon::CPUCTL_IINVAL);
    std::thread::sleep(Duration::from_millis(1));

    println!("  CPUCTL_ALIAS ← {:#x} (STARTCPU)", falcon::CPUCTL_STARTCPU);
    bar.w32(base + falcon::CPUCTL_ALIAS, falcon::CPUCTL_STARTCPU);

    poll_for_boot(bar, base)
}

/// Fallback: try booting via CPUCTL directly.
fn try_boot_cpuctl(bar: &Bar0Map, base: u32, bootvec_val: u32) -> bool {
    println!("  CPUCTL ← {:#x} (HRESET)", falcon::CPUCTL_HRESET);
    bar.w32(base + falcon::CPUCTL, falcon::CPUCTL_HRESET);
    std::thread::sleep(Duration::from_millis(10));

    bar.w32(base + falcon::BOOTVEC, bootvec_val);
    bar.w32(base + falcon::MAILBOX0, 0);

    println!("  CPUCTL ← {:#x} (IINVAL)", falcon::CPUCTL_IINVAL);
    bar.w32(base + falcon::CPUCTL, falcon::CPUCTL_IINVAL);
    std::thread::sleep(Duration::from_millis(1));

    println!("  CPUCTL ← {:#x} (STARTCPU)", falcon::CPUCTL_STARTCPU);
    bar.w32(base + falcon::CPUCTL, falcon::CPUCTL_STARTCPU);

    poll_for_boot(bar, base)
}

fn poll_for_boot(bar: &Bar0Map, base: u32) -> bool {
    let start = Instant::now();
    let delays = [1, 5, 10, 50, 100, 500, 1000, 2000, 5000];

    for &delay_ms in &delays {
        std::thread::sleep(Duration::from_millis(delay_ms));
        let elapsed = start.elapsed();
        let cpuctl = bar.r32(base + falcon::CPUCTL);
        let alias = bar.r32(base + falcon::CPUCTL_ALIAS);
        let sctl = bar.r32(base + falcon::SCTL);
        let mb0 = bar.r32(base + falcon::MAILBOX0);
        let pc = bar.r32(base + falcon::PC);
        let exci = bar.r32(base + falcon::EXCI);

        let sec = sctl & 0x3;
        let halted = (cpuctl | alias) & falcon::CPUCTL_HALTED != 0;
        let hreset = (cpuctl | alias) & falcon::CPUCTL_HRESET != 0;
        let state = if halted {
            "HALT"
        } else if hreset {
            "HRESET"
        } else {
            "RUN?"
        };

        println!(
            "    t={:>6.1}ms: {state:<6} CPUCTL={cpuctl:#010x} ALIAS={alias:#010x} \
             SEC={sec} PC={pc:#010x} MB0={mb0:#010x} EXCI={exci:#010x}",
            elapsed.as_secs_f64() * 1000.0,
        );

        if cpuctl & (falcon::CPUCTL_HALTED | falcon::CPUCTL_HRESET) == 0 {
            println!("    *** CPU IS RUNNING (no HALT/HRESET flags) ***");
            return true;
        }
        if mb0 != 0 {
            println!("    *** MAILBOX RESPONSE: {mb0:#010x} ***");
            return true;
        }
        if sec != 2 && sec != 0 {
            println!("    *** SEC_MODE changed to {sec} ***");
        }
    }
    false
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let target_bdf = extract_arg(&args, "--target").unwrap_or_else(|| "0000:49:00.0".into());
    let control_bdf = extract_arg(&args, "--control").unwrap_or_else(|| "0000:02:00.0".into());

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXP 224: PMU ACR Catalyst — CPUCTL_ALIAS Path              ║");
    println!("║  DO NOT touch ENGCTL (0x3C0) — stay in HS mode              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Target:  {target_bdf}");
    println!("  Control: {control_bdf}");

    // Target GPU: SafeBar0 with ENGCTL deny-list to prevent accidental destruction
    let target_safe = SafeBar0::open_with_deny_list(
        &target_bdf,
        vec![Bar0Domain::pmc(), Bar0Domain::pmu_falcon(), Bar0Domain::fecs_falcon()],
        vec![
            DenyEntry {
                offset: falcon::PMU_BASE + falcon::ENGCTL,
                reason: "ENGCTL destroys falcon security state irreversibly",
            },
            DenyEntry {
                offset: falcon::FECS_BASE + falcon::ENGCTL,
                reason: "ENGCTL destroys falcon security state irreversibly",
            },
        ],
    ).unwrap_or_else(|e| {
        eprintln!("FATAL: cannot open target BAR0: {e}");
        std::process::exit(1);
    });

    // Use inner() for raw Bar0Map access in boot sequences (which need
    // unchecked speed for tight polling loops). SafeBar0 guards the domains.
    let target = target_safe.inner();

    let control_path = format!("/sys/bus/pci/devices/{control_bdf}/resource0");
    let control = Bar0Map::open(&control_path).unwrap_or_else(|e| {
        eprintln!("FATAL: cannot open control BAR0: {e}");
        std::process::exit(1);
    });

    // ── Phase 1: Probe ──────────────────────────────────────────────────────

    println!("\n━━━ Phase 1: GPU Probe ━━━\n");

    let boot0_t = target.r32(falcon::BOOT0);
    let boot0_c = control.r32(falcon::BOOT0);
    let pmc_t = target.r32(falcon::PMC_ENABLE);
    let pmc_c = control.r32(falcon::PMC_ENABLE);

    println!("  Target  BOOT0={boot0_t:#010x}  PMC={pmc_t:#010x}");
    println!("  Control BOOT0={boot0_c:#010x}  PMC={pmc_c:#010x}");

    if boot0_t == falcon::DEAD_LINK || boot0_c == falcon::DEAD_LINK {
        eprintln!("FATAL: GPU link dead (0xFFFFFFFF)");
        std::process::exit(1);
    }

    let t_snap = FalconSnapshot::read(target, falcon::PMU_BASE);
    let c_snap = FalconSnapshot::read(&control, falcon::PMU_BASE);

    println!();
    t_snap.print("Target PMU");
    println!();
    c_snap.print("Control PMU");

    let fecs_cpuctl = target.r32(falcon::FECS_BASE + falcon::CPUCTL);
    println!("\n  Target FECS CPUCTL={fecs_cpuctl:#010x}");

    if t_snap.sec_mode() != 2 {
        eprintln!(
            "\n  WARNING: Target SEC_MODE={} ({}), expected 2 (HS).",
            t_snap.sec_mode(), t_snap.sec_mode_str(),
        );
        eprintln!("  GPU may need a power cycle to restore VBIOS state.");
    }

    // ── Phase 2: PIO Access Test ────────────────────────────────────────────

    println!("\n━━━ Phase 2: PIO Access Test (Target) ━━━\n");

    target.w32(falcon::PMU_BASE + falcon::IMEMC, falcon::IMEMC_AINCR);
    let imem_0 = target.r32(falcon::PMU_BASE + falcon::IMEMD);
    let imem_1 = target.r32(falcon::PMU_BASE + falcon::IMEMD);
    println!("  IMEM[0x0000]: [{imem_0:#010x}, {imem_1:#010x}]");

    target.w32(falcon::PMU_BASE + falcon::DMEMC, falcon::DMEMC_AINCR);
    let dmem_0 = target.r32(falcon::PMU_BASE + falcon::DMEMD);
    println!("  DMEM[0x0000]: {dmem_0:#010x}");

    let sentinel: u32 = 0xCAFE_BEEF;
    let test_off: u32 = 0x1F00;
    target.w32(falcon::PMU_BASE + falcon::DMEMC, falcon::DMEMC_AINCW | test_off);
    target.w32(falcon::PMU_BASE + falcon::DMEMD, sentinel);
    target.w32(falcon::PMU_BASE + falcon::DMEMC, falcon::DMEMC_AINCR | test_off);
    let rb = target.r32(falcon::PMU_BASE + falcon::DMEMD);
    let dmem_ok = rb == sentinel;
    println!("  DMEM write test: {sentinel:#010x} at {test_off:#x} → {rb:#010x} {}", ok(dmem_ok));

    let imem_test_off: u32 = 0x0800;
    target.w32(falcon::PMU_BASE + falcon::IMEMC, falcon::IMEMC_AINCW | imem_test_off);
    target.w32(falcon::PMU_BASE + falcon::IMEMT, imem_test_off >> 8);
    target.w32(falcon::PMU_BASE + falcon::IMEMD, sentinel);
    target.w32(falcon::PMU_BASE + falcon::IMEMC, imem_test_off);
    let rb_i = target.r32(falcon::PMU_BASE + falcon::IMEMD);
    let imem_ok = rb_i == sentinel;
    println!("  IMEM write test: {sentinel:#010x} at {imem_test_off:#x} → {rb_i:#010x} {}", ok(imem_ok));

    if !dmem_ok && !imem_ok {
        println!("\n  PIO appears blocked. Trying with secure flag (BIT28)...");
        target.w32(falcon::PMU_BASE + falcon::IMEMC, falcon::IMEMC_AINCW | falcon::IMEMC_SECURE | 0x0800);
        target.w32(falcon::PMU_BASE + falcon::IMEMT, 0x08);
        target.w32(falcon::PMU_BASE + falcon::IMEMD, sentinel);
        target.w32(falcon::PMU_BASE + falcon::IMEMC, falcon::IMEMC_AINCR | falcon::IMEMC_SECURE | 0x0800);
        let rb_sec = target.r32(falcon::PMU_BASE + falcon::IMEMD);
        println!("  Secure PIO: {rb_sec:#010x} {}", ok(rb_sec == sentinel));
    }

    // ── Phase 3: CPUCTL_ALIAS Responsiveness ────────────────────────────────

    println!("\n━━━ Phase 3: CPUCTL_ALIAS Test ━━━\n");

    let alias_before = target.r32(falcon::PMU_BASE + falcon::CPUCTL_ALIAS);
    let cpuctl_before = target.r32(falcon::PMU_BASE + falcon::CPUCTL);
    println!("  Before: ALIAS={alias_before:#010x} CPUCTL={cpuctl_before:#010x}");

    target.w32(falcon::PMU_BASE + falcon::CPUCTL_ALIAS, falcon::CPUCTL_HRESET);
    std::thread::sleep(Duration::from_millis(10));
    let alias_after = target.r32(falcon::PMU_BASE + falcon::CPUCTL_ALIAS);
    let cpuctl_after = target.r32(falcon::PMU_BASE + falcon::CPUCTL);
    println!("  After ALIAS←0x10: ALIAS={alias_after:#010x} CPUCTL={cpuctl_after:#010x}");

    let alias_responsive = alias_after != alias_before || cpuctl_after != cpuctl_before;
    println!("  ALIAS responsive: {}", yn(alias_responsive));

    // ── Phase 4: Firmware Load ──────────────────────────────────────────────

    println!("\n━━━ Phase 4: Firmware PIO Load ━━━\n");

    let tags_nvidia = [0x0000_0100_u32, 0x0000_0101];
    falcon::pio_upload_imem(target, falcon::PMU_BASE, ACR_BL_IMEM_OFFSET, &ACR_BL_IMEM, &tags_nvidia);

    let imem_errs = falcon::pio_verify_imem(target, falcon::PMU_BASE, ACR_BL_IMEM_OFFSET, &ACR_BL_IMEM);
    println!(
        "  IMEM: {}/{} correct {}",
        ACR_BL_IMEM.len() - imem_errs,
        ACR_BL_IMEM.len(),
        ok(imem_errs == 0),
    );

    falcon::pio_upload_dmem(target, falcon::PMU_BASE, 0, &ACR_DMEM_DESC);
    let dmem_errs = falcon::pio_verify_dmem(target, falcon::PMU_BASE, 0, &ACR_DMEM_DESC);
    println!(
        "  DMEM: {}/{} correct {}",
        ACR_DMEM_DESC.len() - dmem_errs,
        ACR_DMEM_DESC.len(),
        ok(dmem_errs == 0),
    );

    if imem_errs > 0 || dmem_errs > 0 {
        eprintln!("\n  FIRMWARE LOAD FAILED — aborting boot.");
        std::process::exit(1);
    }

    // ── Phase 5: Boot Attempts ──────────────────────────────────────────────

    println!("\n━━━ Phase 5: Boot via CPUCTL_ALIAS (BOOTVEC={ACR_BL_IMEM_OFFSET:#x}) ━━━\n");

    let mut success = try_boot_alias(target, falcon::PMU_BASE, ACR_BL_IMEM_OFFSET);

    if !success {
        println!("\n  ALIAS boot did not succeed. Trying CPUCTL direct...\n");
        success = try_boot_cpuctl(target, falcon::PMU_BASE, ACR_BL_IMEM_OFFSET);
    }

    if !success {
        println!("\n  Primary boot failed. Trying BOOTVEC=0...\n");
        success = try_boot_alias(target, falcon::PMU_BASE, 0);
    }

    if !success {
        println!("\n  Trying nvidia CPUCTL=0x12 (HS ROM) trigger...\n");
        target.w32(falcon::PMU_BASE + falcon::CPUCTL, falcon::CPUCTL_HRESET);
        std::thread::sleep(Duration::from_millis(10));
        target.w32(
            falcon::PMU_BASE + falcon::CPUCTL,
            falcon::CPUCTL_STARTCPU | falcon::CPUCTL_HRESET,
        );
        print!("  CPUCTL=0x12 poll: ");
        let _ = io::stdout().flush();
        for _ in 0..10 {
            std::thread::sleep(Duration::from_millis(200));
            let cpuctl = target.r32(falcon::PMU_BASE + falcon::CPUCTL);
            let sctl = target.r32(falcon::PMU_BASE + falcon::SCTL);
            let mb0 = target.r32(falcon::PMU_BASE + falcon::MAILBOX0);
            print!("[{cpuctl:#x}/{sctl:#x}/{mb0:#x}] ");
            let _ = io::stdout().flush();
            if cpuctl & (falcon::CPUCTL_HALTED | falcon::CPUCTL_HRESET) == 0 || mb0 != 0 {
                println!("\n  *** Response detected ***");
                success = true;
                break;
            }
        }
        println!();
    }

    // ── Phase 6: Final State ────────────────────────────────────────────────

    println!("\n━━━ Phase 6: Final State ━━━\n");

    let t_final = FalconSnapshot::read(target, falcon::PMU_BASE);
    let c_final = FalconSnapshot::read(&control, falcon::PMU_BASE);

    t_final.print("Target PMU (final)");
    println!();
    c_final.print("Control PMU (final)");

    println!();
    let ctrl_ok = c_final.sec_mode() == c_snap.sec_mode();
    let pmu_alive = target.r32(falcon::BOOT0) != falcon::DEAD_LINK;

    println!("  ┌──────────────────────────────────────────┐");
    println!("  │  EXP 224: PMU ACR Catalyst Score          │");
    println!("  │  PMU CPU started:       {}               │", yn(success));
    println!("  │  PIO IMEM verified:     {}               │", yn(imem_errs == 0));
    println!("  │  PIO DMEM verified:     {}               │", yn(dmem_errs == 0));
    println!("  │  Control GPU unchanged: {}               │", yn(ctrl_ok));
    println!("  │  Target GPU alive:      {}               │", yn(pmu_alive));
    println!("  └──────────────────────────────────────────┘");

    // Verify ENGCTL deny-list worked: attempt a write (it should be rejected)
    let engctl_result = target_safe.w32(
        falcon::PMU_BASE + falcon::ENGCTL,
        0,
    );
    if let Err(e) = &engctl_result {
        println!("\n  ENGCTL deny-list active: {e}");
    }

    if success {
        println!("\n  PMU ACR boot succeeded!");
        println!("  Next: FECS/GPCCS boot via the same ALIAS pattern.");
    } else {
        println!("\n  PMU CPU did not start. Analysis:");
        println!("  - ALIAS responsive in Phase 3: {}", yn(alias_responsive));
        println!("  - SEC_MODE at end: {} ({})", t_final.sec_mode(), t_final.sec_mode_str());
        if t_final.sec_mode() == 0 {
            println!("  - GPU fell to NS mode — ENGCTL was inadvertently triggered");
            println!("  - This is UNRECOVERABLE without a power cycle");
        }
    }

    println!("\n═══════════════════════════════════════");
    println!("  EXP 224 complete.");
    println!("═══════════════════════════════════════");
}

fn ok(b: bool) -> &'static str {
    if b { "[OK]" } else { "[FAIL]" }
}

fn yn(b: bool) -> &'static str {
    if b { "YES" } else { "NO " }
}

fn extract_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}
