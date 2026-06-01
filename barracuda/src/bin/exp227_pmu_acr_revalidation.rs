// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 227: PMU ACR Revalidation — ember-wired edition
//!
//! ## Context
//!
//! Rewire of Exp 224 (PMU ACR Catalyst) from direct BAR0 mmap to ember RPCs.
//! All GPU interaction routes through toadstool-ember's fork-isolated MMIO
//! gateway with circuit breaker protection. If the GPU triggers a D-state
//! hang during falcon PIO, ember's child process is killed — the experiment
//! binary and the daemon survive.
//!
//! ## What this re-examines
//!
//! 1. PIO access to PMU IMEM/DMEM in HS mode 2 (blocked in Exp 224)
//! 2. CPUCTL_ALIAS responsiveness for sovereign falcon boot
//! 3. ACR bootloader firmware load and CPU start
//! 4. Control-card comparison (twin Titan V differential)
//!
//! ## Wiring
//!
//! | Operation              | RPC method                        |
//! |------------------------|-----------------------------------|
//! | Register read          | `mmio.read32`                     |
//! | Register write         | `mmio.write32`                    |
//! | Batch writes           | `mmio.batch`                      |
//! | IMEM upload            | `ember.falcon.upload_imem`        |
//! | DMEM upload            | `ember.falcon.upload_dmem`        |
//! | Falcon poll            | `ember.falcon.poll`               |
//!
//! ## Usage
//!
//! ```text
//! sudo ./target/release/exp227_pmu_acr_revalidation \
//!      --bdf 0000:49:00.0                            \
//!      --control 0000:02:00.0                        \
//!      [--ember-socket /run/toadstool/biomeos/compute.sock] \
//!      [--dry-run]
//! ```

use std::time::{Duration, Instant};

use hotspring_barracuda::ember_types::MmioBatchOp;
use hotspring_barracuda::fleet_client::EmberClient;

#[path = "../bin_helpers/sovereignty/mod.rs"]
mod sovereignty;
use sovereignty::connect::{connect_ember, extract_arg};

// ── Falcon v5 register offsets (GV100, relative to engine base) ──────────

const CPUCTL: u32 = 0x100;
const BOOTVEC: u32 = 0x104;
const CPUCTL_ALIAS: u32 = 0x130;
const EXCI: u32 = 0x148;
const IMEMC: u32 = 0x180;
const IMEMD: u32 = 0x184;
const DMEMC: u32 = 0x1C0;
const DMEMD: u32 = 0x1C4;
const SCTL: u32 = 0x240;
const MAILBOX0: u32 = 0x040;
const MAILBOX1: u32 = 0x044;
const PC: u32 = 0x030;
const ENGCTL: u32 = 0x3C0;

const CPUCTL_IINVAL: u32 = 1 << 0;
const CPUCTL_STARTCPU: u32 = 1 << 1;
const CPUCTL_HRESET: u32 = 1 << 4;
const CPUCTL_HALTED: u32 = 1 << 5;

const IMEMC_AINCW: u32 = 0x0100_0000;
const IMEMC_AINCR: u32 = 0x0200_0000;
const DMEMC_AINCW: u32 = 0x0100_0000;
const DMEMC_AINCR: u32 = 0x0200_0000;

const PMU_BASE: u32 = 0x10_A000;
const FECS_BASE: u32 = 0x40_9000;
const BOOT0: u32 = 0x00_0000;
const PMC_ENABLE: u32 = 0x00_0200;

const DEAD_LINK: u32 = 0xFFFF_FFFF;

// ── mmiotrace-extracted ACR bootloader (128 words, IMEM offset 0xFE00) ───

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
    0x00000004,
    0xDD990000, 0x00000001,
    0x00000000,
    0x00000600, 0x00000600,
    0x00006900,
    0x00000000,
    0xDD998000, 0x00000001,
    0x000042F0,
    0x00000034, 0x0000002F,
];

// ── Ember MMIO helpers ──────────────────────────────────────────────────

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

// ── Falcon snapshot via ember ───────────────────────────────────────────

struct FalconState {
    cpuctl: u32,
    cpuctl_alias: u32,
    sctl: u32,
    pc: u32,
    mailbox0: u32,
    mailbox1: u32,
    exci: u32,
}

impl FalconState {
    fn read(ember: &EmberClient, bdf: &str, base: u32) -> Self {
        Self {
            cpuctl: r32(ember, bdf, base + CPUCTL),
            cpuctl_alias: r32(ember, bdf, base + CPUCTL_ALIAS),
            sctl: r32(ember, bdf, base + SCTL),
            pc: r32(ember, bdf, base + PC),
            mailbox0: r32(ember, bdf, base + MAILBOX0),
            mailbox1: r32(ember, bdf, base + MAILBOX1),
            exci: r32(ember, bdf, base + EXCI),
        }
    }

    fn sec_mode(&self) -> u32 { self.sctl & 0x3 }
    fn sec_mode_str(&self) -> &'static str {
        match self.sec_mode() {
            0 => "NS",
            1 => "LS",
            2 => "HS",
            3 => "??",
            _ => "??",
        }
    }

    fn print(&self, label: &str) {
        println!("  {label}:");
        println!("    CPUCTL={:#010x} ALIAS={:#010x}", self.cpuctl, self.cpuctl_alias);
        println!("    SEC_MODE={} ({}) PC={:#010x}", self.sec_mode(), self.sec_mode_str(), self.pc);
        println!("    MB0={:#010x} MB1={:#010x} EXCI={:#010x}", self.mailbox0, self.mailbox1, self.exci);
    }
}

// ── Boot sequences via ember RPCs ───────────────────────────────────────

fn try_boot_alias(ember: &EmberClient, bdf: &str, base: u32, bootvec_val: u32) -> bool {
    println!("  CPUCTL_ALIAS <- {CPUCTL_HRESET:#x} (HRESET)");
    w32(ember, bdf, base + CPUCTL_ALIAS, CPUCTL_HRESET);
    std::thread::sleep(Duration::from_millis(10));

    let alias_rb = r32(ember, bdf, base + CPUCTL_ALIAS);
    let cpuctl_rb = r32(ember, bdf, base + CPUCTL);
    println!("    readback: ALIAS={alias_rb:#010x} CPUCTL={cpuctl_rb:#010x}");

    println!("  BOOTVEC <- {bootvec_val:#x}");
    batch(ember, bdf, &[
        MmioBatchOp::write(base + BOOTVEC, bootvec_val),
        MmioBatchOp::write(base + MAILBOX0, 0),
        MmioBatchOp::write(base + MAILBOX1, 0),
    ]);

    println!("  CPUCTL_ALIAS <- {CPUCTL_IINVAL:#x} (IINVAL)");
    w32(ember, bdf, base + CPUCTL_ALIAS, CPUCTL_IINVAL);
    std::thread::sleep(Duration::from_millis(1));

    println!("  CPUCTL_ALIAS <- {CPUCTL_STARTCPU:#x} (STARTCPU)");
    w32(ember, bdf, base + CPUCTL_ALIAS, CPUCTL_STARTCPU);

    poll_for_boot(ember, bdf, base)
}

fn try_boot_cpuctl(ember: &EmberClient, bdf: &str, base: u32, bootvec_val: u32) -> bool {
    println!("  CPUCTL <- {CPUCTL_HRESET:#x} (HRESET)");
    w32(ember, bdf, base + CPUCTL, CPUCTL_HRESET);
    std::thread::sleep(Duration::from_millis(10));

    batch(ember, bdf, &[
        MmioBatchOp::write(base + BOOTVEC, bootvec_val),
        MmioBatchOp::write(base + MAILBOX0, 0),
    ]);

    println!("  CPUCTL <- {CPUCTL_IINVAL:#x} (IINVAL)");
    w32(ember, bdf, base + CPUCTL, CPUCTL_IINVAL);
    std::thread::sleep(Duration::from_millis(1));

    println!("  CPUCTL <- {CPUCTL_STARTCPU:#x} (STARTCPU)");
    w32(ember, bdf, base + CPUCTL, CPUCTL_STARTCPU);

    poll_for_boot(ember, bdf, base)
}

fn poll_for_boot(ember: &EmberClient, bdf: &str, base: u32) -> bool {
    let start = Instant::now();
    let delays = [1, 5, 10, 50, 100, 500, 1000, 2000, 5000];

    for &delay_ms in &delays {
        std::thread::sleep(Duration::from_millis(delay_ms));
        let elapsed = start.elapsed();
        let cpuctl = r32(ember, bdf, base + CPUCTL);
        let alias = r32(ember, bdf, base + CPUCTL_ALIAS);
        let sctl = r32(ember, bdf, base + SCTL);
        let mb0 = r32(ember, bdf, base + MAILBOX0);
        let pc = r32(ember, bdf, base + PC);
        let exci = r32(ember, bdf, base + EXCI);

        if cpuctl == 0xDEAD_DEAD || alias == 0xDEAD_DEAD {
            println!("    t={:>6.1}ms: EMBER CIRCUIT BREAKER — BAR0 dead", elapsed.as_secs_f64() * 1000.0);
            return false;
        }

        let sec = sctl & 0x3;
        let halted = (cpuctl | alias) & CPUCTL_HALTED != 0;
        let hreset = (cpuctl | alias) & CPUCTL_HRESET != 0;
        let state = if halted { "HALT" } else if hreset { "HRESET" } else { "RUN?" };

        println!(
            "    t={:>6.1}ms: {state:<6} CPUCTL={cpuctl:#010x} ALIAS={alias:#010x} \
             SEC={sec} PC={pc:#010x} MB0={mb0:#010x} EXCI={exci:#010x}",
            elapsed.as_secs_f64() * 1000.0,
        );

        if cpuctl & (CPUCTL_HALTED | CPUCTL_HRESET) == 0 {
            println!("    *** CPU IS RUNNING ***");
            return true;
        }
        if mb0 != 0 {
            println!("    *** MAILBOX RESPONSE: {mb0:#010x} ***");
            return true;
        }
    }
    false
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let target_bdf = extract_arg(&args, "--bdf")
        .or_else(|| extract_arg(&args, "--target"))
        .unwrap_or_else(|| "0000:49:00.0".into());
    let control_bdf = extract_arg(&args, "--control").unwrap_or_else(|| "0000:02:00.0".into());
    let ember_socket = extract_arg(&args, "--ember-socket");
    let dry_run = args.iter().any(|a| a == "--dry-run");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXP 227: PMU ACR Revalidation — ember-wired               ║");
    println!("║  All GPU access via toadstool-ember RPCs (crash protected)  ║");
    println!("║  DO NOT touch ENGCTL (0x3C0) — stay in HS mode             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Target:  {target_bdf}");
    println!("  Control: {control_bdf}");
    println!("  Wiring:  toadstool-ember (fork-isolated MMIO — no raw BAR0)");
    if dry_run {
        println!("  Mode:    DRY RUN (read-only pre-condition check)");
    }

    let ember = connect_ember(&target_bdf, ember_socket.as_deref());
    println!("  Ember:   {}", ember.socket_path().display());

    // ── Phase 1: Probe ──────────────────────────────────────────────────

    println!("\n━━━ Phase 1: GPU Probe (via ember) ━━━\n");

    let boot0_t = r32(&ember, &target_bdf, BOOT0);
    let boot0_c = r32(&ember, &control_bdf, BOOT0);
    let pmc_t = r32(&ember, &target_bdf, PMC_ENABLE);
    let pmc_c = r32(&ember, &control_bdf, PMC_ENABLE);

    println!("  Target  BOOT0={boot0_t:#010x}  PMC={pmc_t:#010x} (pop={})", pmc_t.count_ones());
    println!("  Control BOOT0={boot0_c:#010x}  PMC={pmc_c:#010x} (pop={})", pmc_c.count_ones());

    if boot0_t == DEAD_LINK || boot0_c == DEAD_LINK {
        eprintln!("FATAL: GPU link dead (0xFFFFFFFF) — check VFIO/ember status");
        std::process::exit(1);
    }

    let t_snap = FalconState::read(&ember, &target_bdf, PMU_BASE);
    let c_snap = FalconState::read(&ember, &control_bdf, PMU_BASE);
    println!();
    t_snap.print("Target PMU");
    println!();
    c_snap.print("Control PMU");

    let fecs_cpuctl = r32(&ember, &target_bdf, FECS_BASE + CPUCTL);
    println!("\n  Target FECS CPUCTL={fecs_cpuctl:#010x}");

    if t_snap.sec_mode() != 2 {
        eprintln!(
            "\n  WARNING: Target SEC_MODE={} ({}), expected 2 (HS).",
            t_snap.sec_mode(), t_snap.sec_mode_str(),
        );
        eprintln!("  GPU may need a power cycle to restore VBIOS state.");
    }

    if dry_run {
        println!("\n  DRY RUN complete — exiting without writes.");
        return;
    }

    // ── Phase 2: PIO Access Test ────────────────────────────────────────

    println!("\n━━━ Phase 2: PIO Access Test (via ember) ━━━\n");

    w32(&ember, &target_bdf, PMU_BASE + IMEMC, IMEMC_AINCR);
    let imem_0 = r32(&ember, &target_bdf, PMU_BASE + IMEMD);
    let imem_1 = r32(&ember, &target_bdf, PMU_BASE + IMEMD);
    println!("  IMEM[0x0000]: [{imem_0:#010x}, {imem_1:#010x}]");

    w32(&ember, &target_bdf, PMU_BASE + DMEMC, DMEMC_AINCR);
    let dmem_0 = r32(&ember, &target_bdf, PMU_BASE + DMEMD);
    println!("  DMEM[0x0000]: {dmem_0:#010x}");

    let sentinel: u32 = 0xCAFE_BEEF;
    let test_off: u32 = 0x1F00;

    batch(&ember, &target_bdf, &[
        MmioBatchOp::write(PMU_BASE + DMEMC, DMEMC_AINCW | test_off),
        MmioBatchOp::write(PMU_BASE + DMEMD, sentinel),
    ]);
    w32(&ember, &target_bdf, PMU_BASE + DMEMC, DMEMC_AINCR | test_off);
    let rb = r32(&ember, &target_bdf, PMU_BASE + DMEMD);
    let dmem_ok = rb == sentinel;
    println!("  DMEM write test: {sentinel:#010x} at {test_off:#x} -> {rb:#010x} {}", tag(dmem_ok));

    let imem_test_off: u32 = 0x0800;
    batch(&ember, &target_bdf, &[
        MmioBatchOp::write(PMU_BASE + IMEMC, IMEMC_AINCW | imem_test_off),
        MmioBatchOp::write(PMU_BASE + IMEMD, sentinel),
    ]);
    w32(&ember, &target_bdf, PMU_BASE + IMEMC, imem_test_off);
    let rb_i = r32(&ember, &target_bdf, PMU_BASE + IMEMD);
    let imem_ok = rb_i == sentinel;
    println!("  IMEM write test: {sentinel:#010x} at {imem_test_off:#x} -> {rb_i:#010x} {}", tag(imem_ok));

    if !dmem_ok && !imem_ok {
        println!("\n  PIO blocked in HS mode 2 (expected on GV100 pre-DEVINIT).");
        println!("  This confirms the silicon boundary: host PIO to PMU is");
        println!("  gated by firmware execution, not just register access.");
    }

    // ── Phase 3: CPUCTL_ALIAS Responsiveness ────────────────────────────

    println!("\n━━━ Phase 3: CPUCTL_ALIAS Test (via ember) ━━━\n");

    let alias_before = r32(&ember, &target_bdf, PMU_BASE + CPUCTL_ALIAS);
    let cpuctl_before = r32(&ember, &target_bdf, PMU_BASE + CPUCTL);
    println!("  Before: ALIAS={alias_before:#010x} CPUCTL={cpuctl_before:#010x}");

    w32(&ember, &target_bdf, PMU_BASE + CPUCTL_ALIAS, CPUCTL_HRESET);
    std::thread::sleep(Duration::from_millis(10));
    let alias_after = r32(&ember, &target_bdf, PMU_BASE + CPUCTL_ALIAS);
    let cpuctl_after = r32(&ember, &target_bdf, PMU_BASE + CPUCTL);
    println!("  After ALIAS<-0x10: ALIAS={alias_after:#010x} CPUCTL={cpuctl_after:#010x}");

    let alias_responsive = alias_after != alias_before || cpuctl_after != cpuctl_before;
    println!("  ALIAS responsive: {}", yn(alias_responsive));

    // ── Phase 4: Firmware Load (via ember falcon RPCs) ──────────────────

    println!("\n━━━ Phase 4: Firmware PIO Load (via ember) ━━━\n");

    let imem_bytes: Vec<u8> = ACR_BL_IMEM.iter().flat_map(|w| w.to_le_bytes()).collect();
    match ember.falcon_upload_imem(&target_bdf, PMU_BASE, ACR_BL_IMEM_OFFSET, &imem_bytes, 0, false) {
        Ok(r) if r.ok => println!("  IMEM: {:?} bytes uploaded {}", r.bytes, tag(true)),
        Ok(r) => println!("  IMEM: upload returned ok=false, bytes={:?} {}", r.bytes, tag(false)),
        Err(e) => println!("  IMEM: upload failed: {e} {}", tag(false)),
    }

    let dmem_bytes: Vec<u8> = ACR_DMEM_DESC.iter().flat_map(|w| w.to_le_bytes()).collect();
    match ember.falcon_upload_dmem(&target_bdf, PMU_BASE, 0, &dmem_bytes) {
        Ok(r) if r.ok => println!("  DMEM: {:?} bytes uploaded {}", r.bytes, tag(true)),
        Ok(r) => println!("  DMEM: upload returned ok=false, bytes={:?} {}", r.bytes, tag(false)),
        Err(e) => println!("  DMEM: upload failed: {e} {}", tag(false)),
    }

    // ── Phase 5: Boot Attempts ──────────────────────────────────────────

    println!("\n━━━ Phase 5: Boot via CPUCTL_ALIAS (BOOTVEC={ACR_BL_IMEM_OFFSET:#x}) ━━━\n");

    let mut success = try_boot_alias(&ember, &target_bdf, PMU_BASE, ACR_BL_IMEM_OFFSET);

    if !success {
        println!("\n  ALIAS boot did not succeed. Trying CPUCTL direct...\n");
        success = try_boot_cpuctl(&ember, &target_bdf, PMU_BASE, ACR_BL_IMEM_OFFSET);
    }

    if !success {
        println!("\n  Primary boot failed. Trying BOOTVEC=0...\n");
        success = try_boot_alias(&ember, &target_bdf, PMU_BASE, 0);
    }

    if !success {
        println!("\n  Trying nvidia CPUCTL=0x12 (HS ROM) trigger...\n");
        w32(&ember, &target_bdf, PMU_BASE + CPUCTL, CPUCTL_HRESET);
        std::thread::sleep(Duration::from_millis(10));
        w32(&ember, &target_bdf, PMU_BASE + CPUCTL, CPUCTL_STARTCPU | CPUCTL_HRESET);
        for _ in 0..10 {
            std::thread::sleep(Duration::from_millis(200));
            let cpuctl = r32(&ember, &target_bdf, PMU_BASE + CPUCTL);
            let mb0 = r32(&ember, &target_bdf, PMU_BASE + MAILBOX0);
            print!("[{cpuctl:#x}/{mb0:#x}] ");
            if cpuctl & (CPUCTL_HALTED | CPUCTL_HRESET) == 0 || mb0 != 0 {
                println!("\n  *** Response detected ***");
                success = true;
                break;
            }
        }
        println!();
    }

    // ── Phase 6: Final State ────────────────────────────────────────────

    println!("\n━━━ Phase 6: Final State ━━━\n");

    let t_final = FalconState::read(&ember, &target_bdf, PMU_BASE);
    let c_final = FalconState::read(&ember, &control_bdf, PMU_BASE);

    t_final.print("Target PMU (final)");
    println!();
    c_final.print("Control PMU (final)");

    let ctrl_ok = c_final.sec_mode() == c_snap.sec_mode();
    let pmu_alive = r32(&ember, &target_bdf, BOOT0) != DEAD_LINK;

    // ENGCTL deny-list is now enforced by ember's domain validation,
    // not a client-side SafeBar0. Verify by attempting a write:
    let engctl_blocked = ember.mmio_write(&target_bdf, PMU_BASE + ENGCTL, 0).is_err();

    println!();
    println!("  ┌──────────────────────────────────────────────┐");
    println!("  │  EXP 227: PMU ACR Revalidation Score          │");
    println!("  │  PMU CPU started:       {}                   │", yn(success));
    println!("  │  PIO DMEM accessible:   {}                   │", yn(dmem_ok));
    println!("  │  PIO IMEM accessible:   {}                   │", yn(imem_ok));
    println!("  │  ALIAS responsive:      {}                   │", yn(alias_responsive));
    println!("  │  Control GPU unchanged: {}                   │", yn(ctrl_ok));
    println!("  │  Target GPU alive:      {}                   │", yn(pmu_alive));
    println!("  │  ENGCTL blocked:        {}                   │", yn(engctl_blocked));
    println!("  │  Wiring:                ember RPC            │");
    println!("  └──────────────────────────────────────────────┘");

    if success {
        println!("\n  PMU ACR boot succeeded!");
        println!("  Next: FECS/GPCCS boot via the same ALIAS pattern.");
    } else {
        println!("\n  PMU CPU did not start. Analysis:");
        println!("  - ALIAS responsive: {}", yn(alias_responsive));
        println!("  - SEC_MODE: {} ({})", t_final.sec_mode(), t_final.sec_mode_str());
        if t_final.sec_mode() == 0 {
            println!("  - GPU fell to NS mode — UNRECOVERABLE without power cycle");
        }
        if !dmem_ok && !imem_ok {
            println!("  - PIO blocked in HS mode 2 — host cannot load firmware");
            println!("  - This is the PMU silicon boundary on GV100");
            println!("  - Correct path: Boot Falcon (NVDEC) -> SEC2 -> ACR -> PMU");
            println!("  - toadStool sovereign.init handles this via DMA boot");
        }
    }

    println!("\n═══════════════════════════════════════");
    println!("  EXP 227 complete.");
    println!("═══════════════════════════════════════");
}


fn tag(b: bool) -> &'static str { if b { "[OK]" } else { "[FAIL]" } }
fn yn(b: bool) -> &'static str { if b { "YES" } else { "NO " } }
