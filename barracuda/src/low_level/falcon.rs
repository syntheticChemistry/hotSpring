// SPDX-License-Identifier: AGPL-3.0-or-later

//! Falcon v5 microcontroller register map, snapshot, and PIO helpers.
//!
//! Canonical register offsets verified against toadStool `falcon.rs` and
//! envytools. Used by sovereign boot experiment binaries (exp224+) and
//! available as a library module behind the `low-level` feature gate.
//!
//! # Register Map Corrections (Exp 223)
//!
//! | Offset | Name          | Previously Misidentified As |
//! |--------|---------------|-----------------------------|
//! | 0x104  | BOOTVEC       | "pre-trigger reg"           |
//! | 0x108  | HWCFG         | -                           |
//! | 0x10C  | DMACTL        | "0x10C"                     |
//! | 0x130  | CPUCTL_ALIAS  | (unused)                    |
//! | 0x148  | EXCI          | "HWCFG2"                    |
//! | 0x3C0  | ENGCTL        | "HS_CTRL"                   |

use super::bar0::{Bar0Domain, Bar0Map};
use std::fmt;

// ── Falcon v5 register offsets (relative to engine base) ─────────────────────

pub const IRQSSET: u32 = 0x000;
pub const IRQSCLR: u32 = 0x004;
pub const IRQSTAT: u32 = 0x008;
pub const IRQMSET: u32 = 0x010;
pub const IRQMCLR: u32 = 0x014;
pub const PC: u32 = 0x030;
pub const MAILBOX0: u32 = 0x040;
pub const MAILBOX1: u32 = 0x044;
pub const ITFEN: u32 = 0x048;
pub const CPUCTL: u32 = 0x100;
pub const BOOTVEC: u32 = 0x104;
pub const HWCFG: u32 = 0x108;
pub const DMACTL: u32 = 0x10C;
pub const CPUCTL_ALIAS: u32 = 0x130;
pub const EXCI: u32 = 0x148;
pub const IMEMC: u32 = 0x180;
pub const IMEMD: u32 = 0x184;
pub const IMEMT: u32 = 0x188;
pub const DMEMC: u32 = 0x1C0;
pub const DMEMD: u32 = 0x1C4;
pub const SCTL: u32 = 0x240;
/// Engine reset — writing 1/0 transitions HS→NS **irreversibly**.
/// NEVER write to this register unless you intend to destroy the falcon.
pub const ENGCTL: u32 = 0x3C0;

// ── CPUCTL bit definitions ───────────────────────────────────────────────────

pub const CPUCTL_IINVAL: u32 = 1 << 0;
pub const CPUCTL_STARTCPU: u32 = 1 << 1;
pub const CPUCTL_HRESET: u32 = 1 << 4;
pub const CPUCTL_HALTED: u32 = 1 << 5;

// ── IMEMC/DMEMC control flags ────────────────────────────────────────────────

pub const IMEMC_AINCW: u32 = 0x0100_0000;
pub const IMEMC_AINCR: u32 = 0x0200_0000;
pub const IMEMC_SECURE: u32 = 0x1000_0000;
pub const DMEMC_AINCW: u32 = 0x0100_0000;
pub const DMEMC_AINCR: u32 = 0x0200_0000;

// ── Engine base addresses (GV100) ────────────────────────────────────────────

pub const PMU_BASE: u32 = 0x10_A000;
pub const FECS_BASE: u32 = 0x40_9000;
pub const GPCCS_BASE: u32 = 0x41_A000;
pub const SEC2_BASE: u32 = 0x08_7000;
pub const NVDEC_BASE: u32 = 0x08_4000;

// ── GPU-wide registers ───────────────────────────────────────────────────────

pub const BOOT0: u32 = 0x00_0000;
pub const PMC_ENABLE: u32 = 0x00_0200;

/// MMIO dead-link sentinel — all BAR0 reads return this when PCIe link is down.
pub const DEAD_LINK: u32 = 0xFFFF_FFFF;

// ── Falcon window size (covers all registers up to and including ENGCTL+4) ───

const FALCON_WINDOW: u32 = 0x400;

// ── Bar0Domain presets ───────────────────────────────────────────────────────

impl Bar0Domain {
    #[must_use]
    pub fn pmc() -> Self {
        Self { name: "PMC", start: 0x00_0000, end: 0x00_1000 }
    }

    #[must_use]
    pub fn pmu_falcon() -> Self {
        Self { name: "PMU", start: PMU_BASE, end: PMU_BASE + FALCON_WINDOW }
    }

    #[must_use]
    pub fn fecs_falcon() -> Self {
        Self { name: "FECS", start: FECS_BASE, end: FECS_BASE + FALCON_WINDOW }
    }

    #[must_use]
    pub fn gpccs_falcon() -> Self {
        Self { name: "GPCCS", start: GPCCS_BASE, end: GPCCS_BASE + FALCON_WINDOW }
    }

    #[must_use]
    pub fn sec2_falcon() -> Self {
        Self { name: "SEC2", start: SEC2_BASE, end: SEC2_BASE + FALCON_WINDOW }
    }

    #[must_use]
    pub fn nvdec_falcon() -> Self {
        Self { name: "NVDEC", start: NVDEC_BASE, end: NVDEC_BASE + FALCON_WINDOW }
    }
}

// ── FalconSnapshot ───────────────────────────────────────────────────────────

/// Snapshot of a falcon engine's control registers.
pub struct FalconSnapshot {
    pub cpuctl: u32,
    pub alias: u32,
    pub sctl: u32,
    pub bootvec: u32,
    pub hwcfg: u32,
    pub dmactl: u32,
    pub exci: u32,
    pub pc: u32,
    pub mb0: u32,
    pub mb1: u32,
    pub itfen: u32,
    pub engctl: u32,
}

impl FalconSnapshot {
    /// Read all diagnostic registers from a falcon at `base`.
    pub fn read(bar: &Bar0Map, base: u32) -> Self {
        Self {
            cpuctl: bar.r32(base + CPUCTL),
            alias: bar.r32(base + CPUCTL_ALIAS),
            sctl: bar.r32(base + SCTL),
            bootvec: bar.r32(base + BOOTVEC),
            hwcfg: bar.r32(base + HWCFG),
            dmactl: bar.r32(base + DMACTL),
            exci: bar.r32(base + EXCI),
            pc: bar.r32(base + PC),
            mb0: bar.r32(base + MAILBOX0),
            mb1: bar.r32(base + MAILBOX1),
            itfen: bar.r32(base + ITFEN),
            engctl: bar.r32(base + ENGCTL),
        }
    }

    /// Construct from raw field values (for testing without hardware).
    #[cfg(test)]
    pub fn from_raw(
        cpuctl: u32, alias: u32, sctl: u32, bootvec: u32,
        hwcfg: u32, dmactl: u32, exci: u32, pc: u32,
        mb0: u32, mb1: u32, itfen: u32, engctl: u32,
    ) -> Self {
        Self { cpuctl, alias, sctl, bootvec, hwcfg, dmactl, exci, pc, mb0, mb1, itfen, engctl }
    }

    /// SEC_MODE from SCTL bits [1:0]: 0=NS, 1=LS, 2=HS.
    #[must_use]
    pub fn sec_mode(&self) -> u32 {
        self.sctl & 0x3
    }

    #[must_use]
    pub fn sec_mode_str(&self) -> &'static str {
        match self.sec_mode() {
            0 => "NS",
            1 => "LS",
            2 => "HS",
            _ => "??",
        }
    }

    #[must_use]
    pub fn cpu_state(&self) -> &'static str {
        if self.cpuctl & CPUCTL_HALTED != 0 {
            "HALT"
        } else if self.cpuctl & CPUCTL_HRESET != 0 {
            "HRESET"
        } else {
            "RUN/IDLE"
        }
    }

    /// IMEM size in bytes (from HWCFG bits [8:0], units of 256 bytes).
    #[must_use]
    pub fn imem_size(&self) -> u32 {
        (self.hwcfg & 0x1FF) * 256
    }

    /// DMEM size in bytes (from HWCFG bits [17:9], units of 256 bytes).
    #[must_use]
    pub fn dmem_size(&self) -> u32 {
        ((self.hwcfg >> 9) & 0x1FF) * 256
    }

    /// Print a labeled register dump to stdout.
    pub fn print(&self, label: &str) {
        println!("  {label}:");
        println!(
            "    CPUCTL={:#010x} ({})  ALIAS={:#010x}",
            self.cpuctl, self.cpu_state(), self.alias,
        );
        println!(
            "    SCTL={:#010x} (SEC_MODE={} {})  ENGCTL={:#010x}",
            self.sctl, self.sec_mode(), self.sec_mode_str(), self.engctl,
        );
        println!("    BOOTVEC={:#010x}  PC={:#010x}", self.bootvec, self.pc);
        println!(
            "    HWCFG={:#010x} (IMEM={}B DMEM={}B)  DMACTL={:#010x}",
            self.hwcfg, self.imem_size(), self.dmem_size(), self.dmactl,
        );
        println!(
            "    EXCI={:#010x}  MB0={:#010x}  MB1={:#010x}  ITFEN={:#010x}",
            self.exci, self.mb0, self.mb1, self.itfen,
        );
    }
}

impl fmt::Display for FalconSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CPUCTL={:#010x}({}) SEC={}{} PC={:#010x} MB0={:#010x}",
            self.cpuctl, self.cpu_state(),
            self.sec_mode(), self.sec_mode_str(),
            self.pc, self.mb0,
        )
    }
}

// ── PIO upload/verify helpers ────────────────────────────────────────────────

/// Upload firmware words to falcon IMEM via PIO (IMEMC/IMEMD/IMEMT).
///
/// Tags are written at each 256-byte page boundary. Words beyond the last
/// full page are zero-padded to the boundary.
pub fn pio_upload_imem(bar: &Bar0Map, base: u32, addr: u32, words: &[u32], tags: &[u32]) {
    bar.w32(base + IMEMC, IMEMC_AINCW | addr);
    let mut tag_idx = 0;
    for (i, &word) in words.iter().enumerate() {
        let byte_off = (i * 4) as u32;
        if byte_off % 256 == 0 && tag_idx < tags.len() {
            bar.w32(base + IMEMT, tags[tag_idx]);
            tag_idx += 1;
        }
        bar.w32(base + IMEMD, word);
    }
    let remainder = (words.len() * 4) & 0xFF;
    if remainder != 0 {
        for _ in 0..((256 - remainder) / 4) {
            bar.w32(base + IMEMD, 0);
        }
    }
}

/// Verify IMEM contents against expected words. Returns the error count.
pub fn pio_verify_imem(bar: &Bar0Map, base: u32, addr: u32, words: &[u32]) -> usize {
    bar.w32(base + IMEMC, IMEMC_AINCR | addr);
    let mut errors = 0;
    for (i, &expected) in words.iter().enumerate() {
        let got = bar.r32(base + IMEMD);
        if got != expected {
            if errors < 3 {
                println!("    IMEM[{i}] MISMATCH: {got:#010x} != {expected:#010x}");
            }
            errors += 1;
        }
    }
    errors
}

/// Upload data words to falcon DMEM via PIO (DMEMC/DMEMD).
pub fn pio_upload_dmem(bar: &Bar0Map, base: u32, addr: u32, words: &[u32]) {
    bar.w32(base + DMEMC, DMEMC_AINCW | addr);
    for &word in words {
        bar.w32(base + DMEMD, word);
    }
}

/// Verify DMEM contents against expected words. Returns the error count.
pub fn pio_verify_dmem(bar: &Bar0Map, base: u32, addr: u32, words: &[u32]) -> usize {
    bar.w32(base + DMEMC, DMEMC_AINCR | addr);
    let mut errors = 0;
    for (i, &expected) in words.iter().enumerate() {
        let got = bar.r32(base + DMEMD);
        if got != expected {
            if errors < 3 {
                println!("    DMEM[{i}] MISMATCH: {got:#010x} != {expected:#010x}");
            }
            errors += 1;
        }
    }
    errors
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sec_mode_decode() {
        let snap = FalconSnapshot::from_raw(0, 0, 0x3002, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(snap.sec_mode(), 2);
        assert_eq!(snap.sec_mode_str(), "HS");

        let snap = FalconSnapshot::from_raw(0, 0, 0x3000, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(snap.sec_mode(), 0);
        assert_eq!(snap.sec_mode_str(), "NS");

        let snap = FalconSnapshot::from_raw(0, 0, 0x0001, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(snap.sec_mode(), 1);
        assert_eq!(snap.sec_mode_str(), "LS");

        let snap = FalconSnapshot::from_raw(0, 0, 0xFFFF, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(snap.sec_mode(), 3);
        assert_eq!(snap.sec_mode_str(), "??");
    }

    #[test]
    fn cpu_state_decode() {
        let snap = FalconSnapshot::from_raw(CPUCTL_HALTED, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(snap.cpu_state(), "HALT");

        let snap = FalconSnapshot::from_raw(CPUCTL_HRESET, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(snap.cpu_state(), "HRESET");

        let snap = FalconSnapshot::from_raw(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(snap.cpu_state(), "RUN/IDLE");

        // HALTED takes priority over HRESET
        let snap = FalconSnapshot::from_raw(
            CPUCTL_HALTED | CPUCTL_HRESET, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(snap.cpu_state(), "HALT");
    }

    #[test]
    fn imem_dmem_size_from_hwcfg() {
        // IMEM=256 units × 256 = 65536, DMEM=224 units × 256 = 57344
        let hwcfg: u32 = 0x400e_0100;
        let snap = FalconSnapshot::from_raw(0, 0, 0, 0, hwcfg, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(snap.imem_size(), 65536);
        assert_eq!(snap.dmem_size(), 65536);

        // Matches toadStool falcon.rs: (hwcfg & 0x1FF) * 256
        let hwcfg2 = (3 << 9) | 5;
        let snap2 = FalconSnapshot::from_raw(0, 0, 0, 0, hwcfg2, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(snap2.imem_size(), 5 * 256);
        assert_eq!(snap2.dmem_size(), 3 * 256);
    }

    #[test]
    fn cpuctl_bits_match_toadstool() {
        assert_eq!(CPUCTL_IINVAL, 0x01);
        assert_eq!(CPUCTL_STARTCPU, 0x02);
        assert_eq!(CPUCTL_HRESET, 0x10);
        assert_eq!(CPUCTL_HALTED, 0x20);
        assert_eq!(CPUCTL_STARTCPU | CPUCTL_IINVAL, 0x03);
        assert_eq!(CPUCTL_STARTCPU | CPUCTL_HRESET, 0x12);
    }

    #[test]
    fn engine_base_plus_register_offsets() {
        assert_eq!(PMU_BASE + CPUCTL, 0x10_A100);
        assert_eq!(PMU_BASE + CPUCTL_ALIAS, 0x10_A130);
        assert_eq!(PMU_BASE + SCTL, 0x10_A240);
        assert_eq!(PMU_BASE + ENGCTL, 0x10_A3C0);
        assert_eq!(PMU_BASE + IMEMC, 0x10_A180);

        assert_eq!(FECS_BASE + CPUCTL, 0x40_9100);
        assert_eq!(SEC2_BASE + CPUCTL, 0x08_7100);
        assert_eq!(NVDEC_BASE + CPUCTL, 0x08_4100);
    }

    #[test]
    fn domain_presets_cover_required_offsets() {
        let pmu = Bar0Domain::pmu_falcon();
        assert!(PMU_BASE >= pmu.start);
        assert!(PMU_BASE + ENGCTL + 4 <= pmu.end);
        assert!(PMU_BASE + SCTL + 4 <= pmu.end);
        assert!(PMU_BASE + IMEMC + 4 <= pmu.end);

        let fecs = Bar0Domain::fecs_falcon();
        assert!(FECS_BASE + CPUCTL + 4 <= fecs.end);
        assert!(FECS_BASE + ENGCTL + 4 <= fecs.end);

        let pmc = Bar0Domain::pmc();
        assert!(BOOT0 + 4 <= pmc.end);
        assert!(PMC_ENABLE + 4 <= pmc.end);
    }

    #[test]
    fn display_impl() {
        let snap = FalconSnapshot::from_raw(
            CPUCTL_HALTED, 0, 0x3002, 0, 0x400e_0100, 0, 0, 0x2E, 0x300, 0, 0, 0,
        );
        let s = format!("{snap}");
        assert!(s.contains("HALT"));
        assert!(s.contains("HS"));
        assert!(s.contains("0x0000002e"));
    }
}
