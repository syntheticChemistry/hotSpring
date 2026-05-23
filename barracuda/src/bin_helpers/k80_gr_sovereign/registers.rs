// SPDX-License-Identifier: AGPL-3.0-or-later

// ── PMC / MC registers ─────────────────────────────────────────────────────
pub const BOOT0: u32 = 0x000000;
pub const PMC_ENABLE: u32 = 0x000200;
pub const MC_UNK260: u32 = 0x000260;
pub const PTIMER_LO: u32 = 0x009400; // TIME_0: increments ~32M/sec, readable while running

// ── GR engine PMC_ENABLE bitmask ──────────────────────────────────────────
pub const PMC_GR_BIT: u32 = 1 << 12; // bit 12 = GR engine (PGRAPH)

// ── PGRAPH / GR engine registers ──────────────────────────────────────────
pub const GR_READY: u32 = 0x409800; // FECS boot-ready flag (bit 31)
pub const GR_CTX_SIZE: u32 = 0x409804; // context buffer size (set by FECS at boot)

// ── FECS Falcon registers (base 0x409000) ─────────────────────────────────
pub const FECS_BASE: u32 = 0x409000;
pub const FECS_CPUCTL: u32 = FECS_BASE + 0x100;
pub const FECS_BOOTVEC: u32 = FECS_BASE + 0x104;
pub const FECS_PC: u32 = FECS_BASE + 0x110;
pub const FECS_SCTL: u32 = FECS_BASE + 0x240;
// GK110B: HWCFG at 0x10C (DMACTL in newer Falcon, repurposed in Kepler)
pub const FECS_HWCFG: u32 = FECS_BASE + 0x10C; // 0x40910C
// IMEMC/IMEMD/IMETTAG are handled internally by ember.falcon.upload_imem
pub const FECS_DMEMC: u32 = FECS_BASE + 0x1C0;
pub const FECS_DMEMD: u32 = FECS_BASE + 0x1C4;
pub const FECS_MB0: u32 = FECS_BASE + 0x040;
pub const FECS_MB1: u32 = FECS_BASE + 0x044;

// ── GPCCS Falcon registers (broadcast PIO at 0x41A000) ────────────────────
pub const GPCCS_BASE: u32 = 0x41A000;
pub const GPCCS_CPUCTL: u32 = GPCCS_BASE + 0x100;
pub const GPCCS_BOOTVEC: u32 = GPCCS_BASE + 0x104;
pub const GPCCS_PC: u32 = GPCCS_BASE + 0x110;
pub const GPCCS_HWCFG: u32 = GPCCS_BASE + 0x10C; // 0x41A10C
// IMEMC/IMEMD/IMETTAG are handled internally by ember.falcon.upload_imem
pub const GPCCS_DMEMC: u32 = GPCCS_BASE + 0x1C0;
pub const GPCCS_DMEMD: u32 = GPCCS_BASE + 0x1C4;
pub const GPCCS_MB0: u32 = GPCCS_BASE + 0x040;
pub const GPCCS_MB1: u32 = GPCCS_BASE + 0x044;

// ── GPC status ────────────────────────────────────────────────────────────
pub const GPC0_BOOT0: u32 = 0x418000;
pub const GPC1_BOOT0: u32 = 0x428000;

// ── FECS reset handshake (GR wrapper, not Falcon CPUCTL) ──────────────────
// nouveau's gf100_gr_fecs_reset: arms FECS's internal PRIV ring master unit
// BEFORE MC_UNK260=0. Sequence: 0x70 → 0x30 → poll bit4=0 → 0x10.
pub const FECS_RESET: u32 = FECS_BASE + 0x614; // 0x409614 — promoted to sovereign_stages.rs Phase 2b

// ── FECS init command ─────────────────────────────────────────────────────
// FECS reads MB0 exactly once at startup. MB0=4 → INIT_CTXSW:
//   train PRIV ring, establish GPC links, set GR_READY.
// Promoted from exp184 discovery → sovereign_stages.rs `FECS_CMD_INIT`.
pub const FECS_CMD_INIT: u32 = 0x0000_0004;

// ── DMEMC auto-increment flags ─────────────────────────────────────────────
pub const DMEMC_AINCW: u32 = 0x0100_0000; // bit 24: auto-increment write pointer

// ── FECS DMEM layout (r_hub = 0x600 from gk110b_grctx) ───────────────────
pub const FECS_DMEM_FW_OFFSET: u32 = 0x600;

// ── GPCCS DMEM layout (r_gpc ≈ 0x7ff → aligned to 0x800) ────────────────
pub const GPCCS_DMEM_FW_OFFSET: u32 = 0x800;

// ── IMEM page geometry (for display — PIO is handled internally by ember) ─
pub const IMEM_WORDS_PER_PAGE: usize = 64; // 256 bytes / 4 bytes per word
