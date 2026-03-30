# Data Inventory — hotSpring GPU Cracking Corpus

**Last Updated:** 2026-03-30
**Hardware (biomeGate):** Titan V #1 (0000:03:00.0), Titan V #2 (0000:4a:00.0), RTX 5070 (GB206) (0000:21:00.0), Tesla K80 (GK210)
**Hardware (strandgate):** RTX 3090 (SM86, **GPFIFO operational**), RX 6950 XT (GFX10.3, **24/24 QCD compiled**)

## Capture Matrix

| Dataset | Titan #1 | Titan #2 | Format | Exp | Quality |
|---------|----------|----------|--------|-----|---------|
| Cold VFIO register snapshot | `070/cold_oracle.json` | `070/cold_target.json` | JSON (129 regs) | 070 | GOOD |
| VFIO (post-nouveau) snapshot | `070/config_a_oracle_vfio.json` | `070/config_a_target_vfio.json` | JSON | 070 | GOOD |
| Nouveau-warm snapshot | `070/config_b_oracle_nouveau_warm.json` | *(missing)* | JSON | 070 | GOOD (one card) |
| Nvidia-warm snapshot | `070/config_c_oracle_nvidia_warm.json` | *(missing)* | JSON | 070 | GOOD (one card) |
| Final warm snapshot | `070/config_d_oracle_final_warm.json` | `070/config_d_target_final_warm.json` | JSON | 070 | GOOD |
| Cold-vs-cold cross-card diff | `070/diff_oracle_vs_target_cold.json` | — | JSON diff | 070 | GOOD |
| D0 warm full register dump | — | `target_vfio_d0_warm.txt` | Text (58K lines) | — | GOOD (deep) |
| Recent GlowPlug snapshot | `snapshots/0000-03-00.0_snapshot_*.json` | `snapshots/0000-4b-00.0_snapshot_*.json` | JSON | — | GOOD |
| VBIOS dump | `vbios_0000_03_00_0.bin` | `vbios_0000_4a_00_0.bin` | Binary (127KB) | — | GOOD |
| Nouveau mmiotrace (raw) | `nouveau_mmiotrace_raw.log` | — | mmiotrace | — | **BAD** (98 lines, PCI enum only) |
| Nouveau mmiotrace (writes) | `mmiotrace_writes_*.txt` | — | filtered | — | **EMPTY** (0 bytes) |
| BAR0 warm (oracle) | `oracle_bar0_warm_*.bin` | — | Binary | — | **EMPTY** (0 bytes) |
| Oracle domain/PLL dumps | `scripts/data/oracle_domains_*.txt`, `oracle_root_plls_*.txt` | — | Text | — | GOOD |
| PFIFO diagnostic matrix | `071/titan_v_vfio_cold.json` | — | JSON | 071 | GOOD |
| AMD Radeon VII (reference) | — | `071/radeon_vii_*` | JSON | 071 | GOOD |

## Exp 082 Attempts (Failed)

| Path | Contents | Issue |
|------|----------|-------|
| `082/nouveau_0000_03_00.0_20260324_082638/` | `bar0_cold_vfio.bin` (0 bytes) | pkexec script hung during unbind |
| `082/nouveau_titan1_20260324_082924/` | Empty directory | Script killed before capture |

## Critical Gaps

### 1. No Valid mmiotrace Captures
The only mmiotrace file (`nouveau_mmiotrace_raw.log`, 98 lines) contains only PCI device enumeration — no MMIO write operations were recorded. This likely means the trace was captured too early (before driver bind) or the driver didn't perform BAR0-mapped I/O during the capture window.

**Needed:** Full mmiotrace during nouveau bind and nvidia bind for both Titans. Each should produce 50K-200K write operations if the capture window covers the full driver init.

### 2. Titan #2 Missing Middle States
Titan #2 has cold and final snapshots but no per-driver intermediate states (nouveau-warm, nvidia-warm). These are needed for cross-card comparison to distinguish hardware-specific vs universal init patterns.

### 3. No nvidia_oracle Captures
The `nvidia_oracle.ko` module has never been built or loaded. No captures exist.

### 4. No Warm BAR0 Binary Captures
The `oracle_bar0_warm_*.bin` files are 0 bytes — the BAR0 read failed (likely the GPU was not in a readable state).

## Comparison Results (Generated)

Using `scripts/compare_snapshots.py`:

### Cold → Nouveau-warm (Titan #1)
- **86 registers changed**, 43 unchanged
- Falcon findings: PMU HALTED (CPUCTL 0x20→0x10), PMU BOOTVEC set to 0x10000
- FECS HALTED (CPUCTL 0xbadf→0x10), FECS BOOTVEC = 0 (not booted, but accessible)
- PFIFO fully initialized, PBDMA0/PBDMA2 active with GP entries
- MMU fault buffers configured

### Nouveau-warm → Nvidia-warm (Titan #1)
- **7 falcon registers changed**
- PMU CPUCTL: 0x10 → 0x20 (nouveau left HALTED → nvidia leaves different state)
- FECS: all registers return to `0xbadf1201` (powered down / inaccessible)
- **Conclusion:** nvidia teardown destroys the falcon accessibility that nouveau preserves

### Key Insight for ACR Boot Solver
After nouveau: PMU and FECS are HALTED but **register-accessible** (reads return valid values).
After nvidia: FECS registers return `0xbadf` (BAR0 PRI timeout — engine powered off).
**nouveau is the only viable warm-up path** for leaving falcons in a state where VFIO can subsequently attempt boot sequences.
