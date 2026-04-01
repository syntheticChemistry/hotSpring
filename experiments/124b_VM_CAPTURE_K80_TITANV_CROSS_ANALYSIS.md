# Experiment 124: VM-Based BAR0 Capture — K80 + Titan V Cross-GPU Analysis

**Date:** 2026-03-29 → 2026-03-30  
**Hardware:** Tesla K80 die2 (GK210, `0000:4d:00.0`), Titan V (GV100, `0000:03:00.0`)  
**Depends on:** Exp 123 (K80 strategy), Exp 120 (sovereign DEVINIT), Exp 074 (ember/glowplug swap)  
**Status:** COMPLETE — both GPU captures extracted, cross-analysis done

---

## Hypothesis

By passing GPUs into VMs via VFIO, we can capture a clean BAR0 snapshot
at two critical points: (1) post-VBIOS/BIOS init ("cold" — before the
Linux driver touches the GPU) and (2) post-driver init ("warm"). The
delta reveals exactly what the VBIOS and driver each program, giving us
the sovereign register replay recipe.

## Method

### Pipeline
1. `device.lend` via glowplug IPC → release VFIO fd
2. `ember.release` → detach ember state tracking
3. `virt-install` with `--hostdev $BDF` + cloud-init Ubuntu 24.04 VM
4. SSH into guest → cold BAR0 snapshot (no driver bound)
5. Install NVIDIA driver → warm BAR0 snapshot
6. Extract artifacts via SCP → destroy VM
7. `ember.reacquire` + `device.reclaim` → return GPU to fleet

### K80 Capture (nvidia-470, GK210)
- Guest BDF: `0000:06:00.0` → Host: `0000:4d:00.0`
- IOMMU group 38 (single function, no audio peer)
- BIOS boot (no UEFI needed for Kepler)
- nvidia-470 full install via `.run` package

### Titan V Capture (nvidia-535, GV100)
- Guest BDF: `0000:05:00.0` → Host: `0000:03:00.0`
- IOMMU group 73 (VGA + audio: `03:00.0` + `03:00.1`)
- **Both IOMMU group functions required** for VM boot
- **UEFI without Secure Boot** required (OVMF_CODE_4M.fd, not .secboot.fd)
  - Secure Boot enables kernel lockdown → blocks direct PCI BAR access
  - BIOS boot with VGA passthrough hangs (no DHCP, stuck in display init)
- nvidia-535 full install via `.run` package

### BAR0 Snapshot Regions (both GPUs)

| Region | BAR0 Range | Purpose |
|--------|-----------|---------|
| PMC | 0x000000–0x001000 | Master control, BOOT0, enables |
| PBUS | 0x001000–0x002000 | Bus interface, interrupts |
| PTIMER | 0x009000–0x00A000 | GPU timer |
| PFIFO | 0x002000–0x004000 | DMA engine, channel scheduling |
| PPCI | 0x088000–0x089000 | PCI config space mirror |
| PCLOCK | 0x137000–0x138000 | PLL/clock programming |
| PFB | 0x100000–0x101000 | Framebuffer/memory controller |
| PCOPY | 0x104000–0x105000 | Copy engine |
| PROM | 0x300000–0x301000 | VBIOS ROM |
| PGRAPH | 0x400000–0x402000 | Graphics/compute engine |
| FECS | 0x409000–0x40A000 | Front-End Context Switch falcon |
| GPCCS | 0x41A000–0x41B000 | GPC Context Switch falcon |
| SEC2 | 0x840000–0x841000 | Security Engine 2 |
| PMU | 0x10A000–0x10B000 | Power Management Unit falcon |
| ACR | 0x1FA000–0x1FB000 | Authenticated Code Runner |

---

## Results

### K80 (GK210/Kepler)

| Metric | Value |
|--------|-------|
| Cold BAR0 (post-VBIOS) | 10,283 non-zero, non-BADF registers |
| Host cold BAR0 (pre-VBIOS) | 0 registers (truly cold) |
| VBIOS recipe size | **10,283 register writes** |
| PCLOCK regs captured | 64 (PLL/clock programming) |
| PGRAPH regs | 2,048 (fully initialized by VBIOS) |
| Security barriers | **None** (pre-Maxwell) |

**Sovereign readiness: HIGH** — direct register replay should cold-boot this GPU.

### Titan V (GV100/Volta)

| Metric | Value |
|--------|-------|
| Cold BAR0 raw | 15,363 registers |
| BADF markers (QEMU unmapped) | 8,895 excluded |
| VBIOS recipe size | **6,468 register writes** |
| nvidia-535 driver delta | **255 register changes** |
| PCLOCK regs captured | 99 (PLL/clock programming) |
| PGRAPH regs | 0 (all BADF — falcon-gated) |
| SEC2 visible regs | 0 (all 0xBADF1100) |
| ACR visible regs | 5 |
| FECS visible regs | 42 (of 873 raw) |
| Security barriers | SEC2/ACR/WPR2 (falcon-authenticated boot) |

**Sovereign readiness: MEDIUM** — VBIOS recipe captured, but falcons are opaque.

### Cross-GPU Comparison

|  | K80 (GK210) | Titan V (GV100) |
|--|-------------|-----------------|
| Architecture | Kepler | Volta |
| Recipe size | 10,283 | 6,468 |
| Shared BAR0 offsets | 6,119 | 6,119 |
| Same value at shared offsets | 90 (1.5%) | — |
| Different value | 6,029 (98.5%) | — |
| K80-only offsets | 4,164 | — |
| Titan V-only offsets | — | 349 |
| BOOT0 | 0x0f22d0a1 (GK210) | 0x140000a1 (GV100) |

### Per-Region Shared Analysis

| Region | K80 | Titan V | Shared | Same Value |
|--------|-----|---------|--------|------------|
| PBUS | 949 | 966 | 931 | 7 |
| PCLOCK | 64 | 99 | 48 | 14 |
| PFB | 1,024 | 68 | 68 | 0 |
| PFIFO | 2,048 | 1,999 | 1,999 | 0 |
| PGRAPH | 2,048 | 0 | 0 | 0 |
| PMC | 1,014 | 996 | 995 | 0 |
| PPCI | 101 | 122 | 93 | 56 |
| PROM | 987 | 981 | 955 | 13 |
| PTIMER | 1,024 | 1,019 | 1,019 | 0 |

---

## Key Findings

### 1. VBIOS Does the Heavy Lifting
The nvidia-535 driver only changes **255 registers** on top of VBIOS init.
The VBIOS/BIOS POST sequence programs 98%+ of the GPU's register state.
For sovereign compute, the VBIOS recipe IS the init sequence.

### 2. Titan V Falcon Registers Are Invisible Through BAR0
SEC2 and ACR regions return `0xBADF1100` (QEMU unmapped MMIO). These
falcon engines use **PRAMIN/falcon DMA windows**, not direct BAR0 MMIO.
To inspect them, we need to:
- Map the PRAMIN window (BAR0 0x700000–0x800000) to falcon DMEM/IMEM
- Or intercept the DMA at the IOMMU level

### 3. PGRAPH Is the Architecture Divide
- **K80 (Kepler)**: PGRAPH fully initialized by VBIOS (2,048 regs)
- **Titan V (Volta)**: PGRAPH entirely falcon-gated (0 regs visible)
  
This confirms: Kepler allows direct register programming of the compute
engine. Volta requires FECS/GPCCS falcon firmware authentication through
SEC2/ACR before any PGRAPH access.

### 4. mmiotrace Is Incompatible with VFIO Passthrough
In-guest `mmiotrace` captured only 12 lines. QEMU/KVM maps GPU BARs
directly into the guest physical address space, bypassing the guest
kernel's page-fault mechanism. Host-level mmiotrace would be needed for
full MMIO tracing, but conflicts with VFIO's direct-map model.

### 5. VM Boot Configuration Matters
- **Titan V requires UEFI without Secure Boot**: Secure Boot enables
  kernel lockdown (integrity mode), which blocks `mmap()` of PCI BARs.
  Non-SB OVMF (`OVMF_CODE_4M.fd`) solves this.
- **Both IOMMU group functions must be passed through**: VGA devices
  sharing a group with audio cause BIOS-level hangs if only one function
  is passed.

---

## Artifacts

```
data/k80/nvidia470-vm-captures/
  artifacts/cold_bar0.json      (VM cold BAR0, 10,291 regs)
  artifacts/warm_bar0.json      (VM warm BAR0)
  artifacts/dmesg.log
  artifacts/lspci.log
  artifacts/mmiotrace.log       (12 lines — ineffective in VFIO)
  gk210_full_bios_recipe.json   (10,283 register writes)

data/titanv/nvidia535-vm-captures/
  artifacts/cold_bar0.json      (VM cold BAR0, 15,363 regs)
  artifacts/warm_bar0.json      (VM warm BAR0, 14,437 regs)
  artifacts/dmesg.log
  artifacts/lspci.log
  artifacts/nvidia-smi.log
  gv100_full_bios_recipe.json   (6,468 register writes)
  gv100_nvidia535_driver_delta.json (255 register changes)

data/cross_gpu_comparison_report.json
```

---

## Remaining Deep Debt

### Tier 1: Sovereign Cold Boot (K80 — Ready Now)
- [ ] Build register replay engine in `coral-driver`
- [ ] Replay 10,283-register GK210 VBIOS recipe on cold K80
- [ ] Validate BOOT0, PTIMER, PCLOCK after replay
- [ ] Attempt FECS/GPCCS PIO firmware upload (Exp 123-K1)
- [ ] Dispatch first sovereign compute kernel

### Tier 2: Falcon Microcode Extraction (Titan V)
- [ ] Extract SEC2/ACR microcode binaries from nvidia-535 `.run` package
- [ ] Map PRAMIN window to read falcon DMEM/IMEM state
- [ ] Capture DMA traffic during ACR boot (host-level IOMMU interception)
- [ ] Resolve WPR copy stall (Exp 120 paradigm: DMA path or PRIV ring)

### Tier 3: Pipeline Hardening
- [ ] Automate VM capture pipeline (template-driven, one command per GPU)
- [ ] Add wide BAR0 scan (full 16MB / 32MB instead of curated regions)
- [ ] Host-level BAR0 snapshot for comparison (no VM overhead)
- [ ] Integrate register replay into `ember.init` IPC method

### Tier 4: Ember/Glowplug Evolution
- [ ] `device.lend` / `device.reclaim` → automated VM passthrough
- [ ] Register replay as ember "personality" (alongside vfio, nouveau, nvidia)
- [ ] Cold-boot recovery: ember detects dead GPU → replay → reclaim
- [ ] D-state watchdog hardening (Exp 074 follow-up)
