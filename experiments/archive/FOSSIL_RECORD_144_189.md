# Fossil Record: Experiments 144–189

**Archived:** May 13, 2026
**Era:** coralReef → toadStool migration, sovereign GPU compute evolution
**Active experiments:** 190 (Three-GPU Sovereign Validation), 191 (toadStool S258 PBDMA Validation)

---

## Summary

Experiments 144–189 document the journey from initial GPU register probing through
to proven sovereign compute dispatch across the Titan V, Tesla K80, and RTX 5060.
This era established the warm-catch pattern, proved FECS/PBDMA pipelines, and
motivated the absorption of `coral-ember`/`coral-glowplug`/`coral-driver` into
`toadStool`.

## Timeline & Milestones

### Phase 1: ACR/SEC2/Firmware Exploration (Exp 144–163)

| Exp | Title | Significance |
|-----|-------|-------------|
| 144 | PMC Bit 5 ACR Progress | Discovered SEC2 ACR boot pathway, BL doesn't execute |
| 150 | Crash Vector Hunt | Identified fault vectors in GPU init sequences |
| 151 | Revalidation & Next Stages | Comprehensive revalidation of prior findings |
| 152 | Compute Dispatch Provenance | Validated provenance chain for dispatch pipeline |
| 153 | Ember Flood Resurrection | Proved ember can resurrect after crash floods |
| 154 | SEC2 ACR PMU First Pipeline | First SEC2→ACR→PMU pipeline attempt |
| 155 | K80 Warm FECS Dispatch | Initial K80 FECS dispatch exploration |
| 156 | Reagent Trace Comparison | Cross-compared agent reagent traces |
| 157 | K80 DEVINIT Replay | Replayed K80 DEVINIT initialization sequence |
| 158 | SEC2 Real Firmware | Loaded actual SEC2 firmware (vs synthetic) |
| 159 | Titan V VM POST HBM2 | VM-based Titan V POST with HBM2 training |
| 160 | Titan V MMIOTRACE Capture | Captured nvidia driver init via mmiotrace |
| 161 | Titan V NVDEC Sovereign Attempt | Explored NVDEC as compute entry point |
| 162 | Titan V Sovereign Compute Pipeline | Full sovereign compute pipeline on GV100 |
| 163 | Firmware Boundary | Mapped firmware/hardware responsibility boundary |

### Phase 2: Sovereign Dispatch Proven (Exp 164–170)

| Exp | Title | Significance |
|-----|-------|-------------|
| **164** | **Sovereign Compute Dispatch PROVEN** | **First proof of sovereign GPU compute dispatch** |
| 165 | Sovereign Init Pipeline | End-to-end init pipeline validated |
| 166 | Sovereign Boot Wiring | Wired boot sequence into systemd |
| **167** | **Warm Handoff** | **Nouveau → VFIO warm handoff pattern established** |
| 168 | Sovereign Pipeline Complete | Full pipeline operational |
| **169** | **Warm Handoff Validated** | **Warm handoff cycle proven on hardware** |
| 170 | Sovereign Boot E2E | End-to-end sovereign boot demonstrated |

### Phase 3: Multi-GPU Expansion (Exp 171–180)

| Exp | Title | Significance |
|-----|-------|-------------|
| 171 | K80 Sovereign Init | First sovereign init on Kepler architecture |
| 172 | No-ACR Warm Handoff | Proved warm handoff works without ACR |
| 173 | VM Reagent WPR Capture | Captured WPR state via VM reagent |
| 174 | K80 Sovereign Boot | Full sovereign boot on Tesla K80 |
| **175** | **RTX 5060 Shared Compute** | **Proved wgpu/Vulkan sovereign dispatch on Blackwell** |
| **176** | **QCD Parity Benchmark** | **lattice QCD benchmark across GPU architectures** |
| 177 | Blackwell Dispatch ABI Fixes | Fixed ABI issues for SM120 dispatch |
| 178 | K80 PGOB nvidia470 Analysis | Analyzed K80 power-gating behavior |
| **179** | **K80 Warm FECS Dispatch Pipeline** | **FECS boot + PFIFO runlist operational on K80** |
| **180** | **Three-GPU Hardware Validation** | **First three-GPU validation sweep** |

### Phase 4: Pipeline Hardening & Warm-Catch (Exp 181–189)

| Exp | Title | Significance |
|-----|-------|-------------|
| 181 | Sovereign Dispatch Pipeline Sweep | Cross-GPU dispatch pipeline validation |
| 185 | K80 Nouveau GK210 Chipset Analysis | Deep chipset analysis for Kepler sovereignty |
| 186 | PMU Firmware Extraction Analysis | Analyzed PMU firmware extraction paths |
| 187 | Titan V nvidia580 MMIOTRACE Prep | Prepared mmiotrace capture with driver 580 |
| **188** | **K80 Warm-Catch Breakthrough** | **Proved warm-catch cycle on K80 hardware** |

## Key Architectural Discoveries

1. **Warm-Catch Pattern** (Exp 167–169, 188): Load patched nouveau (NOP'd teardown) → GPU initializes FECS/GDDR5 → unbind nouveau → rebind vfio-pci → probe warm state. This avoids the need for sovereign cold boot firmware.

2. **FECS HS-Lock Issue** (Exp 188, carried to Exp 191): After nouveau unbind on Titan V, FECS transitions to HS-locked mode (CPUCTL bit 4=1, bit 5=0) instead of halted-warm. Root cause under investigation.

3. **K80 PLX D3cold Cascade** (Exp 178, 188): Nouveau's power management triggers K80 CLKREQ# deassert → PLX PEX 8747 removes PCIe clock → entire PLX subtree dies. Requires full AC power cycle to recover.

4. **Sovereign Dispatch Proof** (Exp 164): First demonstration that GPU compute can be dispatched entirely through ecoPrimal Rust code without vendor driver involvement.

5. **Three-GPU Fleet** (Exp 180, 190): Titan V (GV100, oracle/compute), RTX 5060 (GB206, display+compute via wgpu), Tesla K80 (GK210B×2, sovereign compute targets).

## Register Maps & Firmware

Key register discoveries documented across these experiments:
- PMC_ENABLE (0x200): GPU engine enable bitmap
- FECS CPUCTL (0x409400): Falcon execution control
- PBDMA channels (0x040000+): Push buffer DMA engine state
- GPC/TPC config (0x418880): Graphics processing cluster topology
- PRAMIN (0x700000): Instance memory access window

## Migration Context

These experiments were conducted during the coralReef era. The daemon
infrastructure (`coral-ember`, `coral-glowplug`, `coral-driver`) has since
been absorbed into `toadStool` (Phases A–D, Sprints S244–S258). Modern
experiments (190+) reference `toadStool` IPC surfaces exclusively.

---

*This fossil record preserves the experimental lineage that led to sovereign
GPU compute on the biomeGate hardware. The active frontier continues in
experiments 190+ using the modern toadStool stack.*
