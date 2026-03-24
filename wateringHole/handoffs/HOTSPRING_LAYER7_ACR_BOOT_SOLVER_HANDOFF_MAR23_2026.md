# hotSpring → coralReef / toadStool / barraCuda: Layer 7 ACR Boot Solver + Compute Trio Evolution

**Date:** March 23, 2026
**From:** hotSpring (Exp 078-081 — Layer 7 ACR Boot Solver)
**To:** coralReef team, toadStool team, barraCuda team

---

## Context

hotSpring has been driving the sovereign GPU pipeline from 6/10 to 7/10 layers
(Exp 076-077) and is now assaulting Layer 8 — the FECS/GPCCS firmware context
that controls the GR (graphics/compute) engine on GV100. This is the last
compute-blocking layer before shader dispatch.

Four experiments (078-081) produced critical architecture, discoveries, and code
that each primal should absorb according to its domain.

---

## For coralReef Team

### Immediate: Complexity Debt (5 Files >1000 LOC)

| File | LOC | Recommended Split |
|------|-----|-------------------|
| `acr_boot.rs` | 4462 | `firmware.rs` (parsing), `wpr.rs` (WPR construction), `instance_block.rs` (MMU/page tables), keep orchestrator thin |
| `coralctl.rs` | 1649 | Extract subcommand handlers into `cli/` modules |
| `socket.rs` | 1434 | Extract protocol encoding/decoding from socket I/O |
| `mmu_oracle.rs` | 1131 | Extract `oracle_diff.rs` + `oracle_capture.rs` |
| `device.rs` | 1030 | Extract DMA mapping logic |

### Architecture Built (Iteration 63)

- **Falcon Boot Solver** (`acr_boot.rs`): Multi-strategy SEC2→ACR→FECS chain
  - `FalconProbe` / `Sec2Probe` — captures full falcon state
  - `AcrFirmwareSet` — loads all firmware files for boot chain
  - `NvFwBinHeader` / `HsBlDescriptor` — NVIDIA firmware header parsing
  - `FalconBootSolver::boot()` — tries 5 strategies in cost order
  - `sec2_emem_write/read/verify` — SEC2 EMEM PIO interface
- **Falcon Diagnostics** (`diagnostics.rs`): State capture, HWCFG decode, exception info
- **FECS Boot** (`fecs_boot.rs`): Direct firmware upload, ACR-bypass path
- **Register Infrastructure**: SEC2 base corrected, CPUCTL v4+ bits, SCTL/EXCI/TRACEPC

### Key Technical Findings for coralReef

1. **`ctx_dma = VIRT(4)`** not `PHYS_SYS(6)` — Nouveau uses `FALCON_DMAIDX_VIRT` for BL descriptor. This fix advanced the HS ROM PC from 0x14b9 to 0x1505
2. **Full PMC disable+enable** required before SEC2 boot — clearing ITFEN, all interrupts, then PMC disable→falcon reset→PMC enable→memory scrub→BOOT0
3. **Instance block bind_stat** at `0x0dc` bits [14:12] — must reach value 5 (bind complete). This is the current blocker
4. **ACR production signatures** differ from debug — patch `sig_prod_offset`/`sig_prod_count` from HS header
5. **nvfw_bin_hdr** format: magic `0x10DE`, sub-header at `header_offset`, payload at `data_offset`

### Current Blocker

The SEC2 HS ROM is executing (PC is advancing) but the instance block binding
does not complete. Three hypotheses:
1. IOMMU mapping alignment for the instance block DMA buffer
2. V2 page table encoding — aperture bits in PDE/PTE
3. Race condition between bind_inst write and polling

### What to Continue

- Resolve `bind_stat` timeout
- Once SEC2 boots BL: construct WPR with FECS/GPCCS LS images
- Warm handoff (Exp 079 path): capture Nouveau's FECS state via Ember swap
- Split `acr_boot.rs` after strategies stabilize

---

## For toadStool Team

### GPU State Awareness

The falcon diagnostic infrastructure (Exp 078) provides real-time visibility into
GPU microcontroller state. toadStool's hardware learning system (`hw-learn`) can
absorb this:

- **FalconProbe** provides FECS/GPCCS/PMU/SEC2 state in a structured format
- **Sec2State enum**: `HsLocked | CleanReset | Running | Inaccessible`
- **HWCFG decoding**: IMEM/DMEM sizes, security mode per falcon

This data can feed the `observer → distiller → knowledge_store → applicator`
pipeline for GPU initialization state tracking.

### Compute Dispatch Readiness

When Layer 8 resolves, toadStool's dispatch infrastructure will need to handle:
- `ComputeDevice::dispatch()` with a fully initialized GR engine
- Sovereign VFIO path with DMA-mapped shader binary + constant buffers
- The same QMD v2.1 (Volta) descriptors already prototyped

### Session Absorption

toadStool sessions 155+ should track the ACR boot progress. The firmware loading
and DMA patterns discovered here inform how the hardware learning system models
GPU initialization sequences.

---

## For barraCuda Team

### Shader Readiness

barraCuda's 806+ WGSL shaders are ready to dispatch the moment Layer 8 resolves.
No changes needed on the shader side. The precision tiers (f32/df64/f64) are
stable and hardware-validated.

### Potential ACR/Microcode Application

The ACR boot solver probes and manipulates Falcon microcontrollers using DMA
and PIO. barraCuda's WGSL→native ISA compilation pipeline could theoretically
target Falcon ISA for microcode generation. This is speculative but the
architecture exists:

1. Falcon ISA is documented (v4+ instruction set)
2. coralReef's encoder infrastructure is vendor-parameterized
3. A `ShaderModelFalcon` implementation could generate microcode from WGSL

This would be a future evolution path, not a current priority. The immediate
path is using NVIDIA's signed firmware through the ACR chain.

---

## Sovereign Pipeline Status

```
Layer 1-6:  DONE  (PCIe, PFB, PFIFO, scheduler, channel, PBDMA)
Layer 7:    DONE  (MMU page table translation — Exp 076)
Layer 8:    ACTIVE (GR/FECS context — ACR boot solver, Exp 081)
Layer 9:    PENDING (FECS/GPCCS firmware — depends on Layer 8)
Layer 10:   PENDING (shader dispatch — AMD DRM 6/6 PASS, NVIDIA sovereign pending)
```

The compute trio (coralReef WHAT, toadStool WHERE, barraCuda HOW) is architecturally
complete and hardware-validated on AMD. NVIDIA sovereign dispatch awaits Layer 8
resolution. The ACR boot solver is producing increasingly detailed data with each
iteration — the HS ROM PC is advancing, which means we are getting closer to
the bootloader handoff that will unlock FECS.
