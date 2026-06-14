# Pattern: Driver-as-Software

**Status:** Active — foundational pattern for GPU cracking pipeline
**Origin:** hotSpring Exp 070, 082; user feedback on pkexec elimination
**Absorbers:** coralReef (GlowPlug/Ember), toadStool (hw-learn), ecoPrimals standards
**Extended by:** [UNIVERSAL_DRIVER_REAGENT_ARCHITECTURE.md](UNIVERSAL_DRIVER_REAGENT_ARCHITECTURE.md) (open targets, reagent safety, trace-as-default)

## Core Principle

A kernel driver is not infrastructure — it is a **software tool** that the GPU lifecycle
manager invokes on demand, captures output from, and returns the GPU to its managed state.

Every driver binding is a **learning opportunity**. The system captures the full MMIO
register write sequence during init, which becomes a permanent **recipe** in the driver
library. Over time, the system accumulates recipes from every driver version and backend,
building a corpus that the ACR boot solver uses to synthesize sovereign init sequences.

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │         Driver Library                │
                    │  ┌──────┐ ┌──────┐ ┌──────────────┐  │
                    │  │nouv. │ │nvidia│ │nvidia_oracle_*│  │
                    │  │recipe│ │recipe│ │  v525..v580   │  │
                    │  └──────┘ └──────┘ └──────────────┘  │
                    └────────────────┬─────────────────────┘
                                     │ recipes feed solver
                    ┌────────────────▼─────────────────────┐
                    │       ACR Boot Solver                 │
                    │  (synthesizes sovereign init from     │
                    │   observed MMIO write patterns)       │
                    └────────────────┬─────────────────────┘
                                     │ produces sovereign boot
                    ┌────────────────▼─────────────────────┐
                    │       Sovereign Compute               │
                    │  (VFIO + own init = DRM-free GPU)     │
                    └──────────────────────────────────────┘
```

## The Swap-Capture-Return Cycle

```
       VFIO (managed)
           │
    ┌──────▼──────┐
    │ 1. UNBIND   │  GlowPlug/Ember unbinds from vfio-pci
    │    from VFIO │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ 2. ENABLE   │  Ember enables mmiotrace (debugfs)
    │    TRACE    │  (only when --trace requested)
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ 3. BIND     │  Ember binds to target driver (nouveau, nvidia, nvidia_oracle)
    │    DRIVER   │  The driver initializes the GPU — every MMIO write is captured
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ 4. SETTLE   │  Wait for full init (10-12s nouveau, 8s nvidia)
    │    + CAPTURE │  Capture warm BAR0 snapshot, stop trace, save filtered outputs
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ 5. UNBIND   │  Unbind from driver
    │    DRIVER   │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ 6. RETURN   │  Return to vfio-pci, capture residual BAR0
    │    TO VFIO  │  GPU back under GlowPlug management
    └──────┴──────┘
```

## Capture Outputs

Each swap-capture cycle produces:

| Artifact | Description | Use |
|----------|-------------|-----|
| `mmiotrace_raw.txt` | Full kernel mmiotrace output | Archive, deep analysis |
| `mmiotrace_writes.txt` | Write operations only | Init sequence analysis |
| `mmiotrace_falcon_init.txt` | Writes to SEC2/FECS/GPCCS/PMU ranges | ACR boot solver input |
| `mmiotrace_acr_dma.txt` | SEC2 + instance block + bind registers | DMA context analysis |
| `bar0_warm.bin` | BAR0 snapshot while driver is active | Register state reference |
| `bar0_residual.bin` | BAR0 snapshot after driver unbind | Teardown behavior |
| `manifest.json` | Metadata (driver, BDF, timing, kernel, counts) | Indexing, comparison |

## Driver Coexistence: nvidia_oracle

The protected display GPU (RTX 5070) owns `nvidia.ko`. To bind a Titan to an nvidia
driver without conflict, we build a **renamed module**:

- `MODULE_BASE_NAME` patched from `"nvidia"` to `"nvidia_oracle"`
- `NV_MAJOR_DEVICE_NUMBER` patched from `195` to `0` (dynamic allocation)
- Loads as separate kernel module, no conflict with system nvidia

This enables **version-indexed recipes**: `nvidia_oracle_525.ko`, `nvidia_oracle_535.ko`,
`nvidia_oracle_580.ko`. Each version's init sequence may differ in ACR firmware layout,
DMA addressing, and falcon boot vectors. Collecting all versions builds a robust recipe
library.

Build: `hotSpring/scripts/build_nvidia_oracle.sh [VERSION]`

## Cross-Driver Comparison

The comparison tool (`coralctl snapshot diff <BDF>`, formerly `scripts/archive/compare_snapshots.py`)
diffs any two register snapshots and automatically flags ACR/falcon-relevant changes. This
enables systematic analysis:

| Comparison | Reveals |
|-----------|---------|
| cold → nouveau-warm | What nouveau's init does (our warm-up path) |
| cold → nvidia-warm | What nvidia's init does (and destroys on teardown) |
| nouveau-warm → nvidia-warm | Behavioral difference between drivers |
| Titan #1 → Titan #2 (same driver) | Hardware-specific vs universal patterns |
| nvidia_v525 → nvidia_v580 | ACR firmware evolution across versions |

### Key Finding: Nouveau vs Nvidia Falcon State

| Register | After nouveau | After nvidia |
|----------|--------------|--------------|
| PMU_CPUCTL | `0x00000010` (HALTED, accessible) | `0x00000020` (different state) |
| PMU_BOOTVEC | `0x00010000` (set) | `0x00000000` (cleared) |
| FECS_CPUCTL | `0x00000010` (HALTED, accessible) | `0xbadf1201` (powered off) |
| FECS_BOOTVEC | `0x00000000` | `0xbadf1201` (BAR0 PRI timeout) |

**Conclusion:** nvidia's teardown powers off FECS entirely, returning registers to PRI
timeout state. Nouveau leaves FECS and PMU **halted but register-accessible**, which is
the only viable warm-up path for subsequent VFIO boot attempts.

## Recipe Distillation

Once captured, traces flow through the **distillation pipeline**:

```
mmiotrace → hw-learn distiller (toadStool) → oracle_recipe.json → ACR boot solver
```

The distiller extracts:
- Ordered register write sequences per functional block
- Timing relationships between writes
- Polling patterns (read-until-value loops)
- DMA address setup sequences

The ACR boot solver then uses these recipes to construct its own init sequence,
substituting VFIO DMA addresses for the driver's DMA addresses.

## Lifecycle Ownership

| Component | Owner | Role |
|-----------|-------|------|
| Capture scripts + comparison tool | hotSpring | Experiment tooling |
| GlowPlug/Ember trace integration | coralReef | Production implementation |
| hw-learn distiller | toadStool | Recipe extraction |
| ACR boot solver | coralReef | Sovereign init synthesis |
| Pattern docs + gap analysis | hotSpring | Architecture evolution |

## Pattern Evolution

This pattern is **not NVIDIA-specific**. The same swap-capture-return cycle applies to
any driver backend:

- AMD: `amdgpu` init traces for GCN5 sovereign compute
- Intel: `xe`/`i915` for future Intel discrete GPU support
- Custom: any kernel module that initializes GPU hardware
- **Reagent drivers:** even nonsensical combinations (e.g. `amdgpu` on NVIDIA hardware)
  are valid experiments — the failure trace itself is informative

The driver library grows with each new hardware generation and driver version. The ACR
boot solver evolves from hardware-specific to **architecture-general** as the recipe
corpus expands.

## Universal Driver Reagent Extension

The [Universal Driver Reagent Architecture](UNIVERSAL_DRIVER_REAGENT_ARCHITECTURE.md)
extends this pattern with:

- **Open target acceptance:** No allowlist — any driver string on any managed GPU
- **Reagent safety taxonomy:** Protected / Managed / Shared / Native-Compute roles
- **Trace-as-default:** Every swap traces by default (`--no-trace` to opt out)
- **Generic personality:** `Personality::Custom` handles unknown drivers
- **Native-compute roadmap:** Borrow compute from gaming GPUs without swap
  (see [NATIVE_COMPUTE_ROADMAP.md](NATIVE_COMPUTE_ROADMAP.md))

## ecoPrimals Absorption Path

1. **coralReef** absorbs the trace integration spec (handoff delivered)
2. **coralReef** absorbs the open target + reagent model (handoff delivered)
3. **toadStool** absorbs the distillation + recipe format
4. **ecoPrimals standards** formalize the swap-capture-return cycle as a reusable pattern
5. **barraCuda** benefits from sovereign compute path (DRM-free GPU dispatch)
6. **barraCuda** integrates with native-compute mode for gaming GPU borrowing (late-stage)

## Update (April 2026): SovereignInit + Staged Fork Isolation

The driver-as-software pattern evolved into the **SovereignInit pipeline** (Exp 165):
an 8-stage pure Rust replacement for nouveau's initialization subsystem. Firmware blobs
are treated as ingredients (loaded by Rust, executed by GPU hardware). Each stage runs
inside ember's fork-isolated child process. The **nouveau DRM dispatch path is fully
proven** (Exp 164: 5/5 E2E phases pass on Titan V). The VFIO sovereign path has stages
0-5 hardware-validated; falcon boot (stage 6) is blocked by memory controller sleep
after nouveau teardown. See `whitePaper/baseCamp/sovereign_gpu_compute.md` Phase 21.
