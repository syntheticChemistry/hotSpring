# Handoff: RegisterMap Trait Absorption + Sovereign Dispatch Evolution

**Date:** March 18, 2026
**From:** hotSpring (barracuda v0.6.32)
**To:** barraCuda, toadStool
**License:** AGPL-3.0-only
**Covers:** Vendor-agnostic register maps, AMD MI50 register definitions, GlowPlug register RPCs, sovereign dispatch path

---

## Executive Summary

- **`RegisterMap` trait** provides vendor-agnostic GPU register introspection. Implementations exist for NVIDIA GV100 (127 registers) and AMD GFX906/MI50 (Vega 20 registers). Runtime vendor detection via `detect_register_map(vendor_id)`.
- **`RegDef`** struct encodes offset, name, and group for each register. Enables structured JSON dumps with `RegisterDump`/`RegisterEntry` output types.
- **GlowPlug register RPCs** (`device.register_dump`, `device.register_snapshot`) expose BAR0 reads over the existing JSON-RPC socket — no root/pkexec needed for register inspection.
- **AMD MI50 swap path** is now fully functional in `coral-ember`: nouveau↔vfio and amdgpu↔vfio swaps work identically. `hbm2_training_driver()` selects the correct warm driver per vendor.
- **Absorption target**: `RegisterMap` trait, `RegDef`, vendor implementations, and `detect_register_map()` belong in barraCuda long-term. hotSpring will lean on upstream after absorption.

---

## Part 1: RegisterMap Trait (for barraCuda)

### Location

`hotSpring/barracuda/src/register_maps/`

### API

```rust
pub struct RegDef {
    pub offset: u32,
    pub name: &'static str,
    pub group: &'static str,
}

pub trait RegisterMap {
    fn vendor(&self) -> &str;
    fn arch(&self) -> &str;
    fn registers(&self) -> &[RegDef];
    fn decode_temp_c(&self, raw: u32) -> Option<u32>;
    fn decode_boot_id(&self, raw: u32) -> String;
    fn thermal_offset(&self) -> Option<u32>;
}

pub fn detect_register_map(vendor_id: u16) -> Option<Box<dyn RegisterMap>>;
```

### Implementations

| Struct | Vendor | Registers | Thermal Offset | Boot ID Decode |
|--------|--------|-----------|----------------|----------------|
| `NvGv100Map` | NVIDIA | 127 (PMC, PBUS, PFIFO, PBDMA, PFB, FBHUB, PMU, PCLOCK, GR, FECS, GPCCS, LTC, FBPA, PRAMIN, THERM) | `0x020460` | BOOT0 chipset/stepping |
| `AmdGfx906Map` | AMD | ~20 (SRBM, GRBM, MMHUB, GFX, SDMA, IH, THM, SMN, HDP) | `0xC0300E04` (CG_MULT_THERMAL_STATUS) | SRBM_STATUS mapping |

### JSON Output Types

```rust
pub struct RegisterDump {
    pub vendor: String,
    pub arch: String,
    pub bdf: String,
    pub timestamp: String,
    pub registers: Vec<RegisterEntry>,
}

pub struct RegisterEntry {
    pub offset: u32,
    pub name: String,
    pub group: String,
    pub value: u32,
    pub hex: String,
}
```

### Absorption Path

1. **Move trait + RegDef + detect_register_map** to `barraCuda/crates/barracuda/src/register_maps/`
2. **Move NvGv100Map** and **AmdGfx906Map** implementations alongside
3. hotSpring `exp070_register_dump.rs` and `exp070_register_diff.rs` should then lean on upstream
4. toadStool can use `detect_register_map()` in hw-learn to decode health snapshots

---

## Part 2: GlowPlug Register RPCs (for toadStool)

### New RPC Methods

| Method | Params | Returns |
|--------|--------|---------|
| `device.register_dump` | `{bdf, offsets?}` | `{bdf, registers: {offset: value}}` — if offsets empty, uses default 129-register set |
| `device.register_snapshot` | `{bdf}` | `{bdf, registers: {offset: value}}` — returns last pre-swap snapshot (taken automatically before every swap) |

### How toadStool Should Use These

1. **Health monitoring**: Poll `device.register_dump` with thermal offset to track GPU temperature without needing nvidia-smi
2. **Pattern learning**: Feed `device.register_snapshot` (pre-swap state) into hw-learn to correlate register patterns with swap outcomes
3. **Anomaly detection**: Compare successive dumps to detect stuck registers, thermal throttling, or VRAM degradation

### What toadStool Should NOT Do

- Do NOT open `/dev/vfio/*` directly for register reads — use the RPC
- Do NOT cache register values across driver swaps — snapshots are invalidated by personality changes
- Do NOT poll faster than 1/second — VFIO register reads are PCIe round-trips

---

## Part 3: Sovereign Dispatch Path

### Current State

The sovereign dispatch path (hotSpring → coralReef compile → VFIO dispatch) has these components:

| Component | Status | Crate |
|-----------|--------|-------|
| WGSL → SPIR-V → NAK IR → SASS | Working | coralReef (`coral-reef`) |
| VFIO device lifecycle (open, map BAR, hold) | Working | coralReef (`coral-driver`, `coral-ember`) |
| GlowPlug personality management | Working | coralReef (`coral-glowplug`) |
| Register introspection (read BAR0) | Working | coralReef (via `device.register_dump` RPC) |
| PFIFO channel init + GP_PUT DMA | **BLOCKED** | coralReef (Exp 058, GP_PUT reads as 0) |
| FECS/GPCCS firmware load | Research | hotSpring (Exp 068, FECS halts at PC=0x2835) |

### Remaining Blockers for Full Sovereign Dispatch

1. **GP_PUT DMA read**: The USERD GP_PUT register reads as 0 after DMA write. Cache coherency issue. Experiment 058 documented in PFIFO handoff.
2. **FECS firmware halt**: FECS executes from host-loaded IMEM but halts. SEC2 → ACR → FECS chain works until the graphics context handshake.
3. **GPCCS falcon address**: Different from FECS on GV100. Mapped but not yet loaded.

### What toadStool Can Do Now (Without Full Sovereign Dispatch)

- **Register health monitoring** via GlowPlug RPCs
- **Driver swap orchestration** via `device.swap` for warm/cold cycling
- **Dual-GPU management** with heterogeneous vendor pairs (NVIDIA + AMD)
- **Precision brain** integration using existing wgpu/Vulkan path while sovereign dispatch matures

---

## Part 4: Lessons Learned

### AMD vs NVIDIA in VFIO Context

| Aspect | NVIDIA GV100 | AMD MI50 |
|--------|-------------|----------|
| VFIO boot | vfio-pci required at boot (no FLR) | amdgpu can be hot-unbound |
| HBM2 init | Requires nouveau warm cycle | Requires amdgpu warm cycle |
| DRM isolation needed | Yes (nouveau creates DRM nodes) | Yes (amdgpu creates DRM nodes) |
| Register readback | BAR0 mmap works immediately | BAR0 mmap works but different offsets |
| Power management | pin_power prevents D3hot | pin_power works identically |
| IOMMU groups | Often shares with HDA audio | Often solo group on server boards |

### Privilege Model Evolution

```
v1: sudo everything (dangerous)
v2: pkexec for register reads (interrupts workflow)
v3: systemd capabilities (CAP_SYS_ADMIN + CAP_SYS_RAWIO)
v4: capabilities + seccomp + namespace isolation (current)
```

Each step removes attack surface without losing functionality. The `coralctl deploy-udev` tool eliminates the last manual privilege setup step.

---

## License

AGPL-3.0-only. Part of the ecoPrimals sovereign compute fossil record.
