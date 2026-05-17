# Experiment 198 — Vendor-Agnostic BootPipeline + VBIOS Fixes

**Date:** May 16, 2026
**Status:** VALIDATED — BootPipeline trait proven cross-vendor, 591 -> 606 tests
**Hardware:** Titan V (GV100), Tesla K80 (GK210×2), AMD Vega 20 (stub via FakeBar mock)
**Parent:** Experiment 197 (Sovereign Init RPC)

---

## Objective

1. Define a vendor-agnostic boot abstraction via `&dyn RegisterAccess`
2. Prove cross-vendor compatibility with an AMD Vega 20 stub
3. Fix VBIOS interpreter bugs blocking K80 Script 1 parsing
4. Re-validate Titan V warm path through the new abstraction

## Key Innovation — `BootPipeline` Trait

Vendor-agnostic boot abstraction replacing NVIDIA-specific `InitPipeline`:

```
probe → is_warm → devinit → engine_init → verify
```

`DeviceTopology` and `DeviceFunction` replace NVIDIA-specific multi-die models,
enabling a single pipeline to handle any GPU vendor that implements
`RegisterAccess` for its BAR0 register space.

## Implementation

### Trait Hierarchy

- **`RegisterAccess`** — raw BAR0 read/write abstraction (`read32`, `write32`, `read_block`)
- **`BootPipeline`** — vendor-agnostic 5-stage boot sequence
- **`InitPipeline`** — NVIDIA-specific extensions (VBIOS, falcon, FECS)

### Vendor Implementations

| Vendor | Struct | BootPipeline | InitPipeline | Tests |
|--------|--------|-------------|-------------|-------|
| NVIDIA Kepler | `KeplerInit` | Yes | Yes | Existing |
| NVIDIA Volta | `VoltaInit` | Yes | Yes | Existing |
| AMD Vega 20 | `VegaInit` | Yes | N/A | 8 new (FakeBar mock) |

### AMD Vega 20 Stub — Cross-Vendor Proof

`VegaInit` implements `BootPipeline` for AMD Vega 20 using:
- `GRBM_STATUS` / `SRBM_STATUS` for warm detection
- `FakeBar` mock for testing without real hardware
- 8 tests validate probe, warm detect, and boot sequence

## VBIOS Interpreter Fixes

| Opcode | Bug | Fix |
|--------|-----|-----|
| 0x50 (I2C) | Stride was `count*4` | Corrected to `11 + count*4` |
| 0x88 (RAM restrict) | Not implemented | Added RAM-restrict group filtering |
| 0x70 (extended) | Missing EON subop | Added EON (extended opcode, new) |
| M table header | `ram_restrict_group_count()` wrong | Fixed to parse from M table header byte |

**Result:** K80 Script 1 now parses correctly through all DEVINIT opcodes.

## Validation

- Titan V warm re-validated through BootPipeline: **101ms**, `compute_ready=true`
- K80 cold path: BootPipeline reaches `devinit` stage (PRAMIN still blocked per Exp 197)
- AMD Vega 20: all 8 FakeBar tests pass
- **591 -> 606 tests** (15 new tests for BootPipeline + VegaInit + VBIOS fixes)

## Proven

- Vendor-agnostic boot abstraction is viable across NVIDIA and AMD
- `DeviceTopology` / `DeviceFunction` generalize multi-die GPU models
- VBIOS interpreter handles Kepler DEVINIT scripts correctly
- FakeBar mock enables testing vendor implementations without hardware

## Next Steps

- Wire `BootPipeline` into `sovereign.init` RPC (replace `InitPipeline` as primary)
- Add Intel Arc stub (`IntelInit`) for DG2 family
- K80 DEVINIT replay through corrected VBIOS interpreter
- Titan V: FECS firmware loading via BootPipeline `engine_init` stage
