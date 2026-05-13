# Experiment 156: Reagent Trace Comparison

**Status:** Active  
**Date:** 2026-04-07  
**Targets:** Titan V (GV100), Tesla K80 (GK210)  
**Dependencies:** agentReagents captures, exp070 sovereign dumps, exp154/155 pipeline output

## Background

The sovereign compute pipeline (exp 144/145/154/155) drives GPU initialization
through ember IPC. To identify what initialization steps we're missing (causing
SEC2 HS failure on Titan V, untested FECS on K80), we compare our register state
against traces from actual drivers.

agentReagents provides VM-isolated captures: nouveau or proprietary NVIDIA drivers
run inside PCI-passthrough VMs, producing mmiotrace and BAR0 snapshots without
contaminating the host RTX 5070 display stack.

## Approach

1. **Sovereign baseline:** exp070 register dump after our pipeline (warm cycle +
   ember MMIO operations)
2. **Driver baseline:** agentReagents capture (nouveau or nvidia-470) of the same
   GPU after full driver initialization
3. **Diff:** Register-by-register comparison, grouped by hardware domain

## Reagent Templates

| Template | GPU | Driver | Captures |
|----------|-----|--------|----------|
| `reagent-nouveau-titanv.yaml` | Titan V | In-tree nouveau | mmiotrace, BAR0 snapshot |
| `reagent-nvidia470-titanv.yaml` | Titan V | NVIDIA 470.x | mmiotrace, BAR0 snapshot |
| `reagent-nouveau-k80.yaml` | K80 | In-tree nouveau | mmiotrace, BAR0 snapshot |
| `reagent-nvidia470-k80.yaml` | K80 | NVIDIA 470.x | mmiotrace, BAR0 snapshot |

## Input Formats

### exp070 JSON (sovereign dump)
```json
{
  "vendor": "nvidia",
  "arch": "GV100",
  "registers": [
    { "offset": "0x000000", "name": "BOOT0", "group": "PMC", "value": "0x12000291", "raw_offset": 0, "raw_value": 302055057 }
  ]
}
```

### Kernel mmiotrace (driver capture)
```
W 4 0x00000200 0x00000003
R 4 0x00000000 0x12000291
W 4 0x00001C00 0x00000001
```

## Domain Classification

| Offset Range | Domain | Relevance |
|-------------|--------|-----------|
| 0x000000–0x000FFF | PMC | Power management, engine enable |
| 0x10A000–0x10BFFF | PMU | PMU falcon (hypothesis A) |
| 0x300000–0x3FFFFF | BROM | Boot ROM registers (hypothesis C) |
| 0x400000–0x408FFF | PGRAPH | Graphics pipeline |
| 0x409000–0x409FFF | FECS | FECS falcon (K80 primary target) |
| 0x700000–0x7FFFFF | PRAMIN | VRAM access window |
| 0x840000–0x84FFFF | SEC2 | SEC2 falcon (Titan V ACR) |

## Expected Output

Per-domain divergence report:
- **Matches:** Registers where sovereign = driver (confirms our pipeline covers this)
- **Divergences:** Where driver writes a different value (potential missing init step)
- **Reagent-only:** Registers the driver touches that we never read/write (blind spots)

High-value targets are divergences in PMU, BROM, and SEC2 domains for Titan V,
and FECS/PGRAPH for K80.

## Binary

```text
cargo run --release --bin exp156_reagent_compare -- \
  --sovereign data/070/titan_v_warm.json \
  --reagent data/reagent_captures/nouveau_titanv.json \
  --output data/156/diff_report.json
```

## Files

| File | Role |
|------|------|
| `barracuda/src/bin/exp156_reagent_compare.rs` | Comparison binary |
| `barracuda/src/register_maps/nv_gv100.rs` | GV100 register labels |
| `infra/agentReagents/templates/reagent-nouveau-titanv.yaml` | Nouveau capture template |
| `infra/agentReagents/templates/reagent-nvidia470-titanv.yaml` | NVIDIA 470 capture template |

## Results

*To be filled after running captures and comparisons.*
