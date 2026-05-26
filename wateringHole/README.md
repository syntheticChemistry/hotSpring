# wateringHole — PostPrimordial Redirect

**As of Exp 224 (2026-05-26), hotSpring has transitioned to postPrimordial deployment.**

All handoffs have been archived to the ecosystem-level wateringHole:

```
ecoPrimals/infra/wateringHole/handoffs/hotSpring/         # 6 active handoffs
ecoPrimals/infra/wateringHole/handoffs/hotSpring/archive/  # 38 archived handoffs
```

Future handoffs go to `infra/wateringHole/` only. This local directory retains
only lab artifacts (firmware extracts, VBIOS data, mmiotraces) that are too
large or hardware-specific for the centralized hub.

## Remaining Lab Artifacts

| Directory/File | Purpose |
|----------------|---------|
| `mmiotraces/` | Raw GPU MMIO trace captures (gitignored) |
| `gk110/` | GK110/GK210 register reference data (gitignored) |
| `vbios/` | VBIOS binary data for sovereign boot analysis |
| `titanv_*.bin` | Binary firmware extracts from mmiotrace (gitignored) |

## Ecosystem wateringHole

The authoritative hub for all primal taxonomy, composition patterns, NUCLEUS
definitions, cross-primal contracts, and handoffs:

```
ecoPrimals/infra/wateringHole/
```
