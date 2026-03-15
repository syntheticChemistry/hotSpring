# GPU Metal Maps

Machine-generated register maps from BAR0 cartography scans.

Each JSON file captures a complete snapshot of a GPU's register topology, power domain states, engine enumeration, and memory configuration as discovered through the GlowPlug Metal Explorer.

## File Format

```json
{
  "bdf": "0000:4a:00.0",
  "boot0": "0x14000000",
  "chip": "GV100",
  "architecture": "Volta",
  "pmc_enable": "0xffffffff",
  "bar_map": {
    "bar_index": 0,
    "size": 16777216,
    "responsive_bytes": 524288,
    "error_bytes": 16252928,
    "regions": [...]
  },
  "power_domains": {
    "GR": { "active": true, "bit": "0x00001000" },
    "PFIFO": { "active": true, "bit": "0x00000100" }
  }
}
```

## Generated Files

- `titan_v_gv100_metal_map.json` — NVIDIA Titan V (GV100, Volta)
- `mi50_vega20_metal_map.json` — AMD Radeon MI50 (Vega 20) [planned]

## How to Generate

```bash
CORALREEF_VFIO_BDF=0000:XX:00.0 cargo test --test hw_nv_vfio --features vfio \
  -- --ignored vfio_metal_cartography --nocapture
```

Results are automatically written to this directory.
