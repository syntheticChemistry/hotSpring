# Spec: MILC Validation Loop

**Binary**: `milc_validation_loop`  
**Module**: `src/lattice/milc.rs` (implemented)  
**Target reviewer**: TC Chuna (LANL), Alexei Bazavov (MILC collaboration)

---

## Purpose

Prove bidirectional gauge configuration interoperability with the MILC code.
A MILC reviewer loading our config and getting the same plaquette is the
strongest possible external validation.

---

## Interface

```bash
# Export: hotSpring config → MILC v5 binary
cargo run --release --bin milc_validation_loop -- export \
  --beta 6.0 --volume 16 --trajectories 200 \
  --output /tmp/hotspring_b6.0_16x16x16x16.milc

# Import: MILC config → hotSpring measurement
cargo run --release --bin milc_validation_loop -- import \
  --input /path/to/milc_config.milc \
  --measure plaquette,polyakov,wilson

# Round-trip: export → read back → compare
cargo run --release --bin milc_validation_loop -- roundtrip \
  --beta 6.0 --volume 16 --trajectories 200
```

---

## Export Specification

1. Thermalize SU(3) at given β/V using CPU HMC (cpu_mom path)
2. After thermalization, take final configuration
3. Write via `milc::write_milc_config()`:
   - Header: magic number, dims, β, plaquette, timestamp
   - Body: 4 × V matrices in MILC natural order (x-slowest, t-fastest)
   - Each SU(3) matrix: 9 complex numbers as 18 big-endian f32
   - CRC32 checksums (sum29, sum31)
4. Compute and report: ⟨P⟩ from final config
5. Output provenance: BLAKE3 hash of output file

---

## Import Specification

1. Read MILC v5 binary via `milc::read_milc_config()`
2. Verify CRC32 checksums match header
3. Convert to internal `Lattice` representation
4. Compute observables:
   - Wilson plaquette ⟨P⟩
   - Polyakov loop |⟨L⟩|
   - Optional: Wilson loops W(R,T)
5. Compare ⟨P⟩ against header value
6. Report: agreement metric (must be < 10⁻⁶ relative)

---

## Round-trip Specification

1. Generate config (cold start + HMC thermalization)
2. Export to MILC format
3. Read back from MILC format
4. Compare all 4×V×18 floats (accounting for f32 truncation)
5. Re-compute plaquette on imported config
6. Report: max element-wise deviation, plaquette agreement

**Pass criteria**: 
- Element-wise deviation ≤ f32 machine epsilon (≈6×10⁻⁸)
- Plaquette agreement ≤ 10⁻⁶ (limited by f32 storage)

---

## MILC Format Reference

```
MILC v5 binary gauge configuration:
├── Header (ASCII lines, newline-terminated)
│   ├── BEGIN_HEADER
│   ├── HDR_VERSION = 2.0
│   ├── DATATYPE = 4D_SU3_GAUGE
│   ├── DIMENSION_1..4 = nx, ny, nz, nt
│   ├── LINK_TRACE = <average trace / 3>
│   ├── PLAQUETTE = <average plaquette>
│   ├── CHECKSUM = <hex sum29> <hex sum31>
│   ├── ...
│   └── END_HEADER
├── Body (binary, big-endian)
│   └── For each site in MILC natural order (t fastest, x slowest):
│       └── For each direction μ = 0,1,2,3:
│           └── 18 × f32 (re,im for each of 9 matrix elements, row-major)
└── EOF
```

**Site ordering**: MILC natural = x·(ny·nz·nt) + y·(nz·nt) + z·nt + t

Our internal ordering: t·(nx·ny·nz) + x·(ny·nz) + y·nz + z

The conversion is handled in `milc_natural_to_coords()`.

---

## Validation Protocol (for Chuna)

```
1. We generate: hotspring_b6.0_16x16x16x16.milc
2. We report: ⟨P⟩ = X.XXXXXXXX (our measurement)
3. Chuna runs: su3_gauge_read + plaquette measurement in MILC
4. Chuna reports: ⟨P⟩ = Y.YYYYYYYY (MILC measurement)
5. Compare: |X - Y| / X < 10⁻⁶

If match: interoperability proven.
If mismatch: diagnose (byte order? site order? normalization?)
```

---

## Files

- `src/lattice/milc.rs` — Core reader/writer (implemented)
- `src/bin/milc_validation_loop.rs` — Binary (to be written)
- `specs/MILC_VALIDATION_LOOP.md` — This document
