# Fossil Record: strandGate Sovereign Era — Experiments 191–234

**Fossilized:** 2026-08-02
**Origin gate:** strandGate (dual Titan V GV100 at 0000:02:00.0 + 0000:49:00.0)
**Reason:** biomeGate revalidation begins with no assumptions. All strandGate
results are reference, not truth. Different BDFs, different PLX topology,
different driver stack. We retest.

## Why This Is Fossil Record

strandGate ran 44 active experiments (191–234) on a specific hardware
configuration: two Titan V cards at BDFs 02:00.0 and 49:00.0, one K80
pair through PLX bridges, RTX 3090 and RX 6950 XT for silicon routing.
Those results are the product of that specific silicon in that specific
PCIe topology with that specific kernel.

biomeGate has:
- Titan V at 0000:21:00.0 (different BDF, different bridge topology)
- K80 at 0000:4b:00.0 + 0000:4c:00.0 (different PLX path)
- RTX 5060 at 0000:02:00.0 (SM100 — generation not present on strandGate)
- Different kernel (7.0.0-28-generic)
- Different nvidia driver stack (580-series)

Sovereign experiments touch BAR0 registers at specific offsets that may
behave differently on different silicon instances (fused-out GPC, variant
TPC counts, different PRAMIN layouts). We assume nothing.

## Experiments Fossilized

### Proven Infrastructure (carry forward as reference)

| # | Title | Key Finding | Revalidation Status |
|---|-------|-------------|---------------------|
| 191 | toadStool PBDMA validation | Compute trio pipeline | Must re-run on biomeGate |
| 191B | Sovereign dispatch validated | First e2e VFIO dispatch | Must re-run at 21:00.0 |
| 200 | Diesel engine power safety | PowerSafetyProfile | CODE CONFIRMED — fn0 FLR proved need |
| 204 | VBIOS interpreter live | 422 ops on cold Titan V | Must re-run at 21:00.0 |
| 208 | Reboot-efficient sovereign | 183ms warm, fd keepalive | Must re-run |
| 213 | Live hardware warm handoff | Consolidated warm handoff | Must re-run at 21:00.0 |
| 219 | Catalyst driver pattern | 83K BAR0 regs captured | Requires nvsov install |
| 227 | Tier 2 warm compute breakthrough | tpc_alive=true | CRITICAL — must reproduce |
| 230 | Diesel abstraction revalidation | 5 lockup vectors PASS | Code carries forward |

### Proven Failures (carry forward as known-bad)

| # | Title | Root Cause | biomeGate Relevance |
|---|-------|------------|---------------------|
| 199 | K80 fire on reboot | Bulk PMC_ENABLE=0xFFFFFFFF | REPRODUCED on fn0 (FLR) |
| 217 | TPC PRI station creation | Firmware-mediated, not BAR0 | Architecture constraint |
| 225 | Catalyst TPC persistence | vfio-pci FLR destroys warm state | REPRODUCED (no_bus_reset fails) |
| 233 | Hybrid RM dispatch | NOP'd cap system → 0x22 | Architecture constraint |
| 234 | Catalyst minimal NOP | RM kernel deadlock cold GPU | Pending retest |

### Active Frontiers (to be continued on biomeGate)

| # | Title | biomeGate Status |
|---|-------|------------------|
| 226 | SBR bus reset suppression | CODE COMPLETE, hardware pending |
| 228 | Sovereign dispatch sprint | Requires Tier 2 on biomeGate Titan V |
| 231 | K80 cross-gen quench probe | **REVALIDATED** as Exp 236 |
| 234 | Catalyst minimal NOP | Requires nvsov module |

## Abstractions That Carry Forward (Code, Not Results)

The diesel engine abstractions were built during the strandGate era and
are hardware-independent by design. These carry forward as code, validated
by biomeGate experiments:

```
GenerationProfile::for_sm(sm)     ← data-driven, no SM-specific branches
InterruptProfile::for_sm(sm)      ← HARDWARE-PROVEN on biomeGate (Exp 236)
PowerSafetyProfile                ← PROVEN-BY-FAILURE on fn0 (Exp 235)
SovereignStrategy                 ← code carries forward, hardware TBD
HandoffCapabilityProfile          ← code carries forward, hardware TBD
PatchSet::from_recipe_toml()      ← code carries forward
```

## coralReef lower_f64 Fossil Status

The WGSL f64 polyfill PRNG path (`log_f64`, `sqrt_f64`, `cos_f64`) was
proven broken during strandGate era (570σ plaquette error). The `cpu_mom`
workaround and TMU PRNG path are production. coralReef `lower_f64/`
Newton-Raphson lowering is compile-ready but never validated on hardware.
biomeGate carries this forward as an open item.

## Reference Hardware (strandGate)

```
Titan V #1:  0000:02:00.0  (10de:1d81)  SM70
Titan V #2:  0000:49:00.0  (10de:1d81)  SM70
K80 fn0:     unknown BDF
K80 fn1:     unknown BDF
RTX 3090:    unknown BDF                 SM86
RX 6950 XT:  unknown BDF                 AMD RDNA2
```

None of these BDFs are relevant to biomeGate. All register-level results
(BAR0 captures, PMC values, PRAMIN offsets, GPC counts) must be re-derived.
