# Experiment 225 — Catalyst TPC Persistence Test

**Date**: 2026-05-26
**Status**: RESULT — nvidia RM failed to initialize GPU after vfio-pci release; vfio-pci reset-on-release destroyed VBIOS warm state; Tier 0 (Cold) after swap
**Hardware**: Titan V #1 (`0000:02:00.0`) — experiment target; Titan V #2 (`0000:49:00.0`) — control (stayed Tier 1)
**Dependency**: Exp 219 (Catalyst Driver Pattern), Exp 224 (Sovereignty Audit)

## Objective

Test whether TPC PRI ring stations survive the nvidia-470 catalyst unbind →
vfio-pci rebind. This is the critical question for Tier 2 via the catalyst
pattern: if TPC stations persist, we can achieve sovereign compute by loading
nvidia once and replaying the golden state.

## Execution

### Pipeline

Used `sovereign.warm_handoff` RPC with strategy `nvidia_catalyst_titanv`:

1. Released VFIO anchor and BAR0 fds for `0000:02:00.0`
2. DKMS module `nvidia/470.256.02` already built for kernel `6.17.9`
3. Module patched: 13/13 patches applied (`nvidia_catalyst_handoff` set)
4. ksymtab stripped, renamed nvidia → nvsov, relocation normalization
5. `insmod /tmp/toadstool-patched-nvsov.ko` — completed in 400ms
6. 60s settle for RM initialization
7. BAR0 capture: 5,210 alive registers (domain-scoped, 889ms, 22 domains)
8. Fire-and-poll unbind (2s)
9. `driver_override` → `vfio-pci`, `drivers_probe` → bound in 7s
10. PRI ring recovery
11. `rmmod nvsov` (100ms)

Total pipeline: 72s

### Results

#### nvidia RM did NOT initialize the GPU

After 60s settle with nvsov bound to `0000:02:00.0`:

```
PMC_ENABLE       = 0x40000020 (2 engines — RM never ran DEVINIT)
FECS cpuctl      = 0xBADF1201 (PRI fault — never started)
GPCCS cpuctl     = 0xBADF3000 (PRI fault — never started)
PGRAPH status    = 0xBADF1201 (PRI fault)
TPC alive        = false
GPC0-5 TPC0      = 0xBADF3000 FAULT (all 6 GPCs)
PMU cpuctl       = 0x00000020 (idle)
```

#### Root Cause: vfio-pci reset-on-release

dmesg reveals the sequence:

```
[3078.060830] vfio-pci 0000:02:00.0: resetting
[3078.060885] vfio-pci 0000:02:00.1: resetting
[3078.225015] vfio-pci 0000:02:00.0: reset done
[3078.225056] vfio-pci 0000:02:00.1: reset done
[3081.269587] nvsov: module license 'NVIDIA' taints kernel.
[3081.366563] nvsov 0000:02:00.0: vgaarb: VGA decodes changed
[3081.595911] NVRM: The NVIDIA probe routine was not called for 2 device(s).
[3081.595923] NVRM: loading NVIDIA UNIX x86_64 Kernel Module 470.256.02
```

**vfio-pci performed a device reset during unbind** (despite `All device reset
methods disabled by user` at boot). The reset destroyed the VBIOS-initialized
warm state (`PMC_ENABLE 0x5fecdff1 → 0x40000020`). By the time nvsov bound,
the GPU was in a cold/reset state.

nvidia RM *did* bind to `0000:02:00.0` (vgaarb confirms), but RM found a
cold GPU and its probe sequence failed to complete DEVINIT — the "probe
routine was not called for 2 device(s)" refers to Titan V #2 and RTX 5060
audio, not our target card.

#### Post-swap state

After nvsov unbind → vfio-pci rebind:

```
classify_tier: Cold (Tier 0)
PMC_ENABLE     = 0x40001020 (3 engines — PRI ring recovery added PGRAPH)
PMC popcount   = 3
PRAMIN         = not accessible
TPC alive      = false
```

PRI ring recovery ran successfully (PGRAPH re-enabled, FECS/GPCCS accessible
at cpuctl=0x10) but the GPU is Tier 0 — below the hardware line.

#### Catalyst capture (degraded)

The 5,210 alive registers were captured from the degraded state, not from a
fully-initialized GPU:

```
PMC_ENABLE in capture = 0x40000020 (2 engines)
TPC region writes     = 0
Domains              = CE, PBUS, PCLOCK, PDISP, PFB, PGRAPH, PMC, PMU,
                        PRIV_RING, PTHERM, PTIMER, SEC2
```

No TPC registers in the capture because RM never created TPC stations.

#### Control card

Titan V #2 (`0000:49:00.0`) remained at Tier 1 throughout:

```
tier = warm_infrastructure (level 1)
PMC_ENABLE = 0x5fecdff1 (23 engines)
TPC alive  = false (same as before — VBIOS POST doesn't create TPC stations)
```

## Key Discovery: vfio-pci Reset-on-Release

The vfio-pci driver performs a device reset when releasing a device, even when
`disable_default_reset_type=1` is set. This destroys the VBIOS warm state:

- `PMC_ENABLE` drops from `0x5fecdff1` (23 engines) to `0x40000020` (2 engines)
- All falcon firmwares are wiped
- PRI ring routing is destroyed
- The GPU enters a state where nvidia RM cannot complete its probe

### Implications

1. **The catalyst pattern requires solving reset-on-release.** Either:
   - Suppress vfio-pci's reset during the handoff (kernel parameter or
     module parameter `disable_idle_d3=1` or similar)
   - Use a different unbind strategy (raw sysfs unbind without vfio release)
   - Keep an fd open during the swap to prevent vfio from releasing
   - Use `new_id` / `remove_id` instead of driver_override

2. **Without solving reset-on-release, the catalyst can never see TPC alive.**
   Even if RM fully initializes the GPU (which it would on a warm GPU), the
   vfio-pci release reset would destroy TPC stations before vfio-pci rebinds.

3. **Titan V #1 is now Tier 0 (Cold).** Requires power cycle to recover.

## Preserved Artifacts

| File | Description |
|------|-------------|
| `/tmp/toadstool-catalyst-0000-02-00-0.json` | 28MB full BAR0 snapshot (degraded state) |
| `/tmp/toadstool-catalyst-replay-0000-02-00-0.json` | 542KB replay sequence (5,210 writes) |
| `/var/lib/toadstool/catalysts/frozen/nvsov_gv100_470.256.02_k6.17.9-*.ko` | 41MB frozen patched module |
| `/var/lib/toadstool/catalysts/recipes/gv100_nvidia470_patchset.json` | Patch recipe |

## Next Steps

1. **Solve vfio-pci reset-on-release**: Investigate `vfio_pci_core_disable()`
   reset suppression — may need kernel parameter `vfio_pci.noresetrelease=1`
   or a pre-swap anchor hold strategy

2. **Re-run with reset suppressed**: If RM can bind to a warm GPU (VBIOS POST
   state preserved through the swap), it should complete DEVINIT → ACR → GPCCS →
   TPC station creation

3. **Pre-swap TPC snapshot**: While nvsov has the GPU fully initialized, capture
   PRI ring hub/router/station registers — these might be replayable even if
   TPC stations are destroyed by the swap

4. **Power cycle required** to recover Titan V #1 to Tier 1
