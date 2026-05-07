# Sprint C: Hardware Validation + MITM Tracing Handoff

**Date:** 2026-05-06  
**Scope:** First hardware validation of cold-boot sovereign pipelines; warm handoff intermediate solve; BootConversation MITM tracing infrastructure.

## Hardware Validation Results

### Titan V (GV100) — `volta_sovereign_pipeline` on `0000:02:00.0`

| Stage | Result | Detail |
|-------|--------|--------|
| VFIO open + BAR0 | PASS | Bus master enabled, BOOT0=0x140000a1 (SM70) |
| PCI hot reset | PASS | BOOT0 stable after SBR |
| Falcon probe | PASS | FECS: FAULTED (expected), SEC2: CleanReset |
| SEC2 BL load | PASS | 512B code + 256B data uploaded to IMEM/EMEM |
| SEC2 BL exec | **STALL** | PC traces: 0xfd0a→0xfd62→0xfd7a→0xfe0e→0xfe30→0xfe4b→0xfe57 → **PC=0x0001** |
| All 10 ACR strategies | FAIL | SEC2 timeout on every strategy; MB0=0xcafebeef (sentinel never overwritten) |

**Root Cause:** HBM2 is cold (VRAM sentinel test: wrote=0xacb00700, read=0xbad0ac0d, ok=false). SEC2 bootloader starts executing but cannot complete DMA because VRAM is dead. Both instance-block-backed VIRT DMA and physical DMA produce the same PC=0x0001 stall. The BL's DMA initiation instruction fails because there is no accessible memory for the ACR payload.

**Cold-boot prerequisite:** HBM2 training (FBPA initialization + memory controller POST). This requires VBIOS DEVINIT or driver-level memory training — the proprietary sequence not yet replicated in our DEVINIT interpreter.

### Tesla K80 (GK210B) — `kepler_cold_pipeline` on `0000:4b:00.0`

| Stage | Result | Detail |
|-------|--------|--------|
| PRI ring init | PASS | 5 GPC stations, 6 ROP, 1 hub |
| VBIOS DEVINIT | PASS | 6 scripts, 69 ops, 41 writes |
| PMU firmware boot | PASS | Ring queues configured, running=true |
| GPC PLL (0x137000) | **LOCKED** | pll_ctrl_writable=false, both pre and post-PMU |
| PGOB (3 strategies) | PASS | All power steps complete, but gpc_alive=false |
| FECS upload + start | PASS | IMEM verified, STARTCPU issued, cpuctl=running |
| FECS boot poll | **STALL** | mailbox0=0 for 2s, exci=0x00000001 |
| Channel + runlist | PARTIAL | SCHED_ERROR code=8 (GR engine dead) |
| NOP dispatch | FAIL | GP_GET never advances |

**Root Cause:** GPC clock domain PLL CTRL at `0x137000` is hardware write-protected. The PLX bridge topology and cold-reset state leave the GPC clock tree ungated. DEVINIT ran but did not configure this register range. Without GPC clocks, the GPC falcon (GPCCS) PIO is dead, FECS cannot discover GPCs, and the GR engine never initializes.

### Warm Handoff Attempt — K80 via nouveau

| Step | Result | Detail |
|------|--------|--------|
| nouveau modprobe | **CRASH** | Kernel page fault at `ioread32+0x3a/0x80`, CR2=ffffd1c3c600020c |
| K80 PCIe link | DEAD | All BAR0 reads return 0xFFFFFFFF after crash |
| nouveau module | WEDGED | State="Loading", refcount=1, cannot rmmod |

**Root Cause:** nouveau on kernel 6.17.9 crashes during GK210 probe. Known issue — the chipset mapping patch doesn't fully protect against the BAR access pattern nouveau uses during probe on this kernel version.

## Deliverables

### 1. BootConversation MITM Tracing (`boot_follower.rs`)

New types for full driver↔GPU conversation capture:

- **`BootConversation`** — ordered interleaving of all MMIO reads/writes with timestamps, domain classification, and detected poll patterns
- **`MmioOp`** / **`MmioDirection`** — individual operation in the conversation
- **`PollPattern`** — detected poll loops (consecutive reads to same register), with first/final values and duration
- **`BootConversation::writes_after_poll()`** — finds writes that follow a specific poll (reveals "wait for X then configure Y" patterns)
- **`DomainOpsCount`** — per-domain read/write statistics

### 2. mmiotrace Integration in Warm Handoff Scripts

Both `k80_nouveau_post.sh` and `manual_warm_handoff.sh` now support `CAPTURE_MMIOTRACE=1`:

```
sudo CAPTURE_MMIOTRACE=1 ./manual_warm_handoff.sh 0000:02:00.0
```

Flow: enable mmiotrace → load driver → capture trace → disable mmiotrace → save to `/var/lib/coralreef/traces/`.

### 3. K80 BDF Fix

`k80_nouveau_post.sh` updated from stale BDFs (`4c:00.0`/`4d:00.0`) to actual topology (`4b:00.0`/`4c:00.0`). Bridge reset paths corrected to match PLX switch hierarchy.

## Architecture: Learning Cold Boot via Warm Observation

```
┌─────────────────────────────────────────────────────────────────┐
│                   Cold Boot Learning Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. WARM HANDOFF (intermediate solve)                           │
│     nvidia-470 / nouveau → livepatch → vfio-pci swap            │
│     ↳ Proven compute dispatch on warm GPU state                  │
│                                                                  │
│  2. OBSERVE (mmiotrace + oracle)                                │
│     Cold BAR0 snapshot → mmiotrace during driver init            │
│     → Warm BAR0 snapshot → diff → BootConversation JSON          │
│     ↳ Full ordered R/W sequence with poll patterns               │
│                                                                  │
│  3. ANALYZE (boot_follower)                                     │
│     BootConversation → domain-classified sequence                │
│     → PLL lock polls → HBM2 training writes → FBPA config       │
│     → FECS boot handshake → dependency graph                     │
│                                                                  │
│  4. REPLAY (recipe engine)                                      │
│     Training recipe → cold VFIO BAR0 replay                     │
│     → Validate: PTIMER ticking? PLLs locked? GPCs alive?        │
│     ↳ Progressive: replay PLL domain first, then HBM2, etc.     │
│                                                                  │
│  5. SOVEREIGN (cold boot)                                       │
│     Replayed init = no driver dependency                         │
│     ↳ Era-agnostic: recipe is data, not code                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Next Steps — Execution Order

1. **Reboot** to clear wedged nouveau module and dead K80 PCIe link
2. **K80 warm via nvidia-470:** Investigate nvidia-470 support for GK210 (may need `NVreg_OpenRmEnableUnsupportedGpus=1`)
3. **Titan V warm via nvidia-470:** Run `titan-v-module-swap.sh swap-only` from TTY/SSH, then `kepler_cold_pipeline` (detects warm) or direct dispatch test
4. **Capture training recipes:** Run warm handoffs with `CAPTURE_MMIOTRACE=1`, feed traces to `BootConversation::from_trace()` to learn init sequences
5. **Progressive cold replay:** Use captured recipes to replay HBM2/PLL initialization on cold GPU via VFIO BAR0

## Hardware Status

| GPU | BDF | IOMMU | Current State | Next Action |
|-----|-----|-------|---------------|-------------|
| Titan V | 0000:02:00.0 | 69 | vfio-pci (healthy) | nvidia-470 warm handoff + mmiotrace capture |
| K80 die0 | 0000:4b:00.0 | 35 | nouveau WEDGED (PCIe dead) | Reboot → nvidia-470 or patched nouveau warm |
| K80 die1 | 0000:4c:00.0 | 36 | vfio-pci | Warm handoff after die0 solved |
| RTX 5060 | 0000:21:00.0 | — | nvidia-580 (display) | No changes needed |
