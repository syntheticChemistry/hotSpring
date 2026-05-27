# Experiment 228: Sovereign Dispatch Sprint

**Date:** 2026-05-26
**Status:** Partial — pipeline proven, FECS ACR blocks shader execution
**Hardware:** Dual Titan V (GV100, SM70) — 0000:02:00.0 + 0000:49:00.0
**Prerequisite:** Exp 227 Tier 2 (WarmCompute) via catalyst pipeline

## Objective

Validate the full sovereign shader dispatch pipeline on Titan V from Tier 2 state: cold boot → catalyst → FECS channel binding → coralReef SM70 compile → QMD dispatch → non-zero readback.

## Findings

### What Works

1. **Catalyst detection by FECS PC range** — RM firmware PCs at `0x18b3xxxx` vs nouveau at `~0x6000`. On Volta HS, `CPUCTL_ALIAS` reads `0x00000000` (HS security gate zeros the register), so halted/running detection is unreliable. FECS PC >= `0x10000000` is the robust catalyst indicator.

2. **`catalyst_warm` flag in `open_vfio()`** — skips the destructive deferred-boot path (PMC cold detection, FECS FLR handling) to preserve RM-established hardware state.

3. **TPC state survives PRI ungating** — CG sweep + PRI ring recovery + PGOB + force enumerate + GPC MMU init + sw_nonctx.bin replay all execute without destroying TPC PRI stations:
   ```
   tpc_before: [16, 16, 16, 16, 16, 16]
   tpc_after:  [16, 16, 16, 16, 16, 16]
   tpc_survived: true
   ```

4. **FECS INIT_CTXSW / BIND_CHANNEL / COMMIT succeed** — via the PRI-faulted MAILBOX1-trigger path, all three methods return status=0. The FECS method protocol works mechanically.

5. **QMD + GPFIFO dispatch pipeline works** — shader binary uploaded, QMD constructed, pushbuffer submitted via GPFIFO, sync completes (GP_PUT advances). coralReef SM70 compile produces valid 96-byte binary.

6. **shader_info forwarded from compile_result** — `compile_and_submit()` now passes `shader_info` (gpr_count, shared_mem_bytes, barrier_count) through to QMD construction.

### What Doesn't Work

7. **Channel stays PENDING** — `pccsr=0x11000001` (status=1, PENDING). The PFIFO scheduler queues our channel in the runlist but the GR engine never loads it. Without FECS processing context switches, the scheduler can't transition from PENDING → ACTIVE.

8. **PBDMA DEVICE faults** — `intr_0=0x10011111` (DEVICE + ENGINE + PBSEG + SEMAPHORE + top DEVICE). The GR engine doesn't respond to PBDMA requests because no channel context is loaded.

9. **Zero readback** — buffer reads back all zeros. The GPU DMA'd the buffer contents to host memory (the readback path works), but the SMs never executed the shader, so no writes occurred.

### Root Cause: Volta FECS ACR

The fundamental blocker is **Authenticated Code ROM (ACR)** on Volta (GV100) FECS:

- FECS runs in **HS (High Security)** mode
- IMEM is signed and write-protected by ACR
- Only matching-signed firmware (NVIDIA key) can be loaded
- nouveau's unsigned FECS firmware cannot be loaded on HS-locked falcons
- After PGRAPH reset: FECS IMEM clears (`fecs_imem_kb=0`), but the HS bootloader rejects unsigned code
- GPCCS PIO boot consistently times out (`cpuctl=0x00000000, pc=0x00000000`)
- FECS PIO boot "succeeds" but firmware unchanged (PC stays at RM range `0x18b34xxx`)

**Impact:** The RM firmware's FECS idle loop doesn't process our INIT_CTXSW/BIND_CHANNEL/COMMIT protocol. It uses NVIDIA's proprietary RM control messages. Without replacing FECS firmware, we can't get context switching, and without context switching, the scheduler can't load channels for compute dispatch.

## Code Changes

| File | Change |
|------|--------|
| `toadStool/cylinder/src/nv/compute_device.rs` | `catalyst_warm` field, `probe_warm_fecs()` PC range detection, `open_vfio()` catalyst skip + PRI ungating + FECS re-boot |
| `toadStool/cylinder/src/nv/compute_device.rs` | `open_vfio_from_received()` catalyst path for anchor-fd adoption |
| `hotSpring/barracuda/src/compute_dispatch/mod.rs` | `compile_and_submit()` forwards `shader_info` from compile_result |

## Pipeline Flow (Catalyst → Dispatch)

```
1. probe_warm_fecs()
   → FECS PC=0x18b34xxx → is_catalyst_pc=true
   → live_warm (CPUCTL_ALIAS=0x00, running, pmc_pop=23) + catalyst_pc
   → fecs_ready=true, catalyst_warm=true

2. open_vfio() [catalyst_mode=true]
   → Skip deferred boot path (no PMC cold check, no FECS FLR detect)
   → Channel creation (GPFIFO + USERD + GR context)
   → Channel stuck PENDING (scheduler won't load)
   → Catalyst block: PRI ungating sequence
     → CG sweep, PRI recovery, PGOB, force enumerate
     → GPC MMU init, sw_nonctx.bin replay
     → TPC state preserved ✓
   → GPCCS PIO boot: TIMEOUT ✗
   → FECS PIO boot: "success" but firmware unchanged ✗
   → PGRAPH reset fallback
   → FECS alive but still RM firmware
   → Channel setup methods succeed (status=0)

3. dispatch()
   → Shader upload, QMD build, GPFIFO submit
   → sync() — GP_PUT/GP_GET match
   → Readback: all zeros (SMs never executed)
```

## Tier Model Update

| Tier | State | Status |
|------|-------|--------|
| 0 | Cold | ✓ Baseline |
| 1 | WarmInfrastructure | ✓ sovereign.init |
| 2 | WarmCompute | ✓ catalyst pipeline |
| 2.5 | DispatchMechanics | ✓ **NEW** — pipeline works, FECS methods succeed |
| 3 | SovereignExecution | ✗ Blocked by FECS ACR |

## Paths Forward

1. **K80 Tesla (Kepler GK210)** — no ACR on Kepler falcons. FECS firmware can be freely loaded via PIO. The dispatch path already has Kepler branching. First sovereign shader execution likely achievable on K80.

2. **RM channel protocol reverse engineering** — understand how RM's FECS firmware processes channel context switches. Requires analyzing `nvidia-open-gpu-kernel-modules` 470.x FECS firmware source or the RM driver's channel management code.

3. **ACR bypass research** — Volta's ACR is the same generation as GP10x. Research whether the boot ROM can be bypassed via specific register sequences or timing attacks.

4. **Hybrid mode** — keep nvidia-470 loaded for FECS context switching but own the dispatch path via direct BAR0 PBDMA submission. This is not fully sovereign but achieves compute dispatch with vendor-managed initialization.

## Cold-Boot Reproducibility (Post-Reboot Validation)

**Validated:** May 26, 2026 — full AC power cycle, cold start

1. Both Titan Vs confirmed Tier 0 (Cold): `pmc_popcount=4`, `tpc_alive=false`
2. `sovereign.init` → Tier 1 on both (13.9s each, `pmc_popcount=23`)
3. `sovereign.warm_handoff` → Tier 2 on both:
   - Titan V #1: 18,481 alive regs, `tpc_alive=true`, `tpc_status=16`, 76s
   - Titan V #2: 17,934 alive regs, `tpc_alive=true`, `tpc_status=16`, 76s
4. Dispatch on both: `compute.dispatch.submit` with coralReef SM70 binary (96 bytes):
   - Titan V #1: `status=completed`, `non_zero=False`, `word0=0x00000000`
   - Titan V #2: `status=completed`, `non_zero=False`, `word0=0x00000000`
5. Catalyst detection confirmed: `FECS PC=0x18b3456f`, `catalyst=true`
6. TPC preservation confirmed: `tpc_before=[16,16,16,16,16,16] == tpc_after`
7. GPCCS ACR timeout confirmed: `cpuctl=0x00000000, pc=0x00000000`

## Validation Commands

```bash
# Tier 2 confirm
echo '{"jsonrpc":"2.0","method":"sovereign.classify_tier","params":{"bdf":"0000:02:00.0"},"id":1}' | \
  sudo socat - UNIX-CONNECT:/run/toadstool/biomeos/compute.sock

# Dispatch test (returns zeros due to FECS ACR)
echo '{"jsonrpc":"2.0","method":"compute.dispatch.submit","params":{...},"id":1}' | \
  sudo socat -t30 - UNIX-CONNECT:/run/toadstool/biomeos/compute.sock
```
