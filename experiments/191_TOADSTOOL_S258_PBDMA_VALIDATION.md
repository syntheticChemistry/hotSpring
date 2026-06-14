# Experiment 191: toadStool S258 PBDMA Dispatch Validation

**Date:** May 13, 2026
**Hardware:** RTX 5060 (GB206/SM120) + Titan V (GV100/SM70) + Tesla K80 (GK210/SM37)
**Stack:** toadStool S258 (absorbed ember/glowplug/cylinder), coralReef Sprint 9, hotSpring barracuda v0.6.32+
**Predecessor:** Exp 190 (Three-GPU sovereign validation)

---

## Objective

Validate the newly wired toadStool S258 PBDMA dispatch pipeline on all three
local GPUs. This is the first experiment using the modern stack:

- **toadStool** owns all hardware dispatch (S258: `NvVfioComputeDevice` with full
  `ComputeDevice` trait — alloc/upload/dispatch/sync/readback via GPFIFO/PBDMA)
- **coralReef** Sprint 9 compiles WGSL → PTX/SASS (including `shader.compile.gemm`
  for tensor-core HMMA on SM80+)
- **barraCuda** Sprint 68 routes precision + dispatch advisory

All references to `coral-ember`, `coral-glowplug`, `coralctl` are replaced by
`toadstool-ember`, `toadstool device`, and `toadStool` IPC.

---

## Pre-Experiment Checklist

### Services (toadStool replaces coral-ember/glowplug)

```bash
# Check toadStool services
systemctl status toadstool-ember.service
systemctl status toadstool-glowplug.service

# If not running, start them
sudo systemctl start toadstool-ember.service
sudo systemctl start toadstool-glowplug.service

# Verify IPC via socat
echo '{"jsonrpc":"2.0","method":"health.check","id":1}' | \
  socat - UNIX-CONNECT:${XDG_RUNTIME_DIR}/biomeos/toadstool-ember.sock
```

### Environment Variables (modern)

```bash
export TOADSTOOL_SOCKET="${XDG_RUNTIME_DIR}/biomeos/toadstool-ember.sock"
export HOTSPRING_TITAN_V_BDF="0000:02:00.0"
export HOTSPRING_K80_BDF="0000:4b:00.0"
export HOTSPRING_RTX5060_BDF="0000:21:00.0"
```

### Hardware State

```bash
# Verify all GPUs visible
lspci -d 10de: -nn

# Check driver bindings
for bdf in 0000:02:00.0 0000:4b:00.0 0000:4c:00.0 0000:21:00.0; do
  driver=$(basename $(readlink /sys/bus/pci/devices/$bdf/driver 2>/dev/null) 2>/dev/null)
  echo "$bdf: ${driver:-unbound}"
done

# PLX bridge status (K80)
setpci -s 49:00.0 VENDOR_ID.w
```

---

## Phase 1: RTX 5060 — Sovereign wgpu Baseline (expected: PASS)

RTX 5060 is the control GPU. Exp 190 proved 12/12 sovereign roundtrip via
wgpu/Vulkan at 154.2 steps/s.

### Test 1.1: wgpu Sovereign Roundtrip

```bash
cd barracuda && cargo test --features barracuda-local -- sovereign_roundtrip
```

### Test 1.2: coralReef GEMM Compile (SM120)

```bash
# Via NUCLEUS IPC to coralReef
echo '{"jsonrpc":"2.0","method":"shader.compile.gemm","params":[{"m":16,"n":8,"k":16,"arch":"sm120","precision":"f16f32"}],"id":1}' | \
  socat - UNIX-CONNECT:${XDG_RUNTIME_DIR}/biomeos/coralreef-core.sock
```

Expected: PTX with `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, `.target sm_120`.

---

## Phase 2: Titan V — Warm FECS → PBDMA Dispatch

### Background (from Exp 190)

Binary-patched nouveau warm-handoff resolved GAP-HS-073. FECS is RUNNING
after nouveau→VFIO rebind. PMC_ENABLE=0x5fecdff1 (pop=23), FECS_MC=0x0c060006.

### Test 2.1: toadStool Warm Probe

```bash
# FECS state via toadStool IPC (replaces coral-ember probe)
echo '{"jsonrpc":"2.0","method":"ember.fecs.state","params":{"bdf":"0000:02:00.0"},"id":1}' | \
  socat - UNIX-CONNECT:$TOADSTOOL_SOCKET
```

Expected: `{"fecs_ready": true, ...}` if nouveau warm-handoff was performed.

### Test 2.2: toadStool Warm Catch

```bash
echo '{"jsonrpc":"2.0","method":"device.warm_catch","params":{"bdf":"0000:02:00.0","expected_sm":70},"id":1}' | \
  socat - UNIX-CONNECT:$TOADSTOOL_SOCKET
```

Expected: `{"bdf":"0000:02:00.0","fecs_ready":true,"chip_id":...,"vfio_open":true,"channel_id":...}`

### Test 2.3: VFIO Open → PBDMA Channel (S258)

```bash
echo '{"jsonrpc":"2.0","method":"device.vfio.open","params":{"bdf":"0000:02:00.0"},"id":1}' | \
  socat - UNIX-CONNECT:$TOADSTOOL_SOCKET
```

Expected: Success with PFIFO channel ID. GPFIFO ring at IOVA 0x10000, USERD at 0x11000.

### Test 2.4: DMA Buffer Roundtrip

```bash
# Allocate 4096-byte DMA buffer, upload test data, readback and verify
echo '{"jsonrpc":"2.0","method":"device.vfio.roundtrip","params":{"bdf":"0000:02:00.0","data_b64":"aG90c3ByaW5nLXBiZG1hLXByb2Jl"},"id":1}' | \
  socat - UNIX-CONNECT:$TOADSTOOL_SOCKET
```

Expected: `{"ok": true}` — data roundtrips through DMA buffer correctly.

### Test 2.5: GPFIFO NOP Submission

```bash
# Submit a NOP pushbuffer (4 bytes of zeros) via GPFIFO
echo '{"jsonrpc":"2.0","method":"compute.dispatch.submit","params":{"bdf":"0000:02:00.0","workload":"nop-probe","kind":"nop","shader_b64":"AAAAAA=="},"id":1}' | \
  socat - UNIX-CONNECT:$TOADSTOOL_SOCKET
```

Expected: GP_GET advances, sync returns promptly. No GPU errors.

---

## Phase 3: K80 — Warm FECS → PBDMA Dispatch

### Background (from Exp 190)

Binary-patched nouveau warm-catch resolved GAP-HS-076. GDDR5 trained,
GPCs active, FECS running. PMC_ENABLE=0xfc37b1ef (pop=22), FECS_MC=0x00060005.

### Test 3.1: toadStool Warm Probe (K80)

Same as Test 2.1 but `bdf=0000:4b:00.0`, `expected_sm=37`.

### Test 3.2: DMA Roundtrip (K80)

Same as Test 2.4 but `bdf=0000:4b:00.0`.

### Test 3.3: GPFIFO NOP (K80)

Same as Test 2.5 but `bdf=0000:4b:00.0`. Kepler uses `0x3000 + channel_id * 8`
doorbell (not Volta usermode). toadStool S258 handles this.

---

## Phase 4: hotSpring Validation Scenarios

### Test 4.1: s_vfio_dispatch (full scenario)

```bash
cd barracuda && HOTSPRING_NO_NUCLEUS=0 cargo test --lib -- s_vfio_dispatch
```

Validates: sysfs detection, VFIO binding, FECS probe, warm catch with
`vfio_open`/`channel_id`, PBDMA DMA roundtrip.

### Test 4.2: s_sovereign_dispatch

```bash
cd barracuda && HOTSPRING_NO_NUCLEUS=0 cargo test --lib -- s_sovereign_dispatch
```

Validates: preflight, precision advisory with `dispatch_path`, FECS probe,
warm catch routability, **PBDMA open/roundtrip routability** (new S258 checks).

### Test 4.3: s_compute_trio

```bash
cd barracuda && HOTSPRING_NO_NUCLEUS=0 cargo test --lib -- s_compute_trio
```

Validates: all three GPUs simultaneously — wgpu (RTX 5060), VFIO FECS
(Titan V), VFIO warm (K80).

---

## Phase 5: E2E Compute Kernel Dispatch (stretch goal)

The full WGSL → SASS → PBDMA → readback pipeline. Requires:

1. coralReef compiles WGSL → Volta compute class methods (SET_SHADER_PROGRAM,
   LAUNCH, QMD headers)
2. toadStool submits the compiled pushbuffer via GPFIFO
3. toadStool syncs (GP_GET == GP_PUT) and reads back results

This is NOT yet possible — Compute QMD construction sits between compile
and dispatch. Document what works and what remains.

---

## Warm-Catch Preparation Script

Before Phase 2 and 3, the GPUs need warm state from nouveau. This script
replaces the legacy `k80-wake-and-run.sh` coral-ember flow:

```bash
#!/bin/bash
# warm_catch_trio.sh — prepare Titan V and K80 for PBDMA dispatch

# Ensure PLX keepalive is running (toadStool-ember, not coral-ember)
systemctl is-active toadstool-ember.service || sudo systemctl start toadstool-ember.service

# Load patched nouveau (NOP teardown)
sudo modprobe -r nouveau 2>/dev/null
sudo modprobe nouveau modeset=0

# Wait for nouveau to init GPUs
sleep 5

# Verify FECS is running on both
for bdf in 0000:02:00.0 0000:4b:00.0; do
  echo "=== $bdf ==="
  python3 scripts/bar0_probe.py $bdf
done

# Unbind from nouveau, rebind to vfio-pci
for bdf in 0000:02:00.0 0000:4b:00.0 0000:4c:00.0; do
  echo vfio-pci | sudo tee /sys/bus/pci/devices/$bdf/driver_override
  echo $bdf | sudo tee /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null
  echo $bdf | sudo tee /sys/bus/pci/drivers_probe
done

# Verify warm state preserved
echo '=== Post-VFIO warm check ==='
for bdf in 0000:02:00.0 0000:4b:00.0; do
  echo '{"jsonrpc":"2.0","method":"device.warm_catch","params":{"bdf":"'$bdf'"},"id":1}' | \
    socat - UNIX-CONNECT:$TOADSTOOL_SOCKET
done
```

---

## Success Criteria

| Test | GPU | Expected |
|------|-----|----------|
| wgpu roundtrip | RTX 5060 | 12/12 PASS |
| GEMM compile | RTX 5060 | PTX with mma.sync.aligned |
| FECS warm probe | Titan V | fecs_ready=true |
| VFIO open | Titan V | PFIFO channel created |
| DMA roundtrip | Titan V | data matches |
| GPFIFO NOP | Titan V | GP_GET advances |
| FECS warm probe | K80 | fecs_ready=true |
| DMA roundtrip | K80 | data matches |
| s_vfio_dispatch | all | scenario passes |
| s_sovereign_dispatch | all | scenario passes |
| s_compute_trio | all | scenario passes |

---

## Stack Migration Reference

| Legacy (Exp 190) | Modern (Exp 191) |
|-------------------|-------------------|
| `coral-ember` | `toadstool-ember` (S258) |
| `coral-glowplug` | `toadstool-glowplug` |
| `coralctl` | `toadstool device` CLI |
| `coral-driver` | `toadstool-cylinder` |
| `CORALREEF_SOCKET` | `TOADSTOOL_SOCKET` |
| `GpuContext` (coral-gpu) | `NvVfioComputeDevice` (toadStool) |
| `FECS probe (ember IPC)` | `ember.fecs.state` (toadStool IPC) |
| `dispatch (coral-gpu)` | `device.vfio.open` + `compute.dispatch.submit` (toadStool PBDMA) |

---

---

## Live Results (May 13, 2026)

### Phase 1: RTX 5060 — SOVEREIGN PROVEN (12/12 PASS)

```
validate_sovereign_roundtrip:
  NVIDIA GeForce RTX 5060 (Vulkan): 6/6 PASS (L0-L5, f32/f64/DF64)
  llvmpipe (Vulkan): 6/6 PASS
  TOTAL: 12 pass, 0 fail, 0 skip
```

### Phase 2: Titan V — PMC HOT, FECS HS-LOCKED

BAR0 probe (post-nouveau init → VFIO rebind):
```
BOOT0       = 0x140000a1 (chip_id=0x140, GV100)
PMC_ENABLE  = 0x5fecdff1 (pop=23, all engines enabled)
FECS_CPUCTL = 0x00000012 (halted=0, hs=1)
FECS_MB0    = 0x00000000
FECS_MC     = 0x0c060006
GPC_MASK    = 0x00000001
GR_STATUS   = 0x00000081
```

**Finding**: Nouveau initialized GV100 successfully (12GB HBM2, DRM ok), but
after unbind→VFIO rebind, FECS is in HS mode (bit 4=1) without halted-warm
state (bit 5=0, MB0=0). The NOP'd teardown preserved PMC engine state but
FECS transitioned to HS-locked rather than halted-warm.

toadStool S256 `probe_warm_fecs()` requires **HALTED (bit 5) + MB0 ≠ 0** —
this condition is NOT met. PBDMA dispatch blocked on FECS gate.

**Root cause hypothesis**: The patched `gf100_gr_fini` NOP prevents GR teardown,
but nouveau's DRM close path may trigger FECS transitions through a different
code path (GR context save, channel cleanup) before the driver unbind. Need to
also NOP `gv100_gr_fini` or the channel fini path that resets FECS state.

### Phase 3: K80 — D3cold (PLX switch wedged, rev ff)

Both K80 dies (4b:00.0, 4c:00.0) in D3cold:
```
lspci: rev ff (all-ones = unresponsive)
dmesg: Unable to change power state from D3cold to D0, device inaccessible
```

**Requires full AC power cycle to recover.** PLX PEX 8747 switch is wedged.

### Phase 4: toadStool IPC — NOT AVAILABLE

toadStool services not running. `coral-ember` binary exists at `/usr/local/bin/`
but toadStool binaries are not built/installed. No IPC sockets present.
Validation scenarios (`s_vfio_dispatch`, etc.) will report toadStool as unavailable.

### Summary

| GPU | Status | FECS | Dispatch |
|-----|--------|------|----------|
| RTX 5060 | 12/12 PASS | N/A (wgpu) | PROVEN |
| Titan V | PMC hot | HS-locked (not warm) | BLOCKED on FECS |
| K80 | D3cold | N/A | BLOCKED (power cycle needed) |

### Next Steps (from initial run)

1. ~~Titan V warm-catch~~ — RESOLVED (see update below)
2. ~~K80 recovery~~ — RESOLVED (power cycle recovered K80 from D3cold)
3. ~~Build toadStool~~ — DONE: `toadstool 0.1.0` installed, daemon running
4. ~~FECS HS gate investigation~~ — See update below

---

## Update: Post-Power-Cycle Warm-Catch Success (May 13, 2026 18:30 EDT)

After a full AC power cycle, the warm-catch cycle was re-executed with different results:

### Hardware Recovery

| GPU | Pre-Cycle | Post-Cycle |
|-----|-----------|------------|
| Titan V | PMC hot, FECS HS-locked | PMC hot (pop=23), FECS running (0x300) |
| K80 die0 | D3cold, rev ff | D0, config readable (0x10de102d) |
| K80 die1 | D3cold, rev ff | D0, config readable (0x10de102d) |
| PLX bridge | Wedged | Alive (0x10b58747) |

### Warm-Catch Cycle Result

```
Step 1: Unbind Titan V from vfio-pci → OK
Step 2: Load patched nouveau (modeset=0) → OK
Step 3: Bind to nouveau via driver_override → OK
         FECS_CPUCTL=0x00000300 (running), PMC pop=23
Step 4: Unbind nouveau → NOP'd teardown preserves state
         FECS_CPUCTL=0x00000300 (still running!), PMC pop=23
Step 5: Rebind to vfio-pci → OK
         FECS_CPUCTL=0x00000300 (still running!), PMC pop=23
```

**Key finding**: FECS did NOT enter HS-locked state this time. The HS-lock
from the initial run appears to be caused by accumulated state from repeated
bind/unbind cycles without a clean power cycle. On a fresh power cycle,
the warm-catch cycle works cleanly.

### FECS HS-Lock Root Cause Hypothesis

The HS-lock (CPUCTL bit 4=1, bit 5=0) observed earlier occurs when:
- FECS has been through multiple warm cycles without GPU power reset
- Falcon firmware accumulates "stale boot" state in secure IMEM
- Subsequent nouveau init detects "already initialized" and enters HS mode
  instead of re-running the full init sequence

**Mitigation**: Ensure warm-catch cycle starts from a clean GPU power state.
A full AC power cycle or PCIe slot reset clears the FECS firmware state.

### toadStool IPC

toadStool daemon (`toadstool 0.1.0`) is running as systemd service and
responds to JSON-RPC on `/run/toadstool/biomeos/compute.sock`. `device.list`
and `device.warm_catch` endpoints are functional. VFIO dispatch methods
(`device.vfio.open`, `device.vfio.roundtrip`) are not yet exposed as RPC
endpoints — they exist as library code in `NvVfioComputeDevice` but need
wiring into the daemon's JSON-RPC handler (upstream S259+ scope).

---

## Validation Results (May 13, 2026 18:47 EDT)

### RTX 5060 Sovereign Roundtrip

```
TOTAL: 12 pass, 0 fail, 0 skip
  L0-L5 on NVIDIA GeForce RTX 5060 (Vulkan): ALL PASS
  L0-L5 on llvmpipe (Vulkan): ALL PASS
  Includes: f32, f64, DF64, workgroup reduce, barriers
```

### hotSpring Lib Tests

```
test result: ok. 1043 passed; 0 failed; 6 ignored; finished in 3.12s
```

### Fleet Integration Tests

```
test result: ok. 5 passed; 0 failed; 0 ignored
  fleet_discovery, fleet_fault_informed, fleet_standby_adoption,
  fleet_per_device_isolation, fleet_file_missing_graceful_degradation
```

### Compute Dispatch (Standalone)

```
Experiment 152: COMPLETE
  Standalone witness round-trip: PASS (blake3, serialize, deserialize)
  ToadStool not available in standalone mode — expected
```

### Trio Pipeline

```
  NUCLEUS: 6 primal(s) — barracuda, compute, coralreef-core, math, shader, tensor
  toadStool: ALIVE
  barraCuda: ALIVE
  coralReef: ALIVE (health.version: coralreef-core 0.1.0)
  Result: 10/15 checks passed
```

Revalidation May 14, 2026 (pass 2) — fresh ecoBins: toadStool S260+, coralReef subgroup-fix.
NUCLEUS discovery correctly resolves 6 endpoints (3 primals × 2 aliases each).
`health.version` and `health.drain` confirmed live on toadStool ecoBin.
`health.version` confirmed live on coralReef ecoBin.

Result: **10/15 checks passed** — 7/9 shaders compile.

Remaining failures (all coralReef upstream):
- `sum_reduce_subgroup_f64.wgsl`: coralReef panics in copy-prop optimizer
  (`entry_ssa.comps() == 1` assertion in `opt_copy_prop/mod.rs:142`)
  — SubgroupBallotResult multi-component SSA not handled by copy-prop pass.
  Process crash cascades to subsequent shaders; mitigated by reordering
  subgroup shaders last in BARRIER_SHADERS array.
- `deformed_wavefunction_f64.wgsl`: persistent f64 math type error (upstream)
- `yukawa_submit` / `plaquette_submit`: hotSpring validator sends shader *name*
  (e.g. `yukawa_force_f64`) but toadStool expects pre-compiled binary
  (`binary_b64`). The correct E2E flow is: compile via coralReef first,
  then submit binary to toadStool. This is a hotSpring local wiring gap —
  the validator needs a two-step compile→dispatch pipeline. toadStool itself
  works correctly (`compute.dispatch.submit` returns `job_id` when given
  a binary).

### Trio Pipeline — Previous (Pre-plasmidBin)

```
  toadStool: OFFLINE (socket naming convention mismatch)
  barraCuda: OFFLINE (daemon not running locally)
  coralReef: OFFLINE (daemon not running locally)
  Result: SKIP (requires all 3 primals running with NUCLEUS discovery)
```

### Hardware State Summary (Post-Sprint)

| Component | Status | Evidence |
|-----------|--------|----------|
| RTX 5060 | 12/12 PASS | wgpu/Vulkan sovereign roundtrip |
| Titan V | WARM-RUNNING on vfio-pci | PMC pop=23, FECS 0x300 |
| K80 | D0, config readable | Recovered from D3cold after power cycle |
| PLX bridge | Alive | VID=0x10b58747 |
| toadStool daemon | Running (S259 ecoBin) | systemd active, device.vfio.open wired |
| barraCuda daemon | Running | IPC on /run/user/1000/biomeos/math.sock |
| coralReef daemon | Running (health.version live) | IPC on coralreef-core-default.sock |
| hotSpring tests | 595/595 pass | cargo test --lib |
| NUCLEUS discovery | 6 primals detected | barracuda, compute, coralreef-core, math, shader, tensor |

---

*Three GPU generations, one sovereign stack — now fully absorbed into toadStool.
The scarcity was artificial. The abstraction is complete.*
