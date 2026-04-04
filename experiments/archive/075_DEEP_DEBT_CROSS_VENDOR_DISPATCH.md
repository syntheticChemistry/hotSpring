# Experiment 075 — Deep Debt Burndown + Cross-Vendor Dispatch Validation

**Date:** March 18–22, 2026
**Hardware:** biomeGate — 2× Titan V (GV100, `0000:03:00.0` + `0000:4a:00.0`), RTX 5060 (GB206, display head); kernel 6.17.9
**Status:** COMPLETE — 13 deep-debt items resolved, cross-vendor dispatch validated, pkexec-free pipeline proven
**Crates:** coral-driver, coral-ember, coral-glowplug, coral-gpu (coralReef); hotspring-barracuda (hotSpring)
**Experiment:** 075

---

## Context

After the Ember swap pipeline hardening (Exp 074) and PFIFO diagnostic matrix (Exp 071), the sovereign pipeline was functionally correct but carried accumulated engineering debt across concurrency, error handling, documentation, and build configuration. Before proceeding with PMU cracking (Layer 6 MMU page table translation), a systematic burndown was needed to harden the tooling and prevent subtle bugs from poisoning future experiments.

Simultaneously, two new capabilities were validated:
1. **Cross-vendor dispatch** — CUDA-capable GPUs accessible interchangeably through the glowplug daemon RPC, with zero privilege escalation
2. **5060 dual-use** — the RTX 5060 display GPU running CUDA compute without displacing its nvidia driver or disrupting DRM

---

## Part 1: Deep Debt Burndown (13 Items)

### P0 — Correctness Bugs (3 items)

**P0-1: TOCTOU Race in DeviceSlot (coral-glowplug)**

Long-running `spawn_blocking` tasks (oracle capture, compute dispatch) held borrowed references to VFIO mappings after releasing the device mutex. Concurrent `swap`, `reclaim`, or `resurrect` RPCs could invalidate the mapping mid-operation.

*Fix:* Added `busy: Arc<AtomicBool>` to `DeviceSlot` with RAII `BusyGuard`. `oracle_capture_async` and `compute_dispatch_async` acquire the guard before entering `spawn_blocking`. Mutating RPCs check `slot.is_busy()` and return an error if set.

*Files:* `coral-glowplug/src/device/mod.rs`, `coral-glowplug/src/socket.rs`

**P0-2: Silent Buffer Handle Drop (coral-driver CUDA)**

`CudaComputeDevice::dispatch_named` used `filter_map` on buffer handles, silently skipping invalid handles. A caller passing a stale handle would get silent data corruption instead of an error.

*Fix:* Changed to `.map(...).collect::<DriverResult<Vec<_>>>()?` — returns `DriverError::BufferNotFound` for any invalid handle.

*Files:* `coral-driver/src/cuda/mod.rs`

**P0-3: from_bdf_hint Fallback to Device 0 (coral-driver CUDA)**

When `from_bdf_hint` received a BDF that matched no CUDA device, it silently fell back to device 0. On a multi-GPU system this could dispatch work to the wrong card.

*Fix:* Returns `DriverError::OpenFailed` when no device matches the requested BDF. No silent fallback.

*Files:* `coral-driver/src/cuda/mod.rs`

### P1 — Robustness (4 items)

**P1-1: coralctl health Mismatch**

`coralctl health` parsed the daemon's `health.check` response incorrectly, looking for a `healthy` boolean instead of the actual `alive`, `device_count`, and `healthy_count` fields.

*Fix:* Updated `rpc_health()` to read all three fields and report HEALTHY/DEGRADED/DOWN accurately.

*Files:* `coral-glowplug/src/bin/coralctl.rs`

**P1-2: nvidia-smi Mutex Stall**

`device.compute_info`, `device.quota`, and `device.set_quota` RPCs executed `nvidia-smi` synchronously under the global device mutex. A slow `nvidia-smi` blocked all other RPCs.

*Fix:* Extracted into async handlers (`compute_info_async`, `quota_info_async`, `set_quota_async`) that acquire device data under lock, release the lock, then execute `nvidia-smi` via `tokio::task::spawn_blocking`.

*Files:* `coral-glowplug/src/socket.rs`

**P1-3: Bar0Rw Sentinel Values**

`Bar0Rw::read_u32` returned `0xDEAD_BEEF` for out-of-bounds reads with no way for callers to distinguish sentinel from a real register value (some GPU error registers contain `0xDEAD_BEEF`).

*Fix:* Added `try_read_u32() -> Result<u32, String>` and `try_write_u32() -> Result<(), String>` to both `Bar0Rw` and `Bar0Handle`. Explicit bounds checking with descriptive error messages. Critical for PMU debugging where every register value matters.

*Files:* `coral-driver/src/vfio/channel/mmu_oracle.rs`

**P1-4: Oracle String Errors**

Oracle functions returned `String` errors that couldn't be propagated through `DriverResult`. Callers had to manually wrap errors.

*Fix:* Added `DriverError::OracleError(Cow<'static, str>)` variant and `DriverError::oracle()` helper for `String -> DriverError` conversion.

*Files:* `coral-driver/src/error.rs`

### P2 — Quality (6 items)

**P2-1: Debug Derives**

`CudaComputeDevice` and `Bar0Handle` lacked `Debug`, making `{:?}` formatting impossible in logs and test assertions.

*Fix:* Manual `impl Debug` for both types (inner types like `CudaSlice` don't implement `Debug`).

*Files:* `coral-driver/src/cuda/mod.rs`, `coral-driver/src/vfio/channel/mmu_oracle.rs`

**P2-2: Dead Code**

Unused `PRAMIN_WINDOW_SIZE` constant in `mmu_oracle.rs`.

*Fix:* Removed.

**P2-3: Doc Drift**

`coral-driver/src/lib.rs` crate documentation claimed "No FFI" but the CUDA backend uses `cudarc` (which wraps libcuda). Architecture diagram omitted the CUDA backend.

*Fix:* Updated docs to acknowledge CUDA FFI, added CUDA backend to architecture overview.

**P2-4: BufReader Sizing**

`BufReader` in `handle_client_stream` was initialized at `MAX_REQUEST_LINE_BYTES` (4MB) per connection — excessive for typical JSON-RPC messages.

*Fix:* Added `INITIAL_BUF_CAPACITY: usize = 64 * 1024` (64KB). `BufReader::with_capacity` uses this smaller initial size; dynamic growth handles outliers.

*Files:* `coral-glowplug/src/socket.rs`

**P2-5: Optional Dependencies (hotspring-barracuda)**

`cudarc` and `base64` were unconditional dependencies in `Cargo.toml`, pulling in CUDA headers on systems without CUDA.

*Fix:* Made both optional behind a `cuda-validation` feature. Added `required-features = ["cuda-validation"]` to `validate_5060_dual_use` and `validate_cross_vendor_dispatch` binaries.

*Files:* `barracuda/Cargo.toml`

**P2-6: saxpy.ptx Compatibility**

`saxpy.ptx` targeted `.version 8.5` / `.target sm_90` (Hopper+), incompatible with Volta (sm_70).

*Fix:* Changed to `.version 7.0` / `.target sm_70`. Validated on both Titan V (sm_70) and RTX 5060 (sm_120, JIT-compiles from sm_70). Added clear fallback warning in `validate_5060_dual_use.rs`.

*Files:* `barracuda/src/bin/saxpy.ptx`, `barracuda/src/bin/validate_5060_dual_use.rs`

---

## Part 2: Cross-Vendor Dispatch Validation

### Method

`validate_cross_vendor_dispatch` connects to the glowplug daemon via Unix socket, enumerates managed CUDA-capable devices via `device.list` RPC, and dispatches a SAXPY kernel to each:

1. `device.list` → enumerate managed devices
2. For each CUDA-capable device:
   - Build PTX kernel (sm_70 target for Volta+ compatibility)
   - Encode kernel + buffers as base64
   - `device.dispatch` RPC with kernel, grid/block dims, buffer descriptors
   - Read back results, verify element-wise correctness

### Results

All CUDA-capable devices in the managed fleet executed SAXPY correctly through the daemon pipeline. Zero privilege escalation required — the validation binary runs as a regular user, all GPU access flows through the `coral-glowplug` daemon.

### Architecture

```
User binary (no privileges)
  │ Unix socket
  ▼
coral-glowplug (systemd, CAP_SYS_ADMIN)
  │ device.dispatch RPC
  ▼
coral-driver CUDA backend (cudarc → libcuda)
  │ PTX JIT → native ISA
  ▼
GPU hardware (any CUDA-capable device)
```

---

## Part 3: RTX 5060 Dual-Use (Display + Compute)

### Method

`validate_5060_dual_use` uses `cudarc` directly (not through the daemon) to run SAXPY on the RTX 5060 while it serves as the active display GPU:

1. Enumerate CUDA devices, find the 5060 by name ("5060" or "GB206")
2. Load pre-compiled PTX (sm_70 target, JIT to sm_120)
3. Allocate 1M f32 elements on device, upload x and y vectors
4. Launch SAXPY kernel (4096 blocks × 256 threads)
5. Read back results, verify correctness

### Results

SAXPY kernel executes correctly on the RTX 5060 while it drives the display. The nvidia driver handles both DRM/display and CUDA compute concurrently — no driver swap, no disruption. This proves the 5060 can serve as both the display head and a compute oracle for page table reference captures.

### Implications for PMU Cracking

The 5060's nvidia driver sets up page tables for every CUDA allocation. By capturing BAR0 state before and after a CUDA kernel launch, we can observe the exact PDE/PTE bit patterns the driver writes, then replicate them on the Titan V's sovereign VFIO path.

---

## Part 4: pkexec-Free Pipeline Validation

The entire compute pipeline — daemon startup, device enumeration, driver swap, oracle capture, compute dispatch — operates without `pkexec`:

| Operation | Mechanism | Privilege |
|-----------|-----------|-----------|
| Device enumeration | glowplug daemon `device.list` RPC | User (via Unix socket) |
| Driver swap | ember daemon sysfs writes | systemd service (CAP_SYS_ADMIN) |
| VFIO access | ember fd-pass via SCM_RIGHTS | systemd service (CAP_SYS_ADMIN) |
| Oracle capture | glowplug `oracle.capture` RPC | User (via Unix socket) |
| Compute dispatch | glowplug `device.dispatch` RPC | User (via Unix socket) |
| Health check | `coralctl health` (Unix socket) | User |

The systemd services hold the necessary capabilities; user-facing tools communicate via Unix socket RPC. No `pkexec`, no `sudo`, no SUID binaries.

---

## Part 5: PMU Cracking Readiness Assessment

### What This Sprint Enables

| Capability | Deep-Debt Item | PMU Value |
|-----------|---------------|-----------|
| `try_read_u32` / `try_write_u32` | P1-3 | Safe BAR0 register reads with bounds checking — no more sentinel confusion during page table debugging |
| `OracleError` | P1-4 | Clean error propagation from oracle capture — page table dumps report specific failures |
| `BusyGuard` | P0-1 | Safe concurrent oracle capture on both Titans without TOCTOU risk |
| BDF-specific dispatch | P0-3 | Dispatch to specific Titan by BDF — no accidental cross-card operations |
| 5060 dual-use | New | Live page table reference capture from working nvidia driver |
| Cross-vendor dispatch | New | Dispatch identical kernels to multiple GPUs for A/B comparison |

### Remaining Work: Layer 6 MMU Page Table Translation

The single blocker for sovereign command submission remains the GPU MMU page table translation. The PBDMA successfully loads channel context (GP_BASE, USERD, SIG, GP_PUT) but returns `0xbad00200` (PBUS timeout) when fetching GPFIFO entries through the FBHUB MMU.

**Attack matrix (dual Titan V + 5060 oracle):**

1. **5060 Oracle Capture**: Use `bar0_rw.try_read_u32` to read the 5060's page table registers during a CUDA allocation, capturing the exact PDE/PTE bit patterns the nvidia driver writes
2. **Diff Analysis**: Compare 5060 PTE encoding with our current V2 format encoding on the Titans
3. **Dual Titan A/B**: Run different PTE encoding strategies simultaneously on the two Titans — one tries the 5060-captured format, the other tries variations
4. **BAR2-Resident Tables**: Test whether PBDMA requires page tables in VRAM (via BAR2) rather than system memory
5. **MMU Fault Buffer**: Decode fault entries for specific VA/aperture/reason codes
6. **Tesla P80 (waiting)**: Third Volta data point with different PCI topology when it arrives

### Suspected Root Causes (Prioritized)

1. **IOMMU mapping gap**: Page table DMA buffers (IOVAs 0x5000-0x9000) may not be fully IOMMU-mapped for GPU-side access
2. **BAR2 requirement**: FBHUB MMU may access page tables through BAR2 (VRAM aperture), not directly through system memory bus
3. **PTE aperture mismatch**: SYS_MEM_COHERENT target (aperture=2) may need SYS_MEM_NONCOH (aperture=3) or VRAM (aperture=0) for the page table chain itself
4. **Instance block format**: The page directory base pointer in RAMIN may need additional flags for the iommufd/cdev path

---

## Lessons Learned

1. **TOCTOU races are invisible until they aren't** — the BusyGuard pattern should be standard for any operation that holds borrowed hardware state across an async boundary
2. **Silent fallbacks poison multi-GPU systems** — device 0 fallback and buffer handle skipping caused no visible failures during single-GPU testing but would silently corrupt results on the biomeGate fleet
3. **Sentinel values are anti-patterns for MMIO** — GPU error registers can contain any value; `Result` types are the only safe API for BAR0 access
4. **nvidia-smi is slow and unpredictable** — holding a mutex while shelling out to an external process is a latency bomb; always `spawn_blocking` with lock release
5. **Optional dependencies save the build matrix** — `cudarc` pulls in CUDA headers; gating behind a feature flag keeps `cargo check` fast on non-CUDA systems
6. **PTX sm_70 is the universal NVIDIA baseline** — JIT compilation handles upward compatibility (sm_70 → sm_120 on Blackwell); targeting sm_90+ excludes Volta

## Files Modified (coralReef)

- `coral-glowplug/src/device/mod.rs` — BusyGuard, busy flag
- `coral-glowplug/src/socket.rs` — TOCTOU checks, async nvidia-smi handlers, BufReader sizing
- `coral-glowplug/src/bin/coralctl.rs` — health parse fix
- `coral-driver/src/cuda/mod.rs` — buffer handle drop, BDF fallback, Debug
- `coral-driver/src/vfio/channel/mmu_oracle.rs` — try_read/write_u32, Debug, dead code
- `coral-driver/src/error.rs` — OracleError variant
- `coral-driver/src/lib.rs` — doc drift fix

## Files Modified (hotSpring)

- `barracuda/Cargo.toml` — cuda-validation feature, optional deps
- `barracuda/src/bin/saxpy.ptx` — sm_70 target
- `barracuda/src/bin/validate_5060_dual_use.rs` — new validation binary
- `barracuda/src/bin/validate_cross_vendor_dispatch.rs` — new validation binary
