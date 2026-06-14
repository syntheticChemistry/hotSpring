# hotSpring Infrastructure

Local infrastructure artifacts for GPU sovereignty research.

## Directory Structure

### `catalysts/`

Versioned sovereign boot artifacts for the diesel engine pipeline.

- **`recipes/`** — TOML recipes describing how to produce patched kernel modules from DKMS source. Each recipe specifies patch strategies, objcopy offsets, module renames (`nvidia` → `nvsov`), and settle times.
  - `gv100_nvidia470.toml` — Volta/GV100 catalyst recipe for nvidia-470.256.02

- **`reagents/`** — Captured "chemical agents": manifests, ACR register recipes, and pointers to runtime blobs under `/var/lib/toadstool/reagents/`.
  - `gv100_acr_catalyst.json` — ACR boot register recipe (historical v1; superseded by Boot Falcon path in Exp 224+)
  - `gv100_nvidia47025602_k6.17.9/manifest.json` — Reagent manifest for nvidia-470 on kernel 6.17.9

### `golden_state/`

Canonical BAR0 register-write replay sequences for sovereign cold boot.

- `gv100_catalyst_replay.json` — Domain-tagged PMC/PBUS write sequence (~485K lines). Validated via exp170/exp227.

## Ecosystem Tools (external)

These tools live in `ecoPrimals/infra/`, not in this spring:

### `agentReagents/`

Disposable VM "reagent lab" for running hazardous drivers (nvidia-470, patched nouveau) with VFIO GPU passthrough. Captures mmiotrace, BAR0 snapshots, and firmware while vendor drivers are loaded.

- **Capture command**: `toadstool device oracle reagent-capture --bdf <BDF>`
- **Templates**: `reagent-nvidia470-titanv.yaml`, `reagent-nouveau-k80.yaml`
- **Build tools**: `build_nvidia470_kernel617.sh`, `patch_nouveau_teardown.py`

hotSpring consumes agentReagents **outputs** (JSON dumps, mmiotrace) via binaries like `exp156_reagent_compare`, not the VM tooling itself.

### `benchScale/`

Libvirt/QEMU VM orchestration for multi-substrate validation. VFIO passthrough with options like `no_flr` to preserve GPU state across VM shutdown.

- **Warm-handoff automation**: spin VM → load nvidia-470 → train GDDR5/FECS → rebind VFIO → sovereign dispatch
- **CI**: `benchScale/scripts/validate-hotspring-multi.sh`

## Hardware Fleet

| GPU | Role | Driver | Protection |
|-----|------|--------|------------|
| RTX 5060 | Host display + deployment parity | nvidia (proprietary) | `GlowplugDeviceDetail.protected` — swap-immune |
| Titan V A (0000:02:00.0) | Sovereignty target (VFIO) | vfio-pci | Tier 2 HW VALIDATED (Exp 227) |
| Titan V B (0000:49:00.0) | Sovereignty target (VFIO) | vfio-pci | Tier 2 HW VALIDATED (Exp 227) |
| K80 Tesla | Cross-gen target (swap-in) | -- | Exp 231 awaiting hardware |

## Diesel Engine Architecture

The diesel engine is toadStool's driver lifecycle management system:

- **glowplug**: Fleet orchestrator — device.swap, driver rotation, catalyst recipes
- **ember**: Per-GPU sacrificial MMIO gateway — fork-isolated BAR0, circuit breaker
- **cylinder**: Core VFIO/sovereign logic — catalyst pipeline, module patches, register quench

hotSpring interacts with the diesel engine via JSON-RPC over NUCLEUS Unix sockets. See `barracuda/src/glowplug_client/`, `barracuda/src/fleet_client.rs`, `barracuda/src/fleet_ember.rs`.

## Lockup Protection

Crash vector catalog (Exp 229/232) with 6 confirmed kills (A1-A6) and 3 confirmed hangs (B1-B3). All defenses implemented in toadStool; hotSpring validates via `validate_lockup_defense_matrix`.

See `barracuda/src/bin_helpers/sovereignty/lockup_vectors.rs` for the full catalog.
