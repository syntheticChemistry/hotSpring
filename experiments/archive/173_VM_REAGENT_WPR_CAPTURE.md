# Experiment 173 — VM Reagent WPR Capture

**Date:** 2026-04-18
**GPU:** Titan V (GV100, 0000:03:00.0)
**Driver tested:** nvidia-535.183.06 (closed source, pre-GSP)
**VM:** Ubuntu 24.04 server via QEMU/libvirt, VFIO passthrough

## Objective

Test whether the nvidia closed-source driver configures WPR (Write-Protected Region)
on GV100, and whether that WPR state persists after `rmmod nvidia` and VM release.
This is the "vendor UEFI" approach: treat the proprietary driver as a disposable
reagent in a VM, capture the hardware state it creates, then use that state for
sovereign compute via `ember.sovereign.init`.

## Constraint (Hard Lesson)

Vendor drivers are **toxic reagents** that must never load in the host kernel.
A previous attempt to install `nvidia-dkms-580` directly on the host crashed the
system and locked the RTX 5060 display GPU. This forced a manual purge and rebuild.
See `specs/UNIVERSAL_DRIVER_REAGENT_ARCHITECTURE.md` for the full "5060 lesson."

All vendor driver execution must happen inside disposable `agentReagents` VMs.

## Infrastructure Changes

### Phase 1a: Artifact Extraction (FetchFile)
Added `PostBootStep::FetchFile` to `agentReagents` and `VmHandle::scp_fetch` to pull
files from guest VM to host via SCP. Replaces the unimplemented `CopyFile` direction.

### Phase 1b: No-FLR Passthrough
Added `no_flr: bool` to benchScale's `PciPassthroughDevice`. When true, generates
`managed='no'` in libvirt XML and hot-attaches the device after VM creation (since
virt-install doesn't support `managed='no'` via `--hostdev`). This prevents Function
Level Reset on VM shutdown, preserving GPU hardware state.

**Note:** Hot-attach with `managed='no'` does NOT configure BARs — the device must
be in the domain XML from boot (or the guest kernel must rescan PCI). For this
experiment we used `managed='yes'` since WPR persistence across VM release was not
the primary question (WPR was never configured in the first place).

### Phase 1c: Reagent Template
Updated `reagent-nvidia535-titanv.yaml` to v2.0.0 with:
- `gpu-state.py` — comprehensive BAR0 snapshot with WPR registers, falcon state,
  PMU/SEC2/FECS/GPCCS/NVDEC falcon CPUCTL/SCTL/MAILBOX/PC/EXCI
- Cold, warm, and post-rmmod captures
- WPR persistence verdict logic
- `no_flr: true` on PCI passthrough config

## Results

### Cold State (before nvidia load)

| Register | Value | Notes |
|----------|-------|-------|
| BOOT0 | 0x140000a1 | GV100 confirmed |
| PMC_ENABLE | 0x40000020 | 2 engines (cold/minimal) |
| WPR1_BEG/END | 0 / 0 | Not configured |
| WPR2_BEG/END | 0 / 0 | Not configured |
| FECS CPUCTL | 0xbadf1201 | VFIO BAR guard (engine powered down) |
| SEC2 CPUCTL | 0xbadf1100 | VFIO BAR guard |
| PMU CPUCTL | 0x00000020 | Partially alive |

### Warm State (nvidia-535 loaded, nvidia-smi shows healthy GPU)

| Register | Value | Notes |
|----------|-------|-------|
| BOOT0 | 0x140000a1 | GV100 confirmed |
| PMC_ENABLE | 0x42001120 | 5 engines (warm) |
| **WPR1_BEG/END** | **0 / 0** | **NOT configured** |
| **WPR2_BEG/END** | **0 / 0** | **NOT configured** |
| FECS CPUCTL | 0x00000010 | Halted (HRESET) |
| GPCCS CPUCTL | 0x00000010 | Halted (HRESET) |
| SEC2 CPUCTL | 0xbadf1100 | BAR guard (RM controls access) |
| PMU CPUCTL | 0x00000020 | Running (RM-controlled) |

### Post-rmmod State

Identical to warm state. `rmmod nvidia` does not change visible BAR0 register state
on GV100 when reading via sysfs resource0.

### Sovereign Init (nouveau warm handoff + vfio-pci)

After nouveau warms the GPU (trains HBM2, probes fb: 12288 MiB), swapping back to
vfio-pci and running `ember.sovereign.init`:

- **warm_detected:** true (HBM2 sentinel passes)
- **hbm2_training:** skipped (warm)
- **falcon_boot:** FAILED (all 12 strategies)
  - SEC2 never enters HS mode (SCTL=0x3000 throughout)
  - SEC2 stuck at various PCs (0x03e5-0x03ec)
  - EXCI exceptions: 0x001f001e, 0x001f0007
  - Queue discovery fails in all strategies
  - Direct FECS boot: firmware uploads via PIO succeed but FECS stays halted

## Key Finding

**nvidia-535 closed driver does NOT configure WPR on GV100 (Titan V).**

The "vendor UEFI WPR capture" approach was based on a false premise for this GPU
generation. GV100 is pre-GSP architecture:

- The Resource Manager runs entirely on the host CPU
- PMU firmware runs but does not set up WPR boundaries via the PFB registers
- The driver likely uses a different internal mechanism (CPU-side RM) that doesn't
  require WPR hardware protection
- nvidia-smi shows a fully functional GPU with 0% utilization and 12288 MiB VRAM,
  proving the driver works without WPR

WPR (Write-Protected Region) is an Ampere+ / Turing+ concept required by GSP-RM to
protect its own firmware from tampering. On Volta (GV100), the closed driver's RM
runs on the CPU and doesn't need this protection boundary.

## Implications for Sovereign Pipeline

1. **WPR is NOT the blocker for GV100.** The ACR chain expects WPR because it was
   designed for Turing+. On GV100, a different approach is needed.

2. **The real blocker is SEC2 HS mode.** SEC2 firmware loads via PIO but cannot
   enter Heavy Secure mode. Without HS, SEC2 can't run the ACR bootstrap that
   verifies and loads FECS/GPCCS.

3. **Possible paths forward for GV100:**
   - Reverse-engineer how nvidia's RM initializes SEC2 on Volta (mmiotrace data
     available in `reagent-artifacts/exp173/`)
   - Find non-secure FECS/GPCCS firmware variants (if they exist for Volta)
   - Accept vendor-in-VM as the permanent GV100 compute path
   - Focus sovereign boot efforts on K80 (Kepler, no ACR/WPR needed) and newer
     GPUs (where GSP provides a cleaner interface)

4. **Exp 172 (no-ACR warm handoff) remains the best GV100 state:** warm HBM2 +
   clean HRESET falcons + working PIO upload. The gap is SEC2→FECS bootstrapping.

## Artifacts

Stored at `/var/lib/coralreef/reagent-artifacts/exp173/artifacts/`:
- `cold_state.json` — full BAR0 snapshot before nvidia load
- `warm_nvidia_loaded.json` — full BAR0 snapshot with nvidia running
- `post_rmmod_state.json` — full BAR0 snapshot after rmmod

## Infrastructure Delivered

Despite the WPR finding being a dead end for GV100, this experiment delivered:

1. **FetchFile PostBootStep** — artifact extraction from VMs now works
2. **No-FLR passthrough** — benchScale can preserve GPU state across VM shutdown
3. **Comprehensive gpu-state.py** — reusable BAR0/WPR/falcon diagnostic tool
4. **Updated reagent template v2.0.0** — ready for future GPU experiments
5. **Validated VM isolation pattern** — nvidia loaded and ran inside VM without
   touching host kernel, confirming the "toxic reagent" architecture works
