# Experiment 159: Titan V VM-POST for HBM2 Training

## Date: 2026-04-07

## Hypothesis
Boot the Titan V in a reagent VM with nvidia-535 VFIO passthrough.
nvidia-535 will train HBM2, and the training state will persist through
VFIO device release, enabling sovereign SEC2/FECS pipeline from the host.

## Background
- exp158 showed SEC2 ACR bootloader runs but stalls on DMA (HBM2 not trained)
- VBIOS DEVINIT parsing (BIT table, 6 scripts found) too complex for direct replay
- PMU firmware not in linux-firmware for GV100; PMU DEVINIT embedded in VBIOS
- K80 exp157 proved direct DEVINIT replay can brick GPUs (PLL reprogramming)
- Pre-built `reagent-titanv-nvidia535.qcow2` VM image available

## Experiments

### Round 1: QEMU default (FLR, no VBIOS)
- `qemu-system-x86_64 -device vfio-pci,host=0000:03:00.0`
- GPU detected in VM: `[10de:1d81]` at 00:04.0
- No nvidia driver loaded (DKMS not built for kernel 6.8.0-106-generic)
- FLR on attach, FLR on release
- **Result**: VRAM dead (0xbad0ac00 sequential fault pattern)
- PMC_ENABLE: 0x42001120 → 0x40000020 (FLR reset)

### Round 2: QEMU with VBIOS ROM file
- Added `romfile=/tmp/titanv_vbios.rom,x-vga=on`
- SeaBIOS executed VBIOS: `Video device with shadowed ROM at [mem 0x000c0000-0x000dffff]`
- BARs properly mapped (BAR0 16MB, BAR1 256MB, BAR3 32MB)
- Still no nvidia driver
- **Result**: VRAM dead — VBIOS option ROM (x86 real-mode) only sets up VGA display,
  NOT HBM2. The BootROM handles memory training at power-on, not the option ROM.

### Round 3: QEMU with cloud-init auto-nvidia build
- Custom cloud-init seed ISO with `runcmd` to build and load nvidia-535
- nvidia-535.230.02 built with DKMS and loaded successfully
- **nvidia-smi output**:
  ```
  NVIDIA TITAN V  | 00:04.0 | 29% 44C P0 36W/250W | 0MiB/12288MiB | 0% Default
  ```
- **12288MiB (12GB) HBM2 detected and accessible inside VM!**
- nvidia-535 trained HBM2, initialized all GPU subsystems

### Post-VM Sovereign Reclaim
- Shut down VM (QEMU issues FLR on VFIO device release)
- Restarted ember + glowplug
- **VRAM: DEAD** — FLR killed all HBM2 training state
  - PRAMIN: 0xbad0ac00..07 (sequential PRI fault)
  - PFB_NISO_CFG0: 0xbadf5040 (PRI fault)
  - FBPA0: 0xbadf3000 (PRI fault)
  - PMC_ENABLE: 0x40000020 (FLR minimal state)
- Re-enabling PMC_ENABLE to 0x42001120 did NOT help — PFB/VRAM still dead
- Write/read test: wrote 0xDEADBEEF, read back 0xbad0ac0e (fault)

## Key Finding: FLR Kills HBM2

VFIO's Function-Level Reset (FLR), issued by the kernel when the VFIO device fd
is closed, completely resets the GV100 memory controller. HBM2 training state does
NOT survive VFIO device release.

`/sys/bus/pci/devices/0000:03:00.0/reset_method` is empty on kernel 6.17.9,
but the VFIO driver still performs a reset on fd close.

## Architecture for Solving the FLR Problem

### Option A: Ember FdVault Bridge (Preferred)
1. Ember opens VFIO device fd via `/dev/vfio/devices/vfio0`
2. Ember passes the fd to QEMU process via SCM_RIGHTS
3. QEMU uses the fd for GPU passthrough
4. When VM shuts down, QEMU closes its copy of the fd
5. Ember still holds the fd → kernel does NOT reset the device
6. Host-side sovereign access resumes through ember

Requires: Custom QEMU launcher or ember-integrated VM orchestration.

### Option B: In-VM Sovereign Pipeline
1. Boot VM with nvidia-535, let it initialize GPU
2. Unload nvidia-535 inside the VM (no FLR)
3. Load sovereign tools (Rust binaries via 9p/virtio-fs)
4. Run SEC2/FECS/compute pipeline inside the VM
5. Export results via shared filesystem

Requires: Cross-compiling sovereign tools for VM environment.

### Option C: IOMMUFD Shared Device (Kernel 6.17+)
Use IOMMUFD's device-centric API to share a VFIO device between
ember (host) and QEMU (VM) without exclusive group ownership.
Ember holds the fd continuously; QEMU gets a derived fd.

Requires: IOMMUFD support in QEMU (experimental in 6.2).

### Option D: Prevent FLR via Kernel Module
Patch `vfio-pci` to skip `pci_reset_function()` on release for
configured devices. Controlled via module parameter or sysfs.

Requires: Custom kernel module build.

## Files
- `barracuda/src/bin/exp159_titanv_vm_post.rs` — (not created, orchestrated via shell)
- `data/titanv/gv100_vbios_pg500.rom` — Titan V VBIOS dump (130KB)
- `/var/lib/libvirt/images/reagent-titanv-nvidia535.qcow2` — Pre-built reagent VM

## In-VM Post-Unload Analysis (continued)

### HBM2 Survives `rmmod nvidia` Inside VM
After nvidia-535 loads and initializes GPU (nvidia-smi shows 12GB HBM2),
`rmmod nvidia` was executed. Post-unload MMIO probe showed:

| Register        | Value      | Status |
|----------------|------------|--------|
| BOOT0          | 0x140000A1 | OK     |
| PMC_ENABLE     | 0x42001120 | OK     |
| PFB_NISO_CFG0  | 0xFFE00000 | OK     |
| FBPA0_STATUS   | 0x00000043 | OK     |
| PRAMIN_0       | 0x00000000 | OK     |
| SEC2_SCTL      | 0xBADF1100 | FAULT  |
| FECS_SCTL      | 0x00000000 | OK     |
| PGRAPH_STATUS  | 0x00000081 | OK     |
| PRAMIN W/R     | DEADBEEF   | ALIVE  |

**Key result**: HBM2 persists after nvidia unload (no FLR inside VM).
Only SEC2 is faulted (power-gated, not used by nvidia-535).

### FECS IMEM Hardware Lock Confirmed
- Direct IMEMC0/IMEMD0 writes: silently dropped (readback = 0)
- DMA from VRAM: times out (DMATRFCMD trigger never clears)
- PGRAPH reset via PMC_ENABLE toggle: no effect
- Falcon HRESET: no effect
- All access control registers (PRIV_CTRL, SCTL) show 0x0

**Conclusion**: Volta FECS has silicon-enforced IMEM protection.
Only the internal BootROM (started via CPUCTL_ALIAS) can write to IMEM
through its private DMA path.

### NVDEC IMEM IS Writable
mmiotrace of nvidia-535 shows NVDEC Falcon accepts direct IMEMC writes
at 0x084180/0x084184. This makes NVDEC a viable target for sovereign
falcon firmware on Volta.

## Next Steps
1. **Exp160**: Full mmiotrace capture completed (1.16M MMIO ops archived)
2. **Titan V**: Investigate FECS BootROM method replay (need CPUCTL=0x50 state)
3. **Titan V**: NVDEC as sovereign falcon target (IMEM writable)
4. **K80**: Power cycle + nvidia-470 VM POST (Kepler has writable FECS)
5. **Architecture**: Ember FdVault Bridge for HBM2 persistence across VFIO
