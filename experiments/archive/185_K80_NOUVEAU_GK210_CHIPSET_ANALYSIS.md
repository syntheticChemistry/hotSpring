# Experiment 185: K80 Nouveau GK210 Chipset Analysis

**Date:** 2026-05-10
**GPU:** Tesla K80 (GK210GL, PCI ID 10de:102d)
**Kernel:** 6.17.9-76061709-generic
**Status:** Investigation complete. Root cause confirmed. Actionable path identified.

## Objective

Determine why nouveau never initializes GR on the K80 (GK210), and whether
a kernel patch or older kernel can provide live GPCs for the warm-catch
sovereign pipeline.

## Findings

### BOOT0 and chip ID

K80 BOOT0 reads `0x0f22d0a1`, which decodes to chip ID `0xf2` (GK210).

### Upstream nouveau chipset table (kernel 6.17, verified against torvalds/linux master)

The `nvkm_device_chip_init()` switch in `drivers/gpu/drm/nouveau/nvkm/engine/device/base.c`
has entries for:

- `case 0x0f0:` → `nvf0_chipset` (GK110, `.gr = gk110_gr_new`)
- `case 0x0f1:` → `nvf1_chipset` (GK110B, `.gr = gk110b_gr_new`)
- **NO `case 0x0f2:`** — GK210 is not recognized

When chip ID `0xf2` hits the switch, it falls through to:

```c
if (!device->chip) {
    nvdev_error(device, "unknown chipset (%08x)\n", boot0);
    ret = -ENODEV;
    goto done;
}
```

Nouveau prints "unknown chipset (0f22d0a1)" and returns -ENODEV. **No subdevices
are initialized**, including GR, PMU, FIFO, privring, or any engine. The module
probe succeeds at the DRM/PCI level but the GPU is effectively inert.

### Consequence for warm-catch pipeline

Since nouveau never initializes GR, the warm-catch path (nouveau bind → GR init →
FECS firmware load → GPC enrollment → VFIO rebind) cannot work on the K80.
There is no warm GR state to catch — nouveau skips the GPU entirely.

### The `nvf1_chipset` compatibility hypothesis

GK210 is architecturally identical to GK110B with additional VRAM addressing bits.
The `nvf1_chipset` definition uses `gk110b_gr_new` which handles the same
GPC/TPC/SM topology. The known patch:

```c
case 0x0f2: device->chip = &nvf1_chipset; break;
```

would map GK210 to GK110B initialization code. This was discussed on the
nouveau mailing list (April 2024) but never merged upstream. Ilia Mirkin
(nouveau maintainer) noted it is "speculative" and GK210 may have differences
requiring extracted firmware that has never been done for this chipset.

### Hardware differences (GK110B vs GK210)

- GK210 has 24 GiB GDDR5 (vs 12 GiB on GK110B)
- GK210 has wider VRAM bus (384-bit)
- FB (framebuffer controller) may have different channel/partition config
- PMU firmware may have GK210-specific VBIOS tables

Risk: Even with the chipset patch, FB/memory training or PMU operations may
fail on GK210 due to unhandled address widths.

### Patched nouveau.ko on this system

A patched `nouveau.ko` exists alongside `nouveau.ko.stock` (6235376 vs 6236097 bytes).
The patch appears to be the livepatch (NOP fini functions for warm handoff),
not a chipset ID fix. The K80s are bound to vfio-pci at boot via kernel cmdline,
so nouveau never probes them regardless.

## Conclusions

1. **The K80 GPC PGOB blocker is NOT a power gating issue alone** — it is
   fundamentally that nouveau cannot even recognize the chip, so the warm-catch
   path that would initialize GPCs never runs.

2. **The cold sovereign path (VFIO-only, no nouveau)** must solve GPC enrollment
   without nouveau's help. This means either:
   - Running PMU firmware to do the proper PSW + PRI ring enrollment
   - Directly programming the GPC topology registers (risky, undocumented)

3. **Adding `case 0x0f2: device->chip = &nvf1_chipset;` to the kernel** would
   enable the warm-catch path: nouveau would init GR (including PGOB ungate via
   PMU), then we rebind to VFIO with live GPCs. This is the fastest path to
   unblock K80 compute.

## Next Steps

- [ ] Build nouveau with the `0xf2` → `nvf1_chipset` patch and test
- [ ] If nouveau GR comes up: validate GPC count, FECS state, PFIFO via warm-catch
- [ ] If FB/memory fails: investigate GK210-specific FB differences
- [ ] In parallel: continue Kepler PMU firmware extraction for the cold path

## References

- Exp 171: K80 sovereign init (BOOT0=0x0f22d0a1 confirmed)
- Exp 174: K80 sovereign boot (chip_id 0x0F2 → SM 35 mapping)
- Exp 178: K80 PGOB nvidia-470 analysis
- Exp 179: K80 warm FECS dispatch pipeline
- Exp 181: Sovereign dispatch sweep (chip 0xf2 gap noted)
- nouveau mailing list April 2024: Ilia Mirkin on GK210 support
