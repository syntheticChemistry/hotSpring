# Experiment 188: K80 Warm-Catch Breakthrough

**Date:** 2026-05-10
**GPU:** Tesla K80 (GK210GL, BDF 0000:4b:00.0)
**Kernel:** 6.17.9-76061709-generic
**Status:** Major breakthrough — nouveau recognized GK210, GR initialized. Warm-catch sequence needs PLX keepalive fix.

## Breakthrough Result

The patched `nouveau.ko` (binary patch: `cmp $0xf1` → `cmp $0xf2` at offset `b76f8`)
**successfully initialized the K80 GK210:**

```
nouveau 0000:4b:00.0: NVIDIA GK110B (0f22d0a1)
nouveau 0000:4b:00.0: bios: version 80.21.1b.00.01
nouveau 0000:4b:00.0: fb: 12288 MiB GDDR5
nouveau 0000:4b:00.0: drm: VRAM: 12288 MiB
nouveau 0000:4b:00.0: drm: GART: 1048576 MiB
[drm] Initialized nouveau 1.4.0 for 0000:4b:00.0 on minor 1
```

**This is the first time nouveau has ever initialized the GK210 on this system.**

DRM device created (card1, renderD129), pstate shows valid clock configs
(core 324-875 MHz, memory 648-5010 MHz).

## Post-Rebind State

After unbinding nouveau and rebinding to vfio-pci, ember BAR0 reads showed:

| Register | Address | Value | Meaning |
|----------|---------|-------|---------|
| BOOT0 | 0x000000 | 0x0f22d0a1 | GK210 alive |
| PMC_ENABLE | 0x000200 | 0xfc37b1ef | Many engines enabled |
| PRI GPC count | 0x022430 | **0x00000005** | **5 GPCs enrolled!** |
| TPC per GPC | 0x022438 | 0x00000006 | 6 TPCs per GPC |
| FECS method | 0x409604 | 0x00060005 | 6 TPC, 5 GPC topology |
| GPC0 space | 0x418880 | 0xbadf1002 | GPC power-gated |
| GPC space | 0x500000 | 0xbadf1100 | PRI error — stations down |

**Key insight**: The PRI ring knows about 5 GPCs and 6 TPCs (correct K80 topology),
but the GPC stations are power-gated after the nouveau→VFIO transition. The
livepatch that should have preserved GPC power failed to load ("Invalid module format"
due to kernel 6.17 strict relocation enforcement).

## Failure: PLX D3cold

On the second warm-catch attempt, stopping coral-ember (which does PLX keepalive)
caused the PLX switch to power-gate the K80 to D3cold:

```
nouveau 0000:4b:00.0: Unable to change power state from D3cold to D0, device inaccessible
nouveau 0000:4b:00.0: unknown chipset (ffffffff)
```

**Root cause**: Ember's PLX keepalive uses sysfs config space reads on the PLX switch
BDF — it does NOT need VFIO or the K80 device. But the warm-catch script stopped
ember entirely.

**Fix**: Keep ember running during the entire warm-catch. Ember's keepalive targets
the PLX switch (not the K80), so it can continue while nouveau operates the K80.

## Architecture Discovery

1. **Patched nouveau.ko already exists** on the system — someone (previous agent
   session?) already binary-patched the chipset comparison from `0xf1` to `0xf2`.
   Stock module at `nouveau.ko.stock` still has `0xf1`.

2. **Ember PCIe keepalive is independent of VFIO** — it reads/writes sysfs
   `/sys/bus/pci/devices/{bdf}/config` on the PLX switch BDF. Safe to run
   while nouveau has the K80.

3. **Livepatch needs rebuild for kernel 6.17** — the module relocation format
   changed. The livepatch compiled but kernel rejects it at insmod time.

## Next Steps

1. [ ] Power cycle to recover K80 from D3cold
2. [ ] Keep ember running during warm-catch (PLX keepalive continues)
3. [ ] Rebuild livepatch with corrected relocations for 6.17, or use kprobe-based approach
4. [ ] If livepatch works: capture GPC state BEFORE nouveau unbind, verify GPCs stay alive
5. [ ] If GPCs survive: run sovereign dispatch via ember after VFIO rebind

## Alternative: Capture GR State Without Warm-Catch

Even without preserving GPCs across the driver transition, we learned:
- nouveau CAN initialize GR on GK210 (5 GPCs, 6 TPCs per GPC, 30 TPCs total)
- The FECS firmware loads from VBIOS successfully
- The GR configuration (MMIO init table, CSDATA) is valid for GK210

We can capture the complete GR register state while nouveau is running,
then replay those exact registers in the sovereign cold boot path. This
is the "golden state capture" approach.

## References

- Exp 185: GK210 chipset analysis (root cause)
- Exp 186: PMU firmware source (VBIOS confirmed)
- GAP-HS-057: Nouveau chipset ID gap
