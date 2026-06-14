#!/usr/bin/env python3
"""Compare GR register state between nouveau-warm and post-boot-solver.

Scans BAR0 0x400000-0x420000 (PGRAPH/FECS/GPCCS space) via sysfs mmap.
Run twice with different labels, then compare.
"""
import mmap
import struct
import sys
import json

BDF = sys.argv[1] if len(sys.argv) > 1 else "0000:03:00.0"
STATE = sys.argv[2] if len(sys.argv) > 2 else "capture"
BAR0 = f"/sys/bus/pci/devices/{BDF}/resource0"

RANGES = [
    (0x400000, 0x420000, "PGRAPH+FECS+GPCCS"),
    (0x500000, 0x510000, "GPC0"),
]

try:
    with open(BAR0, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0x800000, mmap.MAP_SHARED, mmap.PROT_READ)

        regs = {}
        for start, end, label in RANGES:
            for off in range(start, end, 4):
                try:
                    val = struct.unpack_from("<I", mm, off)[0]
                    if val != 0 and val != 0xBADF5040 and val != 0xBAD00100:
                        regs[f"0x{off:06x}"] = val
                except Exception:
                    pass

        mm.close()

        outfile = f"/tmp/gr_full_{STATE}.json"
        with open(outfile, "w") as jf:
            json.dump(regs, jf, sort_keys=True)
        print(f"{STATE}: {len(regs)} non-zero registers → {outfile}")

        if STATE == "compare":
            for name in ["nouveau-warm", "vfio-postboot"]:
                try:
                    with open(f"/tmp/gr_full_{name}.json") as jf:
                        pass
                except FileNotFoundError:
                    print(f"  Missing /tmp/gr_full_{name}.json — capture both states first")

            try:
                with open("/tmp/gr_full_nouveau-warm.json") as f1:
                    s1 = json.load(f1)
                with open("/tmp/gr_full_vfio-postboot.json") as f2:
                    s2 = json.load(f2)
                
                all_keys = sorted(set(s1.keys()) | set(s2.keys()))
                diffs = []
                only_nouveau = []
                only_vfio = []
                for k in all_keys:
                    v1 = s1.get(k)
                    v2 = s2.get(k)
                    if v1 is not None and v2 is not None:
                        if v1 != v2:
                            diffs.append((k, v1, v2))
                    elif v1 is not None:
                        only_nouveau.append((k, v1))
                    else:
                        only_vfio.append((k, v2))
                
                print(f"\n=== Comparison ===")
                print(f"Nouveau: {len(s1)} regs, VFIO: {len(s2)} regs")
                print(f"Changed: {len(diffs)}, Only-nouveau: {len(only_nouveau)}, Only-vfio: {len(only_vfio)}")
                
                if diffs:
                    print(f"\nChanged registers ({len(diffs)}):")
                    for k, v1, v2 in diffs[:50]:
                        print(f"  {k}: {v1:#010x} → {v2:#010x}")
                    if len(diffs) > 50:
                        print(f"  ... and {len(diffs)-50} more")
                
                if only_nouveau:
                    print(f"\nOnly in nouveau ({len(only_nouveau)}):")
                    for k, v in only_nouveau[:30]:
                        print(f"  {k} = {v:#010x}")
                    if len(only_nouveau) > 30:
                        print(f"  ... and {len(only_nouveau)-30} more")
                
            except Exception as e:
                print(f"  Compare error: {e}")

except PermissionError:
    print(f"Need root: sudo {sys.argv[0]} {BDF} {STATE}")
    sys.exit(1)
