#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Exp 086 — Falcon register profiler via sysfs BAR0 mmap.

Reads SEC2/FECS/GPCCS/PMU falcon registers + binding registers + PFB/WPR
registers directly from /sys/bus/pci/devices/<BDF>/resource0.
Works regardless of which driver is bound (vfio, nouveau, nvidia, none).

Usage:
    sudo python3 exp086_falcon_profiler.py <BDF> [output.json]

Example:
    sudo python3 exp086_falcon_profiler.py 0000:03:00.0 titan1_vfio_cold.json
"""

import ctypes
import json
import mmap
import os
import struct
import sys
import time
from pathlib import Path

BAR0_SIZE = 0x100_0000  # 16 MiB — GV100 BAR0

FALCON_ENGINES = {
    "SEC2":  0x087000,
    "FECS":  0x409000,
    "GPCCS": 0x41A000,
    "PMU":   0x10A000,
}

# Standard falcon registers (offset from engine base)
FALCON_REGS = {
    "IRQSSET":   0x000,
    "IRQSCLR":   0x004,
    "IRQSTAT":   0x008,
    "IRQMSET":   0x010,
    "IRQMCLR":   0x014,
    "TRACEPC":   0x030,
    "MAILBOX0":  0x040,
    "MAILBOX1":  0x044,
    "BIND_INST": 0x054,  # CHANNEL_NEXT — the B1 fix register
    "CHAN_TRIG":  0x058,  # CHANNEL_TRIGGER — B7 load trigger
    "OS":        0x080,
    "UNK090":    0x090,  # B5 trigger (bit 16)
    "ENG_CTRL":  0x0a4,  # B6 trigger (bit 3)
    "BIND_STAT": 0x0dc,  # bind_stat — bits[14:12] == 5 means bound
    "CPUCTL":    0x100,
    "BOOTVEC":   0x104,
    "HWCFG":     0x108,
    "DMACTL":    0x10C,
    "DMA_BASE":  0x110,
    "DMA_MOFFS": 0x114,
    "DMA_CMD":   0x118,
    "DMA_FBOFFS":0x11C,
    "EXCI":      0x148,
    "IMEMC":     0x180,
    "SCTL":      0x240,
    "DMAIDX":    0x604,  # B4 DMAIDX clear register
    "FBIF_624":  0x624,
    "FBIF_BIND": 0x668,  # legacy (wrong) bind_inst location — read for comparison
}

# PFB / memory controller registers for WPR region detection
PFB_REGS = {
    "PFB_PRI_MMU_CTRL":      0x100C80,
    "PFB_PRI_MMU_PHYS_CTRL": 0x100C94,
    "WPR1_BEG":              0x100CE4,
    "WPR1_END":              0x100CE8,
    "WPR2_BEG":              0x100CEC,
    "WPR2_END":              0x100CF0,
}

# PMC / top-level
PMC_REGS = {
    "BOOT0":          0x000000,
    "PMC_ENABLE":     0x000200,
    "PMC_DEV_ENABLE": 0x000204,
}

# GR / graphics engine (useful for context)
GR_REGS = {
    "GR_FECS_CG":  0x4098C8,  # FECS clock gating
    "GR_FECS_CG2": 0x4098CC,
}


def get_current_driver(bdf: str) -> str:
    driver_link = Path(f"/sys/bus/pci/devices/{bdf}/driver")
    if driver_link.is_symlink():
        return driver_link.resolve().name
    return "none"


def read_bar0_u32(mm: mmap.mmap, offset: int) -> int:
    """Read a single 32-bit register from mmap'd BAR0."""
    if offset + 4 > len(mm):
        return 0xDEADDEAD
    mm.seek(offset)
    raw = mm.read(4)
    return struct.unpack("<I", raw)[0]


def profile_falcon(mm: mmap.mmap, name: str, base: int) -> dict:
    """Read all falcon registers for one engine."""
    result = {"engine": name, "base": f"0x{base:06x}", "registers": {}}
    for reg_name, offset in sorted(FALCON_REGS.items(), key=lambda x: x[1]):
        addr = base + offset
        val = read_bar0_u32(mm, addr)
        result["registers"][reg_name] = {
            "offset": f"0x{offset:03x}",
            "addr": f"0x{addr:06x}",
            "raw": val,
            "hex": f"0x{val:08x}",
        }

    cpuctl = result["registers"]["CPUCTL"]["raw"]
    bind_stat_raw = result["registers"]["BIND_STAT"]["raw"]
    sctl = result["registers"]["SCTL"]["raw"]

    result["decoded"] = {
        "halted": bool(cpuctl & 0x20),
        "hreset": bool(cpuctl & 0x10),
        "startcpu_pending": bool(cpuctl & 0x01),
        "hs_mode": bool(sctl & 0x3000),
        "bind_stat_field": (bind_stat_raw >> 12) & 0x7,
        "bind_stat_raw": f"0x{bind_stat_raw:08x}",
    }
    return result


def profile_pfb(mm: mmap.mmap) -> dict:
    result = {}
    for name, addr in sorted(PFB_REGS.items(), key=lambda x: x[1]):
        val = read_bar0_u32(mm, addr)
        result[name] = {
            "addr": f"0x{addr:06x}",
            "raw": val,
            "hex": f"0x{val:08x}",
        }

    wpr1_beg = result["WPR1_BEG"]["raw"]
    wpr1_end = result["WPR1_END"]["raw"]
    wpr2_beg = result["WPR2_BEG"]["raw"]
    wpr2_end = result["WPR2_END"]["raw"]
    result["decoded"] = {
        "wpr1_active": wpr1_end > wpr1_beg,
        "wpr1_range": f"0x{wpr1_beg:08x}..0x{wpr1_end:08x}" if wpr1_end > wpr1_beg else "inactive",
        "wpr2_active": wpr2_end > wpr2_beg,
        "wpr2_range": f"0x{wpr2_beg:08x}..0x{wpr2_end:08x}" if wpr2_end > wpr2_beg else "inactive",
    }
    return result


def profile_pmc(mm: mmap.mmap) -> dict:
    result = {}
    for name, addr in sorted(PMC_REGS.items(), key=lambda x: x[1]):
        val = read_bar0_u32(mm, addr)
        result[name] = {
            "addr": f"0x{addr:06x}",
            "raw": val,
            "hex": f"0x{val:08x}",
        }

    pmc_en = result["PMC_ENABLE"]["raw"]
    result["decoded"] = {
        "sec2_enabled": bool(pmc_en & (1 << 22)),
        "fecs_gr_enabled": bool(pmc_en & (1 << 12)),
        "pmu_enabled": bool(pmc_en & (1 << 13)),
    }
    return result


def profile_gr(mm: mmap.mmap) -> dict:
    result = {}
    for name, addr in sorted(GR_REGS.items(), key=lambda x: x[1]):
        val = read_bar0_u32(mm, addr)
        result[name] = {"addr": f"0x{addr:06x}", "raw": val, "hex": f"0x{val:08x}"}
    return result


def main():
    if len(sys.argv) < 2:
        print(f"Usage: sudo {sys.argv[0]} <BDF> [output.json]", file=sys.stderr)
        sys.exit(1)

    bdf = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    resource0 = f"/sys/bus/pci/devices/{bdf}/resource0"
    if not os.path.exists(resource0):
        print(f"ERROR: {resource0} not found", file=sys.stderr)
        sys.exit(1)

    driver = get_current_driver(bdf)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print(f"=== Exp 086 Falcon Profiler ===")
    print(f"BDF:       {bdf}")
    print(f"Driver:    {driver}")
    print(f"Timestamp: {timestamp}")
    print()

    fd = os.open(resource0, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, BAR0_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
    except Exception as e:
        print(f"ERROR: mmap failed: {e}", file=sys.stderr)
        print("  (may need root, or driver may block BAR0 access)", file=sys.stderr)
        os.close(fd)
        sys.exit(1)

    profile = {
        "meta": {
            "bdf": bdf,
            "driver": driver,
            "timestamp": timestamp,
            "experiment": "086_cross_driver_falcon_profile",
        },
        "pmc": profile_pmc(mm),
        "pfb_wpr": profile_pfb(mm),
        "gr": profile_gr(mm),
        "falcons": {},
    }

    for engine_name, base in FALCON_ENGINES.items():
        falcon_data = profile_falcon(mm, engine_name, base)
        profile["falcons"][engine_name] = falcon_data

        cpuctl_hex = falcon_data["registers"]["CPUCTL"]["hex"]
        bind_stat = falcon_data["decoded"]["bind_stat_field"]
        sctl_hex = falcon_data["registers"]["SCTL"]["hex"]
        exci_hex = falcon_data["registers"]["EXCI"]["hex"]
        print(f"{engine_name:6s} @ 0x{base:06x}: cpuctl={cpuctl_hex} "
              f"bind_stat[14:12]={bind_stat} sctl={sctl_hex} exci={exci_hex}")

    wpr_dec = profile["pfb_wpr"]["decoded"]
    print(f"\nWPR1: {wpr_dec['wpr1_range']}")
    print(f"WPR2: {wpr_dec['wpr2_range']}")

    pmc_dec = profile["pmc"]["decoded"]
    print(f"\nPMC: SEC2={'ON' if pmc_dec['sec2_enabled'] else 'OFF'} "
          f"GR={'ON' if pmc_dec['fecs_gr_enabled'] else 'OFF'} "
          f"PMU={'ON' if pmc_dec['pmu_enabled'] else 'OFF'}")

    mm.close()
    os.close(fd)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"\nSaved: {out}")
    else:
        print(f"\n{json.dumps(profile, indent=2)}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
