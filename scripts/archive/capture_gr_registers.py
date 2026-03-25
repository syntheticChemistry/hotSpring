#!/usr/bin/env python3
"""Capture GR engine registers via BAR0 sysfs mmap.

Reads key PGRAPH / GR / falcon registers to compare nouveau-warm vs vfio-cold
state. Used to determine what GR MMIO init FECS needs.
"""
import mmap
import struct
import sys
import json

BDF = sys.argv[1] if len(sys.argv) > 1 else "0000:03:00.0"
BAR0 = f"/sys/bus/pci/devices/{BDF}/resource0"

# Key register ranges for GR engine
GR_REGS = {
    # PMC
    "PMC_ENABLE": 0x000200,
    "PMC_SUBDEVICE": 0x000208,
    # PGRAPH top-level
    "PGRAPH_STATUS": 0x400700,
    "PGRAPH_TRAPPED_ADDR": 0x400704,
    "PGRAPH_TRAPPED_DATA_LO": 0x400708,
    "PGRAPH_INTR": 0x400100,
    "PGRAPH_INTR_EN": 0x40013C,
    # GR_FE (Front End)
    "GR_FE_OBJECT": 0x404200,
    "GR_FE_MEM_BASE": 0x404204,
    "GR_FE_POWER_MODE": 0x404170,
    # GR configuration
    "GR_PRI_GPC0_GPC_CS": 0x502000,
    "GR_PRI_GPC0_TPC0_SM_ARCH": 0x504330,
    # FECS falcon
    "FECS_CPUCTL": 0x409100,
    "FECS_BOOTVEC": 0x409104,
    "FECS_HWCFG": 0x409108,
    "FECS_DMACTL": 0x40910C,
    "FECS_SCTL": 0x409240,
    "FECS_OS": 0x409080,
    "FECS_TRACEPC": 0x409030,
    "FECS_EXCI": 0x409148,
    "FECS_MAILBOX0": 0x409040,
    "FECS_MAILBOX1": 0x409044,
    "FECS_IRQSTAT": 0x409008,
    "FECS_IRQMASK": 0x409018,
    "FECS_DEBUG1": 0x409090,
    "FECS_CTXSW_MBOX0": 0x409800,
    "FECS_CTXSW_MBOX1": 0x409804,
    "FECS_CTXSW_MBOX2": 0x409840,
    "FECS_ENGCTL": 0x4098AC,
    "FECS_CURCTX": 0x409B00,
    "FECS_NXTCTX": 0x409B04,
    # GPCCS falcon
    "GPCCS_CPUCTL": 0x41A100,
    "GPCCS_BOOTVEC": 0x41A104,
    "GPCCS_TRACEPC": 0x41A030,
    "GPCCS_MAILBOX0": 0x41A040,
    "GPCCS_SCTL": 0x41A240,
    # GR context switch config
    "GR_FECS_CTX_STATE": 0x409B20,
    "GR_FECS_CTX_SAVE_SWM": 0x409B24,
    "GR_FECS_CUR_CTX": 0x409B00,
    "GR_FECS_CTXSW_STATUS": 0x409400,
    "GR_FECS_CTXSW_MAILBOX4": 0x409840,
    "GR_FECS_CTXSW_MAILBOX6": 0x409848,
    "GR_FECS_CTXSW_MAILBOX7": 0x40984C,
    # GR exceptions / traps
    "GR_EXCEPTION": 0x409C18,
    "GR_EXCEPTION_EN": 0x409C24,
    "GR_TRAP_ADDR": 0x400704,
    # Key GPC registers
    "GPC0_GPCCS_NV_GPC_CS": 0x502000,
    "GPC0_PROP0": 0x503000,
    "GPC0_ZCULL0": 0x504000,
    "GPC0_TPC0_TEX": 0x504200,
    "GPC0_TPC0_SM": 0x504300,
    # PRI hub
    "PRI_RINGSTATION_SYS": 0x122000,
    "PRI_RINGSTATION_GPC": 0x128000,
    # NV_PGRAPH global
    "NV_PGRAPH_PRI_BE0": 0x410000,
    "NV_PGRAPH_PRI_CWD": 0x405800,
    "NV_PGRAPH_PRI_SKED": 0x407000,
    "NV_PGRAPH_PRI_PD": 0x406000,
}

# Scan broader PGRAPH range (0x400000-0x420000) for non-zero/non-dead registers
SCAN_RANGES = [
    (0x400000, 0x400200, "PGRAPH_TOP"),
    (0x404000, 0x404400, "GR_FE"),
    (0x409000, 0x409C00, "FECS"),
    (0x409C00, 0x40A000, "FECS_CTXSW"),
    (0x41A000, 0x41A400, "GPCCS"),
]

try:
    with open(BAR0, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0x800000, mmap.MAP_SHARED, mmap.PROT_READ)

        state = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        print(f"=== GR Register Capture: {state} ({BDF}) ===\n")

        result = {}
        print("Key registers:")
        for name, offset in sorted(GR_REGS.items(), key=lambda x: x[1]):
            try:
                val = struct.unpack_from("<I", mm, offset)[0]
                result[name] = val
                if val != 0 and val != 0xBADF1000 and val != 0xBAD00100:
                    print(f"  {name:40s} [{offset:#010x}] = {val:#010x}")
            except Exception:
                result[name] = None

        for start, end, label in SCAN_RANGES:
            nonzero = []
            for off in range(start, end, 4):
                try:
                    val = struct.unpack_from("<I", mm, off)[0]
                    if val != 0 and val != 0xBADF1000 and val != 0xBAD00100:
                        nonzero.append((off, val))
                except Exception:
                    pass
            if nonzero:
                print(f"\n{label} non-zero ({len(nonzero)} regs):")
                for off, val in nonzero[:30]:
                    print(f"  [{off:#010x}] = {val:#010x}")
                if len(nonzero) > 30:
                    print(f"  ... and {len(nonzero)-30} more")

        mm.close()

        outfile = f"/tmp/gr_regs_{state}.json"
        with open(outfile, "w") as jf:
            json.dump({k: v for k, v in result.items() if v is not None}, jf, indent=2)
        print(f"\nSaved to {outfile}")

except PermissionError:
    print(f"Need root: sudo python3 {sys.argv[0]} {BDF} {state}")
    sys.exit(1)
