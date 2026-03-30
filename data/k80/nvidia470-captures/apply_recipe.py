#!/usr/bin/env python3
"""Apply nvidia-470 cold->warm recipe to K80 via BAR0 mmap."""
import json
import mmap
import struct
import sys
import os
import time

BDF = sys.argv[1] if len(sys.argv) > 1 else "0000:4d:00.0"
RESOURCE = f"/sys/bus/pci/devices/{BDF}/resource0"

SKIP_ADDRS = {0x200, 0x204}  # PMC_ENABLE, PMC_SPOON - handled by test
PRI_FAULT_MASKS = {0xBADF0000, 0xBAD00000}

def is_pri_fault(v):
    return (v & 0xBADF0000) == 0xBADF0000 or (v & 0xBAD00000) == 0xBAD00000

def main():
    diff_path = os.path.join(os.path.dirname(__file__), "nvidia470_cold_warm_diff.json")
    with open(diff_path) as f:
        diff = json.load(f)

    fd = os.open(RESOURCE, os.O_RDWR | os.O_SYNC)
    size = os.fstat(fd).st_size
    bar0 = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    def read32(addr):
        bar0.seek(addr)
        return struct.unpack("<I", bar0.read(4))[0]

    def write32(addr, val):
        bar0.seek(addr)
        bar0.write(struct.pack("<I", val))

    boot0 = read32(0x0)
    print(f"BOOT0={boot0:#010x} (BDF={BDF})")

    # Pre-recipe state
    fecs_cpuctl = read32(0x409100)
    clk_base = read32(0x130000)
    pclock_gate = read32(0x137050)
    print(f"Pre-recipe: FECS_cpuctl={fecs_cpuctl:#010x} CLK={clk_base:#010x} PCLOCK_gate={pclock_gate:#010x}")

    writes = 0
    skipped = 0
    errors = 0

    # Apply "added" registers
    for domain, regs in diff.get("added", {}).items():
        for addr_s, val_s in regs.items():
            addr = int(addr_s, 16)
            val = int(val_s, 16)
            if addr in SKIP_ADDRS:
                skipped += 1
                continue
            if is_pri_fault(val):
                skipped += 1
                continue
            if addr >= 0x01000000:
                skipped += 1
                continue
            try:
                write32(addr, val)
                writes += 1
            except Exception as e:
                errors += 1

    # Apply "changed" registers (write warm value)
    for domain, regs in diff.get("changed", {}).items():
        for addr_s, vals in regs.items():
            addr = int(addr_s, 16)
            warm_val = int(vals["warm"], 16)
            if addr in SKIP_ADDRS:
                skipped += 1
                continue
            if is_pri_fault(warm_val):
                skipped += 1
                continue
            if addr >= 0x01000000:
                skipped += 1
                continue
            try:
                write32(addr, warm_val)
                writes += 1
            except Exception as e:
                errors += 1

    print(f"\nRecipe applied: {writes} writes, {skipped} skipped, {errors} errors")
    time.sleep(0.1)

    # Post-recipe state
    fecs_cpuctl = read32(0x409100)
    fecs_sctl = read32(0x409240)
    fecs_pc = read32(0x409110)
    fecs_mb0 = read32(0x409040)
    fecs_hwcfg = read32(0x409108)
    clk_base = read32(0x130000)
    pclock_gate = read32(0x137050)
    pmu_cpuctl = read32(0x10A100)
    pmu_sctl = read32(0x10A240)
    gr_status = read32(0x400700)
    gpccs_cpuctl = read32(0x41A100)

    print(f"\nPost-recipe state:")
    print(f"  FECS  cpuctl={fecs_cpuctl:#010x} sctl={fecs_sctl:#010x} pc={fecs_pc:#010x} mb0={fecs_mb0:#010x} hwcfg={fecs_hwcfg:#010x}")
    print(f"  GPCCS cpuctl={gpccs_cpuctl:#010x}")
    print(f"  PMU   cpuctl={pmu_cpuctl:#010x} sctl={pmu_sctl:#010x}")
    print(f"  CLK   base={clk_base:#010x} PCLOCK_gate={pclock_gate:#010x}")
    print(f"  GR    status={gr_status:#010x}")

    fecs_ok = not is_pri_fault(fecs_cpuctl)
    gpccs_ok = not is_pri_fault(gpccs_cpuctl)
    gr_ok = not is_pri_fault(gr_status)
    print(f"\n  FECS accessible: {fecs_ok}")
    print(f"  GPCCS accessible: {gpccs_ok}")
    print(f"  GR accessible: {gr_ok}")

    bar0.close()
    os.close(fd)

if __name__ == "__main__":
    main()
