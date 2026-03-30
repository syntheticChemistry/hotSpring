#!/usr/bin/env python3
"""
Replay VBIOS DEVINIT scripts to cold K80 via BAR0 MMIO.

Writes the register initialization sequence extracted from the VBIOS
to bring a cold (unPOSTed) GK210 GPU to life. Operates entirely in
userspace through the vfio-pci resource0 sysfs file — no kernel driver
touches the GPU.

Usage:
    python3 replay_devinit.py <bdf> <devinit_scripts.json> [--dry-run]
"""
import json, mmap, os, struct, sys, time

PMC_BOOT_0 = 0x000000
PTIMER_TIME_0 = 0x009400
PTIMER_TIME_1 = 0x009410

BAR0_MAP_SIZE = 0xA00000  # 10MB covers all init register regions


def open_bar0(bdf):
    resource0 = f"/sys/bus/pci/devices/{bdf}/resource0"
    fd = os.open(resource0, os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, BAR0_MAP_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    return fd, mm


def read_u32(mm, offset):
    if offset + 4 > len(mm):
        return None
    return struct.unpack_from("<I", mm, offset)[0]


def write_u32(mm, offset, value):
    if offset + 4 > len(mm):
        return False
    struct.pack_into("<I", mm, offset, value & 0xFFFFFFFF)
    return True


def check_gpu_health(mm, label=""):
    pmc = read_u32(mm, PMC_BOOT_0)
    t0a = read_u32(mm, PTIMER_TIME_0)
    t1a = read_u32(mm, PTIMER_TIME_1)
    time.sleep(0.001)
    t0b = read_u32(mm, PTIMER_TIME_0)
    t1b = read_u32(mm, PTIMER_TIME_1)
    ticking = (t0a != t0b) or (t1a != t1b)
    print(f"  [{label}] PMC_BOOT_0=0x{pmc:08x}  PTIMER={'TICKING' if ticking else 'FROZEN'}  "
          f"(0x{t0a:08x}->0x{t0b:08x})")
    return pmc, ticking


def normalize_op_type(t):
    """Accept both Python format (ZM_REG) and Rust format (ZmReg)."""
    mapping = {
        'ZmReg': 'ZM_REG', 'ZM_REG': 'ZM_REG',
        'NvReg': 'NV_REG', 'NV_REG': 'NV_REG',
        'ZmMaskAdd': 'ZM_MASK_ADD', 'ZM_MASK_ADD': 'ZM_MASK_ADD',
        'Time': 'TIME', 'TIME': 'TIME',
    }
    return mapping.get(t, t)


def replay_script(mm, script, dry_run=False):
    ops = script['ops']
    applied = 0
    failed = 0
    rmw_count = 0

    for op in ops:
        op_type = normalize_op_type(op['type'])

        if op_type == 'ZM_REG':
            reg = op['reg']
            val = op['val']
            if reg + 4 > BAR0_MAP_SIZE:
                failed += 1
                continue
            if dry_run:
                print(f"    [DRY] W 0x{reg:06x} = 0x{val:08x}")
            else:
                write_u32(mm, reg, val)
            applied += 1

        elif op_type == 'NV_REG':
            reg = op['reg']
            mask = op['mask']
            or_val = op['or_val']
            if reg + 4 > BAR0_MAP_SIZE:
                failed += 1
                continue
            if dry_run:
                print(f"    [DRY] RMW 0x{reg:06x} &= 0x{mask:08x} |= 0x{or_val:08x}")
            else:
                current = read_u32(mm, reg)
                if current is not None:
                    new_val = (current & mask) | or_val
                    write_u32(mm, reg, new_val)
                else:
                    failed += 1
                    continue
            applied += 1
            rmw_count += 1

        elif op_type == 'ZM_MASK_ADD':
            reg = op['reg']
            inv_mask = op['inv_mask']
            add_val = op['add_val']
            if reg + 4 > BAR0_MAP_SIZE:
                failed += 1
                continue
            if dry_run:
                print(f"    [DRY] MASKADD 0x{reg:06x} &~0x{inv_mask:08x} += 0x{add_val:08x}")
            else:
                current = read_u32(mm, reg)
                if current is not None:
                    masked = current & ~inv_mask
                    field = masked + add_val
                    new_val = (current & inv_mask) | (field & ~inv_mask)
                    write_u32(mm, reg, new_val)
                else:
                    failed += 1
                    continue
            applied += 1

        elif op_type == 'TIME':
            usec = op['usec']
            if not dry_run:
                time.sleep(usec / 1_000_000)

    return applied, failed, rmw_count


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <bdf> <devinit_scripts.json> [--dry-run]")
        sys.exit(1)

    bdf = sys.argv[1]
    recipe_path = sys.argv[2]
    dry_run = '--dry-run' in sys.argv

    with open(recipe_path) as f:
        scripts = json.load(f)

    print(f"=== DEVINIT Replay {'(DRY RUN)' if dry_run else ''} ===")
    print(f"Target: {bdf}")
    print(f"Scripts: {len(scripts)} ({sum(len(s['ops']) for s in scripts)} total ops)")
    print()

    fd, mm = open_bar0(bdf)

    print("Pre-replay GPU state:")
    pmc, ticking = check_gpu_health(mm, "BEFORE")
    print()

    if pmc == 0xFFFFFFFF:
        print("ERROR: BAR0 reads 0xFFFFFFFF — PCIe link is down. Cannot proceed.")
        mm.close()
        os.close(fd)
        sys.exit(1)

    for script in scripts:
        sid = script['id']
        ops_count = len(script['ops'])
        if ops_count == 0:
            continue

        print(f"--- Script {sid} at {script['addr']}: {ops_count} ops ---")
        applied, failed, rmw = replay_script(mm, script, dry_run=dry_run)
        print(f"    Applied: {applied}, Failed: {failed}, RMW: {rmw}")
        check_gpu_health(mm, f"after script {sid}")
        print()

    print("=== Replay complete ===")
    print("Final GPU state:")
    pmc, ticking = check_gpu_health(mm, "FINAL")

    if ticking:
        print("\nSUCCESS: PTIMER is ticking — GPU is alive!")
    else:
        print("\nPTIMER still frozen — GPU may need additional initialization.")

    mm.close()
    os.close(fd)


if __name__ == "__main__":
    main()
