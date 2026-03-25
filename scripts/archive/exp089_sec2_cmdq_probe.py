#!/usr/bin/env python3
"""Exp 089: SEC2 CMDQ/MSGQ ring probe and GPCCS bootstrap.

Reads SEC2 CMDQ/MSGQ head/tail registers via BAR0 mmap, dumps the DMEM
queue regions, and optionally sends a properly formatted BOOTSTRAP_FALCON
command through the CMDQ ring.

Usage:
    sudo python3 scripts/exp089_sec2_cmdq_probe.py [probe|bootstrap]

Requires VFIO-bound Titan V. Uses sysfs BAR0 mmap.
"""

import mmap
import struct
import sys
import time
import os

BDF = os.environ.get("BDF", "0000:03:00.0")
BAR0_PATH = f"/sys/bus/pci/devices/{BDF}/resource0"

# SEC2 base in BAR0
SEC2_BASE = 0x087000

# From nouveau gp102_sec2_flcn: .cmdq = { 0xa00, 0xa04, 8 }
CMDQ_HEAD_OFF = 0xa00
CMDQ_TAIL_OFF = 0xa04
CMDQ_STRIDE   = 8

# From nouveau gp102_sec2_flcn: .msgq = { 0xa30, 0xa34, 8 }
MSGQ_HEAD_OFF = 0xa30
MSGQ_TAIL_OFF = 0xa34
MSGQ_STRIDE   = 8

# Falcon register offsets
CPUCTL   = 0x100
MAILBOX0 = 0x040
MAILBOX1 = 0x044
HWCFG    = 0x008
DMEMC    = 0x1c0  # DMEM control (port 0)
DMEMD    = 0x1c4  # DMEM data (port 0)

# FECS/GPCCS bases
FECS_BASE  = 0x409000
GPCCS_BASE = 0x41a000

# SEC2 command constants (from nvfw/sec2.h)
NV_SEC2_UNIT_ACR = 0x08
NV_SEC2_ACR_CMD_BOOTSTRAP_FALCON = 0x00
CMD_FLAGS_STATUS = 0x01
CMD_FLAGS_INTR   = 0x02
FALCON_ID_FECS   = 2
FALCON_ID_GPCCS  = 3


def r32(mm, addr):
    """Read a 32-bit register from BAR0."""
    mm.seek(addr)
    return struct.unpack("<I", mm.read(4))[0]


def w32(mm, addr, val):
    """Write a 32-bit register to BAR0."""
    mm.seek(addr)
    mm.write(struct.pack("<I", val & 0xFFFFFFFF))


def dmem_read_word(mm, base, dmem_addr):
    """Read a u32 from falcon DMEM via PIO (DMEMC/DMEMD)."""
    # Set DMEMC: bit 25 = read mode, addr in bits [15:2] (word-aligned)
    ctrl = (1 << 25) | (dmem_addr & 0xFFFC)
    w32(mm, base + DMEMC, ctrl)
    return r32(mm, base + DMEMD)


def dmem_write_word(mm, base, dmem_addr, val):
    """Write a u32 to falcon DMEM via PIO."""
    # Set DMEMC: bit 24 = write mode, auto-increment, addr
    ctrl = (1 << 24) | (dmem_addr & 0xFFFC)
    w32(mm, base + DMEMC, ctrl)
    w32(mm, base + DMEMD, val)


def dmem_read_block(mm, base, start, count_words):
    """Read a block of u32s from DMEM."""
    words = []
    ctrl = (1 << 25) | (start & 0xFFFC)
    w32(mm, base + DMEMC, ctrl)
    for _ in range(count_words):
        words.append(r32(mm, base + DMEMD))
    return words


def dmem_write_block(mm, base, start, words):
    """Write a block of u32s to DMEM."""
    ctrl = (1 << 24) | (start & 0xFFFC)
    w32(mm, base + DMEMC, ctrl)
    for w in words:
        w32(mm, base + DMEMD, w)


def probe(mm):
    """Probe SEC2 CMDQ/MSGQ state."""
    print(f"=== Exp 089: SEC2 CMDQ/MSGQ Ring Probe ===")
    print(f"BDF: {BDF}  SEC2_BASE: {SEC2_BASE:#08x}\n")

    # SEC2 basic state
    cpuctl = r32(mm, SEC2_BASE + CPUCTL)
    mb0 = r32(mm, SEC2_BASE + MAILBOX0)
    mb1 = r32(mm, SEC2_BASE + MAILBOX1)
    hwcfg = r32(mm, SEC2_BASE + HWCFG)
    pc = r32(mm, SEC2_BASE + 0x030)  # TRACEPC
    sctl = r32(mm, SEC2_BASE + 0x240)
    imem_sz = ((hwcfg >> 0) & 0x1FF) << 8
    dmem_sz = ((hwcfg >> 9) & 0x1FF) << 8
    print(f"SEC2 state:")
    print(f"  cpuctl={cpuctl:#010x}  pc={pc:#010x}  sctl={sctl:#010x}")
    print(f"  mb0={mb0:#010x}  mb1={mb1:#010x}")
    print(f"  hwcfg={hwcfg:#010x}  imem={imem_sz}B  dmem={dmem_sz}B")

    running = cpuctl & 0x10 == 0 and cpuctl & 0x20 == 0
    print(f"  Status: {'RUNNING' if running else 'NOT RUNNING'}")

    # CMDQ/MSGQ registers
    print(f"\nCMDQ registers:")
    for idx in range(2):
        h = r32(mm, SEC2_BASE + CMDQ_HEAD_OFF + idx * CMDQ_STRIDE)
        t = r32(mm, SEC2_BASE + CMDQ_TAIL_OFF + idx * CMDQ_STRIDE)
        print(f"  CMDQ[{idx}]: head={h:#010x} tail={t:#010x} {'EMPTY' if h == t else f'PENDING({h-t} bytes)'}")

    print(f"\nMSGQ registers:")
    for idx in range(2):
        h = r32(mm, SEC2_BASE + MSGQ_HEAD_OFF + idx * MSGQ_STRIDE)
        t = r32(mm, SEC2_BASE + MSGQ_TAIL_OFF + idx * MSGQ_STRIDE)
        print(f"  MSGQ[{idx}]: head={h:#010x} tail={t:#010x} {'EMPTY' if h == t else f'HAS DATA({h-t} bytes)'}")

    # Dump DMEM around queue regions
    # First, try to find the init message remnants to identify queue offsets
    print(f"\nDMEM scan (looking for queue structures):")

    # The init message contains queue_info with offsets.
    # Even though it was consumed, the queue memory region is persistent.
    # Common SEC2 queue layout: CMDQ at ~0x0F20, MSGQ at ~0x0B00 (from our dumps)

    # Read head/tail values to find active DMEM regions
    cmdq0_head = r32(mm, SEC2_BASE + CMDQ_HEAD_OFF)
    cmdq0_tail = r32(mm, SEC2_BASE + CMDQ_TAIL_OFF)
    msgq0_head = r32(mm, SEC2_BASE + MSGQ_HEAD_OFF)
    msgq0_tail = r32(mm, SEC2_BASE + MSGQ_TAIL_OFF)

    print(f"  CMDQ[0] head={cmdq0_head:#06x} tail={cmdq0_tail:#06x}")
    print(f"  MSGQ[0] head={msgq0_head:#06x} tail={msgq0_tail:#06x}")

    # Dump DMEM around the queue head/tail positions
    for label, offset in [("CMDQ head region", cmdq0_head),
                          ("CMDQ tail region", cmdq0_tail),
                          ("MSGQ head region", msgq0_head),
                          ("MSGQ tail region", msgq0_tail)]:
        if offset > 0 and offset < dmem_sz:
            start = max(0, (offset - 32) & ~3)
            words = dmem_read_block(mm, SEC2_BASE, start, 24)
            nonzero = [(start + i*4, w) for i, w in enumerate(words) if w != 0]
            if nonzero:
                print(f"\n  {label} (around {offset:#06x}):")
                for addr, val in nonzero:
                    marker = " <-- HEAD" if addr == (offset & ~3) else ""
                    marker = " <-- TAIL" if addr == (cmdq0_tail & ~3) and "CMDQ" in label else marker
                    print(f"    [{addr:#06x}] = {val:#010x}{marker}")
            else:
                print(f"\n  {label} (around {offset:#06x}): all zeros")

    # Wide DMEM scan for non-zero regions (find queue boundaries)
    print(f"\n  Full DMEM non-zero scan (up to {min(dmem_sz, 0x4000):#x}):")
    scan_end = min(dmem_sz, 0x4000)
    ranges = []
    in_range = False
    rstart = 0
    for off in range(0, scan_end, 4):
        val = dmem_read_word(mm, SEC2_BASE, off)
        if val != 0 and val != 0xDEADDEAD:
            if not in_range:
                rstart = off
                in_range = True
        else:
            if in_range:
                ranges.append((rstart, off))
                in_range = False
    if in_range:
        ranges.append((rstart, scan_end))

    for (s, e) in ranges:
        print(f"    [{s:#06x}..{e:#06x}] ({e-s} bytes)")
        # Dump first 8 words of each range
        words = dmem_read_block(mm, SEC2_BASE, s, min(8, (e-s)//4))
        for i, w in enumerate(words):
            print(f"      [{s+i*4:#06x}] = {w:#010x}")

    # FECS/GPCCS state
    print(f"\nFECS state:")
    fecs_cpuctl = r32(mm, FECS_BASE + CPUCTL)
    fecs_pc = r32(mm, FECS_BASE + 0x030)
    print(f"  cpuctl={fecs_cpuctl:#010x}  pc={fecs_pc:#010x}")

    print(f"\nGPCCS state:")
    gpccs_cpuctl = r32(mm, GPCCS_BASE + CPUCTL)
    gpccs_pc = r32(mm, GPCCS_BASE + 0x030)
    gpccs_exci = r32(mm, GPCCS_BASE + 0x148)
    gpccs_sctl = r32(mm, GPCCS_BASE + 0x240)
    print(f"  cpuctl={gpccs_cpuctl:#010x}  pc={gpccs_pc:#010x}  exci={gpccs_exci:#010x}  sctl={gpccs_sctl:#010x}")

    return cmdq0_head, cmdq0_tail, msgq0_head, msgq0_tail


def build_bootstrap_cmd(falcon_id, seq_id=0):
    """Build a nv_sec2_acr_bootstrap_falcon_cmd as bytes."""
    # struct nvfw_falcon_cmd { u8 unit_id, u8 size, u8 ctrl_flags, u8 seq_id }
    # struct nv_sec2_acr_cmd { struct nvfw_falcon_cmd hdr; u8 cmd_type; }
    # struct nv_sec2_acr_bootstrap_falcon_cmd { nv_sec2_acr_cmd cmd; u32 flags; u32 falcon_id; }
    #
    # Total: 4 + 1 + 3(pad) + 4 + 4 = 16 bytes
    unit_id = NV_SEC2_UNIT_ACR
    size = 16
    ctrl_flags = CMD_FLAGS_STATUS | CMD_FLAGS_INTR
    cmd_type = NV_SEC2_ACR_CMD_BOOTSTRAP_FALCON
    flags = 0  # RESET_YES
    return struct.pack("<BBBB BBBx II",
                       unit_id, size, ctrl_flags, seq_id,
                       cmd_type, 0, 0,  # cmd_type + 3 bytes padding
                       flags, falcon_id)


def bootstrap_gpccs(mm, cmdq_head, cmdq_tail):
    """Send BOOTSTRAP_FALCON(GPCCS) via SEC2 CMDQ ring."""
    print(f"\n=== Sending BOOTSTRAP_FALCON(GPCCS) via CMDQ ===")

    # Build command
    cmd_bytes = build_bootstrap_cmd(FALCON_ID_GPCCS, seq_id=0)
    cmd_words = struct.unpack(f"<{len(cmd_bytes)//4}I", cmd_bytes)
    print(f"  Command ({len(cmd_bytes)} bytes): {' '.join(f'{w:#010x}' for w in cmd_words)}")

    # Write command to DMEM at current head position
    write_pos = cmdq_head
    print(f"  Writing to DMEM at {write_pos:#06x}")
    dmem_write_block(mm, SEC2_BASE, write_pos, list(cmd_words))

    # Verify write
    verify = dmem_read_block(mm, SEC2_BASE, write_pos, len(cmd_words))
    match = verify == list(cmd_words)
    print(f"  Verify: {match}  read={' '.join(f'{w:#010x}' for w in verify)}")

    if not match:
        print("  ERROR: DMEM write verification failed!")
        return

    # Advance CMDQ head register
    new_head = write_pos + ((len(cmd_bytes) + 3) & ~3)  # QUEUE_ALIGNMENT = 4
    print(f"  Advancing CMDQ head: {cmdq_head:#06x} -> {new_head:#06x}")
    w32(mm, SEC2_BASE + CMDQ_HEAD_OFF, new_head)

    # Verify head was written
    read_head = r32(mm, SEC2_BASE + CMDQ_HEAD_OFF)
    print(f"  CMDQ head readback: {read_head:#010x}")

    # Poke SEC2 interrupt to wake it up (bit 6 = EXT_IRQ)
    # falcon IRQSSET at offset 0x000
    print(f"  Poking SEC2 interrupt (IRQSSET bit 6)...")
    w32(mm, SEC2_BASE + 0x000, 0x40)

    # Wait and check
    print(f"  Waiting 500ms for SEC2 to process...")
    time.sleep(0.5)

    # Check MSGQ for response
    new_msgq_head = r32(mm, SEC2_BASE + MSGQ_HEAD_OFF)
    new_msgq_tail = r32(mm, SEC2_BASE + MSGQ_TAIL_OFF)
    print(f"  MSGQ after: head={new_msgq_head:#010x} tail={new_msgq_tail:#010x}")

    if new_msgq_head != new_msgq_tail:
        print(f"  MSGQ has response! Reading...")
        resp = dmem_read_block(mm, SEC2_BASE, new_msgq_tail, 8)
        print(f"  Response: {' '.join(f'{w:#010x}' for w in resp)}")
        # Parse: unit_id, size, ctrl_flags, seq_id, msg_type, error_code, falcon_id
        if len(resp) >= 3:
            hdr = resp[0]
            unit_id = hdr & 0xFF
            size = (hdr >> 8) & 0xFF
            error_code = resp[1] if len(resp) > 1 else 0
            fid = resp[2] if len(resp) > 2 else 0
            print(f"  Parsed: unit_id={unit_id:#04x} size={size} error={error_code:#010x} falcon_id={fid}")

        # Advance MSGQ tail to consume the message
        new_tail = new_msgq_tail + ((resp[0] >> 8) & 0xFF) if resp else new_msgq_tail + 16
        w32(mm, SEC2_BASE + MSGQ_TAIL_OFF, new_tail)
    else:
        print(f"  No MSGQ response (queue empty)")

    # Check GPCCS state
    print(f"\n  GPCCS after bootstrap attempt:")
    gpccs_cpuctl = r32(mm, GPCCS_BASE + CPUCTL)
    gpccs_pc = r32(mm, GPCCS_BASE + 0x030)
    gpccs_sctl = r32(mm, GPCCS_BASE + 0x240)
    print(f"    cpuctl={gpccs_cpuctl:#010x}  pc={gpccs_pc:#010x}  sctl={gpccs_sctl:#010x}")

    # Check SEC2 state
    sec2_cpuctl = r32(mm, SEC2_BASE + CPUCTL)
    sec2_pc = r32(mm, SEC2_BASE + 0x030)
    sec2_mb0 = r32(mm, SEC2_BASE + MAILBOX0)
    sec2_mb1 = r32(mm, SEC2_BASE + MAILBOX1)
    print(f"\n  SEC2 after:")
    print(f"    cpuctl={sec2_cpuctl:#010x}  pc={sec2_pc:#010x}")
    print(f"    mb0={sec2_mb0:#010x}  mb1={sec2_mb1:#010x}")

    # FECS state
    fecs_cpuctl = r32(mm, FECS_BASE + CPUCTL)
    fecs_pc = r32(mm, FECS_BASE + 0x030)
    print(f"\n  FECS after:")
    print(f"    cpuctl={fecs_cpuctl:#010x}  pc={fecs_pc:#010x}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"

    try:
        fd = os.open(BAR0_PATH, os.O_RDWR | os.O_SYNC)
    except PermissionError:
        print(f"Permission denied: {BAR0_PATH}")
        print("Run with: sudo python3 scripts/exp089_sec2_cmdq_probe.py")
        sys.exit(1)
    except FileNotFoundError:
        print(f"BAR0 not found: {BAR0_PATH}")
        print(f"Check BDF={BDF} and VFIO binding")
        sys.exit(1)

    fsize = os.fstat(fd).st_size
    if fsize == 0:
        fsize = 32 * 1024 * 1024  # 32MB default BAR0

    mm = mmap.mmap(fd, fsize, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    try:
        cmdq_head, cmdq_tail, msgq_head, msgq_tail = probe(mm)

        if mode == "bootstrap":
            if cmdq_head == 0 and cmdq_tail == 0:
                print("\nCMDQ head/tail both 0 — queue not initialized.")
                print("SEC2 may not have sent its init message, or registers are at different offsets.")
                print("Trying with inferred queue position from DMEM scan...")
            bootstrap_gpccs(mm, cmdq_head, cmdq_tail)
        elif mode == "probe":
            print(f"\nProbe complete. To attempt GPCCS bootstrap:")
            print(f"  sudo python3 scripts/exp089_sec2_cmdq_probe.py bootstrap")
    finally:
        mm.close()
        os.close(fd)


if __name__ == "__main__":
    main()
