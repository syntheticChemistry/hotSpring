#!/usr/bin/env python3
"""Cross-card register oracle — dump key PFIFO/PBDMA registers from nouveau Titan V.

Reads BAR0 registers from the nouveau-bound Titan V at 03:00.0 via sysfs resource0.
Compare these values with VFIO diagnostic matrix output to find discrepancies.

Usage: sudo python3 scripts/cross_card_oracle.py [BDF]
"""

import mmap
import os
import struct
import sys

BDF = sys.argv[1] if len(sys.argv) > 1 else "0000:03:00.0"
RES0 = f"/sys/bus/pci/devices/{BDF}/resource0"

if not os.path.exists(RES0):
    print(f"ERROR: {RES0} not found — is {BDF} correct?")
    sys.exit(1)

fd = os.open(RES0, os.O_RDWR)
bar0 = mmap.mmap(fd, 0x01000000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

def r(off):
    return struct.unpack_from("<I", bar0, off)[0]

def r64(off):
    lo = r(off)
    hi = r(off + 4)
    return lo | (hi << 32)

print(f"╔══ NOUVEAU ORACLE — {BDF} ═══════════════════════════════╗")

# Basic ID
boot0 = r(0x0)
print(f"║ BOOT0:           {boot0:#010x}")
print(f"║ PMC_ENABLE:      {r(0x200):#010x}")

# PBUS
print(f"║ BAR0_WINDOW:     {r(0x1700):#010x}")
print(f"║ BAR1_BLOCK:      {r(0x1704):#010x}")
print(f"║ BIND_STATUS:     {r(0x1710):#010x}")
print(f"║ BAR2_BLOCK:      {r(0x1714):#010x}")

# MMU
print(f"║ MMU_PHYS_SECURE: {r(0x100CB8):#010x}")

# PFIFO
print(f"╠══ PFIFO ═════════════════════════════════════════════════╣")
print(f"║ PFIFO_ENABLE:    {r(0x2200):#010x}")
print(f"║ PFIFO_INTR:      {r(0x2100):#010x}")
print(f"║ SCHED_DISABLE:   {r(0x2630):#010x}")
print(f"║ CHSW_ERROR:      {r(0x256C):#010x}")
print(f"║ PBDMA_MAP:       {r(0x2004):#010x}")
print(f"║ INTR_EN_0:       {r(0x2140):#010x}")
print(f"║ INTR_EN_1:       {r(0x2144):#010x}")

# Engine table
print(f"╠══ ENGINE TABLE ══════════════════════════════════════════╣")
for i in range(16):
    val = r(0x2600 + i * 4)
    if val == 0:
        break
    kind = val & 3
    typ = (val >> 2) & 0x3F
    rl = (val >> 11) & 0x1F
    final_flag = "FINAL" if val & (1 << 31) else ""
    print(f"║ ENGN[{i:2d}]:  {val:#010x}  type={typ} runlist={rl} {final_flag}")

# PBDMA registers for each active PBDMA
print(f"╠══ PBDMA REGISTERS ═══════════════════════════════════════╣")
pbdma_map = r(0x2004)
for pid in range(8):
    if pbdma_map & (1 << pid) == 0:
        continue
    pb = 0x40000 + pid * 0x2000
    runlist = r(0x2390 + pid * 4)
    state = r(pb + 0xB0)
    gp_base_lo = r(pb + 0x40)
    gp_base_hi = r(pb + 0x44)
    gp_put = r(pb + 0x54)
    gp_fetch = r(pb + 0x48)
    gp_state = r(pb + 0x4C)
    userd_lo = r(pb + 0xD0)
    userd_hi = r(pb + 0xD4)
    sig = r(pb + 0xC0)
    config = r(pb + 0xA8)
    chan_info = r(pb + 0xAC)
    acquire = r(pb + 0x30)
    intr = r(0x40000 + pid * 0x2000 + 0x100)
    ctx_userd_lo = r(pb + 0x08)
    ctx_userd_hi = r(pb + 0x0C)
    ctx_sig = r(pb + 0x10)
    ctx_gp_base_lo = r(pb + 0x48)
    ctx_gp_base_hi = r(pb + 0x4C)
    ctx_gp_put = r(pb + 0x54)
    ctx_gp_fetch = r(pb + 0x48)

    print(f"║ ── PBDMA {pid} (runlist={runlist}) ──")
    print(f"║   GP_BASE:   {gp_base_hi:#010x}_{gp_base_lo:#010x}")
    print(f"║   GP_PUT:    {gp_put}")
    print(f"║   GP_FETCH:  {gp_fetch:#010x}")
    print(f"║   GP_STATE:  {gp_state:#010x}")
    print(f"║   USERD:     {userd_hi:#010x}_{userd_lo:#010x}")
    print(f"║   SIGNATURE: {sig:#010x}")
    print(f"║   CONFIG:    {config:#010x}")
    print(f"║   CHAN_INFO:  {chan_info:#010x}")
    print(f"║   ACQUIRE:   {acquire:#010x}")
    print(f"║   STATE:     {state:#010x}")
    print(f"║   INTR:      {intr:#010x}")
    print(f"║   CTX_USERD: {ctx_userd_hi:#010x}_{ctx_userd_lo:#010x}")
    print(f"║   CTX_SIG:   {ctx_sig:#010x}")

# PCCSR channels — find active ones
print(f"╠══ ACTIVE CHANNELS (PCCSR) ═══════════════════════════════╣")
active_count = 0
for ch in range(512):
    chan_val = r(0x800000 + ch * 8 + 4)
    if chan_val & 1:  # ENABLE bit
        inst_val = r(0x800000 + ch * 8)
        status = (chan_val >> 24) & 0xF
        scheduled = (chan_val >> 1) & 1
        busy = (chan_val >> 28) & 1
        status_names = {0: "IDLE", 5: "ON_PBDMA", 6: "ON_PBDMA+ENG", 7: "ON_ENG"}
        sname = status_names.get(status, f"UNK({status})")
        print(f"║ CH[{ch:3d}]: INST={inst_val:#010x} CHAN={chan_val:#010x} status={sname} sched={scheduled} busy={busy}")
        active_count += 1
if active_count == 0:
    print(f"║ (no active channels)")
print(f"║ Active channels: {active_count}")

# RAMFC dump for first active channel (via PRAMIN)
print(f"╠══ RAMFC (first active channel via PRAMIN) ═══════════════╣")
for ch in range(512):
    chan_val = r(0x800000 + ch * 8 + 4)
    if chan_val & 1:
        inst_val = r(0x800000 + ch * 8)
        inst_target = (inst_val >> 28) & 0x3
        inst_addr = (inst_val & 0x0FFFFFFF) << 12
        target_names = {0: "VRAM", 2: "SYS_COH", 3: "SYS_NCOH"}
        print(f"║ Channel {ch}: INST_ADDR={inst_addr:#010x} TARGET={target_names.get(inst_target, '?')}")

        if inst_target == 0:
            saved_window = r(0x1700)
            window_val = inst_addr >> 16
            struct.pack_into("<I", bar0, 0x1700, window_val)
            import time; time.sleep(0.01)
            pm_off = inst_addr & 0xFFFF
            pm = 0x700000 + pm_off

            ramfc_regs = [
                (0x000, "RAMFC[0x000]"),
                (0x004, "RAMFC[0x004]"),
                (0x008, "USERD_LO"),
                (0x00C, "USERD_HI"),
                (0x010, "SIGNATURE"),
                (0x014, "RAMFC[0x014]"),
                (0x018, "RAMFC[0x018]"),
                (0x01C, "RAMFC[0x01C]"),
                (0x020, "RAMFC[0x020]"),
                (0x024, "RAMFC[0x024]"),
                (0x028, "RAMFC[0x028]"),
                (0x02C, "RAMFC[0x02C]"),
                (0x030, "ACQUIRE"),
                (0x034, "RAMFC[0x034]"),
                (0x038, "RAMFC[0x038]"),
                (0x03C, "RAMFC[0x03C]"),
                (0x040, "RAMFC[0x040]"),
                (0x044, "RAMFC[0x044]"),
                (0x048, "GP_BASE_LO"),
                (0x04C, "GP_BASE_HI"),
                (0x050, "GP_FETCH"),
                (0x054, "GP_PUT"),
                (0x058, "GP_GET"),
                (0x05C, "RAMFC[0x05C]"),
                (0x060, "RAMFC[0x060]"),
                (0x064, "RAMFC[0x064]"),
                (0x068, "RAMFC[0x068]"),
                (0x06C, "RAMFC[0x06C]"),
                (0x070, "RAMFC[0x070]"),
                (0x074, "RAMFC[0x074]"),
                (0x078, "RAMFC[0x078]"),
                (0x07C, "RAMFC[0x07C]"),
                (0x080, "RAMFC[0x080]"),
                (0x084, "RAMFC[0x084]"),
                (0x088, "RAMFC[0x088]"),
                (0x08C, "RAMFC[0x08C]"),
                (0x090, "RAMFC[0x090]"),
                (0x094, "RAMFC[0x094]"),
                (0x098, "RAMFC[0x098]"),
                (0x09C, "RAMFC[0x09C]"),
                (0x0A0, "RAMFC[0x0A0]"),
                (0x0A4, "RAMFC[0x0A4]"),
                (0x0A8, "CONFIG"),
                (0x0AC, "CHANNEL_INFO"),
                (0x0B0, "RAMFC[0x0B0]"),
                (0x0B4, "RAMFC[0x0B4]"),
                (0x0B8, "RAMFC[0x0B8]"),
                (0x0BC, "RAMFC[0x0BC]"),
                (0x0C0, "RAMFC[0x0C0]"),
                (0x200, "PAGE_DIR_BASE_LO"),
                (0x204, "PAGE_DIR_BASE_HI"),
                (0x208, "ADDR_LIMIT_LO"),
                (0x20C, "ADDR_LIMIT_HI"),
            ]
            for off, name in ramfc_regs:
                val = r(pm + off)
                if val != 0:
                    print(f"║   [{off:#05x}] {name:20s} = {val:#010x}")

            # Dump subcontext area (0x210-0x2FF)
            print(f"║   ── Subcontexts ──")
            for off in range(0x210, 0x2C0, 4):
                val = r(pm + off)
                if val != 0:
                    print(f"║   [{off:#05x}] = {val:#010x}")

            # Restore PRAMIN window
            struct.pack_into("<I", bar0, 0x1700, saved_window)
        else:
            print(f"║   (instance in SYS_MEM — cannot read via PRAMIN)")
        break

print(f"╚══════════════════════════════════════════════════════════╝")

bar0.close()
os.close(fd)
