#!/usr/bin/env python3
"""Read key PFIFO/PBDMA/PCCSR registers from an NVIDIA GPU's BAR0.

Usage: sudo python3 read_bar0_regs.py <bdf>
Example: sudo python3 read_bar0_regs.py 0000:21:00.0
"""
import mmap
import struct
import sys

REGS = {
    # PMC
    0x000000: "BOOT0 (chip ID)",
    0x000200: "PMC_ENABLE",
    0x000204: "PMC_PBDMA_ENABLE",
    # PFIFO
    0x002004: "PFIFO_PBDMA_MAP",
    0x002100: "PFIFO_INTR",
    0x002140: "PFIFO_INTR_EN",
    0x002270: "RUNLIST_BASE",
    0x002274: "RUNLIST_SUBMIT",
    0x002284: "RUNLIST0_INFO",
    0x002504: "SCHED_EN",
    0x00252C: "BIND_ERROR",
    0x002630: "SCHED_DISABLE",
    0x002634: "PREEMPT",
    0x002640: "ENGN0_STATUS",
    # PBDMA idle status
    0x003080: "PBDMA0_IDLE",
    0x003084: "PBDMA1_IDLE",
    0x003088: "PBDMA2_IDLE",
    0x00308C: "PBDMA3_IDLE",
    # PBDMA per-engine regs
    0x040108: "PBDMA0_INTR",
    0x04010C: "PBDMA0_INTR_EN",
    0x040148: "PBDMA0_HCE",
    0x04014C: "PBDMA0_HCE_EN",
    0x042108: "PBDMA1_INTR",
    0x04210C: "PBDMA1_INTR_EN",
    0x042148: "PBDMA1_HCE",
    0x04214C: "PBDMA1_HCE_EN",
    0x044108: "PBDMA2_INTR",
    0x046108: "PBDMA3_INTR",
    # PBDMA-to-runlist mapping
    0x002390: "PBDMA_RUNL_MAP[0]",
    0x002394: "PBDMA_RUNL_MAP[1]",
    0x002398: "PBDMA_RUNL_MAP[2]",
    0x00239C: "PBDMA_RUNL_MAP[3]",
    # PCCSR channel 0
    0x800000: "PCCSR_INST[0]",
    0x800004: "PCCSR_CHAN[0]",
    0x800008: "PCCSR_INST[1]",
    0x80000C: "PCCSR_CHAN[1]",
    # USERMODE
    0x810000: "USERMODE_CFG",
    0x810004: "USERMODE_4",
    0x810010: "USERMODE_10",
    0x810080: "USERMODE_80",
    0x810090: "USERMODE_NOTIFY_CHAN_PENDING",
    0x8100A0: "USERMODE_A0",
    # MMU
    0x100A2C: "MMU_FAULT_STATUS",
    0x104A20: "MMU_HUBTLB_ERR",
    # PRIV ring
    0x012070: "PRIV_RING_INTR",
    # TOP device info
    0x022700: "TOP_INFO[0]",
    0x022704: "TOP_INFO[1]",
    0x022708: "TOP_INFO[2]",
}

def read_bar0(bdf):
    path = f"/sys/bus/pci/devices/{bdf}/resource0"
    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 16 * 1024 * 1024, mmap.MAP_SHARED, mmap.PROT_READ)
        chip = struct.unpack_from("<I", mm, 0)[0]
        print(f"╔══ BAR0 REGISTER DUMP: {bdf} (BOOT0={chip:#010x}) ══╗")
        for offset in sorted(REGS):
            try:
                val = struct.unpack_from("<I", mm, offset)[0]
                print(f"║ [{offset:#08x}] {REGS[offset]:36s} = {val:#010x}")
            except Exception as e:
                print(f"║ [{offset:#08x}] {REGS[offset]:36s} = ERROR: {e}")
        print(f"╚{'═' * 62}╝")
        mm.close()

if __name__ == "__main__":
    bdf = sys.argv[1] if len(sys.argv) > 1 else "0000:21:00.0"
    read_bar0(bdf)
