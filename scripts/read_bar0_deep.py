#!/usr/bin/env python3
"""Deep BAR0 register dump for PFIFO/PBDMA comparison."""
import mmap, struct, sys

REGS = {
    0x000000: "BOOT0",
    0x000200: "PMC_ENABLE",
    0x000204: "PMC_DEVICE_ENABLE",
    # PFIFO core
    0x002004: "PFIFO_PBDMA_MAP",
    0x002100: "PFIFO_INTR",
    0x002140: "PFIFO_INTR_EN",
    0x002254: "PFIFO_FB_TIMEOUT",
    0x002270: "RUNLIST_BASE",
    0x002274: "RUNLIST_SUBMIT",
    0x002284: "RUNLIST0_INFO",
    0x00228C: "RUNLIST1_INFO",
    0x002504: "SCHED_EN (may not exist on Volta)",
    0x00252C: "BIND_ERROR",
    0x002630: "SCHED_DISABLE",
    0x002634: "PREEMPT",
    0x002640: "ENGN0_STATUS",
    0x002A04: "PFIFO_PBDMA_INTR_EN",
    # PBDMA idle
    0x003080: "PBDMA0_IDLE",
    0x003084: "PBDMA1_IDLE",
    0x003088: "PBDMA2_IDLE",
    0x00308C: "PBDMA3_IDLE",
    # PBDMA-to-runlist
    0x002390: "PBDMA_RUNL_MAP[0]",
    0x002394: "PBDMA_RUNL_MAP[1]",
    0x002398: "PBDMA_RUNL_MAP[2]",
    0x00239C: "PBDMA_RUNL_MAP[3]",
    # PCCSR channels 0-3
    0x800000: "PCCSR_INST[0]",
    0x800004: "PCCSR_CHAN[0]",
    0x800008: "PCCSR_INST[1]",
    0x80000C: "PCCSR_CHAN[1]",
    0x800010: "PCCSR_INST[2]",
    0x800014: "PCCSR_CHAN[2]",
    # PBDMA0 operational (base 0x040000)
    0x040040: "PBDMA0_GP_BASE_LO",
    0x040044: "PBDMA0_GP_BASE_HI",
    0x040048: "PBDMA0_GP_FETCH",
    0x04004C: "PBDMA0_GP_STATE",
    0x040050: "PBDMA0_GP_PUT_HI",
    0x040054: "PBDMA0_GP_PUT_LO",
    0x040084: "PBDMA0_PB_HEADER",
    0x040088: "PBDMA0_METHOD0",
    0x0400A8: "PBDMA0_TARGET",
    0x0400AC: "PBDMA0_SET_CHANNEL_INFO",
    0x0400B0: "PBDMA0_CHANNEL_STATE",
    0x0400C0: "PBDMA0_SIGNATURE",
    0x0400D0: "PBDMA0_USERD_LO",
    0x0400D4: "PBDMA0_USERD_HI",
    0x040108: "PBDMA0_INTR",
    0x04010C: "PBDMA0_INTR_EN",
    0x040148: "PBDMA0_HCE",
    0x04014C: "PBDMA0_HCE_EN",
    # PBDMA1 operational (base 0x042000)
    0x042040: "PBDMA1_GP_BASE_LO",
    0x042044: "PBDMA1_GP_BASE_HI",
    0x042048: "PBDMA1_GP_FETCH",
    0x04204C: "PBDMA1_GP_STATE",
    0x042050: "PBDMA1_GP_PUT_HI",
    0x042054: "PBDMA1_GP_PUT_LO",
    0x042084: "PBDMA1_PB_HEADER",
    0x0420A8: "PBDMA1_TARGET",
    0x0420AC: "PBDMA1_SET_CHANNEL_INFO",
    0x0420B0: "PBDMA1_CHANNEL_STATE",
    0x0420C0: "PBDMA1_SIGNATURE",
    0x0420D0: "PBDMA1_USERD_LO",
    0x0420D4: "PBDMA1_USERD_HI",
    0x042108: "PBDMA1_INTR",
    0x04210C: "PBDMA1_INTR_EN",
    0x042148: "PBDMA1_HCE",
    0x04214C: "PBDMA1_HCE_EN",
    # USERMODE block
    0x810000: "USERMODE_CFG",
    0x810004: "USERMODE_4",
    0x810010: "USERMODE_TIME_LO",
    0x810014: "USERMODE_TIME_HI",
    0x810080: "USERMODE_80",
    0x810090: "USERMODE_NOTIFY_CHAN_PENDING",
    # MMU
    0x100A2C: "MMU_FAULT_STATUS",
    0x100A30: "MMU_FAULT_ADDR_LO",
    0x100A34: "MMU_FAULT_ADDR_HI",
    0x100A38: "MMU_FAULT_INST_LO",
    0x100A3C: "MMU_FAULT_INST_HI",
    0x100A40: "MMU_FAULT_INFO",
    0x100C80: "MMU_PRI_CTRL",
    # PRIV ring
    0x012070: "PRIV_RING_INTR",
}

def read_bar0(bdf):
    path = f"/sys/bus/pci/devices/{bdf}/resource0"
    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 16 * 1024 * 1024, mmap.MAP_SHARED, mmap.PROT_READ)
        chip = struct.unpack_from("<I", mm, 0)[0]
        print(f"╔══ DEEP BAR0 DUMP: {bdf} (BOOT0={chip:#010x}) ══╗")
        for offset in sorted(REGS):
            try:
                val = struct.unpack_from("<I", mm, offset)[0]
                name = REGS[offset]
                notes = ""
                if offset == 0x800004:
                    notes = f"  EN={val&1} PBDMA_FAULT={val>>24&1} ENG_FAULT={val>>25&1} BUSY={val>>28&1}"
                elif offset == 0x800000:
                    target = (val >> 28) & 3
                    bind = (val >> 31) & 1
                    ptr = val & 0x0FFFFFFF
                    tname = ["VRAM","COH","COH","NCOH"][target]
                    notes = f"  PTR={ptr:#x} TARGET={tname} BIND={bind}"
                print(f"║ [{offset:#08x}] {name:42s} = {val:#010x}{notes}")
            except Exception as e:
                print(f"║ [{offset:#08x}] {REGS[offset]:42s} = ERROR: {e}")
        print(f"╚{'═' * 72}╝")
        mm.close()

if __name__ == "__main__":
    bdf = sys.argv[1] if len(sys.argv) > 1 else "0000:4b:00.0"
    read_bar0(bdf)
