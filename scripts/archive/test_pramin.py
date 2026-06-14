#!/usr/bin/env python3
"""Test PRAMIN window read/write to verify VRAM access via BAR0."""
import mmap
import struct
import sys

PRAMIN_BASE = 0x700000
PRAMIN_SIZE = 0x10000   # 64 KiB window
NV_PBUS_BAR0_WINDOW = 0x001700

def test_pramin(bdf):
    path = f"/sys/bus/pci/devices/{bdf}/resource0"
    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 16 * 1024 * 1024, mmap.MAP_SHARED,
                       mmap.PROT_READ | mmap.PROT_WRITE)
        chip = struct.unpack_from("<I", mm, 0)[0]
        print(f"BOOT0: {chip:#010x}")

        # Read current BAR0_WINDOW setting.
        bar0_win = struct.unpack_from("<I", mm, NV_PBUS_BAR0_WINDOW)[0]
        print(f"BAR0_WINDOW (current): {bar0_win:#010x}")

        # Set PRAMIN window to start of VRAM (offset 0).
        struct.pack_into("<I", mm, NV_PBUS_BAR0_WINDOW, 0x00000000)
        bar0_win_after = struct.unpack_from("<I", mm, NV_PBUS_BAR0_WINDOW)[0]
        print(f"BAR0_WINDOW (after set to 0): {bar0_win_after:#010x}")

        # Read first 16 dwords from PRAMIN window.
        print("\nPRAMIN window @ VRAM offset 0:")
        for i in range(16):
            off = PRAMIN_BASE + i * 4
            val = struct.unpack_from("<I", mm, off)[0]
            print(f"  PRAMIN[{i*4:#06x}] = {val:#010x}")

        # Write a test pattern and read back.
        test_pattern = 0xDEAD_BEEF
        struct.pack_into("<I", mm, PRAMIN_BASE, test_pattern)
        readback = struct.unpack_from("<I", mm, PRAMIN_BASE)[0]
        print(f"\nWrite test: wrote {test_pattern:#010x}, read back {readback:#010x}")
        if readback == test_pattern:
            print("PRAMIN WRITE/READBACK: SUCCESS")
        else:
            print("PRAMIN WRITE/READBACK: FAILED (VRAM may be uninitialized or inaccessible)")

        # Restore original BAR0_WINDOW.
        struct.pack_into("<I", mm, NV_PBUS_BAR0_WINDOW, bar0_win)
        mm.close()

if __name__ == "__main__":
    bdf = sys.argv[1] if len(sys.argv) > 1 else "0000:4b:00.0"
    test_pramin(bdf)
