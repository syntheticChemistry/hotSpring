#!/usr/bin/env python3
"""Read a u32 from PCI BAR0 via sysfs resource0 mmap.

Usage: bar0_read.py <bdf> <hex_offset>
       bar0_read.py 0000:03:00.0 0x409100
Output: hex value on stdout, e.g. 0x00000010
"""
import mmap
import os
import struct
import sys

if len(sys.argv) != 3:
    print(f"usage: {sys.argv[0]} <bdf> <hex_offset>", file=sys.stderr)
    sys.exit(1)

bdf = sys.argv[1]
offset = int(sys.argv[2], 0)
resource_path = f"/sys/bus/pci/devices/{bdf}/resource0"

fd = os.open(resource_path, os.O_RDONLY)
try:
    m = mmap.mmap(fd, 16 * 1024 * 1024, mmap.MAP_SHARED, mmap.PROT_READ)
    m.seek(offset)
    val = struct.unpack("<I", m.read(4))[0]
    print(f"0x{val:08x}")
    m.close()
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    os.close(fd)
