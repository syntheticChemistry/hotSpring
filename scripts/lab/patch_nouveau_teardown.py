#!/usr/bin/env python3
"""
patch_nouveau_teardown.py — Binary-patch stock nouveau.ko to NOP teardown functions.

Patches 4 functions to immediately return (preserving GPU state for VFIO handoff):
  1. gf100_gr_fini    → return 0 (preserves GPC state)
  2. nvkm_pmu_fini    → return 0 (keeps PMU falcon alive)
  3. nvkm_mc_disable  → return   (preserves PMC_ENABLE clocks)
  4. nvkm_fifo_fini   → return 0 (preserves PFIFO/FECS)

Each function starts with a 5-byte __fentry__ call (e8 00 00 00 00).
We patch byte +5 onward with xor eax,eax; ret (int fns) or ret (void fns).

Usage:
  sudo python3 patch_nouveau_teardown.py [/path/to/nouveau.ko] [/path/to/output.ko]
"""

import shutil
import struct
import subprocess
import sys
from pathlib import Path

STOCK = Path(f"/lib/modules/{subprocess.check_output(['uname', '-r']).decode().strip()}"
             "/kernel/drivers/gpu/drm/nouveau/nouveau.ko")

TARGETS = [
    ("gf100_gr_fini",   "int"),
    ("nvkm_pmu_fini",   "int"),
    ("nvkm_fifo_fini",  "int"),
    ("nvkm_mc_disable", "void"),
]

RET_INT  = b"\x31\xc0\xc3"  # xor eax,eax; ret
RET_VOID = b"\xc3"           # ret


def find_text_offset(ko_path: Path) -> int:
    out = subprocess.check_output(
        ["readelf", "-S", str(ko_path)], text=True
    )
    for line in out.splitlines():
        parts = line.split()
        if ".text" in parts and "PROGBITS" in parts:
            idx = parts.index("PROGBITS")
            return int(parts[idx + 2], 16)
    raise RuntimeError("cannot find .text section file offset")


def find_func_offsets(ko_path: Path) -> dict[str, tuple[int, int]]:
    out = subprocess.check_output(
        ["objdump", "-t", str(ko_path)], text=True
    )
    result = {}
    for line in out.splitlines():
        for name, _ in TARGETS:
            if f" {name}\n" in line + "\n" or line.strip().endswith(f" {name}"):
                parts = line.split()
                addr = int(parts[0], 16)
                size = int(parts[4], 16) if len(parts) > 4 else 0
                section = parts[3] if len(parts) > 3 else ""
                if section == ".text" and name not in result:
                    result[name] = (addr, size)
    return result


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else STOCK
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/nouveau-patch/nouveau_patched.ko")

    if not src.exists():
        print(f"ERROR: {src} not found")
        sys.exit(1)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    text_file_off = find_text_offset(dst)
    func_offsets = find_func_offsets(dst)

    print(f"Stock module:  {src}")
    print(f"Output:        {dst}")
    print(f".text file offset: 0x{text_file_off:x}")
    print()

    patched = 0
    with open(dst, "r+b") as f:
        for name, ret_type in TARGETS:
            if name not in func_offsets:
                print(f"  SKIP: {name} not found in symbol table")
                continue

            sec_off, size = func_offsets[name]
            file_off = text_file_off + sec_off
            patch_off = file_off + 5  # skip __fentry__ call

            f.seek(file_off)
            preamble = f.read(5)
            if preamble != b"\xe8\x00\x00\x00\x00":
                print(f"  WARN: {name} preamble is {preamble.hex()}, expected e800000000")

            patch = RET_INT if ret_type == "int" else RET_VOID
            f.seek(patch_off)
            original = f.read(len(patch))
            f.seek(patch_off)
            f.write(patch)
            patched += 1

            print(f"  PATCH: {name} @ .text+0x{sec_off:x} (file 0x{patch_off:x})"
                  f"  {original.hex()} → {patch.hex()}"
                  f"  [{ret_type} → {'xor eax,eax; ret' if ret_type == 'int' else 'ret'}]")

    print(f"\n{patched}/{len(TARGETS)} functions patched → {dst}")


if __name__ == "__main__":
    main()
