#!/usr/bin/env python3
"""
Extract DEVINIT init scripts from a GPU's VBIOS via BAR0 PROM.

Pipeline:
1. Read VBIOS from BAR0+0x300000 (PROM region) via resource0
2. Run envytools/nvbios to parse DEVINIT scripts
3. Extract register operations into JSON recipe

Usage:
    python3 extract_devinit.py <bdf> <output_dir> [--nvbios PATH]
"""
import json, mmap, os, re, struct, subprocess, sys, tempfile

PROM_BASE = 0x300000
PROM_SIZE = 0x10000  # 64KB typical VBIOS
VBIOS_SIG = bytes([0x55, 0xAA])


def dump_vbios_from_prom(bdf: str, output_path: str) -> int:
    resource0 = f"/sys/bus/pci/devices/{bdf}/resource0"
    fd = os.open(resource0, os.O_RDWR | os.O_SYNC)
    size = os.path.getsize(resource0)
    map_size = min(size, PROM_BASE + PROM_SIZE)

    if map_size < PROM_BASE + 3:
        os.close(fd)
        raise RuntimeError(f"BAR0 too small ({size}) to reach PROM at 0x{PROM_BASE:x}")

    mm = mmap.mmap(fd, map_size, mmap.MAP_SHARED, mmap.PROT_READ)

    sig = struct.unpack_from("<H", mm, PROM_BASE)[0]
    if sig != 0xAA55:
        mm.close()
        os.close(fd)
        raise RuntimeError(f"No valid VBIOS signature at PROM (got 0x{sig:04x})")

    img_size = mm[PROM_BASE + 2] * 512
    if img_size == 0 or PROM_BASE + img_size > map_size:
        img_size = PROM_SIZE

    rom_data = bytes(mm[PROM_BASE : PROM_BASE + img_size])
    mm.close()
    os.close(fd)

    with open(output_path, "wb") as f:
        f.write(rom_data)

    print(f"VBIOS dumped: {len(rom_data)} bytes -> {output_path}")
    return len(rom_data)


def parse_nvbios(rom_path: str, nvbios_bin: str) -> str:
    result = subprocess.run(
        [nvbios_bin, rom_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout + result.stderr


def extract_scripts(nvbios_text: str) -> list:
    scripts = []
    in_script = False
    current_script = None

    for line in nvbios_text.split("\n"):
        m = re.match(r"Init script (\d+) at (0x[0-9a-f]+):", line)
        if m:
            if current_script:
                scripts.append(current_script)
            current_script = {"id": int(m.group(1)), "addr": m.group(2), "ops": []}
            in_script = True
            continue

        m2 = re.match(r"Some script at (0x[0-9a-f]+):", line)
        if m2:
            if current_script:
                scripts.append(current_script)
            current_script = {"id": -1, "addr": m2.group(1), "ops": []}
            in_script = True
            continue

        if in_script and (
            line.strip() == ""
            or (not line.startswith("0x") and not line.startswith("\t"))
        ):
            if current_script:
                scripts.append(current_script)
                current_script = None
            in_script = False
            continue

        if not in_script or not current_script:
            continue

        m = re.search(
            r"ZM_REG\s+R\[0x([0-9a-fA-F]+)\]\s*=\s*0x([0-9a-fA-F]+)", line
        )
        if m:
            current_script["ops"].append(
                {
                    "type": "ZmReg",
                    "reg": int(m.group(1), 16),
                    "val": int(m.group(2), 16),
                }
            )
            continue

        m = re.search(
            r"NV_REG\s+R\[0x([0-9a-fA-F]+)\]\s*&=\s*0x([0-9a-fA-F]+)\s*\|=\s*0x([0-9a-fA-F]+)",
            line,
        )
        if m:
            current_script["ops"].append(
                {
                    "type": "NvReg",
                    "reg": int(m.group(1), 16),
                    "mask": int(m.group(2), 16),
                    "or_val": int(m.group(3), 16),
                }
            )
            continue

        m = re.search(r"R\[0x([0-9a-fA-F]+)\]\s*=\s*0x([0-9a-fA-F]+)", line)
        if m and "ZM_REG_SEQUENCE" not in line:
            current_script["ops"].append(
                {
                    "type": "ZmReg",
                    "reg": int(m.group(1), 16),
                    "val": int(m.group(2), 16),
                }
            )
            continue

        m = re.search(r"TIME\s+0x([0-9a-fA-F]+)", line)
        if m:
            current_script["ops"].append(
                {"type": "Time", "usec": int(m.group(1), 16)}
            )
            continue

        m = re.search(
            r"ZM_MASK_ADD\s+R\[0x([0-9a-fA-F]+)\]\s*&\s*~0x([0-9a-fA-F]+)\s*\+=\s*0x([0-9a-fA-F]+)",
            line,
        )
        if m:
            current_script["ops"].append(
                {
                    "type": "ZmMaskAdd",
                    "reg": int(m.group(1), 16),
                    "inv_mask": int(m.group(2), 16),
                    "add_val": int(m.group(3), 16),
                }
            )
            continue

    if current_script:
        scripts.append(current_script)

    return scripts


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <bdf> <output_dir> [--nvbios PATH]")
        sys.exit(1)

    bdf = sys.argv[1]
    output_dir = sys.argv[2]
    nvbios_bin = "/tmp/envytools/build/nvbios/nvbios"

    for i, arg in enumerate(sys.argv):
        if arg == "--nvbios" and i + 1 < len(sys.argv):
            nvbios_bin = sys.argv[i + 1]

    os.makedirs(output_dir, exist_ok=True)

    rom_path = os.path.join(output_dir, "vbios_prom.rom")
    print(f"=== DEVINIT Extraction for {bdf} ===\n")

    print("Step 1: Dump VBIOS from PROM...")
    dump_vbios_from_prom(bdf, rom_path)

    print("\nStep 2: Parse with nvbios...")
    nvbios_text = parse_nvbios(rom_path, nvbios_bin)

    nvbios_log = os.path.join(output_dir, "nvbios_output.txt")
    with open(nvbios_log, "w") as f:
        f.write(nvbios_text)
    print(f"  nvbios output: {len(nvbios_text)} chars -> {nvbios_log}")

    print("\nStep 3: Extract init scripts...")
    all_scripts = extract_scripts(nvbios_text)
    total_ops = sum(len(s["ops"]) for s in all_scripts)
    print(f"  Found {len(all_scripts)} scripts, {total_ops} total ops")

    for s in all_scripts:
        op_types = {}
        for op in s["ops"]:
            op_types[op["type"]] = op_types.get(op["type"], 0) + 1
        print(f"    Script {s['id']} at {s['addr']}: {len(s['ops'])} ops  {op_types}")

    all_path = os.path.join(output_dir, "devinit_all_scripts.json")
    with open(all_path, "w") as f:
        json.dump(all_scripts, f, indent=2)

    safe_scripts = [s for s in all_scripts if s["id"] in [0, 1]]
    safe_path = os.path.join(output_dir, "devinit_safe_recipe.json")
    with open(safe_path, "w") as f:
        json.dump(safe_scripts, f, indent=2)

    safe_ops = sum(len(s["ops"]) for s in safe_scripts)
    print(f"\nStep 4: Safe recipe (scripts 0+1 only): {safe_ops} ops -> {safe_path}")
    print(f"  Full dump: {all_path}")
    print("\nDone. Use replay_devinit.py to apply the safe recipe.")


if __name__ == "__main__":
    main()
