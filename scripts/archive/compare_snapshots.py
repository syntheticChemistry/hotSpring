#!/usr/bin/env python3
"""Cross-driver register snapshot comparison tool.

Compares two GV100 register snapshots (JSON or text format) and produces a
structured diff highlighting driver-level behavioral differences. Automatically
flags ACR/falcon/MMU-relevant changes that inform the boot solver.

Input formats:
  - JSON: { "registers": [{ "offset": "0x...", "value": "0x...", "name": "..." }, ...] }
  - Text: lines matching "<GROUP> 0x<OFFSET> 0x<VALUE>" or "[0x<OFFSET>] <NAME> = 0x<VALUE>"
  - 070-style JSON: { "registers": [{ "offset": "0x...", "raw_value": N, "value": "0x..." }] }

Usage:
  python3 compare_snapshots.py <baseline> <warm> [--json] [--filter falcon|acr|mmu|pfifo]
  python3 compare_snapshots.py data/070/cold_oracle.json data/070/config_b_oracle_nouveau_warm.json
  python3 compare_snapshots.py data/070/cold_oracle.json data/070/config_c_oracle_nvidia_warm.json --filter falcon
"""

import json
import re
import sys
from pathlib import Path

FALCON_RANGES = {
    "SEC2":  (0x087000, 0x087FFF),
    "FECS":  (0x409000, 0x409FFF),
    "GPCCS": (0x41A000, 0x41AFFF),
    "PMU":   (0x10A000, 0x10AFFF),
}

ACR_REGISTERS = {
    0x087000: "SEC2_CPUCTL",
    0x087004: "SEC2_SCTL",
    0x087008: "SEC2_ITFEN",
    0x087110: "SEC2_BOOTVEC",
    0x087504: "SEC2_MAILBOX0",
    0x087508: "SEC2_MAILBOX1",
    0x409000: "FECS_CPUCTL",
    0x409004: "FECS_SCTL",
    0x409130: "FECS_PC",
    0x409110: "FECS_BOOTVEC",
    0x41A000: "GPCCS_CPUCTL",
    0x41A130: "GPCCS_PC",
}

MMU_REGISTERS = {
    0x100A2C: "MMU_FAULT_STATUS",
    0x100A30: "MMU_FAULT_ADDR_LO",
    0x100A34: "MMU_FAULT_ADDR_HI",
    0x100A38: "MMU_FAULT_INST_LO",
    0x100A3C: "MMU_FAULT_INST_HI",
    0x100A40: "MMU_FAULT_INFO",
    0x100C80: "MMU_PRI_CTRL",
    0x100CBC: "MMU_TLB_FLUSH",
    0x100E24: "FAULT_BUF0_LO",
    0x100E28: "FAULT_BUF0_HI",
    0x100E2C: "FAULT_BUF0_SIZE",
    0x100E30: "FAULT_BUF0_GET",
    0x100E34: "FAULT_BUF0_PUT",
}

PFIFO_REGISTERS = {
    0x002004: "PFIFO_PBDMA_MAP",
    0x002100: "PFIFO_INTR",
    0x002140: "PFIFO_INTR_EN",
    0x002200: "PFIFO_ENABLE",
    0x002270: "RUNLIST_BASE",
    0x002630: "SCHED_DISABLE",
    0x002634: "PREEMPT",
}

FILTER_GROUPS = {
    "falcon": FALCON_RANGES,
    "acr": ACR_REGISTERS,
    "mmu": MMU_REGISTERS,
    "pfifo": PFIFO_REGISTERS,
}


def parse_json(path: Path) -> dict[int, tuple[int, str]]:
    """Parse JSON register snapshot. Returns {offset: (value, name)}."""
    data = json.loads(path.read_text())
    regs = {}
    for entry in data.get("registers", []):
        off_str = entry.get("offset", "0x0")
        off = int(off_str, 16) if isinstance(off_str, str) else int(off_str)
        if "raw_value" in entry:
            val = int(entry["raw_value"])
        else:
            val_str = entry.get("value", "0x0")
            val = int(val_str, 16) if isinstance(val_str, str) else int(val_str)
        name = entry.get("name", f"REG_{off:#08x}")
        regs[off] = (val, name)
    return regs


TEXT_LINE_PATTERNS = [
    re.compile(r"^\s*(\S+)\s+(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)"),
    re.compile(r"\[(0x[0-9a-fA-F]+)\]\s+(\S+)\s*=\s*(0x[0-9a-fA-F]+)"),
]


def parse_text(path: Path) -> dict[int, tuple[int, str]]:
    """Parse text register dump. Returns {offset: (value, name)}."""
    regs = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("╔") or line.startswith("╚"):
            continue
        line = line.lstrip("║ ")

        m = TEXT_LINE_PATTERNS[0].match(line)
        if m:
            group = m.group(1)
            off = int(m.group(2), 16)
            val = int(m.group(3), 16)
            regs[off] = (val, f"{group}_{off:#08x}")
            continue

        m = TEXT_LINE_PATTERNS[1].match(line)
        if m:
            off = int(m.group(1), 16)
            name = m.group(2)
            val = int(m.group(3), 16)
            regs[off] = (val, name)
    return regs


def load_snapshot(path: Path) -> dict[int, tuple[int, str]]:
    if path.suffix == ".json":
        return parse_json(path)
    return parse_text(path)


def classify_offset(offset: int) -> list[str]:
    """Return which groups this offset belongs to."""
    tags = []
    for name, (start, end) in FALCON_RANGES.items():
        if start <= offset <= end:
            tags.append(f"falcon:{name}")
    if offset in ACR_REGISTERS:
        tags.append("acr")
    if offset in MMU_REGISTERS:
        tags.append("mmu")
    if offset in PFIFO_REGISTERS:
        tags.append("pfifo")
    if 0x040000 <= offset <= 0x04FFFF:
        tags.append("pbdma")
    if 0x800000 <= offset <= 0x800FFF:
        tags.append("pccsr")
    if 0x020000 <= offset <= 0x020FFF:
        tags.append("therm")
    return tags or ["other"]


def offset_in_filter(offset: int, filter_name: str) -> bool:
    if filter_name == "falcon":
        return any(s <= offset <= e for s, e in FALCON_RANGES.values())
    if filter_name == "acr":
        return offset in ACR_REGISTERS or any(s <= offset <= e for s, e in FALCON_RANGES.values())
    if filter_name == "mmu":
        return offset in MMU_REGISTERS or (0x100000 <= offset <= 0x100FFF)
    if filter_name == "pfifo":
        return offset in PFIFO_REGISTERS or (0x002000 <= offset <= 0x002FFF)
    return True


def compare(baseline: dict, warm: dict, filter_name: str | None = None):
    all_offsets = sorted(set(baseline.keys()) | set(warm.keys()))

    changed = []
    appeared = []
    disappeared = []
    unchanged_count = 0

    for off in all_offsets:
        if filter_name and not offset_in_filter(off, filter_name):
            continue

        in_base = off in baseline
        in_warm = off in warm

        if in_base and in_warm:
            bval, bname = baseline[off]
            wval, wname = warm[off]
            name = bname if bname != f"REG_{off:#08x}" else wname
            if bval != wval:
                changed.append((off, name, bval, wval, classify_offset(off)))
            else:
                unchanged_count += 1
        elif in_warm and not in_base:
            wval, wname = warm[off]
            appeared.append((off, wname, wval, classify_offset(off)))
        elif in_base and not in_warm:
            bval, bname = baseline[off]
            disappeared.append((off, bname, bval, classify_offset(off)))

    return changed, appeared, disappeared, unchanged_count


def print_diff(baseline_path, warm_path, changed, appeared, disappeared, unchanged, as_json=False):
    if as_json:
        result = {
            "baseline": str(baseline_path),
            "warm": str(warm_path),
            "changed": [
                {
                    "offset": f"{off:#08x}",
                    "name": name,
                    "baseline": f"{bv:#010x}",
                    "warm": f"{wv:#010x}",
                    "delta": f"{(wv - bv) & 0xFFFFFFFF:#010x}",
                    "tags": tags,
                }
                for off, name, bv, wv, tags in changed
            ],
            "appeared": [
                {"offset": f"{off:#08x}", "name": n, "value": f"{v:#010x}", "tags": t}
                for off, n, v, t in appeared
            ],
            "disappeared": [
                {"offset": f"{off:#08x}", "name": n, "value": f"{v:#010x}", "tags": t}
                for off, n, v, t in disappeared
            ],
            "unchanged_count": unchanged,
            "summary": {
                "changed": len(changed),
                "appeared": len(appeared),
                "disappeared": len(disappeared),
                "unchanged": unchanged,
                "total": len(changed) + len(appeared) + len(disappeared) + unchanged,
            },
        }
        print(json.dumps(result, indent=2))
        return

    print(f"╔══ CROSS-DRIVER COMPARISON ═══════════════════════════════════════╗")
    print(f"║ Baseline: {baseline_path}")
    print(f"║ Warm:     {warm_path}")
    print(f"╠══════════════════════════════════════════════════════════════════╣")

    acr_changes = []

    if changed:
        print(f"║ CHANGED REGISTERS ({len(changed)}):")
        for off, name, bv, wv, tags in changed:
            tag_str = ",".join(tags)
            marker = " ◀ ACR" if "acr" in tags or any("falcon" in t for t in tags) else ""
            print(f"║   [{off:#08x}] {name:32s}  {bv:#010x} → {wv:#010x}  [{tag_str}]{marker}")
            if marker:
                acr_changes.append((off, name, bv, wv))

    if appeared:
        print(f"║ APPEARED ({len(appeared)}):")
        for off, name, val, tags in appeared:
            tag_str = ",".join(tags)
            print(f"║   [{off:#08x}] {name:32s}  (new) {val:#010x}  [{tag_str}]")

    if disappeared:
        print(f"║ DISAPPEARED ({len(disappeared)}):")
        for off, name, val, tags in disappeared:
            tag_str = ",".join(tags)
            print(f"║   [{off:#08x}] {name:32s}  {val:#010x} → (gone)  [{tag_str}]")

    print(f"╠══════════════════════════════════════════════════════════════════╣")
    print(f"║ Changed: {len(changed)}  Appeared: {len(appeared)}  Gone: {len(disappeared)}  Unchanged: {unchanged}")

    if acr_changes:
        print(f"╠══ ACR/FALCON-RELEVANT CHANGES ═══════════════════════════════════╣")
        for off, name, bv, wv in acr_changes:
            acr_name = ACR_REGISTERS.get(off, name)
            print(f"║   {acr_name}: {bv:#010x} → {wv:#010x}")
            if "CPUCTL" in acr_name:
                halted = "HALTED" if wv & (1 << 4) else "running"
                startcpu = "STARTCPU" if wv & (1 << 1) else ""
                print(f"║     → {halted} {startcpu}")
            elif "SCTL" in acr_name:
                sec_mode = (wv >> 20) & 0xF
                modes = {0: "unset", 1: "falcon", 2: "HS", 4: "LS"}
                print(f"║     → security mode: {modes.get(sec_mode, f'unknown({sec_mode})')}")
            elif "BOOTVEC" in acr_name:
                print(f"║     → boot vector: {wv:#010x}")

    print(f"╚══════════════════════════════════════════════════════════════════╝")


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)

    baseline_path = Path(args[0])
    warm_path = Path(args[1])
    as_json = "--json" in args
    filter_name = None
    for i, a in enumerate(args):
        if a == "--filter" and i + 1 < len(args):
            filter_name = args[i + 1]

    baseline = load_snapshot(baseline_path)
    warm = load_snapshot(warm_path)

    if not baseline:
        print(f"ERROR: No registers parsed from {baseline_path}")
        sys.exit(1)
    if not warm:
        print(f"ERROR: No registers parsed from {warm_path}")
        sys.exit(1)

    changed, appeared, disappeared, unchanged = compare(baseline, warm, filter_name)
    print_diff(baseline_path, warm_path, changed, appeared, disappeared, unchanged, as_json)


if __name__ == "__main__":
    main()
