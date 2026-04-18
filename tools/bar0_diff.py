#!/usr/bin/env python3
"""BAR0 register diff analyzer for sovereign GPU experiments.

Compares cold-state and warm-state BAR0 register snapshots (JSON) captured
by gpu-state.py or similar tooling inside VM reagents. Extracts the register
write delta that the vendor driver applied, organized by hardware region.

The output is usable as a TrainingRecipe ingredient for sovereign init replay.

Usage:
    python3 bar0_diff.py <cold.json> <warm.json> [--region FECS] [--format recipe]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


FALCON_REG_NAMES = {
    0x000: "IRQSSET", 0x004: "IRQSCLR", 0x008: "IRQSTAT", 0x010: "IRQMSET",
    0x014: "IRQMCLR", 0x018: "IRQDEST", 0x040: "MAILBOX0", 0x044: "MAILBOX1",
    0x048: "ITFEN", 0x04C: "IDLESTATE", 0x050: "CURCTX", 0x054: "NXTCTX",
    0x060: "ENGCTL", 0x064: "INTR", 0x068: "INTR_ROUTE", 0x080: "OS",
    0x094: "EXCI", 0x098: "TRACEPC", 0x100: "CPUCTL", 0x104: "BOOTVEC",
    0x108: "HWCFG", 0x10C: "DMACTL", 0x110: "DMATRFBASE", 0x114: "DMATRFMOFFS",
    0x118: "DMATRFCMD", 0x11C: "DMATRFFBOFFS", 0x120: "DMATRFSTAT",
    0x128: "DEBUGPC", 0x12C: "DEBUGINFO", 0x130: "CPUCTL_ALIAS",
    0x140: "SEC2", 0x180: "IMEMC", 0x184: "IMEMD", 0x188: "IMEMT",
    0x1C0: "DMEMC", 0x1C4: "DMEMD", 0x240: "SCTL",
}


def load_snapshot(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def region_regs(snapshot: dict, region: str) -> dict[int, int]:
    raw = snapshot.get("regions", {}).get(region, {})
    return {int(k, 16): int(v, 16) for k, v in raw.items()}


def diff_regions(cold_regs: dict[int, int], warm_regs: dict[int, int]):
    all_offsets = sorted(set(cold_regs) | set(warm_regs))
    added, changed, removed = [], [], []

    for off in all_offsets:
        c = cold_regs.get(off)
        w = warm_regs.get(off)
        if c is None and w is not None:
            added.append((off, w))
        elif c is not None and w is None:
            removed.append((off, c))
        elif c != w:
            changed.append((off, c, w))

    return added, changed, removed


def falcon_reg_name(base: int, offset: int) -> str:
    rel = offset - base
    name = FALCON_REG_NAMES.get(rel, "")
    if name:
        return f"+0x{rel:03X} ({name})"
    return f"+0x{rel:03X}"


def analyze_falcon(name: str, base: int, cold: dict, warm: dict):
    cold_regs = region_regs(cold, name)
    warm_regs = region_regs(warm, name)
    added, changed, removed = diff_regions(cold_regs, warm_regs)

    print(f"\n{'=' * 70}")
    print(f"  {name} (base 0x{base:06X})")
    print(f"  Cold: {len(cold_regs)} regs | Warm: {len(warm_regs)} regs")
    print(f"  Delta: +{len(added)} new, ~{len(changed)} changed, -{len(removed)} cleared")
    print(f"{'=' * 70}")

    if changed:
        print(f"\n  --- Changed registers ({len(changed)}) ---")
        for off, old, new in changed:
            rn = falcon_reg_name(base, off)
            print(f"  0x{off:06X} {rn:30s}  {old:#010x} -> {new:#010x}")

    if added:
        print(f"\n  --- New registers ({len(added)}) ---")
        for off, val in added:
            rn = falcon_reg_name(base, off)
            print(f"  0x{off:06X} {rn:30s}  {val:#010x}")

    if removed:
        print(f"\n  --- Cleared registers ({len(removed)}) ---")
        for off, val in removed:
            rn = falcon_reg_name(base, off)
            print(f"  0x{off:06X} {rn:30s}  was {val:#010x}")


def analyze_all(cold: dict, warm: dict, focus: str | None):
    regions = {
        "FECS":  0x409000,
        "GPCCS": 0x41A000,
        "SEC2":  0x840000,
        "PMU":   0x10A000,
        "PMC":   0x000000,
        "PFB":   0x100000,
        "PFIFO": 0x002000,
        "PGRAPH": 0x400000,
    }

    if focus:
        if focus.upper() not in regions:
            print(f"Unknown region: {focus}. Available: {', '.join(regions)}")
            sys.exit(1)
        regions = {focus.upper(): regions[focus.upper()]}

    for name, base in regions.items():
        if name in cold.get("regions", {}) or name in warm.get("regions", {}):
            analyze_falcon(name, base, cold, warm)

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Total register delta (all regions)")
    print(f"{'=' * 70}")
    total_add = total_change = total_remove = 0
    for name in cold.get("regions", {}):
        cold_regs = region_regs(cold, name)
        warm_regs = region_regs(warm, name)
        a, c, r = diff_regions(cold_regs, warm_regs)
        total_add += len(a)
        total_change += len(c)
        total_remove += len(r)
        if a or c or r:
            print(f"  {name:12s}: +{len(a):4d} new, ~{len(c):4d} changed, -{len(r):4d} cleared")
    print(f"  {'TOTAL':12s}: +{total_add:4d} new, ~{total_change:4d} changed, -{total_remove:4d} cleared")


def export_recipe(cold: dict, warm: dict, output: str):
    writes = []
    for name in warm.get("regions", {}):
        cold_regs = region_regs(cold, name)
        warm_regs = region_regs(warm, name)
        added, changed, _ = diff_regions(cold_regs, warm_regs)
        for off, val in added:
            writes.append({"offset": off, "value": val, "region": name, "type": "new"})
        for off, _, val in changed:
            writes.append({"offset": off, "value": val, "region": name, "type": "changed"})

    recipe = {
        "format": "bar0_diff_recipe",
        "version": 1,
        "gpu": warm.get("bdf", "unknown"),
        "total_writes": len(writes),
        "writes": writes,
    }
    with open(output, "w") as f:
        json.dump(recipe, f, indent=2)
    print(f"\nExported {len(writes)} register writes to {output}")


def main():
    parser = argparse.ArgumentParser(description="BAR0 register diff analyzer")
    parser.add_argument("cold", help="Cold-state JSON snapshot")
    parser.add_argument("warm", help="Warm-state JSON snapshot")
    parser.add_argument("--region", "-r", help="Focus on a single region (FECS, PMU, etc.)")
    parser.add_argument("--format", "-f", choices=["text", "recipe"], default="text",
                        help="Output format: text (default) or recipe (JSON)")
    parser.add_argument("--output", "-o", help="Output file for recipe format")
    args = parser.parse_args()

    cold = load_snapshot(args.cold)
    warm = load_snapshot(args.warm)

    if args.format == "recipe":
        output = args.output or args.warm.replace(".json", "_recipe.json")
        export_recipe(cold, warm, output)
    else:
        analyze_all(cold, warm, args.region)


if __name__ == "__main__":
    main()
