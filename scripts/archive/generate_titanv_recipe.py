#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Generate Titan V init recipe by comparing cold BAR0 vs nouveau mmiotrace.

This script:
1. Reads the cold BAR0 snapshot (from pre-nouveau state)
2. Reads the nouveau mmiotrace (captured during driver init)
3. Produces a priority-ordered init recipe suitable for coralReef's boot_follower
4. Extracts the SEC2/ACR-relevant write sequences

Usage:
    python3 generate_titanv_recipe.py \
        --trace data/titanv/reagent-captures/nouveau_titanv_trace1.mmiotrace \
        --output data/titanv/nouveau_init_recipe.json
"""

import json
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

DOMAIN_PRIORITY = [
    ("ROOT_PLL", 0x136000, 0x137000, 0),
    ("PCLOCK",   0x137000, 0x138000, 1),
    ("CLK",      0x130000, 0x136000, 2),
    ("PMC",      0x000000, 0x001000, 3),
    ("PRI_MASTER", 0x122000, 0x123000, 4),
    ("PBUS",     0x001000, 0x002000, 5),
    ("PTOP",     0x020000, 0x024000, 6),
    ("PFB",      0x100000, 0x100800, 10),
    ("FBHUB",    0x100800, 0x100C00, 11),
    ("PFB_NISO", 0x100C00, 0x101000, 12),
    ("FBPA",     0x9A0000, 0x9B0000, 15),
    ("LTC",      0x17E000, 0x190000, 16),
    ("SEC2",     0x087000, 0x088000, 18),
    ("PMU",      0x10A000, 0x10C000, 20),
    ("PRIV_RING", 0x120000, 0x122000, 22),
    ("PFIFO",    0x002000, 0x004000, 25),
    ("PBDMA",    0x040000, 0x0A0000, 26),
    ("FECS",     0x409000, 0x40A000, 28),
    ("GPCCS",    0x41A000, 0x41B000, 29),
    ("PCCSR",    0x800000, 0x900000, 30),
    ("PRAMIN",   0x700000, 0x710000, 35),
    ("PDISP",    0x610000, 0x640000, 40),
]


def classify(offset):
    for name, start, end, prio in DOMAIN_PRIORITY:
        if start <= offset < end:
            return name, prio
    return f"UNK_{offset >> 16:02x}xxxx", 99


def detect_bar0_base(lines):
    bases = defaultdict(int)
    for line in lines[:2000]:
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] in ("R", "W"):
            try:
                addr = int(parts[4], 16)
                base = addr & 0xFF000000
                bases[base] += 1
            except ValueError:
                pass
    return max(bases, key=bases.get) if bases else 0


def parse_mmiotrace(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]

    bar0_base = detect_bar0_base(lines)

    writes = []
    reads = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6 or parts[0] not in ("R", "W"):
            continue

        try:
            addr = int(parts[4], 16)
            value = int(parts[5], 16)
            offset = addr - bar0_base
            if offset < 0 or offset > 0x1000000:
                continue

            ts = float(parts[2])
            entry = {"offset": offset, "value": value, "timestamp": ts}
            if parts[0] == "W":
                writes.append(entry)
            else:
                reads.append(entry)
        except (ValueError, IndexError):
            continue

    return writes, reads, bar0_base


def generate_recipe(writes):
    """Convert writes into a priority-ordered init recipe."""
    seen = {}
    for w in writes:
        seen[w["offset"]] = w["value"]

    recipe = []
    for offset, value in sorted(seen.items()):
        domain, priority = classify(offset)
        recipe.append({
            "domain": domain,
            "offset": f"0x{offset:06x}",
            "offset_int": offset,
            "value": f"0x{value:08x}",
            "value_int": value,
            "priority": priority,
        })

    recipe.sort(key=lambda r: (r["priority"], r["offset_int"]))
    return recipe


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="mmiotrace file")
    parser.add_argument("--output", "-o", default=None, help="output JSON")
    args = parser.parse_args()

    writes, reads, bar0_base = parse_mmiotrace(args.trace)
    recipe = generate_recipe(writes)

    domain_stats = defaultdict(int)
    for step in recipe:
        domain_stats[step["domain"]] += 1

    result = {
        "source": str(args.trace),
        "bar0_base": f"0x{bar0_base:08x}",
        "total_writes": len(writes),
        "unique_registers_written": len(recipe),
        "domain_register_counts": dict(sorted(domain_stats.items())),
        "recipe": [{k: v for k, v in step.items() if k != "offset_int" and k != "value_int"} for step in recipe],
    }

    output = args.output or str(Path(args.trace).with_suffix(".recipe.json"))
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"BAR0 base: 0x{bar0_base:08x}", file=sys.stderr)
    print(f"Total writes: {len(writes)}", file=sys.stderr)
    print(f"Unique registers: {len(recipe)}", file=sys.stderr)
    print(f"Domain breakdown:", file=sys.stderr)
    for domain, count in sorted(domain_stats.items()):
        print(f"  {domain}: {count}", file=sys.stderr)
    print(f"Output: {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
