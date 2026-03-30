#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Parse mmiotrace logs into structured register write sequences.

Extracts MMIO read/write operations from kernel mmiotrace output,
groups by register region, and produces JSON suitable for feeding
to coralReef's ACR boot solver.

Usage:
    python3 parse_mmiotrace.py <trace.mmiotrace> [--bar0-base 0xADDR] [--output out.json]
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REGISTER_REGIONS = {
    "PMC":       (0x000000, 0x001000),
    "PBUS":      (0x001000, 0x002000),
    "PFIFO":     (0x002000, 0x004000),
    "PTIMER":    (0x009000, 0x00A000),
    "PFB":       (0x100000, 0x102000),
    "PCLOCK":    (0x137000, 0x138000),
    "PPCI":      (0x088000, 0x089000),
    "PROM":      (0x300000, 0x301000),
    "PGRAPH":    (0x400000, 0x420000),
    "FECS":      (0x409000, 0x40A000),
    "GPCCS":     (0x41A000, 0x41B000),
    "PMU":       (0x10A000, 0x10B000),
    "SEC2":      (0x840000, 0x841000),
    "ACR":       (0x862000, 0x863000),
    "PRIV_RING": (0x120000, 0x130000),
    "THERM":     (0x020000, 0x021000),
    "FUSE":      (0x021000, 0x022000),
    "PDISP":     (0x610000, 0x640000),
    "PCOPY":     (0x104000, 0x106000),
}


def classify_register(offset):
    for name, (start, end) in REGISTER_REGIONS.items():
        if start <= offset < end:
            return name
    return f"UNK_{offset >> 16:02x}xxxx"


def detect_bar0_base(lines):
    """Heuristic: find the most common high-nibble in addresses."""
    bases = defaultdict(int)
    for line in lines[:2000]:
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] in ("R", "W"):
            addr = int(parts[4], 16)
            base = addr & 0xFF000000
            bases[base] += 1
    if bases:
        return max(bases, key=bases.get)
    return 0


def parse_trace(path, bar0_base=None):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]

    if bar0_base is None:
        bar0_base = detect_bar0_base(lines)

    ops = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        op_type = parts[0]
        if op_type not in ("R", "W"):
            continue

        width = int(parts[1])
        timestamp = float(parts[2])
        pid = int(parts[3])
        virt_addr = int(parts[4], 16)
        value = int(parts[5], 16)

        offset = virt_addr - bar0_base
        if offset < 0 or offset > 0x2000000:
            continue

        region = classify_register(offset)
        ops.append({
            "type": op_type,
            "offset": offset,
            "value": value,
            "region": region,
            "timestamp": timestamp,
            "width": width,
        })

    return ops, bar0_base


def summarize(ops):
    """Produce a summary of the trace."""
    reads = [o for o in ops if o["type"] == "R"]
    writes = [o for o in ops if o["type"] == "W"]

    region_writes = defaultdict(list)
    for w in writes:
        region_writes[w["region"]].append(w)

    summary = {
        "total_ops": len(ops),
        "total_reads": len(reads),
        "total_writes": len(writes),
        "regions": {},
    }

    for region in sorted(region_writes.keys()):
        rw = region_writes[region]
        unique_offsets = len(set(w["offset"] for w in rw))
        summary["regions"][region] = {
            "write_count": len(rw),
            "unique_registers": unique_offsets,
            "first_write_offset": f"0x{rw[0]['offset']:06x}",
            "last_write_offset": f"0x{rw[-1]['offset']:06x}",
        }

    return summary


def extract_write_sequence(ops, region_filter=None):
    """Extract ordered write sequence, optionally filtered by region."""
    writes = [o for o in ops if o["type"] == "W"]
    if region_filter:
        writes = [w for w in writes if w["region"] in region_filter]
    return [
        {"offset": f"0x{w['offset']:06x}", "value": f"0x{w['value']:08x}", "region": w["region"]}
        for w in writes
    ]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="Path to .mmiotrace file")
    parser.add_argument("--bar0-base", help="BAR0 virtual base address (auto-detected if omitted)")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--region", "-r", action="append", help="Filter to specific regions")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    bar0 = int(args.bar0_base, 16) if args.bar0_base else None
    ops, detected_base = parse_trace(args.trace, bar0)

    print(f"BAR0 base: 0x{detected_base:08x}", file=sys.stderr)
    print(f"Total ops: {len(ops)}", file=sys.stderr)

    summary = summarize(ops)

    if args.summary_only:
        result = summary
    else:
        writes = extract_write_sequence(ops, set(args.region) if args.region else None)
        result = {
            "source": str(args.trace),
            "bar0_base": f"0x{detected_base:08x}",
            "summary": summary,
            "write_sequence": writes,
        }

    output = args.output or args.trace.replace(".mmiotrace", "_parsed.json")
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Output: {output} ({len(result.get('write_sequence', []))} writes)", file=sys.stderr)


if __name__ == "__main__":
    main()
