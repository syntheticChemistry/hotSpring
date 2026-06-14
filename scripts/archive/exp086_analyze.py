#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Exp 086 — Cross-driver falcon profile analysis.

Loads all JSON captures from data/086/ and produces:
  1. Per-register comparison matrix (GPU × driver-state)
  2. Delta report: registers that CHANGE between driver warm-ups
  3. Cross-card divergence: Titan #1 vs #2 in same state
  4. WPR region analysis
  5. Binding register residual analysis

Usage:
    python3 exp086_analyze.py [data_dir]

Default data_dir: ../data/086/ (relative to script)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_profiles(data_dir: Path) -> dict:
    """Load all JSON profiles keyed by filename stem."""
    profiles = {}
    for f in sorted(data_dir.glob("*.json")):
        with open(f) as fh:
            profiles[f.stem] = json.load(fh)
    return profiles


def extract_register_table(profiles: dict) -> dict:
    """Build {engine -> {register -> {profile_name -> hex_value}}}."""
    table = defaultdict(lambda: defaultdict(dict))

    for pname, profile in profiles.items():
        for engine_name, falcon in profile.get("falcons", {}).items():
            for reg_name, reg_data in falcon.get("registers", {}).items():
                table[engine_name][reg_name][pname] = reg_data["hex"]

        for reg_name, reg_data in profile.get("pfb_wpr", {}).items():
            if isinstance(reg_data, dict) and "hex" in reg_data:
                table["PFB"][reg_name][pname] = reg_data["hex"]

        for reg_name, reg_data in profile.get("pmc", {}).items():
            if isinstance(reg_data, dict) and "hex" in reg_data:
                table["PMC"][reg_name][pname] = reg_data["hex"]

    return dict(table)


def find_changing_registers(table: dict) -> dict:
    """Find registers whose values differ across profiles."""
    changes = {}
    for engine, regs in table.items():
        for reg_name, profile_values in regs.items():
            unique_vals = set(profile_values.values())
            if len(unique_vals) > 1:
                changes.setdefault(engine, {})[reg_name] = profile_values
    return changes


def find_cross_card_divergence(table: dict, profiles: dict) -> dict:
    """Find registers that differ between Titan #1 and #2 in the same driver state."""
    states = set()
    for pname in profiles:
        state = pname.replace("titan1_", "").replace("titan2_", "")
        states.add(state)

    divergence = {}
    for state in sorted(states):
        t1_key = f"titan1_{state}"
        t2_key = f"titan2_{state}"
        if t1_key not in profiles or t2_key not in profiles:
            continue

        diffs = {}
        for engine, regs in table.items():
            for reg_name, pvals in regs.items():
                v1 = pvals.get(t1_key)
                v2 = pvals.get(t2_key)
                if v1 and v2 and v1 != v2:
                    diffs.setdefault(engine, {})[reg_name] = {"titan1": v1, "titan2": v2}

        if diffs:
            divergence[state] = diffs

    return divergence


def analyze_wpr(profiles: dict) -> list:
    """Summarize WPR region state across all profiles."""
    rows = []
    for pname, profile in sorted(profiles.items()):
        pfb = profile.get("pfb_wpr", {})
        decoded = pfb.get("decoded", {})
        rows.append({
            "profile": pname,
            "driver": profile.get("meta", {}).get("driver", "?"),
            "wpr1": decoded.get("wpr1_range", "?"),
            "wpr2": decoded.get("wpr2_range", "?"),
            "wpr1_active": decoded.get("wpr1_active", False),
            "wpr2_active": decoded.get("wpr2_active", False),
        })
    return rows


def analyze_binding(profiles: dict) -> list:
    """Summarize binding register state for SEC2 across all profiles."""
    rows = []
    for pname, profile in sorted(profiles.items()):
        sec2 = profile.get("falcons", {}).get("SEC2", {})
        regs = sec2.get("registers", {})
        decoded = sec2.get("decoded", {})
        rows.append({
            "profile": pname,
            "driver": profile.get("meta", {}).get("driver", "?"),
            "bind_inst": regs.get("BIND_INST", {}).get("hex", "?"),
            "bind_stat": decoded.get("bind_stat_raw", "?"),
            "bind_stat_field": decoded.get("bind_stat_field", "?"),
            "unk090": regs.get("UNK090", {}).get("hex", "?"),
            "eng_ctrl": regs.get("ENG_CTRL", {}).get("hex", "?"),
            "chan_trig": regs.get("CHAN_TRIG", {}).get("hex", "?"),
            "dmaidx": regs.get("DMAIDX", {}).get("hex", "?"),
            "cpuctl": regs.get("CPUCTL", {}).get("hex", "?"),
        })
    return rows


def analyze_driver_warmup_effect(profiles: dict) -> list:
    """Compare ACR-relevant state: vfio-cold vs post-nouveau vs post-nvidia."""
    keys_of_interest = [
        ("SEC2", "CPUCTL"), ("SEC2", "SCTL"), ("SEC2", "EXCI"),
        ("SEC2", "BIND_STAT"), ("SEC2", "DMACTL"), ("SEC2", "BOOTVEC"),
        ("FECS", "CPUCTL"), ("FECS", "SCTL"), ("FECS", "HWCFG"),
        ("GPCCS", "CPUCTL"), ("GPCCS", "SCTL"),
        ("PMU", "CPUCTL"), ("PMU", "SCTL"),
    ]

    rows = []
    for pname, profile in sorted(profiles.items()):
        row = {"profile": pname, "driver": profile.get("meta", {}).get("driver", "?")}
        for engine, reg in keys_of_interest:
            falcon = profile.get("falcons", {}).get(engine, {})
            val = falcon.get("registers", {}).get(reg, {}).get("hex", "?")
            row[f"{engine}.{reg}"] = val
        rows.append(row)
    return rows


def print_section(title: str):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print()


def print_table(headers: list, rows: list):
    """Simple aligned table printer."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, h in enumerate(headers):
            val = str(row.get(h, ""))
            col_widths[i] = max(col_widths[i], len(val))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in col_widths]))
    for row in rows:
        vals = [str(row.get(h, "")) for h in headers]
        print(fmt.format(*vals))


def main():
    script_dir = Path(__file__).parent
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir / ".." / "data" / "086"
    data_dir = data_dir.resolve()

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    profiles = load_profiles(data_dir)
    if not profiles:
        print(f"ERROR: No JSON files in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(profiles)} profiles from {data_dir}")
    for name in sorted(profiles):
        meta = profiles[name].get("meta", {})
        print(f"  {name}: driver={meta.get('driver', '?')} bdf={meta.get('bdf', '?')}")

    # 1. WPR Region Analysis
    print_section("1. WPR REGION ANALYSIS")
    wpr_rows = analyze_wpr(profiles)
    print_table(["profile", "driver", "wpr1", "wpr2", "wpr1_active", "wpr2_active"], wpr_rows)

    # 2. SEC2 Binding Register State
    print_section("2. SEC2 BINDING REGISTERS")
    bind_rows = analyze_binding(profiles)
    print_table([
        "profile", "driver", "bind_inst", "bind_stat", "bind_stat_field",
        "unk090", "eng_ctrl", "dmaidx", "cpuctl"
    ], bind_rows)

    # 3. Driver Warm-up Effect on Falcon State
    print_section("3. DRIVER WARM-UP EFFECT ON FALCON STATE")
    warmup_rows = analyze_driver_warmup_effect(profiles)
    warmup_headers = ["profile", "driver"]
    if warmup_rows:
        warmup_headers += [k for k in warmup_rows[0] if k not in ("profile", "driver")]
    print_table(warmup_headers, warmup_rows)

    # 4. Changing Registers (across ALL profiles)
    print_section("4. REGISTERS THAT CHANGE ACROSS PROFILES")
    table = extract_register_table(profiles)
    changes = find_changing_registers(table)
    if not changes:
        print("  (no register changes detected — all profiles identical)")
    else:
        total = sum(len(regs) for regs in changes.values())
        print(f"  {total} registers changed across profiles:\n")
        for engine in sorted(changes):
            for reg_name in sorted(changes[engine]):
                vals = changes[engine][reg_name]
                print(f"  {engine:6s}.{reg_name:12s}:")
                for pname in sorted(vals):
                    print(f"    {pname:40s} = {vals[pname]}")
                print()

    # 5. Cross-Card Divergence
    print_section("5. CROSS-CARD DIVERGENCE (TITAN #1 vs #2, SAME STATE)")
    divergence = find_cross_card_divergence(table, profiles)
    if not divergence:
        print("  (no cross-card divergence detected — both Titans identical)")
    else:
        for state in sorted(divergence):
            diffs = divergence[state]
            total = sum(len(regs) for regs in diffs.values())
            print(f"  State: {state} ({total} registers differ)")
            for engine in sorted(diffs):
                for reg_name, vals in sorted(diffs[engine].items()):
                    print(f"    {engine:6s}.{reg_name:12s}: T1={vals['titan1']} T2={vals['titan2']}")
            print()

    # 6. Key Insight Summary
    print_section("6. KEY INSIGHT SUMMARY")

    any_wpr_active = any(r["wpr1_active"] or r["wpr2_active"] for r in wpr_rows)
    nvidia_wpr = [r for r in wpr_rows if "nvidia" in r["driver"]]
    nvidia_has_wpr = any(r["wpr1_active"] or r["wpr2_active"] for r in nvidia_wpr) if nvidia_wpr else False

    print(f"  WPR ever active:       {'YES' if any_wpr_active else 'NO'}")
    print(f"  nvidia sets up WPR:    {'YES' if nvidia_has_wpr else 'NO'}")

    post_nouveau = [r for r in bind_rows if "post_nouveau" in r["profile"]]
    post_nvidia = [r for r in bind_rows if "post_nvidia" in r["profile"]]
    cold = [r for r in bind_rows if "vfio_cold" in r["profile"]]

    def summarize_bind_states(rows, label):
        if not rows:
            return
        stats = set(r["bind_stat_field"] for r in rows)
        print(f"  bind_stat after {label:20s}: {stats}")

    summarize_bind_states(cold, "vfio-cold")
    summarize_bind_states(post_nouveau, "post-nouveau")
    summarize_bind_states(post_nvidia, "post-nvidia")

    if divergence:
        hw_specific = set()
        for state, diffs in divergence.items():
            for engine, regs in diffs.items():
                for reg_name in regs:
                    hw_specific.add(f"{engine}.{reg_name}")
        print(f"\n  Hardware-specific registers (differ between cards): {len(hw_specific)}")
        for r in sorted(hw_specific):
            print(f"    {r}")
    else:
        print("\n  No hardware-specific divergence detected.")

    print()
    print("  INTERPRETATION GUIDE:")
    print("  - If nvidia sets WPR regions → we can read/reuse those addresses")
    print("  - If ACR goes further post-nvidia → WPR hardware state matters")
    print("  - If bind_stat residual differs → warm-up path affects binding")
    print("  - If cross-card divergence is minimal → universal patterns dominate")
    print("  - Registers identical across all states → not driver-dependent")
    print()


if __name__ == "__main__":
    main()
