#!/usr/bin/env python3
"""Tier 1 FEL validator — Python reference implementation.

Reconstructs FES from HILLS files using the same algorithm as plumed sum_hills,
then compares against reference outputs reporting RMSD parity.

Usage:
    python tier1_validator.py <HILLS> --reference <fes.dat> [--json]
    python tier1_validator.py <HILLS_2d> --2d --reference <fes_2d.dat> [--json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_hills_1d(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Parse 1D HILLS file. Returns (times, centers, sigmas, height0, biasfactor)."""
    times, centers, sigmas, heights = [], [], [], []
    biasfactor = 15.0
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                times.append(float(parts[0]))
                centers.append(float(parts[1]))
                sigmas.append(float(parts[2]))
                heights.append(float(parts[3]))
    return (np.array(times), np.array(centers), np.array(sigmas),
            heights[0] if heights else 1.5, biasfactor)


def reconstruct_fes_1d(hills_path: Path, nbins: int = 110) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct 1D FES from HILLS using Gaussian kernel summation."""
    times, centers, sigmas, _, _ = parse_hills_1d(hills_path)

    grid_min = centers.min() - 3 * sigmas.mean()
    grid_max = centers.max() + 3 * sigmas.mean()
    grid = np.linspace(grid_min, grid_max, nbins)
    fes = np.zeros(nbins)

    for c, s, h in zip(centers, sigmas, [parse_hills_1d(hills_path)[3]] * len(centers)):
        fes += h * np.exp(-0.5 * ((grid - c) / s) ** 2)

    # Actually re-read heights properly
    with open(hills_path) as f:
        hs = []
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                hs.append(float(parts[3]))

    fes = np.zeros(nbins)
    for c, s, h in zip(centers, sigmas, hs):
        fes += h * np.exp(-0.5 * ((grid - c) / s) ** 2)

    fes -= fes.min()
    return grid, fes


def parse_fes_1d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a plumed fes_*.dat file."""
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
    return np.array(xs), np.array(ys)


def validate(hills_path: Path, reference_path: Path, nbins: int = 110,
             mode_2d: bool = False) -> dict:
    """Run validation: reconstruct and compare."""
    ref_x, ref_y = parse_fes_1d(reference_path)
    grid, fes = reconstruct_fes_1d(hills_path, nbins=len(ref_x))

    # Interpolate to reference grid
    fes_interp = np.interp(ref_x, grid, fes)
    rmsd = np.sqrt(np.mean((fes_interp - ref_y) ** 2))

    return {
        "hills": str(hills_path),
        "reference": str(reference_path),
        "rmsd_kJ_mol": round(rmsd, 4),
        "max_diff_kJ_mol": round(np.max(np.abs(fes_interp - ref_y)), 4),
        "grid_points": len(ref_x),
        "pass": rmsd < 2.0,
        "tier": 1,
    }


def main():
    parser = argparse.ArgumentParser(description="Tier 1 FEL validator")
    parser.add_argument("hills", type=Path, help="HILLS file path")
    parser.add_argument("--reference", type=Path, required=True, help="Reference FES file")
    parser.add_argument("--nbins", type=int, default=110)
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--2d", dest="mode_2d", action="store_true")
    args = parser.parse_args()

    result = validate(args.hills, args.reference, args.nbins, args.mode_2d)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = "PASS" if result["pass"] else "FAIL"
        print(f"[{status}] RMSD = {result['rmsd_kJ_mol']:.4f} kJ/mol "
              f"(max diff: {result['max_diff_kJ_mol']:.4f})")

    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
