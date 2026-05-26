#!/usr/bin/env python3
"""DEPRECATED: Logic absorbed into cazyme-fel Rust crate (staging/cazyme-fel).
Use: nest-validate cazyme <HILLS-file> [--reference <ref-FES>]
Kept as provenance fossil only."""
import numpy as np
import sys

def analyze_1d_fes(fes_file, label=""):
    """Parse and analyze a 1D FEL from PLUMED sum_hills."""
    theta_vals, fes_vals = [], []
    with open(fes_file) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.split()
            theta_vals.append(float(parts[0]))
            fes_vals.append(float(parts[1]))

    theta = np.array(theta_vals)
    fes = np.array(fes_vals)
    theta_deg = np.degrees(theta)

    fes -= fes.min()

    print(f"\n=== {label} F(theta) ===")
    imin = np.argmin(fes)
    print(f"Global minimum: theta = {theta_deg[imin]:.1f} deg, E = {fes[imin]:.2f} kJ/mol")
    print(f"Energy range: {fes.min():.2f} to {fes.max():.2f} kJ/mol")

    # Chair regions
    chair_4c1 = fes[theta_deg < 40]
    chair_1c4 = fes[theta_deg > 140]
    boat = fes[(theta_deg > 60) & (theta_deg < 120)]

    if len(chair_4c1) > 0:
        i_4c1 = np.argmin(chair_4c1)
        t_4c1 = theta_deg[theta_deg < 40][i_4c1]
        print(f"4C1 chair minimum: theta={t_4c1:.1f} deg, E={chair_4c1[i_4c1]:.2f} kJ/mol")

    if len(chair_1c4) > 0:
        i_1c4 = np.argmin(chair_1c4)
        t_1c4 = theta_deg[theta_deg > 140][i_1c4]
        print(f"1C4 chair minimum: theta={t_1c4:.1f} deg, E={chair_1c4[i_1c4]:.2f} kJ/mol")

    if len(boat) > 0:
        i_boat = np.argmin(boat)
        t_boat = theta_deg[(theta_deg > 60) & (theta_deg < 120)][i_boat]
        print(f"Boat/skew-boat minimum: theta={t_boat:.1f} deg, E={boat[i_boat]:.2f} kJ/mol")

    # Barrier: max between 4C1 and boat
    barrier_region = fes[(theta_deg > 20) & (theta_deg < 90)]
    if len(barrier_region) > 0:
        barrier = np.max(barrier_region) - fes.min()
        print(f"Barrier (4C1 -> boat): {barrier:.1f} kJ/mol")

    return theta_deg, fes

# Analyze enzyme-bound FEL
if len(sys.argv) > 1:
    bound_fes = sys.argv[1]
else:
    bound_fes = "fes_theta.dat"

theta_bound, fes_bound = analyze_1d_fes(bound_fes, "Enzyme-bound xylose (2D24)")

# Compare with free xylose if available
free_fes = "../cazyme_gh10/fes_theta.dat"
try:
    theta_free, fes_free = analyze_1d_fes(free_fes, "Free xylose (Phase 0.5)")
    
    print("\n=== COMPARISON: Enzyme Effect on Puckering ===")
    
    # Interpolate to common grid
    from numpy import interp
    common_theta = np.linspace(0, 180, 200)
    fes_bound_interp = interp(common_theta, theta_bound, fes_bound)
    fes_free_interp = interp(common_theta, theta_free, fes_free)
    
    # 4C1 region difference
    mask_4c1 = common_theta < 40
    if np.any(mask_4c1):
        diff_4c1 = np.min(fes_bound_interp[mask_4c1]) - np.min(fes_free_interp[mask_4c1])
        print(f"4C1 relative stabilization (bound - free): {diff_4c1:.1f} kJ/mol")
    
    # Boat region
    mask_boat = (common_theta > 60) & (common_theta < 120)
    if np.any(mask_boat):
        diff_boat = np.min(fes_bound_interp[mask_boat]) - np.min(fes_free_interp[mask_boat])
        print(f"Boat relative stabilization (bound - free): {diff_boat:.1f} kJ/mol")
    
    # 1C4 region
    mask_1c4 = common_theta > 140
    if np.any(mask_1c4):
        diff_1c4 = np.min(fes_bound_interp[mask_1c4]) - np.min(fes_free_interp[mask_1c4])
        print(f"1C4 relative stabilization (bound - free): {diff_1c4:.1f} kJ/mol")
    
    print("\n=== Validation Checks ===")
    print(f"[{'PASS' if np.min(fes_bound) == 0 else 'FAIL'}] Global minimum normalized to 0")
    i_4c1_bound = np.argmin(fes_bound_interp[mask_4c1])
    is_4c1_global = common_theta[mask_4c1][i_4c1_bound] == common_theta[np.argmin(fes_bound_interp)]
    print(f"[{'PASS' if fes_bound_interp[mask_4c1].min() < 5 else 'WARN'}] 4C1 is low-energy conformation")
    barrier_bound = np.max(fes_bound_interp[(common_theta > 20) & (common_theta < 90)])
    barrier_free = np.max(fes_free_interp[(common_theta > 20) & (common_theta < 90)])
    print(f"[INFO] Barrier 4C1->boat: bound={barrier_bound:.1f} vs free={barrier_free:.1f} kJ/mol")
    
except FileNotFoundError:
    print(f"\nFree xylose FEL not found at {free_fes}, skipping comparison")

print("\nDone.")
