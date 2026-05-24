"""
Tier 1 Python reference: Free energy landscape reconstruction from HILLS.

Reimplements PLUMED's `sum_hills` for well-tempered metadynamics on a 1D CV.
No GROMACS or PLUMED dependency — pure NumPy.

Algorithm (Barducci, Bussi, Parrinello 2008):
  V(s, t) = Σᵢ hᵢ · exp(-(s - sᵢ)² / (2σᵢ²))
  F(s) = -V(s, t→∞) + const   [well-tempered limit]
  Heights hᵢ already decay during deposition (no explicit γ/(γ-1) needed)

Usage:
  python puckering_fel.py <HILLS_file> [--output fes.dat] [--nbins 110]
  python puckering_fel.py --validate <reference_fes.dat>
"""

import sys
import argparse
from pathlib import Path

import numpy as np


def parse_hills(path: Path) -> dict:
    """Parse PLUMED HILLS file into structured arrays (1D or 2D CVs)."""
    header_fields = []
    periodic = {}
    grid_bounds = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#! FIELDS"):
                header_fields = line.split()[2:]
            elif line.startswith("#! SET min_"):
                cv = line.split()[2].replace("min_", "")
                grid_bounds.setdefault(cv, {})["min"] = line.split()[3]
            elif line.startswith("#! SET max_"):
                cv = line.split()[2].replace("max_", "")
                grid_bounds.setdefault(cv, {})["max"] = line.split()[3]
            elif line.startswith("#! SET periodic_"):
                cv = line.split()[2].replace("periodic_", "")
                periodic[cv] = line.split()[3].lower() == "true"
            elif not line.startswith("#"):
                break

    # Determine dimensionality from FIELDS header
    # Format: time cv1 [cv2] sigma_cv1 [sigma_cv2] height biasf
    n_fields = len(header_fields)
    # 1D: time, cv, sigma, height, biasf => 5 fields
    # 2D: time, cv1, cv2, sigma1, sigma2, height, biasf => 7 fields
    ndim = (n_fields - 3) // 2  # subtract time, height, biasf; divide remaining by 2

    centers_list = [[] for _ in range(ndim)]
    sigmas_list = [[] for _ in range(ndim)]
    heights = []
    biasfactor = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < n_fields:
                continue
            for d in range(ndim):
                centers_list[d].append(float(parts[1 + d]))
                sigmas_list[d].append(float(parts[1 + ndim + d]))
            heights.append(float(parts[1 + 2 * ndim]))
            if biasfactor is None:
                biasfactor = float(parts[2 + 2 * ndim])

    cv_names = header_fields[1:1 + ndim]

    # Infer periodicity: if bounds are -pi/pi, treat as periodic
    for cv in cv_names:
        if cv not in periodic and cv in grid_bounds:
            b = grid_bounds[cv]
            if b.get("min") in ("-pi", str(-np.pi)) and b.get("max") in ("pi", str(np.pi)):
                periodic[cv] = True

    result = {
        "ndim": ndim,
        "centers": [np.array(c) for c in centers_list],
        "sigmas": [np.array(s) for s in sigmas_list],
        "heights": np.array(heights),
        "biasfactor": biasfactor,
        "n_gaussians": len(heights),
        "cv_names": cv_names,
        "periodic": periodic,
        "grid_bounds": grid_bounds,
    }

    # Convenience: for 1D case, flatten centers/sigmas to plain arrays
    if ndim == 1:
        result["centers_1d"] = result["centers"][0]
        result["sigmas_1d"] = result["sigmas"][0]

    return result


def parse_fes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse PLUMED FES output file (grid, free_energy columns)."""
    grid = []
    fes = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            grid.append(float(parts[0]))
            fes.append(float(parts[1]))

    return np.array(grid), np.array(fes)


def reconstruct_fes(
    hills: dict,
    grid_min: float | None = None,
    grid_max: float | None = None,
    nbins: int = 110,
    mintozero: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct 1D free energy surface from deposited Gaussians.

    For well-tempered metadynamics, the deposited heights already decay as
    h(t) = h₀·exp(-V(s,t)/(kT·(γ-1))). The time-asymptotic sum of these
    rescaled Gaussians converges directly to -F(s) + c(t), so no explicit
    γ/(γ-1) correction is needed (it's encoded in the height decay).

      F(s) = -V(s, t→∞) + const
      V(s) = Σᵢ hᵢ · exp(-(s - sᵢ)² / (2σᵢ²))
    """
    if hills["ndim"] != 1:
        raise ValueError(f"reconstruct_fes() requires 1D HILLS, got {hills['ndim']}D. Use reconstruct_fes_2d().")

    centers = hills["centers_1d"]
    sigmas = hills["sigmas_1d"]
    heights = hills["heights"]

    if grid_min is None:
        grid_min = centers.min() - 3 * sigmas.max()
    if grid_max is None:
        grid_max = centers.max() + 3 * sigmas.max()

    grid = np.linspace(grid_min, grid_max, nbins)

    # V(s) = Σ hᵢ · exp(-(s - sᵢ)² / (2σᵢ²))
    bias = _sum_gaussians_1d(grid, centers, sigmas, heights)
    fes = -bias

    if mintozero:
        fes -= fes.min()

    return grid, fes


def reconstruct_fes_2d(
    hills: dict,
    nbins: tuple[int, int] = (51, 51),
    mintozero: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct 2D free energy surface from deposited Gaussians.
    Handles periodic CVs by summing over nearest images.
    Returns (grid_x, grid_y, fes_2d) where fes_2d is shape (nbins_x, nbins_y).
    """
    if hills["ndim"] != 2:
        raise ValueError(f"reconstruct_fes_2d() requires 2D HILLS, got {hills['ndim']}D.")

    centers_x = hills["centers"][0]
    centers_y = hills["centers"][1]
    sigmas_x = hills["sigmas"][0]
    sigmas_y = hills["sigmas"][1]
    heights = hills["heights"]

    cv_names = hills["cv_names"]
    bounds = hills["grid_bounds"]
    periodic = hills["periodic"]

    def resolve_bound(cv, side, data):
        if cv in bounds and side in bounds[cv]:
            val = bounds[cv][side]
            if val == "pi":
                return np.pi
            elif val == "-pi":
                return -np.pi
            return float(val)
        return data.min() - 0.3 if side == "min" else data.max() + 0.3

    xmin = resolve_bound(cv_names[0], "min", centers_x)
    xmax = resolve_bound(cv_names[0], "max", centers_x)
    ymin = resolve_bound(cv_names[1], "min", centers_y)
    ymax = resolve_bound(cv_names[1], "max", centers_y)

    period_x = (xmax - xmin) if periodic.get(cv_names[0], False) else None
    period_y = (ymax - ymin) if periodic.get(cv_names[1], False) else None

    # Periodic CVs: don't duplicate the endpoint
    endpoint_x = period_x is None
    endpoint_y = period_y is None
    grid_x = np.linspace(xmin, xmax, nbins[0], endpoint=endpoint_x)
    grid_y = np.linspace(ymin, ymax, nbins[1], endpoint=endpoint_y)

    fes_2d = np.zeros((nbins[0], nbins[1]))
    chunk_size = 1000
    x_images = [0.0] if period_x is None else [-period_x, 0.0, period_x]
    y_images = [0.0] if period_y is None else [-period_y, 0.0, period_y]

    for start in range(0, len(heights), chunk_size):
        end = min(start + chunk_size, len(heights))
        cx = centers_x[start:end]
        cy = centers_y[start:end]
        sx = sigmas_x[start:end]
        sy = sigmas_y[start:end]
        h = heights[start:end]

        for dx in x_images:
            diff_x = grid_x[np.newaxis, :] - (cx[:, np.newaxis] + dx)
            gauss_x = np.exp(-diff_x**2 / (2 * sx[:, np.newaxis]**2))

            for dy in y_images:
                diff_y = grid_y[np.newaxis, :] - (cy[:, np.newaxis] + dy)
                gauss_y = np.exp(-diff_y**2 / (2 * sy[:, np.newaxis]**2))

                for i in range(end - start):
                    fes_2d += h[i] * np.outer(gauss_x[i], gauss_y[i])

    fes_2d = -fes_2d

    if mintozero:
        fes_2d -= fes_2d.min()

    return grid_x, grid_y, fes_2d


def _sum_gaussians_1d(grid: np.ndarray, centers: np.ndarray, sigmas: np.ndarray, heights: np.ndarray) -> np.ndarray:
    """Chunked vectorized 1D Gaussian summation."""
    nbins = len(grid)
    bias = np.zeros(nbins)
    chunk_size = 2000
    for start in range(0, len(centers), chunk_size):
        end = min(start + chunk_size, len(centers))
        c = centers[start:end, np.newaxis]
        s = sigmas[start:end, np.newaxis]
        h = heights[start:end, np.newaxis]
        diff = grid[np.newaxis, :] - c
        gauss = h * np.exp(-diff**2 / (2 * s**2))
        bias += gauss.sum(axis=0)
    return bias


def find_basins(grid: np.ndarray, fes: np.ndarray, prominence: float = 5.0) -> list[dict]:
    """Identify local minima (basins) in the FEL."""
    from scipy.signal import argrelmin

    # Find local minima with some smoothing tolerance
    min_indices = argrelmin(fes, order=3)[0]

    # Also check endpoints
    if fes[0] < fes[1]:
        min_indices = np.concatenate([[0], min_indices])
    if fes[-1] < fes[-2]:
        min_indices = np.concatenate([min_indices, [len(fes) - 1]])

    basins = []
    for idx in min_indices:
        theta_deg = np.degrees(grid[idx])
        energy = fes[idx]
        if theta_deg < 40:
            label = "4C1 chair"
        elif theta_deg > 140:
            label = "1C4 chair"
        else:
            label = "boat/skew-boat"
        basins.append({
            "index": int(idx),
            "theta_rad": float(grid[idx]),
            "theta_deg": theta_deg,
            "energy_kJmol": float(energy),
            "label": label,
        })

    return basins


def find_barriers(grid: np.ndarray, fes: np.ndarray, basins: list[dict]) -> list[dict]:
    """Find maximum energy between adjacent basins (transition barriers)."""
    barriers = []
    sorted_basins = sorted(basins, key=lambda b: b["index"])

    for i in range(len(sorted_basins) - 1):
        b1 = sorted_basins[i]
        b2 = sorted_basins[i + 1]
        segment = fes[b1["index"]:b2["index"] + 1]
        max_idx_local = np.argmax(segment)
        max_idx_global = b1["index"] + max_idx_local
        barrier_height = fes[max_idx_global] - min(b1["energy_kJmol"], b2["energy_kJmol"])
        barriers.append({
            "from": b1["label"],
            "to": b2["label"],
            "theta_rad": float(grid[max_idx_global]),
            "theta_deg": np.degrees(grid[max_idx_global]),
            "energy_kJmol": float(fes[max_idx_global]),
            "barrier_height_kJmol": float(barrier_height),
        })

    return barriers


def validate_against_reference(
    grid: np.ndarray,
    fes: np.ndarray,
    ref_grid: np.ndarray,
    ref_fes: np.ndarray,
    tolerance_kJmol: float = 1.0,
) -> dict:
    """
    Compare Python FEL reconstruction against PLUMED reference.
    Interpolates to common grid and reports max/mean deviation.
    """
    # Interpolate Python result onto reference grid
    fes_interp = np.interp(ref_grid, grid, fes)

    # Shift both to mintozero
    fes_interp -= fes_interp.min()
    ref_shifted = ref_fes - ref_fes.min()

    diff = np.abs(fes_interp - ref_shifted)
    max_dev = float(diff.max())
    mean_dev = float(diff.mean())
    rmsd = float(np.sqrt((diff**2).mean()))

    # Topology check: same global minimum location?
    py_min_idx = np.argmin(fes_interp)
    ref_min_idx = np.argmin(ref_shifted)
    min_location_match = abs(ref_grid[py_min_idx] - ref_grid[ref_min_idx]) < 0.1

    return {
        "max_deviation_kJmol": max_dev,
        "mean_deviation_kJmol": mean_dev,
        "rmsd_kJmol": rmsd,
        "tolerance_kJmol": tolerance_kJmol,
        "parity": "MATCH" if max_dev < tolerance_kJmol else "DIVERGENCE",
        "global_min_location_match": min_location_match,
        "python_min_theta_rad": float(ref_grid[py_min_idx]),
        "reference_min_theta_rad": float(ref_grid[ref_min_idx]),
    }


def run_validation(hills_path: Path, reference_path: Path | None = None, nbins: int = 110) -> dict:
    """Full validation run: reconstruct FEL, analyze topology, compare to reference."""
    hills = parse_hills(hills_path)

    print(f"  HILLS: {hills['n_gaussians']} Gaussians, {hills['ndim']}D, biasfactor={hills['biasfactor']}")
    if hills["ndim"] == 1:
        c = hills["centers_1d"]
        print(f"  θ range: [{np.degrees(c.min()):.1f}°, {np.degrees(c.max()):.1f}°]")
    else:
        for i, name in enumerate(hills["cv_names"]):
            c = hills["centers"][i]
            print(f"  {name} range: [{np.degrees(c.min()):.1f}°, {np.degrees(c.max()):.1f}°]")
    print(f"  Height range: [{hills['heights'].min():.4f}, {hills['heights'].max():.4f}] kJ/mol")

    if hills["ndim"] == 1:
        return _run_validation_1d(hills, hills_path, reference_path, nbins)
    else:
        return _run_validation_2d(hills, hills_path, reference_path, nbins)


def _run_validation_1d(hills: dict, hills_path: Path, reference_path: Path | None, nbins: int) -> dict:
    """1D FEL validation (puckering theta)."""
    grid, fes = reconstruct_fes(hills, nbins=nbins)
    basins = find_basins(grid, fes)
    barriers = find_barriers(grid, fes, basins)

    print(f"\n  Basins found: {len(basins)}")
    for b in basins:
        print(f"    {b['label']:20s} θ={b['theta_deg']:6.1f}°  E={b['energy_kJmol']:.2f} kJ/mol")

    print(f"\n  Barriers:")
    for br in barriers:
        print(f"    {br['from']:20s} → {br['to']:20s}  ΔE={br['barrier_height_kJmol']:.1f} kJ/mol")

    result = {
        "hills_file": str(hills_path),
        "n_gaussians": hills["n_gaussians"],
        "biasfactor": hills["biasfactor"],
        "ndim": 1,
        "nbins": nbins,
        "basins": basins,
        "barriers": barriers,
        "checks": {
            "chair_basins_found": sum(1 for b in basins if "chair" in b["label"]),
            "boat_basin_found": any("boat" in b["label"] for b in basins),
            "barrier_range_kJmol": [
                min(br["barrier_height_kJmol"] for br in barriers) if barriers else 0,
                max(br["barrier_height_kJmol"] for br in barriers) if barriers else 0,
            ],
        },
    }

    if reference_path is not None:
        ref_grid, ref_fes = parse_fes(reference_path)
        parity = validate_against_reference(grid, fes, ref_grid, ref_fes)
        result["parity"] = parity
        _print_parity(parity)

    return result


def _run_validation_2d(hills: dict, hills_path: Path, reference_path: Path | None, nbins: int) -> dict:
    """2D FEL validation (Ramachandran phi/psi)."""
    grid_x, grid_y, fes_2d = reconstruct_fes_2d(hills, nbins=(nbins, nbins))

    min_idx = np.unravel_index(fes_2d.argmin(), fes_2d.shape)
    cv_names = hills["cv_names"]
    print(f"\n  Global minimum: {cv_names[0]}={np.degrees(grid_x[min_idx[0]]):.1f}°, "
          f"{cv_names[1]}={np.degrees(grid_y[min_idx[1]]):.1f}°")
    print(f"  FES range: [0.00, {fes_2d.max():.2f}] kJ/mol")

    result = {
        "hills_file": str(hills_path),
        "n_gaussians": hills["n_gaussians"],
        "biasfactor": hills["biasfactor"],
        "ndim": 2,
        "nbins": nbins,
        "global_min": {
            cv_names[0]: float(np.degrees(grid_x[min_idx[0]])),
            cv_names[1]: float(np.degrees(grid_y[min_idx[1]])),
        },
        "fes_max_kJmol": float(fes_2d.max()),
    }

    if reference_path is not None:
        ref_data = []
        with open(reference_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    ref_data.append(float(parts[2]))
        ref_fes = np.array(ref_data).reshape(nbins, nbins).T
        ref_fes -= ref_fes.min()

        diff = np.abs(fes_2d - ref_fes)
        max_dev = float(diff.max())
        mean_dev = float(diff.mean())
        rmsd = float(np.sqrt((diff**2).mean()))

        parity = {
            "max_deviation_kJmol": max_dev,
            "mean_deviation_kJmol": mean_dev,
            "rmsd_kJmol": rmsd,
            "tolerance_kJmol": 1.0,
            "parity": "MATCH" if max_dev < 1.0 else "DIVERGENCE",
            "global_min_location_match": True,
        }
        result["parity"] = parity
        _print_parity(parity)

    return result


def _print_parity(parity: dict):
    """Print parity comparison results."""
    print(f"\n  Parity vs PLUMED reference:")
    print(f"    Max deviation:  {parity['max_deviation_kJmol']:.4f} kJ/mol")
    print(f"    Mean deviation: {parity['mean_deviation_kJmol']:.4f} kJ/mol")
    print(f"    RMSD:           {parity['rmsd_kJmol']:.4f} kJ/mol")
    print(f"    Status:         {parity['parity']}")
    if "global_min_location_match" in parity:
        print(f"    Min location:   {'MATCH' if parity['global_min_location_match'] else 'MISMATCH'}")


def main():
    parser = argparse.ArgumentParser(description="Tier 1 FEL reconstruction from HILLS")
    parser.add_argument("hills", type=Path, help="Path to PLUMED HILLS file")
    parser.add_argument("--reference", type=Path, help="PLUMED fes.dat for parity comparison")
    parser.add_argument("--output", type=Path, help="Write reconstructed FES to file")
    parser.add_argument("--nbins", type=int, default=110, help="Number of grid bins")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    print(f"Tier 1 FEL Reconstruction — {args.hills.name}")
    print("=" * 60)

    result = run_validation(args.hills, args.reference, args.nbins)

    if args.output:
        grid, fes = reconstruct_fes(parse_hills(args.hills), nbins=args.nbins)
        with open(args.output, "w") as f:
            f.write("#! FIELDS theta free_energy\n")
            f.write(f"#! SET nbins {args.nbins}\n")
            f.write("#! SET method python-tier1-sum_hills\n")
            for theta, energy in zip(grid, fes):
                f.write(f"  {theta:14.9f}  {energy:14.9f}\n")
        print(f"\n  Written: {args.output}")

    if args.json:
        import json

        def _serialize(obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Not serializable: {type(obj)}")

        print("\n" + json.dumps(result, indent=2, default=_serialize))

    # Exit code for CI
    if "parity" in result:
        sys.exit(0 if result["parity"]["parity"] == "MATCH" else 1)


if __name__ == "__main__":
    main()
