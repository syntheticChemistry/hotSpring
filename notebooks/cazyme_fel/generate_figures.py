#!/usr/bin/env python3
"""Generate publication-quality figures for pseudoSpore visual evidence layer.

Reads FES data from outputs/ and produces:
- 1D theta comparison (free vs enzyme-bound)
- 2D qx/qy heatmaps (free and enzyme-bound)
- Combined panel figure (poster-style)

Output: figures/ directory with SVG + PNG at 300 DPI.

Usage:
    python generate_figures.py --pseudospore <path>
    python generate_figures.py --outputs <outputs_dir> --figures <output_dir>
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Install with: pip install matplotlib")


def parse_fes_1d(path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def parse_fes_2d(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, zs = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
                zs.append(float(parts[2]))

    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

    # Determine grid dimensions (x-fast in PLUMED output)
    unique_y = len(set(np.round(ys, 8)))
    unique_x = len(xs) // unique_y if unique_y > 0 else 1

    X = xs.reshape(unique_y, unique_x)
    Y = ys.reshape(unique_y, unique_x)
    Z = zs.reshape(unique_y, unique_x)
    return X, Y, Z


def theta_to_conformation(theta_rad: float) -> str:
    if theta_rad < 0.5:
        return "⁴C₁"
    elif theta_rad > 2.6:
        return "¹C₄"
    else:
        return "boat/twist"


def plot_1d_comparison(free_path: Path, enzyme_path: Path, output_dir: Path):
    """1D theta FEL comparison: free vs enzyme-bound."""
    free_x, free_y = parse_fes_1d(free_path)
    enz_x, enz_y = parse_fes_1d(enzyme_path)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(np.degrees(free_x), free_y, 'b-', linewidth=2, label='Free xylose')
    ax.plot(np.degrees(enz_x), enz_y, 'r-', linewidth=2, label='Enzyme-bound (-1 subsite)')

    ax.set_xlabel('Cremer-Pople θ (degrees)', fontsize=12)
    ax.set_ylabel('Free energy (kJ/mol)', fontsize=12)
    ax.set_title('Conformational Free Energy Landscape\nβ-D-xylopyranose puckering (PDB 2D24, GH10 xylanase)', fontsize=13)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_xlim(0, 180)

    # Annotate conformations
    ax.axvspan(0, 30, alpha=0.05, color='green', label='_')
    ax.axvspan(150, 180, alpha=0.05, color='orange', label='_')
    ax.text(10, ax.get_ylim()[1] * 0.9, '⁴C₁', fontsize=14, color='green', fontweight='bold')
    ax.text(155, ax.get_ylim()[1] * 0.9, '¹C₄', fontsize=14, color='orange', fontweight='bold')
    ax.text(80, ax.get_ylim()[1] * 0.9, 'boat', fontsize=11, color='gray', ha='center')

    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fel_1d_comparison.svg', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fel_1d_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] fel_1d_comparison.svg/png")


def plot_2d_heatmap(fes_path: Path, title: str, output_name: str, output_dir: Path):
    """2D qx/qy FEL heatmap."""
    X, Y, Z = parse_fes_2d(fes_path)

    # Cap high energies for visualization
    z_cap = min(Z.max(), 60.0)
    Z_viz = np.clip(Z, 0, z_cap)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    cmap = LinearSegmentedColormap.from_list('fel',
        ['#000033', '#0000aa', '#0066ff', '#00cccc', '#66ff66', '#ffff00', '#ff6600', '#ff0000', '#ffffff'])

    im = ax.pcolormesh(X * 10, Y * 10, Z_viz, cmap=cmap, shading='auto')
    cbar = plt.colorbar(im, ax=ax, label='Free energy (kJ/mol)')

    # Add contour lines
    contour_levels = np.arange(0, z_cap, 5)
    ax.contour(X * 10, Y * 10, Z_viz, levels=contour_levels, colors='white', linewidths=0.3, alpha=0.5)

    # Mark global minimum
    min_idx = np.unravel_index(Z.argmin(), Z.shape)
    ax.plot(X[min_idx] * 10, Y[min_idx] * 10, 'w*', markersize=15, markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('qx (Å⁻¹ × 10)', fontsize=12)
    ax.set_ylabel('qy (Å⁻¹ × 10)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(output_dir / f'{output_name}.svg', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] {output_name}.svg/png")


def plot_2d_comparison(free_path: Path, enzyme_path: Path, output_dir: Path):
    """Side-by-side 2D FEL comparison."""
    X_f, Y_f, Z_f = parse_fes_2d(free_path)
    X_e, Y_e, Z_e = parse_fes_2d(enzyme_path)

    z_cap = 50.0
    Z_f_viz = np.clip(Z_f, 0, z_cap)
    Z_e_viz = np.clip(Z_e, 0, z_cap)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cmap = LinearSegmentedColormap.from_list('fel',
        ['#000033', '#0000aa', '#0066ff', '#00cccc', '#66ff66', '#ffff00', '#ff6600', '#ff0000', '#ffffff'])

    im1 = ax1.pcolormesh(X_f * 10, Y_f * 10, Z_f_viz, cmap=cmap, shading='auto', vmin=0, vmax=z_cap)
    im2 = ax2.pcolormesh(X_e * 10, Y_e * 10, Z_e_viz, cmap=cmap, shading='auto', vmin=0, vmax=z_cap)

    for ax, Z, label in [(ax1, Z_f, 'Free xylose'), (ax2, Z_e, 'Enzyme-bound')]:
        min_idx = np.unravel_index(Z.argmin(), Z.shape)
        ax.plot(X_f[min_idx] * 10, Y_f[min_idx] * 10, 'w*', markersize=12, markeredgecolor='black')
        ax.set_xlabel('qx (Å⁻¹ × 10)', fontsize=11)
        ax.set_ylabel('qy (Å⁻¹ × 10)', fontsize=11)
        ax.set_title(label, fontsize=13)
        ax.set_aspect('equal')

    plt.colorbar(im2, ax=[ax1, ax2], label='Free energy (kJ/mol)', shrink=0.8)
    fig.suptitle('2D Cremer-Pople Puckering FEL — Free vs Enzyme-Bound\n(PDB 2D24, GH10 xylanase, WT-MetaD 20ns)', fontsize=14)

    plt.tight_layout()
    fig.savefig(output_dir / 'fel_2d_comparison.svg', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fel_2d_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] fel_2d_comparison.svg/png")


def generate_all(pseudospore_path: Path, output_dir: Path = None):
    """Generate all figures from a pseudoSpore directory."""
    if not HAS_MPL:
        print("ERROR: matplotlib required. pip install matplotlib")
        return

    outputs = pseudospore_path / 'outputs'
    if output_dir is None:
        output_dir = pseudospore_path / 'figures'
    output_dir.mkdir(exist_ok=True)

    print(f"=== Generating visual evidence layer ===")
    print(f"  Source: {outputs}")
    print(f"  Output: {output_dir}")
    print()

    # 1D comparison
    free_1d = outputs / 'xylose-puckering-fel' / 'fes_theta.dat'
    enz_1d = outputs / 'enzyme-bound-puckering' / 'fes_theta.dat'
    if free_1d.exists() and enz_1d.exists():
        plot_1d_comparison(free_1d, enz_1d, output_dir)

    # 2D individual heatmaps
    free_2d = outputs / 'free-xylose-2d' / 'fes_2d.dat'
    enz_2d = outputs / 'enzyme-bound-2d' / 'fes_2d.dat'

    if free_2d.exists():
        plot_2d_heatmap(free_2d, 'Free β-D-xylose — 2D Cremer-Pople (qx, qy)\n20ns WT-MetaD',
                       'fel_2d_free_xylose', output_dir)
    if enz_2d.exists():
        plot_2d_heatmap(enz_2d, 'Enzyme-bound xylose — 2D Cremer-Pople (qx, qy)\nGH10 xylanase -1 subsite, 20ns WT-MetaD',
                       'fel_2d_enzyme_bound', output_dir)

    # 2D side-by-side comparison
    if free_2d.exists() and enz_2d.exists():
        plot_2d_comparison(free_2d, enz_2d, output_dir)

    print()
    print(f"  Done. {len(list(output_dir.glob('*.png')))} PNG + {len(list(output_dir.glob('*.svg')))} SVG generated.")


def main():
    parser = argparse.ArgumentParser(description="Generate pseudoSpore visual evidence layer")
    parser.add_argument("--pseudospore", type=Path, help="Path to pseudoSpore directory")
    parser.add_argument("--outputs", type=Path, help="Path to outputs/ directory (alternative)")
    parser.add_argument("--figures", type=Path, help="Output directory for figures")
    args = parser.parse_args()

    if args.pseudospore:
        generate_all(args.pseudospore, args.figures)
    elif args.outputs:
        # Create a minimal pseudospore-like structure
        class FakePS:
            def __init__(self, outputs):
                self.outputs = outputs
            def __truediv__(self, other):
                if other == 'outputs':
                    return self.outputs
                return self.outputs.parent / other
        generate_all(args.outputs.parent, args.figures)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
