#!/usr/bin/env bash
set -euo pipefail

# v1.7.0 Finalization Script
# Run this after both enzyme-bound simulations complete (10 ns 1D + 20 ns 2D)

ROOT="$(cd "$(dirname "$0")" && pwd)"
GS="$ROOT/pseudoSpore_hotSpring-CompChem-GuideStone_v1.7.0"
REFRESH="$ROOT/control/gromacs_fel/guidestone_refresh"
NEST="$ROOT/control/plumed_nest/nest-validate/target/release/nest-validate"

echo "=== v1.7.0 Finalization ==="
echo "Checking simulations completed..."

# Verify 1D completed
if [ ! -f "$REFRESH/enzyme_bound_1d/HILLS" ] || [ $(wc -l < "$REFRESH/enzyme_bound_1d/HILLS") -lt 5000 ]; then
    echo "ERROR: 1D simulation not complete (need >=5000 HILLS entries)"
    exit 1
fi

# Verify 2D completed  
if [ ! -f "$REFRESH/enzyme_bound_2d/HILLS_2d" ] || [ $(wc -l < "$REFRESH/enzyme_bound_2d/HILLS_2d") -lt 10000 ]; then
    echo "ERROR: 2D simulation not complete (need >=10000 HILLS entries)"
    exit 1
fi

echo "Both simulations complete."

# Copy new data to pseudoSpore modules
echo "Copying simulation data to v1.7.0 modules..."
cp "$REFRESH/enzyme_bound_1d/HILLS" "$GS/modules/05_enzyme_bound_1d/"
cp "$REFRESH/enzyme_bound_1d/COLVAR" "$GS/modules/05_enzyme_bound_1d/"
cp "$REFRESH/enzyme_bound_2d/HILLS_2d" "$GS/modules/06_enzyme_bound_2d/"
cp "$REFRESH/enzyme_bound_2d/COLVAR_2d" "$GS/modules/06_enzyme_bound_2d/"

# Run nest-validate pipeline
echo "Running guidestone finalize (FES reconstruction + figures)..."
$NEST guidestone finalize "$GS"

echo "Running guidestone validate..."
$NEST guidestone validate "$GS"

echo "Running guidestone hash..."
$NEST guidestone hash "$GS"

echo "Running guidestone emit..."
$NEST guidestone emit "$GS"

echo "Running guidestone deploy..."
$NEST guidestone deploy "$GS"

echo ""
echo "=== v1.7.0 COMPLETE ==="
echo "Tarball at: $ROOT/pseudoSpore_hotSpring-CompChem-GuideStone_v1.7.0.tar.gz"
