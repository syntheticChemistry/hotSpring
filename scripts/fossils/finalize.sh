#!/usr/bin/env bash
# DEPRECATED: Logic has been absorbed into:
#   nest-validate guidestone finalize <guidestone-dir> --refresh-dir <this-dir>
#
# Kept as reference only. Use the Rust binary instead:
#   nest-validate guidestone finalize ../../../pseudoSpore_hotSpring-CompChem-GuideStone_v1.5.0/ \
#     --refresh-dir .
#
# Original: Finalize GuideStone after all ABG FEL simulations complete
set -euo pipefail

SPRING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REFRESH_DIR="$SPRING_ROOT/control/gromacs_fel/guidestone_refresh"
GS_DIR="$SPRING_ROOT/pseudoSpore_hotSpring-CompChem-GuideStone_v1.5.0"
NEST_VALIDATE="$SPRING_ROOT/control/plumed_nest/nest-validate/target/release/nest-validate"
CAZYME_FEL="$SPRING_ROOT/staging/cazyme-fel/target/release/cazyme-fel"
OLD_FREE_XYLOSE="$SPRING_ROOT/control/gromacs_fel/cazyme_gh10_v2"
OLD_ENZYME="$SPRING_ROOT/control/gromacs_fel/cazyme_2d24"

export PLUMED_KERNEL=/home/strandgate/miniconda3/envs/gromacs-fel/lib/libplumedKernel.so

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  GuideStone Finalization — Post-Simulation                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# --- Step 1: Verify all simulations completed ---
echo "┌─ Step 1: Verify simulation completion ─────────────────────┐"
for sys in free_xylose_1d free_xylose_2d enzyme_bound_1d enzyme_bound_2d; do
    if [[ "$sys" == *"_2d" ]]; then
        HILLS="$REFRESH_DIR/$sys/HILLS_2d"
    else
        HILLS="$REFRESH_DIR/$sys/HILLS"
    fi
    if [ -f "$HILLS" ]; then
        LINES=$(wc -l < "$HILLS")
        echo "  [OK] $sys: $LINES Gaussians deposited"
    else
        echo "  [MISSING] $sys: HILLS not found!"
        exit 1
    fi
done
echo

# --- Step 2: Generate FES from fresh HILLS ---
echo "┌─ Step 2: Generate FES with plumed sum_hills ───────────────┐"

# Free xylose 2D
cd "$REFRESH_DIR/free_xylose_2d"
conda run -n gromacs-fel plumed sum_hills --hills HILLS_2d --mintozero \
    --bin 100,100 --outfile fes_2d.dat 2>/dev/null
echo "  [OK] Free xylose 2D FES generated"

# Enzyme-bound 1D
cd "$REFRESH_DIR/enzyme_bound_1d"
conda run -n gromacs-fel plumed sum_hills --hills HILLS --mintozero \
    --bin 110 --outfile fes_theta.dat 2>/dev/null
echo "  [OK] Enzyme-bound 1D FES generated"

# Enzyme-bound 2D
cd "$REFRESH_DIR/enzyme_bound_2d"
conda run -n gromacs-fel plumed sum_hills --hills HILLS_2d --mintozero \
    --bin 100,100 --outfile fes_2d.dat 2>/dev/null
echo "  [OK] Enzyme-bound 2D FES generated"
echo

# --- Step 3: Run parity checks ---
echo "┌─ Step 3: Parity checks (fresh vs v0.7 reference) ──────────┐"

echo -n "  Free xylose 1D: "
RESULT=$("$CAZYME_FEL" "$REFRESH_DIR/free_xylose_1d/HILLS" \
    --reference "$OLD_FREE_XYLOSE/fes_theta.dat" --json 2>/dev/null)
RMSD=$(echo "$RESULT" | python3 -c "import json,sys; r=json.load(sys.stdin); print(f'{r[\"parity\"][\"rmsd_kjmol\"]:.2f}')")
echo "RMSD = $RMSD kJ/mol"

echo -n "  Free xylose 2D: "
RESULT=$("$CAZYME_FEL" "$REFRESH_DIR/free_xylose_2d/HILLS_2d" --2d \
    --reference "$OLD_FREE_XYLOSE/fes_2d.dat" \
    --grid-min -0.12,-0.12 --grid-max 0.12,0.12 --json 2>/dev/null)
RMSD=$(echo "$RESULT" | python3 -c "import json,sys; r=json.load(sys.stdin); print(f'{r[\"parity\"][\"rmsd_kjmol\"]:.2f}')")
echo "RMSD = $RMSD kJ/mol"

echo -n "  Enzyme-bound 1D: "
RESULT=$("$CAZYME_FEL" "$REFRESH_DIR/enzyme_bound_1d/HILLS" \
    --reference "$OLD_ENZYME/fes_theta.dat" --json 2>/dev/null)
RMSD=$(echo "$RESULT" | python3 -c "import json,sys; r=json.load(sys.stdin); print(f'{r[\"parity\"][\"rmsd_kjmol\"]:.2f}')")
echo "RMSD = $RMSD kJ/mol"

echo -n "  Enzyme-bound 2D: "
RESULT=$("$CAZYME_FEL" "$REFRESH_DIR/enzyme_bound_2d/HILLS_2d" --2d \
    --reference "$OLD_ENZYME/fes_2d.dat" \
    --grid-min -0.12,-0.12 --grid-max 0.12,0.12 --json 2>/dev/null)
RMSD=$(echo "$RESULT" | python3 -c "import json,sys; r=json.load(sys.stdin); print(f'{r[\"parity\"][\"rmsd_kjmol\"]:.2f}')")
echo "RMSD = $RMSD kJ/mol"
echo

# --- Step 4: Populate GuideStone modules ---
echo "┌─ Step 4: Populate GuideStone modules ───────────────────────┐"

# Module 04: Free xylose 2D
cp "$REFRESH_DIR/free_xylose_2d/HILLS_2d" "$GS_DIR/modules/04_free_xylose_2d/HILLS_2d"
cp "$REFRESH_DIR/free_xylose_2d/fes_2d.dat" "$GS_DIR/modules/04_free_xylose_2d/fes_2d.dat"
cp "$REFRESH_DIR/free_xylose_2d/plumed.dat" "$GS_DIR/modules/04_free_xylose_2d/plumed.dat"
cp "$REFRESH_DIR/free_xylose_2d/COLVAR_2d" "$GS_DIR/modules/04_free_xylose_2d/COLVAR_2d" 2>/dev/null || true
echo "  [OK] Module 04 populated"

# Module 05: Enzyme-bound 1D
cp "$REFRESH_DIR/enzyme_bound_1d/HILLS" "$GS_DIR/modules/05_enzyme_bound_1d/HILLS"
cp "$REFRESH_DIR/enzyme_bound_1d/fes_theta.dat" "$GS_DIR/modules/05_enzyme_bound_1d/fes_theta.dat"
cp "$REFRESH_DIR/enzyme_bound_1d/plumed.dat" "$GS_DIR/modules/05_enzyme_bound_1d/plumed.dat"
cp "$REFRESH_DIR/enzyme_bound_1d/COLVAR" "$GS_DIR/modules/05_enzyme_bound_1d/COLVAR" 2>/dev/null || true
echo "  [OK] Module 05 populated"

# Module 06: Enzyme-bound 2D
cp "$REFRESH_DIR/enzyme_bound_2d/HILLS_2d" "$GS_DIR/modules/06_enzyme_bound_2d/HILLS_2d"
cp "$REFRESH_DIR/enzyme_bound_2d/fes_2d.dat" "$GS_DIR/modules/06_enzyme_bound_2d/fes_2d.dat"
cp "$REFRESH_DIR/enzyme_bound_2d/plumed.dat" "$GS_DIR/modules/06_enzyme_bound_2d/plumed.dat"
cp "$REFRESH_DIR/enzyme_bound_2d/COLVAR_2d" "$GS_DIR/modules/06_enzyme_bound_2d/COLVAR_2d" 2>/dev/null || true
echo "  [OK] Module 06 populated"
echo

# --- Step 5: Generate data.toml (BLAKE3 hashes) ---
echo "┌─ Step 5: Generate BLAKE3 integrity manifest ────────────────┐"
"$NEST_VALIDATE" guidestone hash "$GS_DIR" > "$GS_DIR/data.toml"
echo "  [OK] data.toml written"
echo

# --- Step 6: Re-emit liveSpore.json ---
echo "┌─ Step 6: Emit provenance ──────────────────────────────────┐"
"$NEST_VALIDATE" guidestone emit "$GS_DIR"
echo

# --- Step 7: Verify integrity ---
echo "┌─ Step 7: Self-verification ────────────────────────────────┐"
"$NEST_VALIDATE" guidestone verify "$GS_DIR"
echo

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  GUIDESTONE FINALIZATION COMPLETE                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
