#!/bin/bash
# Run all 9 PP Yukawa DSF cases through LAMMPS/Kokkos-OpenMP
# Matches barracuda/src/md/config.rs::dsf_pp_cases exactly

LAMMPS_BIN="/home/biomegate/Development/ecoPrimals/lammps/build-kokkos-omp/lmp"
RESULTS_DIR="$(dirname "$0")/lammps_results"
mkdir -p "$RESULTS_DIR"

OMP_THREADS=$(nproc)
N_PARTICLES=2048  # Closest FCC lattice to N=2000 (8^3 * 4)

# density = 3/(4*pi) in reduced units (a_ws = 1)
DENSITY="0.238732414637843"

declare -a LABELS=("k1_G14" "k1_G72" "k1_G217" "k2_G31" "k2_G158" "k2_G476" "k3_G100" "k3_G503" "k3_G1510")
declare -a KAPPAS=(1.0 1.0 1.0 2.0 2.0 2.0 3.0 3.0 3.0)
declare -a GAMMAS=(14.0 72.0 217.0 31.0 158.0 476.0 100.0 503.0 1510.0)
declare -a CUTOFFS=(8.0 8.0 8.0 6.5 6.5 6.5 6.0 6.0 6.0)

echo "================================================================"
echo "  LAMMPS/Kokkos-OpenMP Yukawa OCP Benchmark"
echo "  N=$N_PARTICLES, OMP_THREADS=$OMP_THREADS, 9 DSF cases"
echo "================================================================"

SUMMARY_FILE="$RESULTS_DIR/summary.json"
echo '{"results": [' > "$SUMMARY_FILE"
FIRST=true

for i in "${!LABELS[@]}"; do
    LABEL="${LABELS[$i]}"
    KAPPA="${KAPPAS[$i]}"
    GAMMA="${GAMMAS[$i]}"
    CUTOFF="${CUTOFFS[$i]}"

    # Temperature in LJ units: T* = 1/Gamma
    TEMP=$(python3 -c "print(1.0/${GAMMA})")

    # A = 1.0 (coupling amplitude in LJ units, Gamma absorbed into temperature)
    AMPLITUDE="1.0"

    EQUIL_STEPS=2000
    PROD_STEPS=10000
    DT="0.01"
    THERMO=100
    TDAMP=$(python3 -c "print(100.0 * ${DT})")

    INPUT_FILE="$RESULTS_DIR/in.${LABEL}"
    LOG_FILE="$RESULTS_DIR/log.${LABEL}"

    cat > "$INPUT_FILE" << LAMMPS_EOF
# Yukawa OCP: ${LABEL} (kappa=${KAPPA}, Gamma=${GAMMA})
# Matches hotSpring/barraCuda dsf_pp_cases

package         kokkos
units           lj
atom_style      atomic
boundary        p p p

lattice         fcc ${DENSITY}
region          box block 0 8 0 8 0 8
create_box      1 box
create_atoms    1 box
mass            1 1.0

pair_style      yukawa ${KAPPA} ${CUTOFF}
pair_coeff      1 1 ${AMPLITUDE}

velocity        all create ${TEMP} 42 dist gaussian

# Equilibration with Berendsen-like thermostat
fix             therm all nvt temp ${TEMP} ${TEMP} ${TDAMP}
timestep        ${DT}
thermo_style    custom step temp pe ke etotal press
thermo          ${THERMO}

run             ${EQUIL_STEPS}

# Switch to NVE for production
unfix           therm
fix             nve all nve
reset_timestep  0
thermo          ${THERMO}

run             ${PROD_STEPS}
LAMMPS_EOF

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Case $((i+1))/9: ${LABEL} (κ=${KAPPA}, Γ=${GAMMA})"
    echo "────────────────────────────────────────────────────────────"

    START_NS=$(date +%s%N)
    OMP_NUM_THREADS=$OMP_THREADS "$LAMMPS_BIN" \
        -k on t $OMP_THREADS \
        -sf kk \
        -in "$INPUT_FILE" \
        -log "$LOG_FILE" 2>&1 | tail -5
    END_NS=$(date +%s%N)
    WALL_MS=$(( (END_NS - START_NS) / 1000000 ))
    WALL_S=$(python3 -c "print(${WALL_MS}/1000.0)")

    # Extract final energy from log
    FINAL_LINE=$(tail -2 "$LOG_FILE" | head -1)
    echo "    Wall time: ${WALL_S}s, Final: ${FINAL_LINE}"

    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo ',' >> "$SUMMARY_FILE"
    fi
    echo "  {\"label\": \"${LABEL}\", \"kappa\": ${KAPPA}, \"gamma\": ${GAMMA}, \"wall_seconds\": ${WALL_S}, \"n_particles\": ${N_PARTICLES}}" >> "$SUMMARY_FILE"
done

echo '' >> "$SUMMARY_FILE"
echo ']}' >> "$SUMMARY_FILE"

echo ""
echo "================================================================"
echo "  LAMMPS/Kokkos results saved to: $RESULTS_DIR"
echo "================================================================"
