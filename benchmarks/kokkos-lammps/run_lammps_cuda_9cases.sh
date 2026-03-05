#!/bin/bash
# Run all 9 PP Yukawa DSF cases through LAMMPS/Kokkos-CUDA on Titan V
# GPU vs GPU comparison target for barraCuda validation

LAMMPS_BIN="/home/biomegate/Development/ecoPrimals/lammps/build-kokkos-cuda/lmp"
RESULTS_DIR="$(dirname "$0")/lammps_cuda_results"
mkdir -p "$RESULTS_DIR"

N_PARTICLES=2048
DENSITY="0.238732414637843"

declare -a LABELS=("k1_G14" "k1_G72" "k1_G217" "k2_G31" "k2_G158" "k2_G476" "k3_G100" "k3_G503" "k3_G1510")
declare -a KAPPAS=(1.0 1.0 1.0 2.0 2.0 2.0 3.0 3.0 3.0)
declare -a GAMMAS=(14.0 72.0 217.0 31.0 158.0 476.0 100.0 503.0 1510.0)
declare -a CUTOFFS=(8.0 8.0 8.0 6.5 6.5 6.5 6.0 6.0 6.0)

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_NAME=${GPU_NAME:-"unknown"}

echo "================================================================"
echo "  LAMMPS/Kokkos-CUDA Yukawa OCP Benchmark"
echo "  N=$N_PARTICLES, GPU: $GPU_NAME, 9 DSF cases"
echo "================================================================"

if ! nvidia-smi -L 2>/dev/null | head -1 | grep -qi "GPU"; then
    echo "ERROR: nvidia-smi can't see any GPU. Is nvidia.ko loaded?"
    exit 1
fi

CUDA_DEVICE=$(nvidia-smi -L 2>/dev/null | head -1 | grep -oP 'GPU \K[0-9]+')
echo "  Using CUDA device $CUDA_DEVICE"

SUMMARY_FILE="$RESULTS_DIR/summary_cuda.json"
echo "{\"backend\": \"Kokkos-CUDA\", \"gpu\": \"$GPU_NAME\", \"results\": [" > "$SUMMARY_FILE"
FIRST=true

for i in "${!LABELS[@]}"; do
    LABEL="${LABELS[$i]}"
    KAPPA="${KAPPAS[$i]}"
    GAMMA="${GAMMAS[$i]}"
    CUTOFF="${CUTOFFS[$i]}"

    TEMP=$(python3 -c "print(1.0/${GAMMA})")
    AMPLITUDE="1.0"

    EQUIL_STEPS=5000
    PROD_STEPS=30000
    DT="0.01"
    THERMO=1000
    TDAMP=$(python3 -c "print(100.0 * ${DT})")

    INPUT_FILE="$RESULTS_DIR/in.${LABEL}"
    LOG_FILE="$RESULTS_DIR/log.${LABEL}"

    cat > "$INPUT_FILE" << LAMMPS_EOF
# Yukawa OCP: ${LABEL} (kappa=${KAPPA}, Gamma=${GAMMA})
# Kokkos-CUDA on Titan V — GPU vs GPU comparison with barraCuda

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

fix             therm all nvt temp ${TEMP} ${TEMP} ${TDAMP}
timestep        ${DT}
thermo_style    custom step temp pe ke etotal press
thermo          ${THERMO}

run             ${EQUIL_STEPS}

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
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE "$LAMMPS_BIN" \
        -k on g 1 \
        -sf kk \
        -in "$INPUT_FILE" \
        -log "$LOG_FILE" 2>&1 | tail -10
    END_NS=$(date +%s%N)
    WALL_MS=$(( (END_NS - START_NS) / 1000000 ))
    WALL_S=$(python3 -c "print(${WALL_MS}/1000.0)")

    FINAL_LINE=$(tail -2 "$LOG_FILE" | head -1)
    STEPS_PER_S=$(python3 -c "print(round(${PROD_STEPS}/${WALL_S}, 1))" 2>/dev/null || echo "?")
    echo "    Wall: ${WALL_S}s, Steps/s: ${STEPS_PER_S}, Final: ${FINAL_LINE}"

    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo ',' >> "$SUMMARY_FILE"
    fi
    echo "  {\"label\": \"${LABEL}\", \"kappa\": ${KAPPA}, \"gamma\": ${GAMMA}, \"wall_seconds\": ${WALL_S}, \"steps_per_s\": ${STEPS_PER_S}, \"n_particles\": ${N_PARTICLES}, \"prod_steps\": ${PROD_STEPS}}" >> "$SUMMARY_FILE"
done

echo '' >> "$SUMMARY_FILE"
echo ']}' >> "$SUMMARY_FILE"

echo ""
echo "================================================================"
echo "  Kokkos-CUDA results: $RESULTS_DIR/summary_cuda.json"
echo "================================================================"
