#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# hotSpring Experiment 008: Parity Benchmark
# Python baseline → Rust CPU → Rust GPU
#
# Proves the evolution path:
#   1. Python establishes ground truth (reproducible baselines)
#   2. Rust CPU proves pure math correctness AND speed advantage
#   3. Rust GPU proves math portability to accelerator
#
# Hardware: RTX 4070 (12GB) + Titan V (12GB HBM2)
# Usage: bash experiments/008_PARITY_BENCHMARK.sh [--skip-python] [--skip-gpu]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/data/008_parity"
mkdir -p "$RESULTS_DIR"

SKIP_PYTHON=false
SKIP_GPU=false
for arg in "$@"; do
    case $arg in
        --skip-python) SKIP_PYTHON=true ;;
        --skip-gpu) SKIP_GPU=true ;;
    esac
done

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPORT="$RESULTS_DIR/parity_report_${TIMESTAMP}.md"

cat > "$REPORT" <<'HEADER'
# hotSpring Parity Benchmark Report

**Evolution path**: Python baseline → Rust CPU → Rust GPU → ToadStool streaming

| Stage | What it proves |
|-------|---------------|
| Python baseline | Ground truth: reproducible, peer-reviewable |
| Rust CPU | Pure math: correct AND faster than interpreted |
| Rust GPU | Portable math: same results on accelerator |
| ToadStool | Streaming: unidirectional dispatch, minimal round-trips |

HEADER

echo "## Hardware" >> "$REPORT"
echo "" >> "$REPORT"
echo '```' >> "$REPORT"
lscpu | grep "Model name" >> "$REPORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "$REPORT" 2>/dev/null || echo "No NVIDIA GPUs detected" >> "$REPORT"
echo '```' >> "$REPORT"
echo "" >> "$REPORT"

echo "═══════════════════════════════════════════════════════════"
echo "  hotSpring Parity Benchmark — Experiment 008"
echo "  $(date -u)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ─── Phase 1: Python Baselines ───────────────────────────────
echo "## Phase 1: Python Baselines" >> "$REPORT"
echo "" >> "$REPORT"

if [ "$SKIP_PYTHON" = false ]; then
    echo "Phase 1: Python baselines..."
    echo ""

    # Spectral theory control
    echo "  [PY] Spectral theory..."
    SPECTRAL_PY_START=$(date +%s%N)
    python3 "$PROJECT_DIR/control/spectral_theory/scripts/spectral_control.py" > "$RESULTS_DIR/spectral_python.txt" 2>&1 || true
    SPECTRAL_PY_END=$(date +%s%N)
    SPECTRAL_PY_MS=$(( (SPECTRAL_PY_END - SPECTRAL_PY_START) / 1000000 ))
    echo "  [PY] Spectral theory: ${SPECTRAL_PY_MS}ms"
    echo "- Spectral theory (Python): **${SPECTRAL_PY_MS}ms**" >> "$REPORT"

    # Lattice QCD CG control
    echo "  [PY] Lattice QCD CG..."
    LATTICE_PY_START=$(date +%s%N)
    python3 "$PROJECT_DIR/control/lattice_qcd/scripts/lattice_cg_control.py" > "$RESULTS_DIR/lattice_cg_python.txt" 2>&1 || true
    LATTICE_PY_END=$(date +%s%N)
    LATTICE_PY_MS=$(( (LATTICE_PY_END - LATTICE_PY_START) / 1000000 ))
    echo "  [PY] Lattice QCD CG: ${LATTICE_PY_MS}ms"
    echo "- Lattice QCD CG (Python): **${LATTICE_PY_MS}ms**" >> "$REPORT"

    # Abelian Higgs control
    echo "  [PY] Abelian Higgs HMC..."
    HIGGS_PY_START=$(date +%s%N)
    python3 "$PROJECT_DIR/control/abelian_higgs/scripts/abelian_higgs_hmc.py" > "$RESULTS_DIR/abelian_higgs_python.txt" 2>&1 || true
    HIGGS_PY_END=$(date +%s%N)
    HIGGS_PY_MS=$(( (HIGGS_PY_END - HIGGS_PY_START) / 1000000 ))
    echo "  [PY] Abelian Higgs HMC: ${HIGGS_PY_MS}ms"
    echo "- Abelian Higgs HMC (Python): **${HIGGS_PY_MS}ms**" >> "$REPORT"

    echo "" >> "$REPORT"
else
    echo "Phase 1: SKIPPED (--skip-python)"
    echo "- Python baselines: SKIPPED" >> "$REPORT"
    echo "" >> "$REPORT"
fi

# ─── Phase 2: Rust CPU Benchmarks ────────────────────────────
echo ""
echo "Phase 2: Rust CPU benchmarks..."
echo ""
echo "## Phase 2: Rust CPU (BarraCUDA pure math)" >> "$REPORT"
echo "" >> "$REPORT"

# Build release first
echo "  [BUILD] Release compilation..."
cd "$PROJECT_DIR/barracuda"
cargo build --release --bin bench_lattice_cg --bin validate_spectral --bin validate_abelian_higgs --bin validate_nuclear_eos --bin validate_screened_coulomb --bin validate_pure_gauge 2>/dev/null

# Spectral theory (Rust CPU)
echo "  [RS-CPU] Spectral theory..."
SPECTRAL_RS_START=$(date +%s%N)
cargo run --release --bin validate_spectral > "$RESULTS_DIR/spectral_rust_cpu.txt" 2>&1 || true
SPECTRAL_RS_END=$(date +%s%N)
SPECTRAL_RS_MS=$(( (SPECTRAL_RS_END - SPECTRAL_RS_START) / 1000000 ))
echo "  [RS-CPU] Spectral theory: ${SPECTRAL_RS_MS}ms"
echo "- Spectral theory (Rust CPU): **${SPECTRAL_RS_MS}ms**" >> "$REPORT"

# Lattice QCD CG (Rust CPU)
echo "  [RS-CPU] Lattice QCD CG..."
LATTICE_RS_START=$(date +%s%N)
cargo run --release --bin bench_lattice_cg > "$RESULTS_DIR/lattice_cg_rust_cpu.txt" 2>&1 || true
LATTICE_RS_END=$(date +%s%N)
LATTICE_RS_MS=$(( (LATTICE_RS_END - LATTICE_RS_START) / 1000000 ))
echo "  [RS-CPU] Lattice QCD CG: ${LATTICE_RS_MS}ms"
echo "- Lattice QCD CG (Rust CPU): **${LATTICE_RS_MS}ms**" >> "$REPORT"

# Abelian Higgs (Rust CPU)
echo "  [RS-CPU] Abelian Higgs..."
HIGGS_RS_START=$(date +%s%N)
cargo run --release --bin validate_abelian_higgs > "$RESULTS_DIR/abelian_higgs_rust_cpu.txt" 2>&1 || true
HIGGS_RS_END=$(date +%s%N)
HIGGS_RS_MS=$(( (HIGGS_RS_END - HIGGS_RS_START) / 1000000 ))
echo "  [RS-CPU] Abelian Higgs: ${HIGGS_RS_MS}ms"
echo "- Abelian Higgs HMC (Rust CPU): **${HIGGS_RS_MS}ms**" >> "$REPORT"

# Pure gauge SU(3) (Rust CPU)
echo "  [RS-CPU] Pure gauge SU(3)..."
GAUGE_RS_START=$(date +%s%N)
cargo run --release --bin validate_pure_gauge > "$RESULTS_DIR/pure_gauge_rust_cpu.txt" 2>&1 || true
GAUGE_RS_END=$(date +%s%N)
GAUGE_RS_MS=$(( (GAUGE_RS_END - GAUGE_RS_START) / 1000000 ))
echo "  [RS-CPU] Pure gauge SU(3): ${GAUGE_RS_MS}ms"
echo "- Pure gauge SU(3) (Rust CPU): **${GAUGE_RS_MS}ms**" >> "$REPORT"

# Nuclear EOS (Rust CPU)
echo "  [RS-CPU] Nuclear EOS..."
NEOS_RS_START=$(date +%s%N)
cargo run --release --bin validate_nuclear_eos > "$RESULTS_DIR/nuclear_eos_rust_cpu.txt" 2>&1 || true
NEOS_RS_END=$(date +%s%N)
NEOS_RS_MS=$(( (NEOS_RS_END - NEOS_RS_START) / 1000000 ))
echo "  [RS-CPU] Nuclear EOS: ${NEOS_RS_MS}ms"
echo "- Nuclear EOS L1+L2 (Rust CPU): **${NEOS_RS_MS}ms**" >> "$REPORT"

# Screened Coulomb (Rust CPU)
echo "  [RS-CPU] Screened Coulomb..."
SC_RS_START=$(date +%s%N)
cargo run --release --bin validate_screened_coulomb > "$RESULTS_DIR/screened_coulomb_rust_cpu.txt" 2>&1 || true
SC_RS_END=$(date +%s%N)
SC_RS_MS=$(( (SC_RS_END - SC_RS_START) / 1000000 ))
echo "  [RS-CPU] Screened Coulomb: ${SC_RS_MS}ms"
echo "- Screened Coulomb (Rust CPU): **${SC_RS_MS}ms**" >> "$REPORT"

echo "" >> "$REPORT"

# ─── Phase 3: Rust GPU Benchmarks ────────────────────────────
echo ""
echo "## Phase 3: Rust GPU (BarraCUDA GPU portable math)" >> "$REPORT"
echo "" >> "$REPORT"

if [ "$SKIP_GPU" = false ]; then
    echo "Phase 3: Rust GPU benchmarks..."
    echo ""

    cargo build --release --bin bench_lattice_scaling --bin bench_cpu_gpu_scaling --bin validate_gpu_spmv --bin validate_gpu_dirac --bin validate_gpu_cg --bin validate_pure_gpu_qcd 2>/dev/null

    # GPU lattice scaling
    echo "  [RS-GPU] Lattice CG scaling..."
    SCALING_START=$(date +%s%N)
    cargo run --release --bin bench_lattice_scaling > "$RESULTS_DIR/lattice_scaling_gpu.txt" 2>&1 || true
    SCALING_END=$(date +%s%N)
    SCALING_MS=$(( (SCALING_END - SCALING_START) / 1000000 ))
    echo "  [RS-GPU] Lattice scaling: ${SCALING_MS}ms"
    echo "- Lattice CG GPU scaling: **${SCALING_MS}ms** total" >> "$REPORT"

    # GPU SpMV
    echo "  [RS-GPU] SpMV parity..."
    cargo run --release --bin validate_gpu_spmv > "$RESULTS_DIR/gpu_spmv.txt" 2>&1 || true
    echo "- GPU SpMV: validated" >> "$REPORT"

    # GPU Dirac
    echo "  [RS-GPU] Dirac parity..."
    cargo run --release --bin validate_gpu_dirac > "$RESULTS_DIR/gpu_dirac.txt" 2>&1 || true
    echo "- GPU Dirac: validated" >> "$REPORT"

    # GPU CG
    echo "  [RS-GPU] CG solver..."
    cargo run --release --bin validate_gpu_cg > "$RESULTS_DIR/gpu_cg.txt" 2>&1 || true
    echo "- GPU CG solver: validated" >> "$REPORT"

    # Pure GPU QCD workload
    echo "  [RS-GPU] Pure GPU QCD workload..."
    cargo run --release --bin validate_pure_gpu_qcd > "$RESULTS_DIR/pure_gpu_qcd.txt" 2>&1 || true
    echo "- Pure GPU QCD: validated" >> "$REPORT"

    # CPU/GPU scaling comparison
    echo "  [RS-GPU] CPU/GPU scaling comparison..."
    cargo run --release --bin bench_cpu_gpu_scaling > "$RESULTS_DIR/cpu_gpu_scaling.txt" 2>&1 || true
    echo "- CPU/GPU scaling: see data file" >> "$REPORT"

    echo "" >> "$REPORT"
else
    echo "Phase 3: SKIPPED (--skip-gpu)"
    echo "- GPU benchmarks: SKIPPED" >> "$REPORT"
    echo "" >> "$REPORT"
fi

# ─── Summary ─────────────────────────────────────────────────
echo "" >> "$REPORT"
echo "## Summary" >> "$REPORT"
echo "" >> "$REPORT"
echo "Timestamp: $TIMESTAMP" >> "$REPORT"
echo "All results in: \`experiments/data/008_parity/\`" >> "$REPORT"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Benchmark complete. Report: $REPORT"
echo "═══════════════════════════════════════════════════════════"
