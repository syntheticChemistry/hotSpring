#!/bin/bash
# Titan V Timing Attack — FECS warm handoff via nouveau with rapid BAR0 polling.
#
# Research script: supplements `coralctl warm-fecs` with 100ms BAR0 CPUCTL
# polling during nouveau load to capture the exact FECS boot moment.
#
# All driver transitions go through coralctl (ember). Livepatch is managed
# by the warm-fecs flow automatically. Direct sysfs writes are prohibited
# per scripts/README.md safety policy.
#
# For production warm-fecs, use: coralctl warm-fecs <BDF>
set -euo pipefail

BDF="${1:-0000:03:00.0}"
SOCKET="/run/coralreef/glowplug.sock"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BAR0_READ="python3 ${SCRIPT_DIR}/bar0_read.py"

echo "=== Titan V Timing Attack ==="
echo "BDF: ${BDF}"
echo ""

# Step 0: Pre-swap state capture
echo "step 0: checking ember/glowplug connectivity..."
coralctl --socket "${SOCKET}" status >/dev/null 2>&1 || {
    echo "ERROR: glowplug not running. Start with: sudo systemctl start coral-glowplug"
    exit 1
}
echo "  glowplug connected"

# Capture pre-swap state
echo ""
echo "step 0b: pre-swap FECS state..."
FECS_PRE=$(${BAR0_READ} "${BDF}" 0x409100 2>/dev/null || echo "0xDEADDEAD")
SCTL_PRE=$(${BAR0_READ} "${BDF}" 0x409240 2>/dev/null || echo "0xDEADDEAD")
echo "  FECS CPUCTL=${FECS_PRE}  SCTL=${SCTL_PRE}"

# Step 1: Swap to nouveau
echo ""
echo "step 1: swapping ${BDF} -> nouveau..."
coralctl --socket "${SOCKET}" swap "${BDF}" nouveau 2>&1 || true
echo "  swap sent"

# Step 2: Poll FECS CPUCTL rapidly
echo ""
echo "step 2: polling FECS CPUCTL every 100ms..."
POLL_START=$(date +%s%N)
MAX_POLLS=300  # 30 seconds max
CAUGHT_RUNNING=false

for i in $(seq 1 ${MAX_POLLS}); do
    CPUCTL=$(${BAR0_READ} "${BDF}" 0x409100 2>/dev/null || echo "0xERROR")
    
    ELAPSED_MS=$(( ($(date +%s%N) - POLL_START) / 1000000 ))
    
    # Parse: bit4=halted, bit5=stopped
    if [ "${CPUCTL}" = "0xERROR" ]; then
        [ $((i % 50)) -eq 0 ] && echo "  [${ELAPSED_MS}ms] read error"
    else
        VAL=$((CPUCTL))
        HALTED=$(( (VAL >> 4) & 1 ))
        STOPPED=$(( (VAL >> 5) & 1 ))
        DEAD=0
        [ "${CPUCTL}" = "0xdeaddead" ] && DEAD=1
        
        if [ ${DEAD} -eq 0 ] && [ ${HALTED} -eq 0 ] && [ ${STOPPED} -eq 0 ] && [ ${VAL} -ne 0 ]; then
            echo "  [${ELAPSED_MS}ms] *** FECS RUNNING *** CPUCTL=${CPUCTL}"
            CAUGHT_RUNNING=true
            break
        fi
        
        # Print state changes (not every poll)
        if [ $((i % 20)) -eq 0 ] || [ ${i} -eq 1 ]; then
            echo "  [${ELAPSED_MS}ms] CPUCTL=${CPUCTL} halted=${HALTED} stopped=${STOPPED}"
        fi
    fi
    
    sleep 0.1
done

if ! ${CAUGHT_RUNNING}; then
    CPUCTL_FINAL=$(${BAR0_READ} "${BDF}" 0x409100 2>/dev/null || echo "0xERROR")
    echo "  FECS never caught running. Final CPUCTL=${CPUCTL_FINAL}"
fi

# Step 3: Swap to vfio via coralctl (ember handles livepatch automatically
# when warm_handoff is used; for manual swap we enable it explicitly)
echo ""
echo "step 3: swapping ${BDF} -> vfio via coralctl..."
coralctl --socket "${SOCKET}" swap "${BDF}" vfio 2>&1 || {
    echo "  ERROR: swap to vfio failed. Check: coralctl --socket ${SOCKET} status"
    echo "  Do NOT attempt manual sysfs recovery (D-state risk)."
    exit 1
}

# Step 4: Post-swap FECS state
echo ""
echo "step 4: post-swap FECS state..."
sleep 1
FECS_POST=$(${BAR0_READ} "${BDF}" 0x409100 2>/dev/null || echo "0xDEADDEAD")
SCTL_POST=$(${BAR0_READ} "${BDF}" 0x409240 2>/dev/null || echo "0xDEADDEAD")
PC_POST=$(${BAR0_READ} "${BDF}" 0x409030 2>/dev/null || echo "0xDEADDEAD")
MB0_POST=$(${BAR0_READ} "${BDF}" 0x409040 2>/dev/null || echo "0xDEADDEAD")
PMC_POST=$(${BAR0_READ} "${BDF}" 0x200 2>/dev/null || echo "0xDEADDEAD")
echo "  FECS CPUCTL=${FECS_POST}  SCTL=${SCTL_POST}"
echo "  PC=${PC_POST}  MB0=${MB0_POST}  PMC=${PMC_POST}"

VAL=$((FECS_POST))
POST_HALTED=$(( (VAL >> 4) & 1 ))
POST_STOPPED=$(( (VAL >> 5) & 1 ))
echo "  halted=${POST_HALTED} stopped=${POST_STOPPED}"

echo ""
echo "=== Timing Attack Results ==="
echo "  Caught FECS running during nouveau: ${CAUGHT_RUNNING}"
echo "  Post-swap FECS CPUCTL: ${FECS_POST}"
if [ ${POST_HALTED} -eq 0 ] && [ ${POST_STOPPED} -eq 0 ] && [ "${FECS_POST}" != "0xdeaddead" ]; then
    echo "  >>> FECS SURVIVED AND RUNNING <<<"
elif [ "${FECS_POST}" != "0xdeaddead" ]; then
    echo "  FECS survived but halted/stopped"
else
    echo "  FECS state lost"
fi
echo "=== done ==="
