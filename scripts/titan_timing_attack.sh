#!/bin/bash
# Titan V Timing Attack — FECS warm handoff via nouveau with rapid polling.
#
# 1. Swap to nouveau (loads ACR → FECS firmware)
# 2. Poll FECS CPUCTL via BAR0 mmap every 100ms
# 3. As soon as FECS is running (not halted/stopped), enable livepatch and swap to vfio
# 4. Check post-swap FECS state
set -euo pipefail

BDF="${1:-0000:03:00.0}"
SOCKET="/run/coralreef/glowplug.sock"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BAR0_READ="python3 ${SCRIPT_DIR}/bar0_read.py"

echo "=== Titan V Timing Attack ==="
echo "BDF: ${BDF}"
echo ""

# Step 0: Disable livepatch before nouveau load
echo "step 0: disabling livepatch for clean nouveau init..."
echo 0 > /sys/kernel/livepatch/livepatch_nvkm_mc_reset/enabled 2>/dev/null || true
sleep 2
# Wait for livepatch transition to complete
for i in $(seq 1 10); do
    if [ "$(cat /sys/kernel/livepatch/livepatch_nvkm_mc_reset/transition 2>/dev/null)" = "0" ] 2>/dev/null; then
        break
    fi
    sleep 1
done
echo "  livepatch disabled"

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

# Step 3: Enable livepatch (freeze teardown)
echo ""
echo "step 3: enabling livepatch..."
modprobe livepatch_nvkm_mc_reset 2>/dev/null || true
echo 1 > /sys/kernel/livepatch/livepatch_nvkm_mc_reset/enabled 2>/dev/null || true
sleep 1
LP_ENABLED=$(cat /sys/kernel/livepatch/livepatch_nvkm_mc_reset/enabled 2>/dev/null || echo "?")
echo "  livepatch enabled: ${LP_ENABLED}"

# Step 4: Swap to vfio
echo ""
echo "step 4: swapping ${BDF} -> vfio..."
coralctl --socket "${SOCKET}" swap "${BDF}" vfio 2>&1 || {
    echo "  swap failed — trying manual recovery"
    /usr/local/bin/coralreef-sysfs-write "/sys/bus/pci/devices/${BDF}/driver_override" vfio-pci 2>/dev/null || true
    /usr/local/bin/coralreef-sysfs-write "/sys/bus/pci/devices/${BDF}/driver/unbind" "${BDF}" 2>/dev/null || true
    sleep 2
    /usr/local/bin/coralreef-sysfs-write "/sys/bus/pci/drivers/vfio-pci/bind" "${BDF}" 2>/dev/null || true
    echo "  manual recovery attempted"
}

# Step 5: Post-swap FECS state
echo ""
echo "step 5: post-swap FECS state..."
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
