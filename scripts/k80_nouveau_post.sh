#!/usr/bin/env bash
# k80_nouveau_post.sh — POST K80 via patched nouveau, then swap to vfio-pci.
#
# The K80's GK210 PLLs are hardware-gated at cold boot and can only be
# configured by a driver that executes VBIOS DEVINIT scripts (nouveau).
# Upstream nouveau doesn't recognize GK210 (chip_id 0xF2) — our patched
# nouveau.ko maps it to nvf1_chipset (GK110B equivalent).
#
# Flow: unbind vfio-pci → load nouveau → nouveau cold POSTs K80 → unbind
# nouveau → rebind vfio-pci → coral-driver open_legacy() for warm compute.
#
# Usage:
#   sudo ./k80_nouveau_post.sh [die0|die1|both]
#   sudo ./k80_nouveau_post.sh              # defaults to die0

set -euo pipefail

MODE="${1:-die0}"
K80_DIE0="0000:4c:00.0"
K80_DIE1="0000:4d:00.0"
SETTLE_SECS="${SETTLE_SECS:-10}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[k80]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

SYSFS_TIMEOUT=30
CORAL_PROBE="${CORAL_PROBE:-coral-probe}"

gpu_read() {
    local bdf="$1" addr="$2"
    if command -v "$CORAL_PROBE" &>/dev/null; then
        "$CORAL_PROBE" read "$bdf" "$addr" 2>/dev/null \
            | grep -oP '0x[0-9a-fA-F]+$' || echo "0xDEADDEAD"
    else
        python3 -c "
import mmap, struct, sys
f = open('/sys/bus/pci/devices/$bdf/resource0', 'r+b')
mm = mmap.mmap(f.fileno(), 0x1000000)
mm.seek(int('$addr', 0))
v = struct.unpack('<I', mm.read(4))[0]
print(f'{v:#010x}')
mm.close(); f.close()
" 2>/dev/null || echo "0xDEADDEAD"
    fi
}

gpu_write() {
    local bdf="$1" addr="$2" val="$3"
    python3 -c "
import mmap, struct
f = open('/sys/bus/pci/devices/$bdf/resource0', 'r+b')
mm = mmap.mmap(f.fileno(), 0x1000000)
mm.seek(int('$addr', 0))
mm.write(struct.pack('<I', int('$val', 0)))
mm.close(); f.close()
" 2>/dev/null
}

sysfs_write() {
    local target="$1" content="$2" desc="${3:-sysfs write}"
    if ! timeout "$SYSFS_TIMEOUT" bash -c "echo '$content' > '$target'" 2>/dev/null; then
        fail "$desc TIMED OUT after ${SYSFS_TIMEOUT}s — kernel may be stuck on $target"
    fi
}

case "$MODE" in
    die0) TARGETS=("$K80_DIE0") ;;
    die1) TARGETS=("$K80_DIE1") ;;
    both) TARGETS=("$K80_DIE0" "$K80_DIE1") ;;
    *)    fail "Usage: $0 [die0|die1|both]" ;;
esac

[[ $EUID -eq 0 ]] || fail "Must run as root"

# ── Phase 1: Preflight ──────────────────────────────────────
log "Phase 1: Preflight"

for BDF in "${TARGETS[@]}"; do
    [[ -d "/sys/bus/pci/devices/$BDF" ]] || fail "$BDF not found on PCI bus"

    BOOT0=$(gpu_read "$BDF" 0x0)
    log "$BDF BOOT0=$BOOT0"
    [[ "$BOOT0" != "0xffffffff" ]] || fail "$BDF is dead (link down)"

    DRV="(none)"
    [[ -L "/sys/bus/pci/devices/$BDF/driver" ]] && \
        DRV="$(basename "$(readlink "/sys/bus/pci/devices/$BDF/driver")")"
    log "$BDF driver=$DRV"
done

# ── Phase 2: Unbind current driver ──────────────────────────
log "Phase 2: Unbinding current drivers"

for BDF in "${TARGETS[@]}"; do
    SYSFS="/sys/bus/pci/devices/$BDF"
    if [[ -L "$SYSFS/driver" ]]; then
        DRV="$(basename "$(readlink "$SYSFS/driver")")"
        sysfs_write "/sys/bus/pci/drivers/$DRV/unbind" "$BDF" "unbind $BDF from $DRV"
        ok "Unbound $BDF from $DRV"
    fi
    sysfs_write "$SYSFS/driver_override" "" "clear driver_override for $BDF"
done

# ── Phase 3: Load livepatch BEFORE nouveau ──────────────────
# CRITICAL DISCOVERY: gk110_pmu_pgob writes to PPWR registers (0x0205xx)
# that don't exist on K80/GK210B. These writes cause PRIVRING faults that
# CORRUPT the PRI ring, making GPC stations unreachable (0xbadf1100).
# By loading the livepatch FIRST, we NOP nvkm_pmu_pgob before nouveau's
# GR init can issue the corrupting writes. The livepatch targets
# .name="nouveau" so it patches functions when nouveau loads.
log "Phase 3: Loading livepatch BEFORE nouveau (prevents PRI ring corruption)"

LIVEPATCH_DIR="$(cd "$(dirname "$0")/livepatch" && pwd)"
LIVEPATCH_KO="$LIVEPATCH_DIR/livepatch_nvkm_mc_reset.ko"

# Unload nouveau first if loaded (so we get a clean init with livepatch)
if grep -q "^nouveau " /proc/modules 2>/dev/null; then
    log "Unloading existing nouveau to ensure clean livepatch-first init..."
    rmmod nouveau 2>/dev/null || warn "rmmod nouveau failed (may have dependents)"
    sleep 1
fi

# Unload old livepatch if present
if grep -q "^livepatch_nvkm_mc_reset " /proc/modules 2>/dev/null; then
    echo 0 > /sys/kernel/livepatch/livepatch_nvkm_mc_reset/enabled 2>/dev/null
    sleep 2
    rmmod livepatch_nvkm_mc_reset 2>/dev/null
    sleep 1
fi

# Load livepatch FIRST — it will wait for nouveau and patch on load
if [[ -f "$LIVEPATCH_KO" ]]; then
    insmod "$LIVEPATCH_KO" 2>&1 || warn "livepatch insmod failed"
    if grep -q "^livepatch_nvkm_mc_reset " /proc/modules 2>/dev/null; then
        ok "livepatch loaded (waiting for nouveau to patch)"
    fi
else
    warn "livepatch module not found at $LIVEPATCH_KO!"
fi

# Disable runtime PM on K80 devices BEFORE loading nouveau.
for BDF in "${TARGETS[@]}"; do
    PM_CTRL="/sys/bus/pci/devices/$BDF/power/control"
    if [[ -f "$PM_CTRL" ]]; then
        echo "on" > "$PM_CTRL" 2>/dev/null
    fi
done

# ── Phase 3.5: Load nouveau (livepatch will patch it on load) ────
log "Phase 3.5: Loading nouveau (livepatch will intercept gk110_pmu_pgob)"

if grep -q "^nouveau " /proc/modules 2>/dev/null; then
    log "nouveau already loaded"
else
    modprobe --ignore-install nouveau 2>&1 || fail "modprobe nouveau failed"
    ok "nouveau loaded"
fi

# Check livepatch transition
LP_TRANS="/sys/kernel/livepatch/livepatch_nvkm_mc_reset/transition"
if [[ -f "$LP_TRANS" ]]; then
    for i in $(seq 1 10); do
        if [[ "$(cat "$LP_TRANS" 2>/dev/null)" == "0" ]]; then
            ok "livepatch fully transitioned"
            break
        fi
        sleep 1
    done
fi

# Re-assert runtime PM disable (in case nouveau's probe re-enabled it)
for BDF in "${TARGETS[@]}"; do
    PM_CTRL="/sys/bus/pci/devices/$BDF/power/control"
    if [[ -f "$PM_CTRL" ]]; then
        echo "on" > "$PM_CTRL" 2>/dev/null
    fi
done

# CRITICAL CHECKPOINT: check GPCs after nouveau with livepatch-protected init
for BDF in "${TARGETS[@]}"; do
    GPC0_IMM=$(gpu_read "$BDF" 0x500000)
    FECS_IMM=$(gpu_read "$BDF" 0x409100)
    PMU_IMM=$(gpu_read "$BDF" 0x10A100)
    PMC_IMM=$(gpu_read "$BDF" 0x200)
    NSTATIONS=$(gpu_read "$BDF" 0x120070)
    log "$BDF CHECKPOINT: GPC0=$GPC0_IMM FECS=$FECS_IMM PMU=$PMU_IMM PMC=$PMC_IMM NSTATIONS=$NSTATIONS"
done

sleep 1

for BDF in "${TARGETS[@]}"; do
    SYSFS="/sys/bus/pci/devices/$BDF"
    if [[ ! -L "$SYSFS/driver" ]]; then
        log "Probing $BDF..."
        sysfs_write "/sys/bus/pci/drivers_probe" "$BDF" "drivers_probe $BDF (nouveau)"
    fi
done

log "Waiting ${SETTLE_SECS}s for nouveau to cold-POST K80..."
sleep "$SETTLE_SECS"

# Verify nouveau claimed devices
for BDF in "${TARGETS[@]}"; do
    SYSFS="/sys/bus/pci/devices/$BDF"
    if [[ -L "$SYSFS/driver" ]]; then
        DRV="$(basename "$(readlink "$SYSFS/driver")")"
        if [[ "$DRV" == "nouveau" ]]; then
            ok "nouveau claimed $BDF"
        else
            warn "$BDF claimed by $DRV (expected nouveau)"
        fi
    else
        warn "nouveau did not claim $BDF"
        dmesg | grep -i "nouveau.*${BDF##*:}\|chipset\|error" | tail -10
        fail "nouveau probe failed for $BDF"
    fi

    # Check if GPCs are accessible NOW (with livepatch protecting from re-gate)
    GPC0_LIVE=$(gpu_read "$BDF" 0x500000)
    FECS_LIVE=$(gpu_read "$BDF" 0x409100)
    PMU_LIVE=$(gpu_read "$BDF" 0x10A100)
    log "$BDF after nouveau init: GPC0=$GPC0_LIVE FECS_CPUCTL=$FECS_LIVE PMU_CPUCTL=$PMU_LIVE"
done

# ── Phase 4: Verify PLLs are alive ──────────────────────────
log "Phase 4: Verifying PLL state after POST"

for BDF in "${TARGETS[@]}"; do
    PLL0=$(gpu_read "$BDF" 0x130000)
    PLL_COEF=$(gpu_read "$BDF" 0x130004)
    PTIMER=$(gpu_read "$BDF" 0x9400)
    PMC=$(gpu_read "$BDF" 0x200)
    log "$BDF: PLL0=$PLL0 PLL_COEF=$PLL_COEF PTIMER=$PTIMER PMC=$PMC"

    if [[ "$PLL0" == "0xbadf"* ]]; then
        warn "$BDF PLLs still faulting — POST may have failed"
    else
        ok "$BDF PLLs accessible — POST successful!"
    fi
done

# ── Phase 4.5: Run PGOB to power up GPCs ──
# On GK110/GK210B, GPCs are power-gated at boot. The PGOB (Power Gate Off
# Block) sequence ungate them by writing to PPWR power domain registers.
# PMU must remain RUNNING — it processes the 0x10a78c handshake that
# coordinates the power ungating. Halting PMU before PGOB prevents the
# handshake from completing.
log "Phase 4.5: Running PGOB with PMU alive to power up GPCs"

for BDF in "${TARGETS[@]}"; do
    GPC0_PRE=$(gpu_read "$BDF" 0x500000)
    PMU_CPUCTL=$(gpu_read "$BDF" 0x10a100)
    log "$BDF pre-PGOB: GPC0=$GPC0_PRE PMU_CPUCTL=$PMU_CPUCTL"

    # Full gk110_pmu_pgob() matching kernel gk110.c (with 0x0205xx power steps).
    # PMU must be RUNNING for the 0x10a78c handshake. The 0x0205xx registers
    # DO respond on GK210B but may need nvidia-470 mmiotrace to find what's
    # different in the proprietary driver's sequence.
    python3 -c "
import mmap, struct, time

bdf = '$BDF'
f = open(f'/sys/bus/pci/devices/{bdf}/resource0', 'r+b')
mm = mmap.mmap(f.fileno(), 0x1000000)

def rd(reg):
    mm.seek(reg)
    return struct.unpack('<I', mm.read(4))[0]

def wr(reg, val):
    mm.seek(reg)
    mm.write(struct.pack('<I', val & 0xFFFFFFFF))

def mask(reg, clr, set_bits):
    v = rd(reg)
    wr(reg, (v & ~clr) | set_bits)

pmc_before = rd(0x200)
pmu_psw = rd(0x10a78c)
pmu_cpuctl = rd(0x10a100)
gpc0_before = rd(0x500000)
nstations_pre = rd(0x120070)
print(f'Pre: PMC={pmc_before:#010x} PSW={pmu_psw:#010x} PMU={pmu_cpuctl:#010x} GPC0={gpc0_before:#010x} NSTATIONS={nstations_pre}')

# Kernel gk110_pmu_pgob() steps:
# 1. Disable PGRAPH (PMC bit 12)
mask(0x200, 0x1000, 0)
rd(0x200)
# 2. Set PMC bit 27 (PGOB gate)
mask(0x200, 0x08000000, 0x08000000)
time.sleep(0.05)
# 3. PMU PSW: set bit 1, pulse bit 0
mask(0x10a78c, 0x2, 0x2)
mask(0x10a78c, 0x1, 0x1)
mask(0x10a78c, 0x1, 0x0)
# 4. NOP mask on 0x0206b4 (hardware sync)
mask(0x0206b4, 0, 0)
# 5. Power steps with bit-31 polling (2s timeout each)
STEPS = [
    (0x020520, 0xFFFFFFFC), (0x020524, 0xFFFFFFFE), (0x020524, 0xFFFFFFFC),
    (0x020524, 0xFFFFFFF8), (0x020524, 0xFFFFFFE0), (0x020530, 0xFFFFFFFE),
    (0x02052C, 0xFFFFFFFA), (0x02052C, 0xFFFFFFF0), (0x02052C, 0xFFFFFFC0),
    (0x02052C, 0xFFFFFF00), (0x02052C, 0xFFFFFC00), (0x02052C, 0xFFFCFC00),
    (0x02052C, 0xFFF0FC00), (0x02052C, 0xFF80FC00), (0x020528, 0xFFFFFFFE),
    (0x020528, 0xFFFFFFFC),
]
for i, (addr, data) in enumerate(STEPS):
    pre = rd(addr)
    wr(addr, data)
    for _ in range(200):
        time.sleep(0.01)
        if rd(addr) & 0x80000000 == 0:
            break
    post = rd(addr)
    if i == 0 or i == len(STEPS)-1:
        print(f'  Step[{i}] {addr:#08x}: pre={pre:#010x} post={post:#010x}')
# 6. PMU PSW: clear bit 1, pulse bit 0
mask(0x10a78c, 0x2, 0x0)
mask(0x10a78c, 0x1, 0x1)
mask(0x10a78c, 0x1, 0x0)
# 7. Clear PMC bit 27, re-enable PGRAPH
mask(0x200, 0x08000000, 0x0)
mask(0x200, 0x1000, 0x1000)
rd(0x200)
time.sleep(0.05)

# Re-enumerate PRI ring
wr(0x12004c, 0x02)
time.sleep(0.02)
wr(0x12004c, 0x04)
for _ in range(200):
    time.sleep(0.01)
    if rd(0x120058) & 0x80000000 == 0:
        break

pmc_after = rd(0x200)
nstations_after = rd(0x120070)
gpc0_after = rd(0x500000)
fecs_after = rd(0x409100)
gr_hub = rd(0x400000)
print(f'Post: PMC={pmc_after:#010x} NSTATIONS={nstations_after} GR_HUB={gr_hub:#010x}')
print(f'Post: GPC0={gpc0_after:#010x} FECS={fecs_after:#010x}')

mm.close()
f.close()
" 2>&1 | while IFS= read -r line; do log "$BDF $line"; done

    GPC0_POST=$(gpu_read "$BDF" 0x500000)
    log "$BDF post-PGOB: GPC0=$GPC0_POST"
    if [[ "$GPC0_POST" == "0xbadf"* ]]; then
        warn "$BDF GPCs still power-gated after PGOB"
    else
        ok "$BDF GPCs accessible after PGOB!"
    fi
done

# ── Phase 4.75: Dump FECS/GPCCS firmware from IMEM ──────────
# Nouveau loaded its firmware into FECS/GPCCS IMEM during GR init.
# Extract it before unbinding so we can reload it via VFIO.
# FECS IMEM: read via 0x409180 (addr) / 0x409184 (data)
# GPCCS IMEM: read via 0x41A180 (addr) / 0x41A184 (data)
FW_DIR="${CORAL_FW_DIR:-$(cd "$(dirname "$0")/../../../primals/coralReef/crates/coral-driver/firmware/gk110" 2>/dev/null && pwd || echo "/home/biomegate/Development/ecoPrimals/primals/coralReef/crates/coral-driver/firmware/gk110")}"
for BDF in "${TARGETS[@]}"; do
    python3 -c "
import mmap, struct, sys

bdf = '$BDF'
fw_dir = '$FW_DIR'

f = open(f'/sys/bus/pci/devices/{bdf}/resource0', 'r+b')
mm = mmap.mmap(f.fileno(), 0x1000000)

def rd(off):
    mm.seek(off)
    return struct.unpack('<I', mm.read(4))[0]

def wr(off, val):
    mm.seek(off)
    mm.write(struct.pack('<I', val))

def dump_imem(base, max_words):
    data = bytearray()
    for i in range(max_words):
        wr(base + 0x180, i * 4)
        w = rd(base + 0x184)
        if w == 0xdeaddead or w == 0xbadf1100:
            break
        data.extend(struct.pack('<I', w))
    return bytes(data)

# FECS IMEM (0x409000 base)
fecs_code = dump_imem(0x409000, 16384)
# FECS DMEM (0x409000 base, DMEM at +0x1C0/+0x1C4)
fecs_data = bytearray()
for i in range(4096):
    wr(0x4091C0, i * 4)
    w = rd(0x4091C4)
    if w == 0xdeaddead or w == 0xbadf1100:
        break
    fecs_data.extend(struct.pack('<I', w))
fecs_data = bytes(fecs_data)

# GPCCS IMEM (0x41A000 base)
gpccs_code = dump_imem(0x41A000, 16384)
# GPCCS DMEM
gpccs_data = bytearray()
for i in range(4096):
    wr(0x41A1C0, i * 4)
    w = rd(0x41A1C4)
    if w == 0xdeaddead or w == 0xbadf1100:
        break
    gpccs_data.extend(struct.pack('<I', w))
gpccs_data = bytes(gpccs_data)

if len(fecs_code) > 0:
    with open(f'{fw_dir}/gk110_fecs_code.bin', 'wb') as out:
        out.write(fecs_code)
    print(f'FECS code: {len(fecs_code)} bytes')
else:
    print('FECS code: EMPTY (falcon may not be initialized)')

if len(fecs_data) > 0:
    with open(f'{fw_dir}/gk110_fecs_data.bin', 'wb') as out:
        out.write(fecs_data)
    print(f'FECS data: {len(fecs_data)} bytes')

if len(gpccs_code) > 0:
    with open(f'{fw_dir}/gk110_gpccs_code.bin', 'wb') as out:
        out.write(gpccs_code)
    print(f'GPCCS code: {len(gpccs_code)} bytes')
else:
    print('GPCCS code: EMPTY')

if len(gpccs_data) > 0:
    with open(f'{fw_dir}/gk110_gpccs_data.bin', 'wb') as out:
        out.write(gpccs_data)
    print(f'GPCCS data: {len(gpccs_data)} bytes')

mm.close()
f.close()
" 2>&1 | while IFS= read -r line; do ok "$BDF firmware: $line"; done
done

# ── Phase 5: Swap to vfio-pci ───────────────────────────────
log "Phase 5: Swapping to vfio-pci"

for BDF in "${TARGETS[@]}"; do
    SYSFS="/sys/bus/pci/devices/$BDF"

    # Probe FECS + GPC state BEFORE unbind
    FECS_CPUCTL=$(gpu_read "$BDF" 0x409100)
    FECS_SCRATCH0=$(gpu_read "$BDF" 0x409500)
    FECS_PC=$(gpu_read "$BDF" 0x409030)
    GPC0_PRE=$(gpu_read "$BDF" 0x500000)
    GPC1_PRE=$(gpu_read "$BDF" 0x508000)
    GPC_BCAST_PRE=$(gpu_read "$BDF" 0x418000)
    PMC_PRE=$(gpu_read "$BDF" 0x200)
    log "$BDF BEFORE unbind: CPUCTL=$FECS_CPUCTL SCRATCH0=$FECS_SCRATCH0 PC=$FECS_PC"
    log "$BDF BEFORE unbind: GPC0=$GPC0_PRE GPC1=$GPC1_PRE BCAST=$GPC_BCAST_PRE PMC=$PMC_PRE"

    # Before unbind: force nouveau to initialize the GR engine.
    # mmiotrace reveals nouveau NEVER inits GR on headless K80 (zero
    # PGRAPH register accesses). We must force GR init by opening the
    # render node and allocating a GEM buffer, which triggers
    # nouveau's gf100_gr_init → FECS/GPCCS boot.
    log "Forcing GR engine init via DRM render node..."

    if [[ -f "$SYSFS/power/control" ]]; then
        echo "on" > "$SYSFS/power/control" 2>/dev/null || true
    fi

    # Find this GPU's DRM render node.
    RENDER_NODE=""
    for CARD_DIR in /sys/class/drm/renderD*; do
        [[ -d "$CARD_DIR/device" ]] || continue
        CARD_BDF="$(basename "$(readlink -f "$CARD_DIR/device")")"
        if [[ "$CARD_BDF" == "$BDF" ]]; then
            RENDER_NODE="/dev/dri/$(basename "$CARD_DIR")"
            break
        fi
    done

    if [[ -n "$RENDER_NODE" && -c "$RENDER_NODE" ]]; then
        log "$BDF render node: $RENDER_NODE — forcing GR engine init"

        # Preferred path: compiled libdrm_nouveau tool (handles all ioctl details).
        FORCE_GR_TOOL="$(cd "$(dirname "$0")/tools" && pwd)/k80_force_gr_init"
        if [[ -x "$FORCE_GR_TOOL" ]]; then
            "$FORCE_GR_TOOL" "$RENDER_NODE" 2>&1 | while IFS= read -r line; do log "  $line"; done || true
        else
            warn "k80_force_gr_init not compiled — using raw ioctl fallback"
            # Fallback: raw DRM ioctl for channel allocation.
            #
            # struct drm_nouveau_channel_alloc (88 bytes):
            #   offset  0: fb_ctxdma_handle  (u32) = 0
            #   offset  4: tt_ctxdma_handle  (u32) = 0x01 (NOUVEAU_FIFO_ENGINE_GR for Kepler!)
            #   offset  8: channel           (s32) = output
            #   offset 12: pushbuf_domains   (u32) = 0x06 (VRAM|GART)
            #   offset 16: notifier_handle   (u32) = 0
            #   offset 20: subchan[8]        (64 bytes) = output
            #   offset 84: nr_subchan        (u32) = output
            #
            # DRM_IOCTL_NOUVEAU_CHANNEL_ALLOC = DRM_IOWR('d', 0x42, 88) = 0xC0586442
            #   DRM_COMMAND_BASE=0x40 + DRM_NOUVEAU_CHANNEL_ALLOC=0x02 = 0x42
            python3 -c "
import fcntl, struct, os, errno

fd = os.open('$RENDER_NODE', os.O_RDWR)
print(f'Opened render node fd={fd}')

buf = bytearray(88)
# fb_ctxdma=0, tt_ctxdma=NOUVEAU_FIFO_ENGINE_GR(0x01), channel=0(output), pushbuf=VRAM|GART(0x6)
struct.pack_into('<IIiI', buf, 0, 0, 0x01, 0, 0x06)

try:
    fcntl.ioctl(fd, 0xC0586442, buf)
    channel_id = struct.unpack_from('<i', buf, 8)[0]
    nr_subchan = struct.unpack_from('<I', buf, 84)[0]
    print(f'CHANNEL_ALLOC ok: channel={channel_id} nr_subchan={nr_subchan}')
    for i in range(min(nr_subchan, 8)):
        handle, grclass = struct.unpack_from('<II', buf, 20 + i*8)
        print(f'  subchan[{i}]: handle=0x{handle:08x} grclass=0x{grclass:04x}')
    print('GR engine initialized')
except OSError as e:
    print(f'CHANNEL_ALLOC failed: errno={e.errno} ({os.strerror(e.errno)})')

os.close(fd)
" 2>&1 | while IFS= read -r line; do log "  $line"; done
        fi

        sleep 3
    else
        warn "$BDF no render node found — GR init skipped (nouveau may not have created DRM device)"
    fi

    # Re-check GPCs and FECS after GR init attempt
    GPC0_WAKE=$(gpu_read "$BDF" 0x500000)
    FECS_POST_GR=$(gpu_read "$BDF" 0x409100)
    GR_HUB_POST=$(gpu_read "$BDF" 0x400000)
    log "$BDF after GR init: GPC0=$GPC0_WAKE FECS_CPUCTL=$FECS_POST_GR GR_HUB=$GR_HUB_POST"

    # Unbind from nouveau (livepatch prevents FECS IMEM wipe)
    if [[ -L "$SYSFS/driver" ]]; then
        sysfs_write "/sys/bus/pci/drivers/nouveau/unbind" "$BDF" "unbind $BDF from nouveau"
        ok "Unbound $BDF from nouveau"
    fi

    # Check GPCs immediately after unbind, before vfio bind
    sleep 0.5
    GPC0_POSTUNBIND=$(gpu_read "$BDF" 0x500000)
    PMC_POSTUNBIND=$(gpu_read "$BDF" 0x200)
    log "$BDF RIGHT AFTER unbind (no driver): GPC0=$GPC0_POSTUNBIND PMC=$PMC_POSTUNBIND"

    # Disable ALL reset methods — device AND upstream bridges.
    # vfio-pci does a secondary bus reset via the bridge when opening/closing
    # the group. This destroys all GPU state (PMC, PRI ring, power domains).
    sysfs_write "$SYSFS/reset_method" "" "disable reset methods for $BDF"
    
    # Disable bridge resets on the entire PCIe path to prevent bus-level reset.
    PARENT_BRIDGE="$(basename "$(readlink "$SYSFS/.." 2>/dev/null)" 2>/dev/null)"
    if [[ -n "$PARENT_BRIDGE" ]]; then
        for ancestor in "$SYSFS/.." "$SYSFS/../.." "$SYSFS/../../.."; do
            bridge_dev="$(readlink -f "$ancestor" 2>/dev/null)"
            if [[ -f "$bridge_dev/reset_method" ]]; then
                echo "" > "$bridge_dev/reset_method" 2>/dev/null || true
                log "Disabled reset on bridge $(basename "$bridge_dev")"
            fi
        done
    fi
    # Also disable by known BDF for this K80 topology
    for bridge in 0000:4b:08.0 0000:4b:10.0 0000:4a:00.0 0000:40:01.3; do
        if [[ -f "/sys/bus/pci/devices/$bridge/reset_method" ]]; then
            echo "" > "/sys/bus/pci/devices/$bridge/reset_method" 2>/dev/null || true
        fi
    done
    ok "All reset methods disabled (device + bridges)"

    # Set driver_override and bind to vfio-pci
    sysfs_write "$SYSFS/driver_override" "vfio-pci" "set driver_override to vfio-pci for $BDF"
    if ! timeout "$SYSFS_TIMEOUT" bash -c "echo '$BDF' > /sys/bus/pci/drivers/vfio-pci/bind" 2>/dev/null; then
        log "direct vfio-pci bind timed out, trying drivers_probe..."
        sysfs_write "/sys/bus/pci/drivers_probe" "$BDF" "drivers_probe $BDF (vfio-pci fallback)"
    fi
    sleep 1

    # NOTE: Do NOT read registers after vfio-pci bind!
    # gpu_read would open the VFIO group, and when this script exits,
    # the group release triggers a secondary bus reset that destroys
    # all GPU state. The first VFIO open must be the actual test program.
    log "$BDF on vfio-pci — skipping register reads to avoid VFIO group open"

    if [[ -L "$SYSFS/driver" ]]; then
        FINAL_DRV="$(basename "$(readlink "$SYSFS/driver")")"
        if [[ "$FINAL_DRV" == "vfio-pci" ]]; then
            ok "$BDF on vfio-pci"
        else
            warn "$BDF on $FINAL_DRV (expected vfio-pci)"
        fi
    else
        warn "$BDF has no driver"
    fi
done

# Disable bus mastering on each K80 die BEFORE unloading nouveau.
# nouveau's warm handoff leaves PMU firmware running, which generates
# periodic DMA to IOVA 0x200. With no VFIO container open, these hit
# the IOMMU as IO_PAGE_FAULT. Accumulated faults can stall the AMD-Vi,
# freezing translations for the display GPU and locking the UI.
# Our coral-driver re-enables bus mastering on VfioDevice::open().
for BDF in "${TARGETS[@]}"; do
    setpci -s "$BDF" COMMAND=0000:0004  # clear bit 2 (bus master)
    ok "$BDF bus mastering disabled (prevents idle IO_PAGE_FAULT)"
done

# Unload nouveau (no longer needed)
rmmod nouveau 2>/dev/null || true

# Unload livepatch (its job is done — falcon state preserved)
if grep -q "^livepatch_nvkm_mc_reset " /proc/modules 2>/dev/null; then
    echo 0 > /sys/kernel/livepatch/livepatch_nvkm_mc_reset/enabled 2>/dev/null || true
    sleep 2
    rmmod livepatch_nvkm_mc_reset 2>/dev/null || true
    ok "livepatch unloaded"
fi

# ── Phase 6: Skip post-handoff register reads ─────────────
# Reading registers via VFIO after bind would open the VFIO group.
# When this script exits, the group release triggers a secondary bus
# reset that destroys all GPU state. Verification is done by the test.
log "Phase 6: Skipped — register reads would trigger VFIO group open/reset"

echo ""
ok "════════════════════════════════════════════════════════"
ok "  K80 POST + warm handoff complete"
ok "  Ready for: coral-driver NvVfioComputeDevice::open_legacy()"
ok "════════════════════════════════════════════════════════"
