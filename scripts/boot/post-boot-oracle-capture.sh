#!/bin/bash
# post-boot-oracle-capture.sh — After reboot, validate bindings and capture oracle state
#
# Run AFTER reboot with the dual Titan V boot config installed.
# Captures:
#   1. Oracle (nouveau-warm) BAR0 → binary dump for register analysis
#   2. Validates VFIO target binding and IOMMU group permissions
#   3. Optionally enables mmiotrace for nouveau init sequence capture
#
# Usage: sudo ./scripts/boot/post-boot-oracle-capture.sh [--mmiotrace]

set -euo pipefail

ORACLE="0000:03:00.0"
VFIO_TARGET="0000:4a:00.0"
OUTDIR="$(cd "$(dirname "$0")/.." && pwd)/data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DO_MMIOTRACE="${1:-}"

mkdir -p "$OUTDIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║ Post-Boot Oracle Capture                                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Validate driver bindings
echo ">>> Step 1: Validating driver bindings..."
ORACLE_DRV=$(basename "$(readlink "/sys/bus/pci/devices/$ORACLE/driver" 2>/dev/null)" 2>/dev/null || echo "none")
VFIO_DRV=$(basename "$(readlink "/sys/bus/pci/devices/$VFIO_TARGET/driver" 2>/dev/null)" 2>/dev/null || echo "none")
DISPLAY_DRV=$(basename "$(readlink "/sys/bus/pci/devices/0000:21:00.0/driver" 2>/dev/null)" 2>/dev/null || echo "none")

echo "  Oracle  ($ORACLE):     $ORACLE_DRV"
echo "  VFIO    ($VFIO_TARGET): $VFIO_DRV"
echo "  Display (21:00.0):        $DISPLAY_DRV"

BINDING_OK=true
if [ "$ORACLE_DRV" != "nouveau" ]; then
    echo "  ⚠ Oracle not on nouveau! Expected: nouveau, got: $ORACLE_DRV"
    BINDING_OK=false
fi
if [ "$VFIO_DRV" != "vfio-pci" ]; then
    echo "  ⚠ VFIO target not on vfio-pci! Expected: vfio-pci, got: $VFIO_DRV"
    BINDING_OK=false
fi

if [ "$BINDING_OK" = false ]; then
    echo ""
    echo "Driver bindings incorrect. Check boot config or use: coralctl swap <BDF> vfio"
    echo "Continuing anyway for diagnostic value..."
fi

# Step 2: Oracle warm state verification
echo ""
echo ">>> Step 2: Verifying oracle warm state..."
WARM_RESULT=$(python3 -c "
import mmap, os, struct, sys
try:
    fd = os.open('/sys/bus/pci/devices/${ORACLE}/resource0', os.O_RDONLY)
    m = mmap.mmap(fd, 0x01000000, mmap.MAP_SHARED, mmap.PROT_READ)
    boot0 = struct.unpack_from('<I', m, 0x000)[0]
    pmc = struct.unpack_from('<I', m, 0x200)[0]
    pfifo = struct.unpack_from('<I', m, 0x2200)[0]
    pbdma = struct.unpack_from('<I', m, 0x2004)[0]
    pclock = struct.unpack_from('<I', m, 0x137000)[0]
    nvpll = struct.unpack_from('<I', m, 0x137050)[0]
    mempll = struct.unpack_from('<I', m, 0x137100)[0]
    pramin = struct.unpack_from('<I', m, 0x700000)[0]
    m.close(); os.close(fd)
    warm = pmc != 0x40000020 and pfifo != 0xbad0da00
    print(f'BOOT0={boot0:#010x}')
    print(f'PMC_ENABLE={pmc:#010x}')
    print(f'PFIFO_ENABLE={pfifo:#010x}')
    print(f'PBDMA_MAP={pbdma:#010x}')
    print(f'PCLOCK_CTL={pclock:#010x}')
    print(f'NVPLL_CTL={nvpll:#010x}')
    print(f'MEMPLL_CTL={mempll:#010x}')
    print(f'PRAMIN[0]={pramin:#010x}')
    print(f'WARM={warm}')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" 2>&1)
echo "$WARM_RESULT"

# Step 3: Full BAR0 binary dump (16MB)
echo ""
echo ">>> Step 3: Capturing oracle BAR0 binary dump..."
BAR0_FILE="$OUTDIR/oracle_bar0_${TIMESTAMP}.bin"
if dd if="/sys/bus/pci/devices/${ORACLE}/resource0" of="$BAR0_FILE" bs=4096 count=4096 status=progress 2>&1; then
    SIZE=$(stat -c%s "$BAR0_FILE")
    echo "  Saved: $BAR0_FILE ($SIZE bytes)"
else
    echo "  ⚠ BAR0 dump failed"
fi

# Step 4: Root PLL register dump (0x136000-0x137FFF — the always-on domain)
echo ""
echo ">>> Step 4: Dumping root PLL registers (0x136000-0x137FFF)..."
ROOT_PLL_FILE="$OUTDIR/oracle_root_plls_${TIMESTAMP}.txt"
python3 -c "
import mmap, os, struct
fd = os.open('/sys/bus/pci/devices/${ORACLE}/resource0', os.O_RDONLY)
m = mmap.mmap(fd, 0x01000000, mmap.MAP_SHARED, mmap.PROT_READ)
with open('${ROOT_PLL_FILE}', 'w') as f:
    f.write('# Oracle root PLL dump: ${ORACLE} at ${TIMESTAMP}\n')
    f.write('# Format: offset value\n')
    live = 0
    dead = 0
    for off in range(0x136000, 0x138000, 4):
        val = struct.unpack_from('<I', m, off)[0]
        if val != 0 and (val >> 16) != 0xBADF and (val >> 16) != 0xBAD0:
            f.write(f'0x{off:06x} 0x{val:08x}\n')
            live += 1
        else:
            dead += 1
    f.write(f'# live={live} dead={dead}\n')
m.close(); os.close(fd)
print(f'  Root PLLs: {live} live, {dead} dead registers')
print(f'  Saved: ${ROOT_PLL_FILE}')
" 2>&1

# Step 5: Critical domain register dump (for oracle diff)
echo ""
echo ">>> Step 5: Dumping critical domain registers..."
DOMAINS_FILE="$OUTDIR/oracle_domains_${TIMESTAMP}.txt"
python3 -c "
import mmap, os, struct

RANGES = [
    ('PMC',        0x000000, 0x001000),
    ('PBUS',       0x001000, 0x002000),
    ('PFIFO',      0x002000, 0x004000),
    ('PTOP',       0x022000, 0x023000),
    ('PFB',        0x100000, 0x102000),
    ('FBPA0',      0x9A0000, 0x9A1000),
    ('FBPA1',      0x9A4000, 0x9A5000),
    ('FBPA_BC',    0x9A8000, 0x9A9000),
    ('LTC0',       0x17E000, 0x17F000),
    ('LTC1',       0x180000, 0x181000),
    ('PCLOCK',     0x130000, 0x138000),
    ('PMU',        0x10A000, 0x10B000),
    ('PFB_NISO',   0x100C00, 0x100E00),
    ('PMEM',       0x1FA000, 0x1FB000),
    ('FUSE',       0x021000, 0x022000),
    ('FBHUB',      0x100800, 0x100A00),
    ('PRI_MASTER', 0x122000, 0x123000),
    ('PRAMIN',     0x700000, 0x701000),
]

fd = os.open('/sys/bus/pci/devices/${ORACLE}/resource0', os.O_RDONLY)
m = mmap.mmap(fd, 0x01000000, mmap.MAP_SHARED, mmap.PROT_READ)

with open('${DOMAINS_FILE}', 'w') as f:
    f.write('# Oracle domain register dump: ${ORACLE} at ${TIMESTAMP}\n')
    total_live = 0
    for name, start, end in RANGES:
        live = 0
        for off in range(start, end, 4):
            val = struct.unpack_from('<I', m, off)[0]
            if val != 0 and (val >> 16) != 0xBADF and (val >> 16) != 0xBAD0 and val != 0xFFFFFFFF:
                f.write(f'{name} 0x{off:06x} 0x{val:08x}\n')
                live += 1
        total_live += live
        print(f'  {name}: {live} live registers')
    f.write(f'# total_live={total_live}\n')

m.close(); os.close(fd)
print(f'  Total: {total_live} live registers')
print(f'  Saved: ${DOMAINS_FILE}')
" 2>&1

# Step 6: VFIO target cold state (for differential comparison)
echo ""
echo ">>> Step 6: Reading VFIO target cold state..."
VFIO_IOMMU_GROUP=$(basename "$(readlink "/sys/bus/pci/devices/$VFIO_TARGET/iommu_group")" 2>/dev/null || echo "?")
chmod 666 "/dev/vfio/$VFIO_IOMMU_GROUP" 2>/dev/null || true
echo "  VFIO group $VFIO_IOMMU_GROUP ready for GlowPlug"

VFIO_PWR=$(cat "/sys/bus/pci/devices/$VFIO_TARGET/power_state" 2>/dev/null || echo "?")
echo "  VFIO target power state: $VFIO_PWR"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║ Capture Complete                                        ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ Oracle BAR0:    $BAR0_FILE"
echo "║ Root PLLs:      $ROOT_PLL_FILE"
echo "║ Domain regs:    $DOMAINS_FILE"
echo "║ VFIO target:    group $VFIO_IOMMU_GROUP ($VFIO_PWR)"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ Next steps:"
echo "║   cargo test --test hw_nv_vfio --features vfio \\"
echo "║     -- --ignored vfio_oracle_differential --nocapture"
echo "╚══════════════════════════════════════════════════════════╝"
