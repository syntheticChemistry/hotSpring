#!/usr/bin/env python3
"""
exp192: Post-reboot warm-state Boot Falcon (NVDEC0) initialization
==================================================================
After system reboot:
- UEFI has initialized GPU fully (memory trained, security context live)
- GPU is under vfio-pci via kernel cmdline vfio-pci.ids=10de:1d81
- We access via BAR0 resource0

Key fixes from exp191 analysis:
- IRQMASK is at BF+0x018 (READ), set via IRQMSET (BF+0x010)
- BF+0x014 is IRQMCLR (WRITE to CLEAR bits) - was wrongly used before
- UNK10C (BF+0x10c) is writable in warm state = write 0 as per mmiotrace

Flow:
1. Verify warm state (PMU should be running, IRQMASK should be writable)
2. Soft reset NVDEC0 / Boot Falcon
3. Load NS+HS hybrid IMEM from scrubber.bin (per mmiotrace)
4. Set IRQMSET = 0x0001ffff, BOOTVEC = 0, UNK10C = 0
5. Write MBOX0 sentinel = 0
6. STARTCPU
7. Poll MBOX0 for success code (~5 second timeout)
"""
import os, mmap, struct, time, sys

BAR0_RES = '/sys/bus/pci/devices/0000:02:00.0/resource0'
SCRUBBER = '/lib/firmware/nvidia/gv100/nvdec/scrubber.bin'

fd = os.open(BAR0_RES, os.O_RDWR|os.O_SYNC)
m  = mmap.mmap(fd, 16*1024*1024, mmap.MAP_SHARED, mmap.PROT_READ|mmap.PROT_WRITE)

def r32(o):      m.seek(o); return struct.unpack('<I', m.read(4))[0]
def w32(o, v):   m.seek(o); m.write(struct.pack('<I', v & 0xFFFFFFFF))
def poll(cond, timeout=5.0, interval=0.01):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond(): return True
        time.sleep(interval)
    return False

NVDEC0 = 0x084000  # Boot Falcon / NVDEC0 base

def bf_r(off): return r32(NVDEC0 + off)
def bf_w(off, v): w32(NVDEC0 + off, v)

print('='*60)
print('exp192: Post-reboot warm-state BF/NVDEC0 init')
print('='*60)

# ─────────────────────────────────────────────────────────────
# Step 0: Verify warm state
# ─────────────────────────────────────────────────────────────
print('\n[0] Warm state check:')
pmc_en   = r32(0x200)
pmc_b0   = r32(0x000)
pmu_cpu  = r32(0x10a100)
priv_st  = r32(0x12006c)
wpr_cd4  = r32(0x100cd4)
bf_cpu   = bf_r(0x100)
bf_mbox0 = bf_r(0x040)
bf_irqm  = bf_r(0x018)

print(f'  PMC_BOOT_0  = 0x{pmc_b0:08x}  (expect 0x140000a1)')
print(f'  PMC_ENABLE  = 0x{pmc_en:08x}')
print(f'  PMU CPUCTL  = 0x{pmu_cpu:08x}')
print(f'  PRIV_RING   = 0x{priv_st:08x}  (0x00 = healthy)')
print(f'  WPR_CD4     = 0x{wpr_cd4:08x}')
print(f'  BF CPUCTL   = 0x{bf_cpu:08x}')
print(f'  BF MBOX0    = 0x{bf_mbox0:08x}')
print(f'  BF IRQMASK  = 0x{bf_irqm:08x}  (via BF+0x018)')

# Quick writability test for IRQMSET
bf_w(0x010, 0x0001ffff)  # IRQMSET: set all 17 bits
new_mask = bf_r(0x018)
bf_w(0x014, 0x0001ffff)  # IRQMCLR: clear back
print(f'  IRQMSET test→ 0x{new_mask:08x}  {"WRITABLE ✓" if new_mask != 0 else "REJECTED ✗"}')

if pmc_b0 != 0x140000a1:
    print('  ⚠  Wrong chip ID — is this the right device?')
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# Step 1: Load scrubber.bin
# ─────────────────────────────────────────────────────────────
print('\n[1] Loading scrubber.bin firmware...')
with open(SCRUBBER, 'rb') as f:
    fw_data = f.read()

# Parse header
hdr_sig    = struct.unpack_from('<I', fw_data, 0x00)[0]
fw_size    = struct.unpack_from('<I', fw_data, 0x08)[0]
code_off   = struct.unpack_from('<I', fw_data, 0x10)[0]  # byte offset to code
code_sz    = struct.unpack_from('<I', fw_data, 0x14)[0]  # code size in bytes
print(f'  sig=0x{hdr_sig:08x}  fw_size={fw_size}  code_off=0x{code_off:x}  code_sz=0x{code_sz:x}')

code_words = code_sz // 4
code = struct.unpack_from(f'<{code_words}I', fw_data, code_off)
print(f'  Code: {code_words} words ({code_sz} bytes)')

# IMEM block size: 64 words per block (256 bytes), confirmed by trace
# NS segment: block 0 = words 0-63 (first 64 words)
# HS segment: blocks 1-N = words 64+ with IMEMC secure bit
BLOCK_W   = 64   # words per IMEM block

NS_WORDS  = 64   # first 64 words = NS (block 0)
HS_START  = NS_WORDS
HS_WORDS  = code_words - NS_WORDS

print(f'  NS: words 0-{NS_WORDS-1} ({NS_WORDS} words)')
print(f'  HS: words {NS_WORDS}-{code_words-1} ({HS_WORDS} words)')

# ─────────────────────────────────────────────────────────────
# Step 2: Soft reset NVDEC0 / Boot Falcon
# ─────────────────────────────────────────────────────────────
print('\n[2] Soft reset NVDEC0...')
bf_w(0x3c0, 0x1)   # assert reset
time.sleep(0.005)
bf_w(0x3c0, 0x0)   # deassert

# Poll UNK10C for reset to complete (should become 1 in warm state)
ok = poll(lambda: bf_r(0x10c) != 0, timeout=0.5)
print(f'  UNK10C after reset = 0x{bf_r(0x10c):08x}  ({"ready" if ok else "stuck at 0"})')

# Also clear any pending IRQs
bf_w(0x008, 0xffffffff)  # IRQSCLR: clear all pending
time.sleep(0.002)

# ─────────────────────────────────────────────────────────────
# Step 3: Load NS IMEM (block 0, words 0-63)
# ─────────────────────────────────────────────────────────────
print('\n[3] Loading NS IMEM (block 0, 64 words)...')
# IMEMC: [bit 28]=HS, [bit 24]=NS/mark, [bits 15:4]=word_offset, [bit 0]=AINCR
# For NS block 0: imemc = 0x00000001 (AINCR=1, block=0, NS)
bf_w(0x180, 0x00000001)  # IMEMC: block 0, NS, auto-increment
for i, w in enumerate(code[:NS_WORDS]):
    bf_w(0x184, w)  # IMEMD: write word

print(f'  Loaded {NS_WORDS} NS words. IMEMC = 0x{bf_r(0x180):08x}')

# ─────────────────────────────────────────────────────────────
# Step 4: Load HS IMEM (blocks 1+, words 64-959)
# ─────────────────────────────────────────────────────────────
print('\n[4] Loading HS IMEM (blocks 1+, 896 words)...')
# Rewind to block 1 with HS bit set
# IMEMC for HS block 1: bit 28=1 (HS), word_offset = 64 << ? 
# Block 1 starts at word 64: word_offset in IMEMC = (64 * 4) = 256 = 0x100
# IMEMC format: [31:28]=type, [17:4]=byte_offset_in_imem, [0]=AINCR
# byte_offset = 64 * 4 = 256 = 0x100 → shifted to bits [17:4]: 0x100 << 4 = 0x1000? No...
# Actually: IMEMC[17:4] = word address (not byte). So word 64 = 0x40 → 0x40 << 4 = 0x400
# imemc = 0x10000000 (HS) | (0x40 << 4) (word=64) | 0x01 (AINCR)
# = 0x10000401
hs_imemc = 0x10000000 | (HS_START << 4) | 0x01
print(f'  HS IMEMC = 0x{hs_imemc:08x}')
bf_w(0x180, hs_imemc)
for i, w in enumerate(code[NS_WORDS:]):
    bf_w(0x184, w)
    if i < 2:
        pass  # we'll show tag progress
    # Write IMETT after each HS block (every BLOCK_W words)
    if (i + 1) % BLOCK_W == 0:
        block_idx = (i + 1) // BLOCK_W
        bf_w(0x188, block_idx - 1)  # IMETT: tag = block_idx - 1

print(f'  Loaded {HS_WORDS} HS words. IMEMC = 0x{bf_r(0x180):08x}  IMETT = 0x{bf_r(0x188):08x}')

# ─────────────────────────────────────────────────────────────
# Step 5: Set up DMEM descriptor (from mmiotrace)
# ─────────────────────────────────────────────────────────────
print('\n[5] Setting DMEM descriptor...')
# From mmiotrace: nouveau writes these DMEM words at offset 0:
# word 0: magic/version
# word 1: code type flags
# etc. — minimal descriptor for scrubber boot
# The scrubber uses a flcn_bl_dmem_desc_v1 (21 words):
# [0]=magic, [1]=dmem_load_off, [2]=imem_load_off, [3]=imem_virt_base,
# [4]=imem_size, [5]=imem_entry_point, ... [20]=reserved
DMEM_DESC = [0] * 21  # zero-fill; scrubber may not need explicit DMEM
bf_w(0x1c0, 0x01000001)  # DMEMC: offset 0, auto-increment
for w in DMEM_DESC:
    bf_w(0x1c4, w)  # DMEMD
print(f'  DMEM filled. DMEMC = 0x{bf_r(0x1c0):08x}')

# ─────────────────────────────────────────────────────────────
# Step 6: Final setup + STARTCPU
# ─────────────────────────────────────────────────────────────
print('\n[6] Final setup and STARTCPU...')
bf_w(0x040, 0x00000000)  # MBOX0: clear sentinel before start
bf_w(0x044, 0x00000000)  # MBOX1: clear
bf_w(0x010, 0x0001ffff)  # IRQMSET: enable all 17 interrupt sources
bf_w(0x104, 0x00000000)  # BOOTVEC: start at word 0
bf_w(0x10c, 0x00000000)  # UNK10C: write 0 as per mmiotrace

print(f'  IRQMASK = 0x{bf_r(0x018):08x}  BOOTVEC = 0x{bf_r(0x104):08x}  MBOX0 = 0x{bf_r(0x040):08x}')

# STARTCPU: write bit 1 of CPUCTL
bf_w(0x100, 0x00000002)  # STARTCPU

t_start = time.time()
print(f'  STARTCPU issued. Polling...')
time.sleep(0.01)
print(f'  CPUCTL  = 0x{bf_r(0x100):08x}  (0x10=HALTED, 0x02=RUNNING, 0x12=started+halted)')
print(f'  MBOX0   = 0x{bf_r(0x040):08x}  IRQSTAT = 0x{bf_r(0x00c):08x}')

# ─────────────────────────────────────────────────────────────
# Step 7: Poll MBOX0 for completion
# ─────────────────────────────────────────────────────────────
print('\n[7] Polling MBOX0 for completion (timeout=30s)...')
last_report = 0
success = False
t0 = time.time()
while time.time() - t0 < 30:
    mbox0 = bf_r(0x040)
    cpuctl = bf_r(0x100)
    irqstat = bf_r(0x00c)
    now = time.time() - t0
    if now - last_report >= 2.0:
        print(f'  t={now:.1f}s  CPUCTL=0x{cpuctl:08x}  MBOX0=0x{mbox0:08x}  IRQ=0x{irqstat:08x}')
        last_report = now
    if mbox0 != 0x00000000:
        success = True
        print(f'  ★ MBOX0 changed! t={now:.2f}s  MBOX0=0x{mbox0:08x}  CPUCTL=0x{cpuctl:08x}')
        break
    if cpuctl == 0x10:  # pure HALTED, never ran
        print(f'  ✗ HALTED without starting at t={now:.2f}s')
        break
    time.sleep(0.05)

if not success:
    print('\n  TIMEOUT — Boot Falcon did not complete')

# ─────────────────────────────────────────────────────────────
# Step 8: Final state dump
# ─────────────────────────────────────────────────────────────
print('\n[8] Final state:')
print(f'  BF CPUCTL   = 0x{bf_r(0x100):08x}')
print(f'  BF MBOX0    = 0x{bf_r(0x040):08x}')
print(f'  BF MBOX1    = 0x{bf_r(0x044):08x}')
print(f'  BF IRQSTAT  = 0x{bf_r(0x00c):08x}')
print(f'  BF IRQMASK  = 0x{bf_r(0x018):08x}')
print(f'  WPR_CD4     = 0x{r32(0x100cd4):08x}  (WPR state)')
print(f'  PMC_ENABLE  = 0x{r32(0x200):08x}')
print(f'  PRIV_RING   = 0x{r32(0x12006c):08x}')
# VRAM test via PRAMIN
w32(0x001700, 0)
w32(0x700000, 0x55aa55aa)
time.sleep(0.002)
vram_rb = r32(0x700000)
print(f'  VRAM test   = 0x{vram_rb:08x}  {"✓ ACCESSIBLE" if vram_rb==0x55aa55aa else "✗ inaccessible"}')

m.close()
os.close(fd)
