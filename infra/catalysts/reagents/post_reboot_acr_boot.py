#!/usr/bin/env python3
"""
SUPERSEDED — Use barracuda exp224_pmu_acr_catalyst (Rust) instead.

This v2 Python script had the correct approach (CPUCTL_ALIAS, no ENGCTL)
but has been ported to idiomatic Rust in exp224_pmu_acr_catalyst.rs for:
  - Compile-time type safety (no jelly-string register offsets)
  - Proper error propagation (no silent failures)
  - Survives partial lockups better than Python mmap
  - Integrated into barracuda diesel engine for evolution

Rust replacement:
  sudo cargo run --release --features low-level \\
      --bin exp224_pmu_acr_catalyst -- \\
      --target 0000:49:00.0 --control 0000:02:00.0

Kept as fossil record.

───────────────────────────────────────────────────────────────────

Post-Reboot ACR Boot Pipeline — v2 (CPUCTL_ALIAS + no ENGCTL)

Key changes from v1:
  - Uses CPUCTL_ALIAS (0x130) instead of CPUCTL (0x100)
  - Does NOT touch ENGCTL (0x3C0) — avoids irreversible NS transition
  - Tests PIO in HS mode 2 (no engine reset needed)
  - Uses correct falcon register map (BOOTVEC=0x104, DMACTL=0x10C)
"""

import argparse
import json
import mmap
import os
import struct
import sys
import time
from pathlib import Path

PMU = 0x10A000

# Falcon register offsets (from toadStool falcon.rs, v5)
CPUCTL      = 0x100
BOOTVEC     = 0x104
HWCFG       = 0x108
DMACTL      = 0x10C
CPUCTL_ALIAS = 0x130
IMEMC       = 0x180
IMEMD       = 0x184
IMEMT       = 0x188
DMEMC       = 0x1C0
DMEMD       = 0x1C4
SCTL        = 0x240
ENGCTL      = 0x3C0
MAILBOX0    = 0x040
MAILBOX1    = 0x044
MB2         = 0x048
PC          = 0x030
EXCI        = 0x148
IRQSTAT     = 0x008
IRQMSET     = 0x010
IRQMCLR     = 0x014

PMU_FBIF    = 0xE00   # PMU FBIF offset from PMU_BASE

# mmiotrace-extracted ACR bootloader firmware (128 words at IMEM offset 0xFE00)
ACR_BL_IMEM = [
    0x00A000D0, 0x0004FE00, 0x107EA4BD, 0x02F80100,
    0x00000089, 0x98099E98, 0x12F90A9D, 0xB6129B98,
    0x0EFD049C, 0x00BD11FE, 0x010F26F0, 0xD000B604,
    0x0004FEA0, 0x00000089, 0x98099E98, 0x12F90A9D,
    0xB6129B98, 0x0EFD049C, 0x00BD11FE, 0x0627F001,
    0x04B60410, 0x00A0D000, 0xA4BD0002, 0x99E49898,
    0x9D129809, 0x98B6120A, 0x9C0EF909, 0xFE00BD04,
    0x01010F11, 0xD004B604, 0x0004FEA0, 0x00000089,
    0x98099E98, 0x12F90A9D, 0xB6129B98, 0x0EFD049C,
    0x00BD11FE, 0x010F26F0, 0xD000B604, 0x0002A0A0,
    0x9898A4BD, 0x0999E498, 0x0A9D1298, 0x0998B612,
    0x049C0EF9, 0x11FE00BD, 0x0401010F, 0xA0D004B6,
    0x89000200, 0x98000000, 0x98099E98, 0x9D12F90A,
    0x98B6129B, 0xFD049C0E, 0xBD11FE00, 0x2EF00100,
    0x00B60410, 0x03A0D000, 0xA4BD0002, 0x99E49898,
    0x9D129809, 0x98B6120A, 0x9C0EF909, 0xFE00BD04,
    0x01010F11, 0xD004B604, 0x0002A0A0, 0x00000089,
    0x98099E98, 0x12F90A9D, 0xB6129B98, 0x0EFD049C,
    0x00BD11FE, 0x010F36F0, 0xD000B604, 0x000200A0,
    0x9898A4BD, 0x0999E498, 0x0A9D1298, 0x0998B612,
    0x049C0EF9, 0x11FE00BD, 0x0401010F, 0xA0D004B6,
    0x89000400, 0x98000000, 0x98099E98, 0x9D12F90A,
    0x98B6129B, 0xFD049C0E, 0xBD11FE00, 0x3EF00100,
    0x00B60410, 0xA0D00000, 0xBD000200, 0xA4BD00A4,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
]

# ACR DMEM descriptor from mmiotrace (21 words)
ACR_DMEM_DESC = [
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000004, 0xDD990000, 0x00000001, 0x00000000,
    0x00000600, 0x00000600, 0x00006900, 0x00000000,
    0xDD998000, 0x00000001, 0x000042F0, 0x00000034,
    0x0000002F,
]


class Bar0:
    def __init__(self, bdf):
        self.bdf = bdf
        self.fd = os.open(f'/sys/bus/pci/devices/{bdf}/resource0', os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 16 * 1024 * 1024, access=mmap.ACCESS_WRITE)

    def r32(self, off):
        return struct.unpack('<I', self.mm[off:off + 4])[0]

    def w32(self, off, val):
        self.mm[off:off + 4] = struct.pack('<I', val & 0xFFFFFFFF)

    def close(self):
        self.mm.close()
        os.close(self.fd)


def probe_pmu(bar, label):
    cpuctl   = bar.r32(PMU + CPUCTL)
    alias    = bar.r32(PMU + CPUCTL_ALIAS)
    bootvec  = bar.r32(PMU + BOOTVEC)
    hwcfg    = bar.r32(PMU + HWCFG)
    dmactl   = bar.r32(PMU + DMACTL)
    sctl     = bar.r32(PMU + SCTL)
    exci     = bar.r32(PMU + EXCI)
    mb0      = bar.r32(PMU + MAILBOX0)
    mb1      = bar.r32(PMU + MAILBOX1)
    mb2      = bar.r32(PMU + MB2)
    pc_val   = bar.r32(PMU + PC)
    engctl   = bar.r32(PMU + ENGCTL)
    pmc      = bar.r32(0x000200)
    boot0    = bar.r32(0x000000)

    sec_mode = sctl & 0x3
    sec_names = {0: 'NS', 1: 'LS', 2: 'HS', 3: '??'}
    halted = bool(cpuctl & 0x10)
    running = not halted and (cpuctl & 0x20 or cpuctl == 0x20)
    state_str = "HALT" if halted else "RUN" if cpuctl & 0x20 else "???"

    print(f"\n  {label} ({bar.bdf}):")
    print(f"    Boot0=0x{boot0:08X}  PMC=0x{pmc:08X}")
    print(f"    CPUCTL=0x{cpuctl:08X}  ALIAS=0x{alias:08X}  ({state_str})")
    print(f"    SCTL=0x{sctl:08X}  SEC_MODE={sec_mode}({sec_names.get(sec_mode, '?')})")
    print(f"    BOOTVEC=0x{bootvec:08X}  PC=0x{pc_val:08X}")
    print(f"    HWCFG=0x{hwcfg:08X}  DMACTL=0x{dmactl:08X}")
    print(f"    EXCI=0x{exci:08X}  ENGCTL=0x{engctl:08X}")
    print(f"    MB0=0x{mb0:08X}  MB1=0x{mb1:08X}  MB2=0x{mb2:08X}")

    imem_size = (hwcfg & 0x1FF) * 256
    dmem_size = ((hwcfg >> 9) & 0x1FF) * 256
    print(f"    IMEM={imem_size}B  DMEM={dmem_size}B")

    return {
        'cpuctl': cpuctl, 'alias': alias, 'sctl': sctl, 'sec_mode': sec_mode,
        'bootvec': bootvec, 'hwcfg': hwcfg, 'dmactl': dmactl,
        'exci': exci, 'mb0': mb0, 'mb1': mb1, 'mb2': mb2,
        'pc': pc_val, 'engctl': engctl, 'pmc': pmc,
        'halted': halted,
    }


def test_pio_access(bar, label):
    """Test PIO read/write to IMEM and DMEM WITHOUT touching ENGCTL."""
    print(f"\n── PIO Access Test ({label}) ──")

    # Read IMEM at offset 0 (should contain VBIOS firmware)
    bar.w32(PMU + IMEMC, 0x02000000)  # read mode + auto-inc
    imem_words = [bar.r32(PMU + IMEMD) for _ in range(8)]
    nonzero = sum(1 for w in imem_words if w != 0)
    print(f"  IMEM[0x0000] read: {nonzero}/8 non-zero, first=0x{imem_words[0]:08X}")

    # Read IMEM at offset 0xFE00
    bar.w32(PMU + IMEMC, 0x0200FE00)
    imem_fe = [bar.r32(PMU + IMEMD) for _ in range(8)]
    nonzero_fe = sum(1 for w in imem_fe if w != 0)
    print(f"  IMEM[0xFE00] read: {nonzero_fe}/8 non-zero, first=0x{imem_fe[0]:08X}")

    # Read DMEM at offset 0
    bar.w32(PMU + DMEMC, 0x02000000)
    dmem_words = [bar.r32(PMU + DMEMD) for _ in range(8)]
    nonzero_d = sum(1 for w in dmem_words if w != 0)
    print(f"  DMEM[0x0000] read: {nonzero_d}/8 non-zero, first=0x{dmem_words[0]:08X}")

    # Try writing a sentinel to DMEM (safe — non-critical area)
    SENTINEL = 0xCAFE_BEEF
    SENTINEL_OFF = 0x1F00
    bar.w32(PMU + DMEMC, 0x01000000 | SENTINEL_OFF)  # write mode
    bar.w32(PMU + DMEMD, SENTINEL)
    bar.w32(PMU + DMEMC, 0x02000000 | SENTINEL_OFF)  # read mode
    readback = bar.r32(PMU + DMEMD)
    dmem_ok = readback == SENTINEL
    print(f"  DMEM write test: wrote 0x{SENTINEL:08X} at 0x{SENTINEL_OFF:04X} → readback 0x{readback:08X} {'OK' if dmem_ok else 'FAIL'}")

    # Try writing a sentinel to IMEM (high offset, unlikely to conflict)
    IMEM_TEST_OFF = 0x0800
    bar.w32(PMU + IMEMC, 0x01000000 | IMEM_TEST_OFF)
    bar.w32(PMU + IMEMT, IMEM_TEST_OFF >> 8)
    bar.w32(PMU + IMEMD, SENTINEL)
    bar.w32(PMU + IMEMC, IMEM_TEST_OFF)  # read at offset
    readback_i = bar.r32(PMU + IMEMD)
    imem_ok = readback_i == SENTINEL
    print(f"  IMEM write test: wrote 0x{SENTINEL:08X} at 0x{IMEM_TEST_OFF:04X} → readback 0x{readback_i:08X} {'OK' if imem_ok else 'FAIL'}")

    return dmem_ok, imem_ok


def test_alias(bar, label):
    """Test CPUCTL_ALIAS in current security mode."""
    print(f"\n── CPUCTL_ALIAS Test ({label}) ──")

    alias = bar.r32(PMU + CPUCTL_ALIAS)
    print(f"  Initial ALIAS: 0x{alias:08X}")

    # Try HRESET via ALIAS
    bar.w32(PMU + CPUCTL_ALIAS, 0x10)
    time.sleep(0.01)
    alias_after = bar.r32(PMU + CPUCTL_ALIAS)
    cpuctl_after = bar.r32(PMU + CPUCTL)
    print(f"  After ALIAS←0x10: ALIAS=0x{alias_after:08X} CPUCTL=0x{cpuctl_after:08X}")

    alias_works = (alias_after != 0) or (alias != alias_after)
    cpuctl_changed = cpuctl_after & 0x10  # HRESET set
    print(f"  ALIAS responsive: {alias_works or cpuctl_changed}")

    return alias_works or cpuctl_changed


def load_firmware_pio(bar, label):
    """Load ACR BL into IMEM and descriptor into DMEM via PIO."""
    print(f"\n── Firmware PIO Load ({label}) ──")

    # IMEM: 128 words at offset 0xFE00 with nvidia tags 0x100/0x101
    bar.w32(PMU + IMEMC, 0x0100FE00)  # auto-inc-write, offset 0xFE00
    bar.w32(PMU + IMEMT, 0x00000100)  # tag for first 256-byte block
    for i, word in enumerate(ACR_BL_IMEM):
        if i == 64:
            bar.w32(PMU + IMEMT, 0x00000101)  # tag for second block
        bar.w32(PMU + IMEMD, word)

    # Verify
    bar.w32(PMU + IMEMC, 0x0200FE00)  # read mode
    errors = 0
    for i in range(128):
        got = bar.r32(PMU + IMEMD)
        if got != ACR_BL_IMEM[i]:
            if errors < 3:
                print(f"  IMEM[{i}] MISMATCH: 0x{got:08X} != 0x{ACR_BL_IMEM[i]:08X}")
            errors += 1
    print(f"  IMEM: {128 - errors}/128 correct")

    # DMEM: 21-word ACR descriptor at offset 0
    bar.w32(PMU + DMEMC, 0x01000000)  # auto-inc-write, offset 0
    for word in ACR_DMEM_DESC:
        bar.w32(PMU + DMEMD, word)

    # Verify
    bar.w32(PMU + DMEMC, 0x02000000)  # read mode
    d_errors = 0
    for i, expected in enumerate(ACR_DMEM_DESC):
        got = bar.r32(PMU + DMEMD)
        if got != expected:
            if d_errors < 3:
                print(f"  DMEM[{i}] MISMATCH: 0x{got:08X} != 0x{expected:08X}")
            d_errors += 1
    print(f"  DMEM: {len(ACR_DMEM_DESC) - d_errors}/{len(ACR_DMEM_DESC)} correct")

    return errors == 0 and d_errors == 0


def try_boot_alias(bar, label, bootvec_val=0xFE00):
    """Try booting via CPUCTL_ALIAS: HRESET → BOOTVEC → IINVAL → STARTCPU."""
    print(f"\n── Boot via CPUCTL_ALIAS ({label}) ──")

    bar.w32(PMU + CPUCTL_ALIAS, 0x10)  # HRESET
    time.sleep(0.01)
    print(f"  HRESET: ALIAS=0x{bar.r32(PMU + CPUCTL_ALIAS):08X} CPUCTL=0x{bar.r32(PMU + CPUCTL):08X}")

    bar.w32(PMU + BOOTVEC, bootvec_val)
    bv = bar.r32(PMU + BOOTVEC)
    print(f"  BOOTVEC←0x{bootvec_val:X}: readback=0x{bv:08X}")

    bar.w32(PMU + MAILBOX0, 0)
    bar.w32(PMU + MAILBOX1, 0)

    bar.w32(PMU + CPUCTL_ALIAS, 0x01)  # IINVAL
    time.sleep(0.001)

    bar.w32(PMU + CPUCTL_ALIAS, 0x02)  # STARTCPU

    start = time.monotonic()
    for delay in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        time.sleep(delay)
        elapsed = time.monotonic() - start
        alias   = bar.r32(PMU + CPUCTL_ALIAS)
        cpuctl  = bar.r32(PMU + CPUCTL)
        sctl    = bar.r32(PMU + SCTL)
        mb0     = bar.r32(PMU + MAILBOX0)
        mb1     = bar.r32(PMU + MAILBOX1)
        pc_val  = bar.r32(PMU + PC)
        exci    = bar.r32(PMU + EXCI)
        intr    = bar.r32(PMU + IRQSTAT)

        sec = sctl & 0x3
        running = bool(cpuctl & 0x20) or bool(alias & 0x20)
        halted = bool(cpuctl & 0x10) or bool(alias & 0x10)
        state = "RUN" if running else "HALT" if halted else "???"

        print(f"  t={elapsed:.3f}s: {state} ALIAS=0x{alias:08X} CPUCTL=0x{cpuctl:08X} "
              f"SEC={sec} PC=0x{pc_val:08X} MB0=0x{mb0:08X} INTR=0x{intr:08X}")

        if running:
            print("  *** CPU IS RUNNING! ***")
            return True
        if mb0 != 0:
            print("  *** MAILBOX RESPONSE! ***")
            return True

    return False


def try_boot_cpuctl(bar, label, bootvec_val=0xFE00):
    """Fallback: try booting via CPUCTL directly (not alias)."""
    print(f"\n── Boot via CPUCTL direct ({label}) ──")

    bar.w32(PMU + CPUCTL, 0x10)  # HRESET
    time.sleep(0.01)
    bar.w32(PMU + BOOTVEC, bootvec_val)
    bar.w32(PMU + MAILBOX0, 0)

    bar.w32(PMU + CPUCTL, 0x01)  # IINVAL
    time.sleep(0.001)
    bar.w32(PMU + CPUCTL, 0x02)  # STARTCPU

    for delay in [0.01, 0.1, 1.0]:
        time.sleep(delay)
        cpuctl = bar.r32(PMU + CPUCTL)
        pc_val = bar.r32(PMU + PC)
        mb0    = bar.r32(PMU + MAILBOX0)
        running = bool(cpuctl & 0x20)
        state = "RUN" if running else "HALT" if cpuctl & 0x10 else "???"
        print(f"  t={delay}s: {state} CPUCTL=0x{cpuctl:08X} PC=0x{pc_val:08X} MB0=0x{mb0:08X}")
        if running or mb0 != 0:
            return True

    # Try nvidia's CPUCTL=0x12 trigger
    print("  Trying CPUCTL=0x12 (HS ROM trigger)...")
    bar.w32(PMU + CPUCTL, 0x10)
    time.sleep(0.01)
    bar.w32(PMU + CPUCTL, 0x12)
    for delay in [0.01, 0.1, 1.0]:
        time.sleep(delay)
        cpuctl = bar.r32(PMU + CPUCTL)
        sctl   = bar.r32(PMU + SCTL)
        mb0    = bar.r32(PMU + MAILBOX0)
        print(f"  t={delay}s: CPUCTL=0x{cpuctl:08X} SCTL=0x{sctl:08X} MB0=0x{mb0:08X}")

    return False


def run_pipeline(target_bdf, control_bdf, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    target = Bar0(target_bdf)
    control = Bar0(control_bdf)

    print("=" * 60)
    print("  ACR Sovereign Boot Pipeline v2 (CPUCTL_ALIAS)")
    print("=" * 60)

    # ── Phase 1: Probe ──
    print("\n── Phase 1: Probe ──")
    t_state = probe_pmu(target, "TARGET")
    c_state = probe_pmu(control, "CONTROL")

    # Verify fresh state
    if t_state['sec_mode'] != 2:
        print(f"\n  WARNING: Target SEC_MODE={t_state['sec_mode']}, expected 2 (HS)")
        print("  GPU may not be in fresh VBIOS state!")
    if c_state['sec_mode'] != 2:
        print(f"\n  WARNING: Control SEC_MODE={c_state['sec_mode']}, expected 2 (HS)")

    # ── Phase 2: PIO access test on TARGET ──
    dmem_ok, imem_ok = test_pio_access(target, "TARGET")

    if not dmem_ok and not imem_ok:
        print("\n  PIO BLOCKED in HS mode — trying secure flag...")
        # Try with secure flag (BIT28)
        bar = target
        bar.w32(PMU + IMEMC, 0x11000800)  # secure + auto-inc-write + offset 0x800
        bar.w32(PMU + IMEMT, 0x08)
        bar.w32(PMU + IMEMD, 0xCAFEBEEF)
        bar.w32(PMU + IMEMC, 0x12000800)  # secure + auto-inc-read
        readback = bar.r32(PMU + IMEMD)
        print(f"  Secure PIO: 0x{readback:08X} {'OK' if readback == 0xCAFEBEEF else 'FAIL'}")

    # ── Phase 3: CPUCTL_ALIAS test ──
    alias_ok = test_alias(target, "TARGET")

    # ── Phase 4: Load firmware ──
    fw_ok = load_firmware_pio(target, "TARGET")

    if not fw_ok:
        print("\n  FIRMWARE LOAD FAILED — aborting")
        target.close()
        control.close()
        return

    # ── Phase 5: Boot attempts ──
    # Try ALIAS path first (preferred)
    if alias_ok:
        success = try_boot_alias(target, "TARGET", bootvec_val=0xFE00)
    else:
        print("\n  CPUCTL_ALIAS not responsive, trying direct CPUCTL...")
        success = try_boot_cpuctl(target, "TARGET", bootvec_val=0xFE00)

    if not success:
        # Try BOOTVEC=0 with open-source firmware path
        print("\n  Primary boot failed, trying BOOTVEC=0 path...")
        success = try_boot_alias(target, "TARGET-BV0", bootvec_val=0)
        if not success:
            success = try_boot_cpuctl(target, "TARGET-BV0", bootvec_val=0)

    # ── Phase 6: Final state ──
    print("\n── Phase 6: Final State ──")
    t_final = probe_pmu(target, "TARGET-FINAL")
    c_final = probe_pmu(control, "CONTROL-FINAL")

    # Save results
    results = {
        'target_bdf': target_bdf,
        'control_bdf': control_bdf,
        'success': success,
        'target_initial': {k: f'0x{v:08X}' if isinstance(v, int) else v for k, v in t_state.items()},
        'target_final': {k: f'0x{v:08X}' if isinstance(v, int) else v for k, v in t_final.items()},
        'control_initial': {k: f'0x{v:08X}' if isinstance(v, int) else v for k, v in c_state.items()},
        'control_final': {k: f'0x{v:08X}' if isinstance(v, int) else v for k, v in c_final.items()},
    }
    with open(f'{out_dir}/pipeline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_dir}/pipeline_results.json")

    target.close()
    control.close()

    if success:
        print("\n  *** ACR SOVEREIGN BOOT SUCCEEDED ***")
    else:
        print("\n  Boot did not succeed — see state dumps above for analysis")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-reboot ACR boot pipeline v2')
    parser.add_argument('--target', default='0000:49:00.0')
    parser.add_argument('--control', default='0000:02:00.0')
    parser.add_argument('--out', default='/tmp/acr_catalyst_v2')
    args = parser.parse_args()

    run_pipeline(args.target, args.control, args.out)
