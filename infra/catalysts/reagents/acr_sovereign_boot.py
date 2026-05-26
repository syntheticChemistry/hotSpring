#!/usr/bin/env python3
"""
SUPERSEDED — Use barracuda exp224_pmu_acr_catalyst (Rust) instead.

This v1 script contains the FLAWED HS-unlock (ENGCTL) approach that
irreversibly transitions PMU from HS to NS mode, permanently killing
CPU execution. See Exp 223 for full post-mortem.

Kept as fossil record; DO NOT EXECUTE.

Rust replacement:
  sudo cargo run --release --features low-level \\
      --bin exp224_pmu_acr_catalyst -- \\
      --target 0000:49:00.0 --control 0000:02:00.0

───────────────────────────────────────────────────────────────────

ACR Sovereign Boot Catalyst — Minimal Reusable GPU Initialization (v1 BROKEN)

Performs the full PMU ACR secure boot chain on a GV100 (Titan V) GPU
through BAR0 MMIO register writes, using firmware data extracted from
the nvidia-535 mmiotrace recipe.

This catalyst is frozen and reusable: given a GPU in post-VBIOS state
(PMU running in HS mode 2), it performs:

  1. PRAMIN-based firmware staging in VRAM
  2. HS unlock (HS2 → HS0) for IMEM/DMEM access    ← DESTRUCTIVE
  3. ACR bootloader load into PMU IMEM
  4. ACR descriptor load into PMU DMEM
  5. HS ROM trigger (CPUCTL=0x12) for authenticated boot
  6. Post-boot verification

Based on analysis of:
  - mmiotrace recipe: nv535_recipe.json (steps 32591-32833)
  - Experiments 206, 211, 217, 219, 222
  - nova-core falcon v5 register documentation
"""

import argparse
import json
import mmap
import os
import struct
import sys
import time
from pathlib import Path

PMU_BASE = 0x10A000
FECS_BASE = 0x409000
GPCCS_BASE = 0x41A000
PRAMIN_BASE = 0x700000
PRAMIN_SIZE = 0x100000  # 1MB PRAMIN window
BAR0_WINDOW = 0x001700


class GpuBar0:
    """BAR0 MMIO accessor for GPU register reads/writes."""

    def __init__(self, bdf: str):
        self.bdf = bdf
        self.path = f"/sys/bus/pci/devices/{bdf}/resource0"
        self.fd = os.open(self.path, os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 16 * 1024 * 1024, access=mmap.ACCESS_WRITE)

    def r32(self, off: int) -> int:
        return struct.unpack("<I", self.mm[off : off + 4])[0]

    def w32(self, off: int, val: int):
        self.mm[off : off + 4] = struct.pack("<I", val & 0xFFFFFFFF)

    def close(self):
        self.mm.close()
        os.close(self.fd)


class FalconState:
    """Snapshot of a falcon engine's control state."""

    def __init__(self, bar: GpuBar0, base: int, name: str):
        self.name = name
        self.cpuctl = bar.r32(base + 0x100)
        self.sctl = bar.r32(base + 0x240)
        self.hwcfg2 = bar.r32(base + 0x148)
        self.bootvec = bar.r32(base + 0x120)
        self.mb0 = bar.r32(base + 0x040)
        self.mb1 = bar.r32(base + 0x044)
        self.mb2 = bar.r32(base + 0x048)
        self.intr = bar.r32(base + 0x008)

    @property
    def is_running(self) -> bool:
        return bool(self.cpuctl & 0x20)

    @property
    def is_halted(self) -> bool:
        return bool(self.cpuctl & 0x10)

    @property
    def hs_mode(self) -> int:
        return self.sctl & 0xF

    @property
    def cpu_operational(self) -> bool:
        return bool(self.hwcfg2 & 0x20000000)

    def __str__(self):
        status = "RUN" if self.is_running else "HALT" if self.is_halted else "IDLE"
        return (
            f"{self.name}: CPUCTL=0x{self.cpuctl:08X}({status}) "
            f"SCTL=HS{self.hs_mode} HWCFG2=0x{self.hwcfg2:08X}(bit29={int(self.cpu_operational)}) "
            f"MB0=0x{self.mb0:08X} INTR=0x{self.intr:08X}"
        )


# ── Extracted from nv535_recipe.json steps 32636-32764 ──
# ACR bootloader: 128 words (512 bytes) loaded at IMEM offset 0xFE00
ACR_BL_IMEM_OFFSET = 0xFE00
ACR_BL_IMEM_WORDS = [
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
    # ── second 256-byte block (tag 0x101) ──
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

# ── Extracted from nv535_recipe.json steps 32613-32633 ──
# ACR descriptor: 21 words (84 bytes) loaded at DMEM offset 0
ACR_DMEM_DESCRIPTOR = [
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000004,  # flags/mode
    0xDD990000,  # VRAM addr low: ACR ucode image
    0x00000001,  # VRAM addr high (64-bit: 0x1_DD990000)
    0x00000000,
    0x00000600,  # ACR ucode size: 1536 bytes
    0x00000600,  # ACR ucode aligned size
    0x00006900,  # Total payload size: 26880 bytes
    0x00000000,
    0xDD998000,  # VRAM addr low: firmware load table
    0x00000001,  # VRAM addr high (64-bit: 0x1_DD998000)
    0x000042F0,  # Load table size: 17136 bytes
    0x00000034,  # Load table entries: 52
    0x0000002F,  # Falcon count: 47
]

# VRAM addresses referenced by ACR descriptor (need firmware staged here)
ACR_UCODE_VRAM_ADDR = 0x1_DD990000
ACR_UCODE_SIZE = 0x600  # 1536 bytes
ACR_PAYLOAD_SIZE = 0x6900  # 26880 bytes
ACR_LOADTABLE_VRAM_ADDR = 0x1_DD998000
ACR_LOADTABLE_SIZE = 0x42F0  # 17136 bytes

# ── Register setup from nv535_recipe.json steps 32596-32611 ──
PMU_REGISTER_SETUP = [
    (0x040, 0x00000000),   # MB0 clear
    (0x134, 0x00020008),   # DMACTL
    # FBIF registers (absolute offsets)
]

PMU_FBIF_SETUP = [
    (0x10AA74, 0x00000000),
    (0x10AE74, 0x00000000),
    (0x10AA70, 0x00000045),
]

PMU_INTR_SETUP = [
    (0x014, 0x0000FFFF),   # INTR_MASK
    (0x01C, 0xFF81FF52),   # NTSTATUS
    (0x010, 0x000000F3),   # INTR_EN
    (0x014, 0x0000FFFF),   # INTR_MASK (repeated)
]

PMU_FALCON_CONFIG = [
    (0x10AE00, 0x00000004),
    (0x10AE04, 0x00000000),
    (0x10AE08, 0x00000004),
    (0x10AE0C, 0x00000005),
    (0x10AE10, 0x00000006),
    (0x10A10C, 0x00000000),
    (0x10AE24, 0x00000190),
]


def probe(bar: GpuBar0):
    """Probe and report GPU state without modifying anything."""
    print(f"=== GPU Probe: {bar.bdf} ===")
    print(f"Boot0: 0x{bar.r32(0x000000):08X}")
    print(f"PMC_ENABLE: 0x{bar.r32(0x000200):08X}")
    print(f"PMC_ENABLE_1: 0x{bar.r32(0x000204):08X}")

    print("\nFalcon Engines:")
    for base, name in [
        (PMU_BASE, "PMU"),
        (FECS_BASE, "FECS"),
        (GPCCS_BASE, "GPCCS"),
    ]:
        try:
            state = FalconState(bar, base, name)
            print(f"  {state}")
        except Exception:
            print(f"  {name}: INACCESSIBLE")

    print("\nPTIMER:")
    t_lo = bar.r32(0x009400)
    t_hi = bar.r32(0x009410)
    print(f"  TIME: 0x{t_hi:08X}_{t_lo:08X}")

    # VRAM check via PRAMIN sentinel
    bar0_window = bar.r32(BAR0_WINDOW)
    print(f"\nPRAMIN:")
    print(f"  BAR0_WINDOW: 0x{bar0_window:08X}")
    pramin_word = bar.r32(PRAMIN_BASE)
    print(f"  PRAMIN[0]: 0x{pramin_word:08X}")

    # Boot readiness assessment
    pmu = FalconState(bar, PMU_BASE, "PMU")
    ready = pmu.is_running and pmu.hs_mode == 2
    print(f"\nBoot readiness: {'READY' if ready else 'NOT READY'}")
    if not ready:
        reasons = []
        if not pmu.is_running:
            reasons.append(f"PMU not running (CPUCTL=0x{pmu.cpuctl:08X})")
        if pmu.hs_mode != 2:
            reasons.append(f"PMU not in HS mode 2 (HS{pmu.hs_mode})")
        for r in reasons:
            print(f"  - {r}")
    return ready


def stage_firmware_pramin(bar: GpuBar0, vram_addr: int, data: bytes):
    """Stage firmware data in VRAM via the PRAMIN window.

    PRAMIN maps a 64KB window of VRAM into BAR0 at offset 0x700000.
    The window target is set by BAR0_WINDOW register (0x001700).
    BAR0_WINDOW value = (vram_physical_addr >> 16) with enable bit.
    """
    page_size = 0x10000  # 64KB PRAMIN window

    for offset in range(0, len(data), page_size):
        vram_page = (vram_addr + offset) >> 16
        window_val = (vram_page << 0) | 0x1  # enable bit
        bar.w32(BAR0_WINDOW, window_val)
        time.sleep(0.001)

        chunk = data[offset : offset + page_size]
        for i in range(0, len(chunk), 4):
            if i + 4 <= len(chunk):
                word = struct.unpack("<I", chunk[i : i + 4])[0]
                bar.w32(PRAMIN_BASE + (offset % page_size) + i, word)

    print(f"  Staged {len(data)} bytes at VRAM 0x{vram_addr:X}")


def hs_unlock(bar: GpuBar0):
    """Transition PMU from HS mode 2 to HS mode 0 (register access unlocked).

    Sequence from nv535_recipe.json steps 32591-32595.
    WARNING: This is irreversible without hardware reset (power cycle or SBR).
    After unlock, CPU execution is disabled (HWCFG2 bit 29 clears).
    """
    bar.w32(PMU_BASE + 0x014, 0x0000FFFF)   # INTR_MASK
    bar.w32(PMU_BASE + 0x048, 0x00000104)   # MB2
    bar.w32(PMU_BASE + 0x0A4, 0x00000002)   # UNK_0A4
    bar.w32(PMU_BASE + 0x3C0, 0x00000001)   # HS_CTRL = 1
    time.sleep(0.001)
    bar.w32(PMU_BASE + 0x3C0, 0x00000000)   # HS_CTRL = 0
    time.sleep(0.001)

    sctl = bar.r32(PMU_BASE + 0x240)
    mode = sctl & 0xF
    print(f"  HS unlock: SCTL=0x{sctl:08X} (HS{mode})")
    return mode


def load_imem(bar: GpuBar0, offset: int, words: list, tag_base: int = 0):
    """Load firmware words into PMU IMEM at the given offset.

    Uses IMEMC auto-increment mode (bit 24) for sequential writes.
    Sets IMEMT tags at 256-byte (64-word) boundaries.
    """
    bar.w32(PMU_BASE + 0x180, 0x01000000 | offset)  # IMEMC: offset + aincw
    tag = tag_base
    bar.w32(PMU_BASE + 0x188, tag)  # IMEMT

    for i, word in enumerate(words):
        if i > 0 and i % 64 == 0:
            tag += 1
            bar.w32(PMU_BASE + 0x188, tag)
        bar.w32(PMU_BASE + 0x184, word)

    # Verify first word
    bar.w32(PMU_BASE + 0x180, offset)
    v = bar.r32(PMU_BASE + 0x184)
    ok = v == words[0]
    print(f"  IMEM: {len(words)} words at 0x{offset:X}, "
          f"verify[0]=0x{v:08X} {'OK' if ok else 'MISMATCH'}")
    return ok


def load_dmem(bar: GpuBar0, offset: int, words: list):
    """Load data words into PMU DMEM at the given offset."""
    bar.w32(PMU_BASE + 0x1C0, 0x01000000 | offset)  # DMEMC: offset + aincw
    for word in words:
        bar.w32(PMU_BASE + 0x1C4, word)

    bar.w32(PMU_BASE + 0x1C0, offset)
    v = bar.r32(PMU_BASE + 0x1C4)
    ok = v == words[0]
    print(f"  DMEM: {len(words)} words at 0x{offset:X}, "
          f"verify[0]=0x{v:08X} {'OK' if ok else 'MISMATCH'}")
    return ok


def setup_registers(bar: GpuBar0):
    """Apply PMU register configuration from mmiotrace steps 32596-32611."""
    for off, val in PMU_REGISTER_SETUP:
        bar.w32(PMU_BASE + off, val)

    for off, val in PMU_FBIF_SETUP:
        bar.w32(off, val)

    for off, val in PMU_INTR_SETUP:
        bar.w32(PMU_BASE + off, val)

    for off, val in PMU_FALCON_CONFIG:
        bar.w32(off, val)

    print("  Register setup complete")


def trigger_hs_rom(bar: GpuBar0, timeout: float = 5.0):
    """Trigger HS ROM by writing CPUCTL=0x12.

    The HS ROM is fuse-resident secure boot code that:
    1. Re-enables CPU execution (transitions HS0 → HS2/3)
    2. Authenticates the ACR bootloader from IMEM[0xFE00]
    3. Starts the ACR bootloader if authentication passes
    4. The ACR bootloader loads and authenticates falcon firmware from VRAM
    """
    # Pre-start register (step 32765)
    bar.w32(PMU_BASE + 0x104, 0x00010000)
    time.sleep(0.001)

    # Trigger HS ROM
    bar.w32(PMU_BASE + 0x100, 0x00000012)
    time.sleep(0.0001)

    # Post-trigger acknowledge (step 32767)
    bar.w32(PMU_BASE + 0x004, 0x00000010)

    print("  CPUCTL=0x12 written, polling...")

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        cpuctl = bar.r32(PMU_BASE + 0x100)
        sctl = bar.r32(PMU_BASE + 0x240)
        mb0 = bar.r32(PMU_BASE + 0x040)

        status = "RUN" if cpuctl & 0x20 else "HALT" if cpuctl & 0x10 else "IDLE"
        hs = sctl & 0xF
        elapsed = time.monotonic() - start

        if cpuctl & 0x20:
            print(f"  t={elapsed:.3f}s: PMU RUNNING! SCTL=HS{hs} MB0=0x{mb0:08X}")
            return True

        if hs != 0:
            print(f"  t={elapsed:.3f}s: HS mode changed to {hs}! "
                  f"CPUCTL=0x{cpuctl:08X} MB0=0x{mb0:08X}")

        if mb0 != 0:
            print(f"  t={elapsed:.3f}s: MB0=0x{mb0:08X} (HS ROM response)")

        time.sleep(0.01)

    cpuctl = bar.r32(PMU_BASE + 0x100)
    sctl = bar.r32(PMU_BASE + 0x240)
    print(f"  TIMEOUT: CPUCTL=0x{cpuctl:08X} SCTL=HS{sctl & 0xF}")
    return False


def post_boot_cleanup(bar: GpuBar0):
    """Apply post-HS-ROM register cleanup from mmiotrace steps 32802-32833.

    nvidia zeros out all PMU control registers after the HS ROM completes,
    then sets BOOTVEC=0 and starts the LS firmware.
    """
    # Zero PMU registers 0x000-0x0F8 (step 32770-32801)
    for off in range(0x000, 0x100, 8):
        bar.w32(PMU_BASE + off, 0x00000000)

    # CPUCTL = 0 (step 32802)
    bar.w32(PMU_BASE + 0x100, 0x00000000)

    # Zero remaining control registers (steps 32803-32833)
    for off in range(0x108, 0x200, 8):
        bar.w32(PMU_BASE + off, 0x00000000)

    # Set BOOTVEC = 0 for LS firmware (step 32806)
    bar.w32(PMU_BASE + 0x120, 0x00000000)

    print("  Post-boot cleanup complete")


def capture_state(bar: GpuBar0, output_path: str):
    """Capture full PMU and falcon state for analysis."""
    state = {
        "bdf": bar.bdf,
        "timestamp": time.time(),
        "boot0": bar.r32(0x000000),
        "pmc_enable": bar.r32(0x000200),
    }

    for base, name in [
        (PMU_BASE, "pmu"),
        (FECS_BASE, "fecs"),
    ]:
        falcon = {}
        for off, reg_name in [
            (0x000, "reg_000"), (0x008, "intr"),
            (0x010, "intr_en"), (0x014, "intr_mask"),
            (0x01C, "ntstatus"), (0x040, "mb0"),
            (0x044, "mb1"), (0x048, "mb2"),
            (0x100, "cpuctl"), (0x104, "reg_104"),
            (0x108, "reg_108"), (0x10C, "reg_10c"),
            (0x120, "bootvec"), (0x134, "dmactl"),
            (0x148, "hwcfg2"), (0x240, "sctl"),
        ]:
            val = bar.r32(base + off)
            falcon[reg_name] = f"0x{val:08X}"
        state[name] = falcon

    with open(output_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"State captured to {output_path}")
    return state


def boot(bar: GpuBar0, recipe_path: str = None):
    """Execute the full ACR sovereign boot sequence.

    Requires the GPU to be in post-VBIOS state (PMU running, HS mode 2).
    """
    print(f"=== ACR Sovereign Boot: {bar.bdf} ===\n")

    # Phase 0: Verify GPU readiness
    print("Phase 0: Readiness check")
    pmu = FalconState(bar, PMU_BASE, "PMU")
    print(f"  {pmu}")

    if not pmu.is_running:
        print("  ERROR: PMU is not running. GPU needs fresh boot (power cycle).")
        return False
    if pmu.hs_mode != 2:
        print(f"  ERROR: PMU is in HS mode {pmu.hs_mode}, need HS mode 2.")
        print("  GPU needs fresh boot (power cycle) to restore HS mode 2.")
        return False

    # Verify PMC_ENABLE has PMU clock
    pmc = bar.r32(0x200)
    if not (pmc & 0x1000):
        print(f"  WARNING: PMU clock not enabled in PMC_ENABLE (0x{pmc:08X})")

    print(f"  GPU ready: PMU running in HS2, PMC_ENABLE=0x{pmc:08X}")

    # Phase 1: HS Unlock
    print("\nPhase 1: HS Unlock (HS2 → HS0)")
    print("  WARNING: This is irreversible without hardware reset!")
    mode = hs_unlock(bar)
    if mode != 0:
        print(f"  ERROR: Expected HS mode 0, got HS{mode}")
        return False

    # Verify IMEM/DMEM are now accessible
    bar.w32(PMU_BASE + 0x1C0, 0x00000000)
    dmem_check = bar.r32(PMU_BASE + 0x1C4)
    print(f"  DMEM accessible: 0x{dmem_check:08X}")

    # Phase 2: Register setup
    print("\nPhase 2: Register setup")
    setup_registers(bar)

    # Phase 3: Load ACR descriptor into DMEM
    print("\nPhase 3: DMEM descriptor")
    if not load_dmem(bar, 0x00000000, ACR_DMEM_DESCRIPTOR):
        print("  ERROR: DMEM load failed")
        return False

    # Phase 4: Load ACR bootloader into IMEM
    print("\nPhase 4: IMEM firmware")
    if not load_imem(bar, ACR_BL_IMEM_OFFSET, ACR_BL_IMEM_WORDS, tag_base=0x100):
        print("  ERROR: IMEM load failed")
        return False

    # Phase 5: Trigger HS ROM
    print("\nPhase 5: HS ROM trigger")
    success = trigger_hs_rom(bar, timeout=5.0)

    if success:
        print("\n  HS ROM boot succeeded!")
        # Phase 6: Post-boot cleanup and LS firmware start
        print("\nPhase 6: Post-boot cleanup")
        post_boot_cleanup(bar)
    else:
        print("\n  HS ROM boot did not succeed.")
        print("  This is expected if firmware is not staged in VRAM.")
        print("  VRAM addresses needed:")
        print(f"    ACR ucode:     0x{ACR_UCODE_VRAM_ADDR:X} ({ACR_UCODE_SIZE} bytes)")
        print(f"    ACR payload:   0x{ACR_UCODE_VRAM_ADDR:X} ({ACR_PAYLOAD_SIZE} bytes)")
        print(f"    Load table:    0x{ACR_LOADTABLE_VRAM_ADDR:X} ({ACR_LOADTABLE_SIZE} bytes)")

    # Final state
    print("\n=== Final State ===")
    pmu = FalconState(bar, PMU_BASE, "PMU")
    print(f"  {pmu}")
    print(f"  PMC_ENABLE: 0x{bar.r32(0x200):08X}")
    print(f"  HWCFG2 bit29: {pmu.cpu_operational}")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="ACR Sovereign Boot Catalyst for GV100"
    )
    parser.add_argument(
        "--bdf", required=True,
        help="PCI BDF of the GPU (e.g., 0000:02:00.0)"
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Probe GPU state without modification"
    )
    parser.add_argument(
        "--boot", action="store_true",
        help="Execute the full ACR boot sequence"
    )
    parser.add_argument(
        "--capture", action="store_true",
        help="Capture GPU state to JSON"
    )
    parser.add_argument(
        "--recipe", type=str, default=None,
        help="Path to nv535_recipe.json (optional, uses embedded data)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for captured state"
    )
    args = parser.parse_args()

    if not any([args.probe, args.boot, args.capture]):
        parser.print_help()
        sys.exit(1)

    bar = GpuBar0(args.bdf)

    try:
        if args.probe:
            probe(bar)
        if args.capture:
            out = args.output or f"/tmp/acr_catalyst_state_{args.bdf.replace(':', '-')}.json"
            capture_state(bar, out)
        if args.boot:
            success = boot(bar, args.recipe)
            sys.exit(0 if success else 1)
    finally:
        bar.close()


if __name__ == "__main__":
    main()
