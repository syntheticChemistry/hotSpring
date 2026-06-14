# Experiment 061: mmiotrace Capture — Sovereign Devinit Investigation

**Date**: March 15, 2026
**Hardware**: 2× Titan V (GV100, SM70) — `0000:03:00.0` (oracle/nouveau), `0000:4a:00.0` (target/vfio-pci)
**Software**: coralReef `coral-driver`, Linux mmiotrace, diagnostic matrix
**Status**: KEY FINDINGS — three paths forward identified
**Chat**: [mmiotrace devinit investigation](28732f32-750e-4053-a1ae-a8d39a738d7a)

---

## Objective

Capture the exact register-write sequence that nouveau uses to initialize a
Titan V GPU, so we can replicate it in pure Rust for sovereign GPU POST
(HBM2 PHY/DRAM training). The user's original idea was to use an empty PCIe
slot as a "probe target" — we used Linux mmiotrace instead, which gives the
same result without hardware.

---

## Background: The HBM2 Training Problem

When a Titan V is accessed via `vfio-pci`, the HBM2 VRAM is often
inaccessible (reads return `0xBAD0_ACxx`). This happens because:

1. **UEFI/BIOS POST** runs devinit at boot, which trains HBM2 PHY/DRAM
2. **vfio-pci bind** unconditionally transitions the GPU to D3hot (shallow
   sleep), which wipes HBM2 training state
3. **No re-POST mechanism** exists in vfio-pci — the devinit is lost

Previous attempts:
- PCIe SBR triggers device reset but NOT boot ROM re-execution
- `disable_idle_d3=Y` module param does not prevent D3hot on bind
- Oracle register cloning (1529 registers) changed error pattern but didn't
  revive VRAM — sequence and timing matter, not just final state

---

## Experiment: mmiotrace Capture of Nouveau Init

### What is mmiotrace?

A Linux kernel facility that hooks `ioremap()` and captures every MMIO
read/write to PCI BAR regions. It records the exact physical address, value,
width, timestamp, and PID for each operation.

### Setup

```
Oracle: 0000:03:00.0 — Titan V on nouveau (warm GPU, HBM2 trained)
BAR0 physical: 0xd7000000 (16MB)
```

### Procedure

1. Unbind nouveau from oracle card
2. Enable mmiotrace: `echo mmiotrace > /sys/kernel/debug/tracing/current_tracer`
3. Start `trace_pipe` reader in background
4. Rebind nouveau → captures full initialization
5. Stop trace, analyze

### Raw Results

| Metric | Value |
|--------|-------|
| Total trace lines | 206,375 |
| 4-byte operations | 97,326 |
| 8-byte operations | 108,958 |
| BAR0 4-byte writes | 18,928 |
| BAR0 4-byte reads | 39,489 |
| Unique BAR0 offsets written | 359 |
| MAP entries (ioremap calls) | 79 |

### Write Distribution by Region

| Region | Writes | Notes |
|--------|--------|-------|
| PMC (0x000-0x1000) | 10,282 | Engine enable, interrupt clear |
| OTHER 0x0008xx | 2,753 | PBDMA/channel setup |
| MISC 0x002xxx-0x00Fxxx | 2,620 | PFIFO, PTOP, etc. |
| OTHER 0x0012xx | 2,560 | Likely PGRAPH init |
| PFB (0x100000-0x102000) | 195 | MMU flush, BAR0_WINDOW |
| PRAMIN (0x070000) | 162 | VRAM page table writes |
| PDISP (0x610000) | 152 | Display init |
| OTHER 0x0064xx | 97 | Display registers |
| OTHER 0x0004-0x0005 | 84 | Timer, interrupt routing |

---

## Critical Finding: Nouveau Does NOT POST the GPU

### What the trace shows

**Zero PMU FALCON writes.** Nouveau does not upload firmware, does not execute
devinit scripts, and does not perform HBM2 training. The GPU was already
POST'd by UEFI at system boot.

Nouveau's 18,928 BAR0 writes are entirely:
- PMC_ENABLE (engine clocking)
- PFIFO configuration (interrupt clearing, PBDMA setup)
- MMU page table construction (PRAMIN writes)
- Display initialization (PDISP)
- Channel/context setup

### What the trace does NOT show

- No writes to FBPA (0x9A0000) — frame buffer partition agent
- No writes to LTC (0x17E000) — L2 cache controller
- No writes to PCLOCK (0x137000) — core clock PLL
- No writes to PMU FALCON (0x10A000) — power management unit
- No reads of NV_DEVINIT_DONE (0x022500) or DEVINIT_STATUS (0x02240C)

### Interpretation

Nouveau **assumes the GPU is POST'd**. It does not check devinit status,
does not verify HBM2 training, and does not have a fallback. On a system
where UEFI POST'd the GPU (the normal case), this works. On a VFIO system
where D3hot wipes HBM2 training, nouveau has no mechanism to fix it — it
relies on the kernel PCI core bringing the device back to D0 and hoping
the boot ROM re-executes.

This explains why our oracle register cloning approach failed: the registers
we copied from nouveau were the SOFTWARE-SIDE configuration (page tables,
interrupts, channels), not the HARDWARE-SIDE POST sequence (HBM2 PHY training,
clock PLL setup, memory controller timing).

---

## Finding: Oracle Register Cloning (Prior Attempt)

Applied 1,529 register diffs from the warm oracle to the cold VFIO card:

| Region | Diffs Applied | Notes |
|--------|--------------|-------|
| FBPA0 | 637 | Frame buffer partition |
| LTC | 583 | L2 cache controller |
| PMU | ~100 | Power management |
| PCLOCK | ~50 | Clock PLLs |
| Others | ~159 | PMC, PFB, etc. |

**Result**: VRAM error changed from `0xBAD0_ACxx` (VRAM access error) to
`0xBAD0_0100` (PRI access error). Register writes reached hardware but
incorrect sequencing caused a clock domain or power gate misalignment.

35 registers were "stuck" (readback didn't match written value), indicating
hardware-managed or read-only registers that require specific preconditions.

---

## Finding: VBIOS Init Scripts Are Plaintext

The VBIOS (read from PROM at BAR0+0x300000) contains a BIT table with:

| BIT Entry | Pointer | Size | Content |
|-----------|---------|------|---------|
| 'I' (init) | 0x4934 | — | Init script 1 (boot-time register writes) |
| 'I' script_off | 0x76A2 | 18,100 bytes | Boot scripts (PMU-format opcodes) |
| 'I' opcode table | 0x48F4 | 1,829 bytes | Opcode/macro definitions |
| 'p' (PMU) | encrypted | — | PMU DEVINIT firmware (FALCON binary) |

**Key insight**: The PMU DEVINIT firmware is encrypted (hardware-decrypted by
FALCON), but the **init scripts it interprets** are plaintext. The firmware is
just an interpreter — the actual register-write sequences are in the scripts.

Init script 1 starts at offset 0x4934 with opcode `0xCA` (INIT_SPREAD —
spread spectrum configuration). Known opcodes in the scripts include:

| Opcode | Name | Function |
|--------|------|----------|
| 0x69 | INIT_ZM_REG | Write value to register |
| 0x6E | INIT_NV_REG | Masked register write |
| 0x71 | INIT_DONE | End of script |
| 0x72 | INIT_RESUME | Resume after condition skip |
| 0x75 | INIT_SUB_DIRECT | Call sub-script |
| 0x76 | INIT_JUMP | Jump to script offset |
| 0x78 | INIT_CONDITION | Conditional execution |
| 0x5B | INIT_SCRIPT | Call script by table index |
| 0xB8 | INIT_DEVINIT_DONE | Signal devinit completion |

These are documented in envytools and implemented in nouveau's
`nvkm/subdev/bios/init.c` (~2000 lines, full host-side interpreter).

---

## Three Paths Forward

### Path A: VBIOS Script Interpreter (Recommended)

Build a Rust interpreter for VBIOS init scripts, modeled on nouveau's
`nvkm/subdev/bios/init.c`. The scripts are plaintext, the opcodes are
well-documented, and the interpreter is ~50 opcodes.

**Pros**:
- Scripts contain the actual HBM2 training sequence
- No encrypted firmware needed — host executes scripts directly
- Reusable across all NVIDIA GPUs with VBIOS (pre-GSP era)
- nouveau already proves host-side interpretation works

**Cons**:
- Some Volta scripts may use PMU-only registers (need testing)
- ~2000 lines of interpreter code
- Need condition tables, PLL tables, RAM timing tables from VBIOS

**Effort**: ~2-3 days for core opcodes, ~1 week for full coverage

### Path B: mmiotrace of UEFI POST (Hardware-Dependent)

If we can capture the UEFI POST sequence (which DOES train HBM2), we'd
have the exact register sequence. This requires:

1. Custom UEFI payload or EFI shell script that enables mmiotrace
2. Or: force GPU through D3cold→D0 cycle while mmiotrace is active

**Challenge**: UEFI POST happens before Linux kernel is loaded, so standard
mmiotrace can't capture it. Would need a custom EFI driver or FPGA interposer.

**Effort**: Complex, hardware-dependent, not recommended for first attempt.

### Path C: PMU FALCON Upload + Hardware Decrypt

Upload the encrypted PMU DEVINIT firmware from VBIOS to the FALCON, which has
hardware decryption. The FALCON then executes the firmware, which interprets
the init scripts and trains HBM2.

**How it works**:
1. Read encrypted firmware from VBIOS (BIT 'p' entry)
2. Upload to FALCON IMEM/DMEM via BAR0 registers (0x10A180-0x10A1C4)
3. Start FALCON execution (0x10A100 CPUCTL)
4. FALCON hardware decrypts + executes → runs devinit → trains HBM2

**Pros**:
- Exactly what the GPU's boot ROM does
- Hardware handles all sequencing and timing
- Known to work (it's the standard boot path)

**Cons**:
- parse_pmu_table currently fails (BIT 'p' pointer misinterpretation)
- Need to fix the BIT 'p' parsing to find the encrypted firmware blob
- FALCON upload protocol must be exact (nouveau's gm200.c is reference)

**Effort**: ~1-2 days to fix parsing + upload logic

### Recommended Approach: Path C first, Path A as fallback

Path C is simpler (upload existing firmware, let hardware do the work) and
only requires fixing the BIT 'p' table parser. If it fails (e.g., FALCON
rejects the firmware after vfio-pci bind), fall back to Path A.

---

## Lessons Learned

1. **mmiotrace is a powerful reverse-engineering tool** — captured 206K
   operations in one bind cycle, takes seconds to set up

2. **Nouveau is not a POST driver** — it assumes UEFI already initialized
   the GPU. There is no devinit code path in nouveau for Volta that runs
   from the host. The `gm200_pmu_nofw()` stub means "no firmware, assume
   POST'd".

3. **Register state != initialization sequence** — blindly copying 1529
   registers from a warm GPU to a cold one changed the error pattern but
   made things worse. HBM2 training requires a specific sequence of writes
   with timing, polling, and conditional logic.

4. **VBIOS init scripts are the Rosetta Stone** — they contain the actual
   training sequence in a well-documented opcode format. The encrypted PMU
   firmware is just the interpreter for these scripts.

5. **D3hot on vfio-pci bind is the root cause** — confirmed multiple times.
   The GPU is warm after UEFI POST, and D3hot wipes HBM2 training. All
   downstream failures stem from this single transition.

6. **System stability risk**: mmiotrace + pkexec + SBR operations left the
   system sluggish with hanging processes. Future experiments should use a
   dedicated test script with proper cleanup.

---

## Files Created

| File | Purpose |
|------|---------|
| `/tmp/nouveau_mmiotrace_buf.log` | Full mmiotrace capture (206K lines) |
| `/tmp/nouveau_init_writes.json` | 18,928 BAR0 writes extracted as JSON |
| `/tmp/mmiotrace_capture.sh` | First capture attempt (empty trace) |
| `/tmp/mmiotrace_v2.sh` | Working capture script (trace_pipe method) |
| `/tmp/mmiotrace_cold_post.sh` | Cold POST attempt (oracle went D3cold) |

---

## Additional Experiments (Session 2)

### Cold POST mmiotrace Attempt

Attempted to capture nouveau's POST of a _cold_ GPU by SBR-resetting the
oracle and then rebinding nouveau with mmiotrace active.

**Result**: SBR caused the oracle to drop to D3cold. With no driver keeping
the card powered, the kernel auto-suspended it. Nouveau rebind failed.
All register reads returned `0xFFFFFFFF`. The oracle had to be recovered via
PCI remove/rescan + nouveau rebind after system reboot.

**Lesson**: SBR on a driverless card triggers D3cold. To capture a cold POST,
we'd need to either:
- Keep a stub driver bound during SBR to prevent power-down
- Or use the VFIO card (which we control) as the cold target

### VBIOS Init Script Partial Decode

Decoded the init script at offset 0x4934 from the oracle VBIOS PROM:
- Starts with opcode `0xCA` (INIT_SPREAD — spread spectrum configuration)
- Contains variable-length encoded register write sequences
- Scripts reference sub-scripts via `0x5B` (INIT_SCRIPT), `0x75` (SUB_DIRECT)
- Termination markers `0x71` (INIT_DONE) confirmed present
- The opcode table at 0x48F4 is a structured header (version, entry size,
  count), not direct opcodes — likely a function/macro dispatch table

The boot scripts at 0x76A2 (18,100 bytes) contain the bulk of initialization
logic. These are in PMU script format but are **plaintext** — not encrypted.

### System Stability Notes

- mmiotrace + pkexec + SBR operations accumulate stale processes
- Reboot cleared all `/tmp/` trace data (206K-line mmiotrace capture lost)
- **Important**: Save trace data to persistent storage before experiments
- Future work should use `~/Development/ecoPrimals/hotSpring/data/` for
  trace captures, not `/tmp/`

---

## Summary of All Findings

| Finding | Impact |
|---------|--------|
| nouveau does NOT POST the GPU | Cannot replay nouveau init for HBM2 |
| mmiotrace captures 206K ops in one bind | Powerful tool for future analysis |
| VBIOS init scripts are plaintext | Can be interpreted from host (Path A) |
| PMU DEVINIT firmware is encrypted | FALCON hardware decrypts it (Path C) |
| Register state copy (1529 regs) fails | Sequencing matters, not just values |
| Error changed 0xBAD0_ACxx → 0xBAD0_0100 | Writes reach HW, wrong sequence |
| SBR on driverless card → D3cold | Must keep card powered during reset |
| vfio-pci bind → D3hot is root cause | Single transition wipes HBM2 training |

---

## Next Steps

1. **Save traces to persistent storage** — rerun mmiotrace and save to
   `hotSpring/data/` instead of `/tmp/` (lost 206K-line capture to reboot)

2. **Path C: Fix BIT 'p' table parser** in `devinit.rs` — the current code
   misinterprets `data_size=4` as table size; it's actually a pointer size.
   The pointer leads to an offset in the VBIOS where the PMU firmware table
   lives.

3. **Path C: Implement FALCON upload** — using nouveau's `gm200.c`:
   reset FALCON, upload code to IMEM, upload data to DMEM, write boot
   vector, start execution, poll for completion.

4. **Path A fallback: VBIOS script interpreter** — start with opcodes:
   - `0x69` INIT_ZM_REG (direct register write)
   - `0x6E` INIT_NV_REG (masked register write)
   - `0x78` INIT_CONDITION (conditional execution)
   - `0x71` INIT_DONE (script termination)
   - `0x5B` INIT_SCRIPT (sub-script call)
   Reference: nouveau `nvkm/subdev/bios/init.c` (~2000 lines, ~50 opcodes)

5. **Cross-card validation**: Post-reboot, both Titan Vs are unbound in D0
   with fresh BIOS POST. Validate devinit status on both before any
   experiments.

6. **Persist polkit rule** for passwordless pkexec to avoid auth friction:
   ```
   /usr/share/polkit-1/rules.d/50-coralreef-vfio.rules
   ```

---

## BREAKTHROUGH: D3hot → D0 Force via PCI PMCSR (2026-03-15)

### The Discovery

**HBM2 training is NOT lost during D3hot.** VRAM data and memory controller
state persist — BAR0 access is simply disabled in the D3hot power state.
Writing D0 (bits 1:0 = 00) to the PCI PM Control/Status Register (PMCSR)
immediately restores BAR0 access and VRAM is alive.

### Evidence

1. **Fresh boot**: Both Titan Vs warm (BIOS POST), VRAM alive, devinit=SET
2. **Bind vfio-pci**: GPU → D3hot, BAR0 reads 0xFFFFFFFF, VRAM inaccessible
3. **Write PMCSR[1:0]=00**: GPU → D0, BAR0 accessible, **VRAM alive**
4. **Unbind vfio-pci**: VRAM still alive (devinit survived entire round-trip)
5. **Full Rust pipeline**: `force_pci_d0()` → VFIO open → GlowPlug → BAR2 → **Warm**

### Mechanism

The PCI PM capability lives at a device-specific offset in config space.
For the Titan V: PM cap at config+0x60, PMCSR at config+0x64.

```
PMCSR[1:0] = 0b11 → D3hot (BAR0 disabled, VRAM retained)
PMCSR[1:0] = 0b00 → D0    (BAR0 enabled, VRAM accessible)
```

After the D3hot → D0 transition, the PCI spec requires a 10ms recovery delay.

### Impact

- **No VBIOS scripts needed** for normal operation (BIOS POST trains HBM2)
- **No oracle cards needed** (HBM2 survives D3hot)
- **No nouveau dependency** (sovereign path: bind vfio-pci → force D0 → dispatch)
- **VBIOS scripts still valuable** as fallback when HBM2 is truly lost (D3cold)
- **D3cold power cycle** works as last resort (remove → rescan → boot ROM re-POSTs)

### Integration

- `devinit::force_pci_d0(bdf)` — reads PCI config, walks cap chain, writes PMCSR
- `RawVfioDevice::open()` — calls `force_pci_d0` before VFIO device setup
- `GlowPlug::warm()` — Strategy 0: D0 force when initial state is D3Hot
- `GlowPlug` strategy order: D0 force → PMC enable → PFIFO reset → D3cold cycle
  → VBIOS scripts → oracle clone → PMU FALCON → FB probe

### VBIOS Script Scanner (Path A)

The scanner found **577 register writes** in the boot script region:
- FBPA: 122 writes (HBM2 memory controller)
- PTOP/FUSE: 145 writes (topology/configuration)
- CLK: 61 writes (clock configuration)
- PCLOCK: 43 writes (PLL configuration)
- LTC: 10 writes (L2 cache)
- **237 HBM2-critical writes** total

These are available as fallback for the rare case when HBM2 training is
truly erased (D3cold without boot ROM re-POST).

---

*The answer was simpler than we thought. The GPU never forgot its training.
It just went to sleep. We needed to wake it up, not re-teach it.*
