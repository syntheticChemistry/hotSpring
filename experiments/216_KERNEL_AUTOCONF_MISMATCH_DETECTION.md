# Experiment 216 — Kernel autoconf.h Mismatch Detection

**Date**: 2026-05-21
**Status**: COMPLETE (root cause identified, fix validated, abstraction planned)
**Hardware**: 2x NVIDIA Titan V (GV100), BDFs `0000:02:00.0`, `0000:49:00.0`
**Kernel**: 6.17.9-76061709-generic (built Dec 2, 2025)
**Dependency**: Exp 215 (Sovereign Warm Compute Tier 2)

## Objective

Identify, isolate, and abstract the root cause of persistent `Invalid relocation
target` errors when loading freshly compiled kernel modules (DKMS nvidia-470,
test modules) on kernel 6.17.9 — errors that appeared to be about ELF relocations
but were actually caused by a `struct module` layout mismatch.

## Symptom

Every freshly compiled kernel module failed to load with:

```
insmod: ERROR: could not insert module: Invalid module format
dmesg: module: Invalid relocation target, existing value is nonzero for type 1
```

Static analysis of the `.ko` file showed **zero** nonzero `R_X86_64_64` relocation
targets. The corruption was invisible at the file level — it only manifested during
the kernel's in-memory relocation pass.

## Root Cause: `autoconf.h` Corruption

The file `/usr/src/linux-headers-$(uname -r)/include/generated/autoconf.h` had been
overwritten on May 3, 2026, by an out-of-tree build operation. The running kernel
binary was compiled on December 2, 2025, with a different set of `CONFIG_*` options.

### The Critical Config Delta

| Config | Running Kernel (Dec 2025) | Corrupted Headers (May 3) |
|--------|--------------------------|--------------------------|
| `CONFIG_DEBUG_INFO_BTF_MODULES` | **enabled** | **missing** |

This single config option controls whether `struct module` includes BTF-related
fields. Its absence shifted `struct module` field offsets by 24 bytes:

| Field | Running Kernel Offset | Corrupted Headers Offset |
|-------|----------------------|-------------------------|
| `source_list` | 0x488 | 0x478 |
| `target_list` | 0x498 | 0x488 |
| `exit` | **0x4a8** | **0x490** |

### The Clobbering Mechanism

During module load, the kernel initializes `struct module` fields using its
own internal offsets (from the Dec 2025 binary). When it executed:

```c
INIT_LIST_HEAD(&mod->source_list);  // at kernel's offset 0x488
```

This wrote a `prev` pointer into offset 0x490 (0x488 + sizeof(void*)).
In the newly compiled module's layout, 0x490 was the `exit` function pointer's
relocation target. The kernel's `apply_relocate_add` then found a nonzero
value where it expected zeros:

```c
// arch/x86/kernel/module.c
if (memcmp(loc, &zero, size))  // loc=0x490 is nonzero (prev pointer)
    goto invalid_relocation;
```

This produced the misleading `Invalid relocation target, existing value is
nonzero for type 1` error — an error that appears to be about ELF file
integrity but is actually about struct layout divergence.

## Detection Methodology

### Phase 1: Elimination of File-Level Causes

A Python script verified all `R_X86_64_64` relocation targets in the DKMS-built
`nvidia.ko` were already zero on disk. This ruled out file corruption.

### Phase 2: Kprobe Instrumentation

Planted kprobes on `apply_relocate_add` and `__write_relocate_add` to capture
the exact ELF section, relocation index, and memory address where the check
failed. This identified `.gnu.linkonce.this_module` as the failing section
and the `exit` field relocation as the specific target.

### Phase 3: Struct Layout Comparison

Compiled a minimal kernel module that stores `offsetof(struct module, exit)`
in a `.note` ELF section. Compared against the working `nvidia-580` module
(compiled with the original, correct headers at kernel build time):

| Module | `exit` offset |
|--------|--------------|
| nvidia-580.ko (Dec 2025 build) | 0x4a8 |
| Test module (current headers) | 0x490 |

The 24-byte delta (0x4a8 - 0x490 = 0x18) matched exactly one `struct list_head`
(16 bytes) plus alignment — the BTF modules field.

### Phase 4: autoconf.h Forensics

```bash
stat /usr/src/linux-headers-6.17.9-76061709-generic/include/generated/autoconf.h
# Modified: May 3, 2026

stat /boot/vmlinuz-6.17.9-76061709-generic
# Modified: Dec 2, 2025

diff <(dpkg -S autoconf.h | ...) ...
# CONFIG_DEBUG_INFO_BTF_MODULES missing from current file
```

### Phase 5: Repair and Validation

Extracted the original `autoconf.h` from the cached `.deb` package:

```bash
dpkg-deb -x /var/cache/apt/archives/linux-headers-6.17.9-76061709-generic_*.deb /tmp/headers-extract
sudo cp /tmp/headers-extract/.../autoconf.h /usr/src/linux-headers-.../include/generated/autoconf.h
```

After restoration, the test module compiled with `exit offset: 0x4a8` — matching
the running kernel. DKMS modules loaded successfully.

## Key Insight

**This class of bug is invisible to static analysis of the `.ko` file.** All
relocation targets read as zero in the file. The corruption manifests only at
runtime when the kernel's `INIT_LIST_HEAD` writes into an in-memory struct
whose layout no longer matches what the module was compiled against.

Any system where kernel headers have been modified after the kernel was built
is vulnerable. Common causes:
- Out-of-tree module builds that regenerate `autoconf.h`
- Partial kernel header package updates
- Manual `make menuconfig` / `make syncconfig` in the headers tree

## Abstraction: Kernel Build Environment Health Check

This discovery motivated a three-layer detection system for toadStool:

1. **Layer 1 — Freshness check**: Compare `autoconf.h` mtime against kernel
   image. If headers are newer than the kernel binary, flag as suspicious.
2. **Layer 2 — Struct layout probe**: Compile a minimal module, read
   `offsetof(struct module, init/exit)` from its `.note` section, compare
   against a reference module known to load successfully.
3. **Layer 3 — Cross-check**: Parse `.gnu.linkonce.this_module` RELA entries
   from any loaded module's `.ko` to determine what layout the kernel expects.

Repair strategy: extract original `autoconf.h` from cached `.deb` package
(fastest, no network, most reliable).

Implementation: `cylinder/src/vfio/kernel_health.rs` — see Exp 216 follow-up
code in toadStool.

## Files Modified

| File | Change |
|------|--------|
| `/usr/src/linux-headers-.../include/generated/autoconf.h` | Restored from .deb |
| `cylinder/src/vfio/kernel_health.rs` | New module (this experiment's abstraction) |

## Post-Fix Audit (May 21, 2026)

Systematic scan of all DKMS-built `.ko` files using the new `kernel_health.rs`
infrastructure to identify the full blast radius of the corruption.

### Corruption Window

**May 3, 2026** (autoconf.h modified) → **May 21, 2026** (restored from .deb)

### Affected Modules (exit=0x490, expected 0x4a8)

| Module | Built | Status | Impact |
|--------|-------|--------|--------|
| `acpi-call/1.2.2/acpi_call.ko` | May 8 | **INSTALLED** | Would fail if loaded |
| `nvidia/470.256.02/nvidia.ko` | May 20 | DKMS only | Could not load (Exp 211 failures) |
| `nvidia/470.256.02/nvidia-drm.ko` | May 20 | DKMS only | Could not load |
| `nvidia/470.256.02/nvidia-modeset.ko` | May 20 | DKMS only | Could not load |
| `nvidia/470.256.02/nvidia-peermem.ko` | May 20 | DKMS only | Could not load |
| `nvidia/470.256.02/nvidia-uvm.ko` | May 20 | DKMS only | Could not load |

### Unaffected Modules (correct exit=0x4a8)

| Module | Built | Reason |
|--------|-------|--------|
| `nvidia/580.126.18/*` | Apr 18 | Pre-corruption window |
| `nvsov/470.256.02/*` | May 21 | Post-fix rebuild |
| `system76/*` | Feb 20 | Pre-corruption window |
| `nouveau.ko` (stock) | Dec 2025 | Stock kernel module, never recompiled |
| `nouveau.ko` (patched) | May 11 | Binary patched (bytes only), RELA from stock |

### Experiment Impact Assessment

**No experiment conclusions invalidated.** Experiments 210-215 used binary-patched
stock `nouveau.ko` modules. Stock `.ko` files carry correct struct module offsets
from the kernel build (Dec 2025). Binary patching modifies function bytes but does
not touch ELF RELA entries or section headers.

The corruption only impacted DKMS-compiled module loading:
- Exp 211: nvidia-470 DKMS build failed to load — initially misattributed to
  proprietary module incompatibility. True cause was autoconf.h corruption.
- The `acpi_call.ko` was silently installed with corrupted offsets but never
  loaded during the corruption window.

### Remediation

All corrupted modules rebuilt with correct headers:

```
sudo dkms remove acpi-call/1.2.2 -k $(uname -r)
sudo dkms build acpi-call/1.2.2 -k $(uname -r)
sudo dkms install acpi-call/1.2.2 -k $(uname -r)

sudo dkms remove nvidia/470.256.02 -k $(uname -r)
sudo dkms build nvidia/470.256.02 -k $(uname -r)
```

Post-remediation scan: **20/20 DKMS modules OK, 10/10 installed modules OK.**

## Outcome

- Root cause identified: corrupted `autoconf.h` causing 24-byte `struct module` shift
- Detection methodology validated: kprobe + struct probe + autoconf forensics
- Fix confirmed: `.deb` package restore of `autoconf.h` resolves all module load failures
- Abstraction designed: 3-layer kernel health check for toadStool preflight
- Post-fix audit: 6 corrupted modules found, all rebuilt, zero remaining corruption
- `toadstool kernel-health` CLI and `sovereign.kernel_health` RPC validated on live system
