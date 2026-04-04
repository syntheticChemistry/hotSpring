# Exp 083: Nouveau Source Analysis — bind_inst Register Discovery

**Date:** 2026-03-24
**Type:** Source code archaeology (no hardware)
**Goal:** Resolve Gap 2 (SYS_MEM_COH_TARGET) and investigate Gap 1 (bind_stat)
**Sources:** Linux kernel nouveau driver (upstream torvalds/linux), envytools rnndb

---

## Executive Summary

Source analysis of nouveau's falcon instance block binding reveals **four
discrepancies** between nouveau's implementation and coralReef's ACR boot
strategies. Any one of these could explain why `bind_stat` never reaches 5.

---

## Finding 1: CRITICAL — Wrong Register Offset (`0x668` vs `0x054`)

**Nouveau** (`gm200_flcn_bind_inst` in `nvkm/falcon/gm200.c`):

```c
nvkm_falcon_mask(falcon, 0x604, 0x00000007, 0x00000000);  // clear DMAIDX
nvkm_falcon_wr32(falcon, 0x054, (1 << 30) | (target << 28) | (addr >> 12));
```

The instance block bind register is at **falcon_base + 0x054**.
For SEC2 (base `0x87000`), this means BAR0 address **`0x87054`**.

**coralReef** (`instance_block.rs`):

```rust
pub(crate) const SEC2_FLCN_BIND_INST: usize = 0x668;
```

This writes to BAR0 address **`0x87668`** — a completely different register.

**Additionally**, `strategy_chain.rs` also writes to `0x480`:

```rust
w(SEC2_FLCN_BIND_INST, inst_val); // 0x668
w(0x480, inst_val);               // 0x480
```

Neither `0x668` nor `0x480` matches nouveau's `0x054`.

**GV100 uses `gp102_sec2_flcn` ops**, which points `bind_inst` at
`gm200_flcn_bind_inst` — confirming that `0x054` is the correct offset
for SEC2 on Volta.

**Impact:** If `0x668` is not the bind register, every bind attempt writes
to the wrong address. The falcon never sees the instance block pointer,
so `bind_stat` at `0x0dc` never advances.

---

## Finding 2: Missing Bit 30 in Bind Value

**Nouveau** always sets bit 30:

```c
nvkm_falcon_wr32(falcon, 0x054, (1 << 30) | (target << 28) | (addr >> 12));
```

**coralReef** `strategy_sysmem.rs`:

```rust
let inst_bind_val = ((sysmem_iova::INST >> 12) as u32) | (SYS_MEM_COH_TARGET << 28);
```

Bit 30 is NOT set. The purpose of bit 30 is undocumented in envytools but
nouveau always includes it. It may be an "enable" or "valid" bit for the
instance block binding.

**Impact:** Even if the register offset were correct, the missing bit could
cause the bind to be ignored.

---

## Finding 3: SYS_MEM_COH_TARGET — `strategy_chain.rs` Uses Wrong Value

**Nouveau** (`gm200_flcn_fw_load`):

```c
switch (nvkm_memory_target(fw->inst)) {
case NVKM_MEM_TARGET_VRAM: target = 0; break;
case NVKM_MEM_TARGET_HOST: target = 2; break;  // coherent system memory
case NVKM_MEM_TARGET_NCOH: target = 3; break;  // non-coherent system memory
}
```

| Value | Nouveau Meaning |
|-------|-----------------|
| 0 | VRAM |
| 2 | System memory, coherent (HOST) — standard for IOMMU-mapped DMA |
| 3 | System memory, non-coherent (NCOH) |

**coralReef discrepancy:**

| Strategy | Target bits [29:28] | Comment | Correct? |
|----------|---------------------|---------|----------|
| `strategy_sysmem.rs` | **2** | "SYS_MEM_COH target" | YES |
| `strategy_chain.rs` | **3** | "target=3 for SYS_MEM_COH" | **NO** — 3 is NCOH |
| `strategy_vram.rs` | **0** | VRAM target | YES (for VRAM path) |
| `strategy_hybrid.rs` | **0** | VRAM target | YES (for VRAM inst block) |

**Impact:** `strategy_chain.rs` labels 3 as "SYS_MEM_COH" but it's actually
non-coherent. For IOMMU-mapped system memory, coherent (2) is the correct
target. This would cause cache coherency issues even if the bind succeeded.

---

## Finding 4: Missing DMAIDX Clear at `0x604`

**Nouveau** clears the DMA index register BEFORE writing the bind register:

```c
nvkm_falcon_mask(falcon, 0x604, 0x00000007, 0x00000000);  // DMAIDX_VIRT → 0
```

**coralReef** does not write to `0x604` in any strategy. Grep confirms zero
hits for `0x604` across the entire codebase.

**Impact:** The DMAIDX register selects which DMA path the falcon uses. If
it's not cleared to 0 (VIRT mode), the instance block binding may not take
effect because the falcon is in a different DMA addressing mode.

---

## envytools Corroboration

envytools `g80_defs.xml` defines `g80_mem_target`:

```xml
<enum name="g80_mem_target" inline="yes">
    <value value="0" name="VRAM"/>
    <!-- 1 is some other sysram. -->
    <value value="2" name="SYSRAM"/>
    <value value="3" name="SYSRAM_NO_SNOOP"/>  <!-- XXX: check -->
</enum>
```

This confirms: 2 = SYSRAM (snooped/coherent), 3 = SYSRAM_NO_SNOOP (non-coherent).

envytools `falcon.xml` documents `falcon_memif` PORT TYPE with:
- 5 = SYSRAM (snooped)
- 6 = SYSRAM_NO_SNOOP

Note: falcon_memif uses a DIFFERENT encoding than PTE target — do not mix them.

envytools does NOT document `0x054` as a named register in the falcon block,
and does NOT document `0x668` at all. The `PSEC2` definition in `psec.xml`
only includes `falcon_base`, not `falcon_memif` — so SEC2's DMA interface
is poorly documented in envytools. Nouveau source is the authority.

---

## Recommended Fix Order for coralReef

1. **Change `SEC2_FLCN_BIND_INST` from `0x668` to `0x054`** — this is the
   single most likely fix for the bind_stat failure
2. **Add bit 30** to the bind value: `(1 << 30) | (target << 28) | (addr >> 12)`
3. **Add DMAIDX clear**: write `0x604` with mask `(val & !0x07)` before bind
4. **Fix `strategy_chain.rs`** `SYS_MEM_COH_TARGET` from 3 to 2

---

## Confidence Assessment

| Finding | Confidence | Evidence |
|---------|-----------|----------|
| Register offset 0x054 | **HIGH** — direct nouveau source | `gm200_flcn_bind_inst` → `nvkm_falcon_wr32(falcon, 0x054, ...)` |
| Bit 30 required | **MEDIUM** — nouveau always sets it, purpose unclear | Could be "valid" flag, could be reserved-but-needed |
| SYS_MEM_COH = 2, not 3 | **HIGH** — nouveau enum + envytools match | `NVKM_MEM_TARGET_HOST` = 2, `NCOH` = 3 |
| DMAIDX clear needed | **MEDIUM** — nouveau does it, unclear if default is OK | May already be 0 after reset |

---

## What This Means for the ACR Boot Solver

If Finding 1 is confirmed (wrong register), then ALL previous `bind_stat`
experiments were writing to the wrong address. The falcon never received
the instance block pointer. This would explain:

- `bind_stat` stuck at 0 (no bind ever initiated)
- HS ROM PC advancing but BL not executing (BL needs DMA via instance block)
- `0x668` reads back 0 (not the bind register, just an unused address)

The fix requires only one line change in `instance_block.rs`. Once the correct
register is used with the correct value (including bit 30) and DMAIDX is cleared,
the bind should succeed — and the HS bootloader should be able to DMA the ACR
firmware from system memory.

---

## Next Steps (hotSpring)

1. Cross-reference with Exp 081 logs — did `0x668` reads ever return the written value?
2. Check if envytools `rnndb` has any documentation for offset `0x054` under SEC2
3. Prepare updated experiment plan for Exp 084 (test corrected bind sequence)
4. Add these findings to the coralReef evolution roadmap at parent wateringHole
