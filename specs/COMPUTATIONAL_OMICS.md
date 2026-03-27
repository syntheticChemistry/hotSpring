# Computational -omics: Environmental Genomics for Sovereign Hardware

**Date:** 2026-03-25
**Status:** Specification — long-term evolution target for the compute trio
**Owner:** hotSpring (methodology) → coralReef (implementation) → barraCuda (math)
**Cross-spring:** wetSpring (algorithmic patterns), toadStool (hardware surface)

---

## The Thesis

A GPU is not a single device. It is an ecosystem of microcontrollers (falcons),
memory subsystems, DMA engines, and firmware payloads that interact through
multiple information substrates. The sovereign pipeline crisis (123 experiments,
6 systematic dead ends before finding the PDE slot bug) revealed that we are
treating this ecosystem as a monolithic system when it is not.

Environmental genomics solved an analogous problem: understanding complex microbial
communities where hundreds of species interact through shared chemical substrates,
each with their own genome, transcription machinery, and metabolic products. The
isomorphism is mathematical, not metaphorical.

## Information Substrates

### The Cell ↔ GPU Isomorphism

| Biological Structure | GPU Equivalent | Encoding | Access Rules |
|---------------------|---------------|----------|--------------|
| **Nuclear DNA** | Firmware binary in WPR | Falcon ISA, signed | Immutable once sealed. HS mode protects. Authentication = immune recognition |
| **Mitochondrial DNA** | Page tables + instance block | GV100 MMU v2 PDE format | Separate encoding from firmware. Host-constructed. Different "codon table" |
| **Ribosomal RNA** | Falcon bootloader (BL) | Falcon ISA, trusted | Translation machinery. Reads firmware and executes it |
| **mRNA** | DMEM data (ACR descriptor, BL descriptor) | Structured binary | Transcribed by host, consumed by BL. Short-lived, purpose-specific |
| **tRNA** | DMA descriptors + IOMMU mappings | IOVA ↔ physical translation | Adapter molecules between address spaces |
| **Plasmids** | WPR2 registers, MAILBOX, config state | Register protocol | Small autonomous information. Can be set by different "organisms" (BIOS, PMU, host) |
| **Epigenetic marks** | SCTL, FBIF_TRANSCFG, security fuses | Bit fields | Same code, different behavior. Modify expression without changing the binary |
| **Metabolites** | DMA buffer contents, ring data | Application-specific | Products of computation. Flow between subsystems |

### Why This Matters

Exp 104 was stuck for 6 experiments because we confused mitochondrial DNA (page
tables) with nuclear DNA (firmware). The page table format has its own "codon
table" (16-byte PDE entries, pointer in upper 8 bytes) that is completely
independent of the firmware encoding. We were applying nuclear DNA rules to
mitochondrial DNA — putting the information in the wrong reading frame.

Similarly, the WPR2 indexed register (0x100CD4) is a **plasmid** — a separate
information substrate that carries the same logical data (WPR boundaries) as
our direct register writes (0x100CEC/CF0), but through a different pathway.
The firmware "reads" the plasmid, not our chromosome. Writing to the wrong
substrate is like putting a gene in a plasmid when the organism reads it from
the chromosome.

## Techniques

### 1. Register Trace Alignment (PCR Amplification)

**Biological analog:** PCR amplifies a specific DNA region for analysis.
**Computational analog:** Extract and align register write sequences across drivers.

```
nouveau mmiotrace   (32,507 ops)  →  "genome A"
nvidia-open trace   (521B header) →  "genome B" (GSP-mediated, mostly invisible)
sovereign boot      (~200 ops)    →  "genome C"

Sequence alignment: find conserved regions, divergent regions, our gaps.
```

**Method:** Treat each register write as a "base" with a 3-tuple encoding:
`(address: u32, value: u32, relative_time: u64)`. Align using a modified
Smith-Waterman with a GPU-specific substitution matrix:

- Same address + same value = match (score +3)
- Same address + different value = mismatch (score -1, important: reveals config differences)
- Address in same functional block (e.g., both SEC2 0x87xxx) = weak match (score +1)
- Missing write = gap (score -2, critical: reveals substrate we never populate)

**Output:** Gap list — registers that nouveau writes but we never touch.
These are candidate "missing genes" for HS authentication.

**Data we already have:** Exp 086 register profiles, Exp 092 mmiotrace capture
(32,507 operations, all four phases visible). This is immediately actionable.

### 2. Differential Register Expression

**Biological analog:** RNA-seq differential expression — which genes are
active in condition A vs condition B.
**Computational analog:** Which registers are "hot" (frequently read/written)
during each boot phase.

From the mmiotrace, partition the 32,507 operations into temporal phases:
1. PRIV_RING topology discovery
2. PMC_ENABLE / engine gating
3. PRAMIN / PFB initialization
4. SEC2 boot (ACR chain)
5. PFIFO / scheduler
6. Display engine

For each phase, compute a "register expression profile" — frequency of
access to each functional block. Compare our sovereign boot profile against
nouveau's. Phases where our profile diverges are candidate problem areas.

### 3. Firmware Motif Search (BLAST for Binary)

**Biological analog:** BLAST searches a query sequence against a genome database.
**Computational analog:** Search for instruction patterns from crash PCs in
the full firmware blob.

The firmware binary at crash/halt locations contains specific instruction
sequences. Extract 16-64 byte windows around key PCs:
- PC 0x0500 (pre-fix crash site)
- PC 0x55C6 (post-fix idle loop)
- PC 0xFD75 (BL entry point)
- The HS entry point (non_sec_code_size offset)

Search for these patterns throughout the firmware blob using local alignment.
Matching regions reveal the function being executed, cross-references to
other call sites, and the data structures being accessed.

### 4. Subsystem Phylogeny

**Biological analog:** Phylogenetic trees from 16S rRNA sequences reveal
evolutionary relationships between organisms.
**Computational analog:** Track how falcon firmware registers and boot
sequences evolved across GPU generations.

```
GM200 (Maxwell)  →  GP102 (Pascal)  →  GV100 (Volta)  →  GA100 (Ampere)  →  AD102 (Ada)
```

Nouveau source provides the "fossil record" — each generation has a
`gXXX_acr.c` / `gXXX_sec2.c` file. Extract the register write sequences
from each and build a phylogenetic tree of the boot protocol. Conserved
regions across all generations are fundamental. Volta-specific changes are
where our bugs likely live.

### 5. Transcription Profiling (Time-Series -omics)

**Biological analog:** Time-series RNA-seq during development.
**Computational analog:** Snapshot falcon state at multiple points during boot.

Currently we sample at two points: pre-boot and post-crash. This is like
measuring gene expression only at birth and death. We need intermediate
snapshots:

- After BL loads but before STARTCPU
- Immediately after STARTCPU (LS mode)
- During NS code execution (if we can interrupt)
- At HS transition attempt
- After HS success/failure

The TRACEPC dump provides 31 data points from the post-fix run. Each PC
is a "gene expression measurement" — which code region was executing.
Mapping these PCs to firmware functions gives us a transcription timeline.

## Implementation Plan

### Phase 1: Immediate (Exp 105-106)

**Register trace alignment** using existing data:
1. Parse the 32,507-operation nouveau mmiotrace from Exp 092
2. Extract our sovereign boot register write sequence from `strategy_sysmem.rs`
3. Align and identify gaps
4. Focus on SEC2 boot phase — registers nouveau writes that we skip

This is the "PCR amplification" step. It directly targets HS authentication
by finding missing register writes.

**Owner:** hotSpring experiment, coralReef implementation.
**Tools needed:** Simple sequence alignment (no wetSpring dependency yet).

### Phase 2: Near-term (post-HS authentication)

**Differential register expression** across drivers:
1. Build register frequency profiles from mmiotrace
2. Compare across vfio-cold, nouveau-warm, nvidia-warm
3. Identify phase-specific register patterns

**Firmware motif search:**
1. Parse the firmware binary into instruction-sized windows
2. Index by opcode pattern
3. Search for patterns around key PCs

**Owner:** hotSpring methodology, coralReef `coral-driver` tooling.

### Phase 3: Long-term (compute trio evolution)

**Full -omics toolkit:**
1. Port wetSpring alignment algorithms to binary domain (Smith-Waterman with
   GPU-specific substitution matrix)
2. Build "genome browser" for GPU firmware — interactive visualization of
   firmware structure, register maps, boot sequences
3. Subsystem phylogeny across GPU generations
4. Time-series transcription profiling with hardware interrupt snapshots

**Owner:** coralReef (implementation), barraCuda (math for alignment/scoring),
toadStool (hardware surface integration), hotSpring (methodology + validation).

**wetSpring consultation:** Algorithm patterns (alignment scoring, motif
discovery, phylogenetic tree construction). The domain knowledge to interpret
results must stay in the compute trio where GPU expertise lives.

## Why This Is Not a Distraction

The biological metaphor is grounded in a real mathematical isomorphism:
both biological systems and GPU firmware ecosystems are **multi-substrate
information processing systems** where:

1. Different information types (DNA/RNA/protein ↔ IMEM/DMEM/registers)
   follow different encoding and access rules
2. Cross-substrate translation (transcription/translation ↔ DMA/MMU) is
   where errors concentrate
3. Authentication (immune recognition ↔ HS signature verification) operates
   on specific substrates and rejects well-formed data from the wrong one
4. Environmental context (epigenetics ↔ register state) modifies behavior
   without changing the underlying code

The PDE slot bug (Exp 104) was a **reading frame error** — correct data in
the wrong position within the encoding substrate. This is the most common
class of errors in both domains.

The -omics framework provides:
- **Systematic hypothesis generation** (register gap analysis replaces manual guessing)
- **Cross-driver comparison** at scale (32,507 operations aligned, not hand-inspected)
- **Evolutionary context** (Volta-specific vs conserved patterns)
- **Vocabulary for the team** (substrate confusion, reading frame errors, missing genes)

## Relationship to Existing Work

| Existing Spec | How -omics Extends It |
|--------------|----------------------|
| `GPU_CRACKING_GAP_TRACKER.md` | Gap analysis becomes systematic via alignment, not manual |
| `DRIVER_AS_SOFTWARE.md` | Driver personalities become "organisms" with distinct genomes |
| `UNIVERSAL_DRIVER_REAGENT_ARCHITECTURE.md` | Reagent recipes become "metabolic pathways" |
| `NATIVE_COMPUTE_ROADMAP.md` | Late-stage sovereign compute benefits from phylogenetic firmware analysis |
| `MULTI_BACKEND_DISPATCH.md` | Backend comparison becomes differential expression analysis |

## Cross-Spring Connections

| Spring | Connection |
|--------|-----------|
| **wetSpring** | Source of alignment algorithms, motif search, phylogenetic methods. Consultation role. |
| **neuralSpring** | Spectral methods for register pattern analysis (eigenvalue decomposition of access matrices) |
| **airSpring** | Environmental monitoring patterns — GPU thermal/power as "environmental factors" affecting "gene expression" |
| **groundSpring** | Measurement noise characterization — register read variability, timing jitter as noise sources |

---

## License

**AGPL-3.0** — GNU Affero General Public License v3.0
