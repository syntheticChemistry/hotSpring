# Infra Maturity & Ecosystem Handoff

**Date:** May 11, 2026
**From:** hotSpring deep-debt evolution sprint
**To:** primalPsing audit, upstream primals teams, springs teams
**Scope:** benchScale, agentReagents, hotSpring barracuda, composition patterns

---

## Executive Summary

benchScale and agentReagents have completed three evolution passes (plan phases
1–4, modernization, deep debt) and are ready for formal infra adoption. The
codebase is modern idiomatic Rust with zero TODO/FIXME markers, zero unsafe in
production paths, and full test coverage. This handoff documents what changed,
what patterns emerged, and what upstream teams should absorb.

---

## 1. benchScale — Infra Readiness

### What It Is

Pure-Rust VM lifecycle management for libvirt/QEMU. Creates, configures, and
tears down builder VMs with GPU passthrough, cloud-init, and SSH orchestration.

### Evolution Completed

| Area | Before | After |
|------|--------|-------|
| VM creation | `virt-install` CLI shell-out | `virt` crate FFI — pure XML generation |
| ISO generation | `genisoimage` subprocess | `hadris-iso` — pure Rust |
| SSH/SCP | `Command::new("ssh")` everywhere | `russh` (SshClient) with CLI fallback |
| File copy | `Command::new("cp")` | `std::fs::copy` |
| Interface discovery | `sh -c "ssh ... awk"` pipe | `SshClient::exec_stdout` async |
| Boot diagnostics | Hardcoded user list `["ubuntu","reagent","builder","cosmic"]` | Configurable via `BenchScaleConfig`, async russh |
| DHCP FFI | Duplicate raw pointer iteration in two files | Consolidated `LeaseList::ipv4_leases()` with RAII wrapper |
| Configuration | Deprecated `LibvirtConfig` still live | Fully migrated to `BenchScaleConfig` — deprecated field removed |
| GPU passthrough | `PciPassthroughDevice` (legacy) | `VfioPassthrough` with attach_mode, ROM BAR, managed, QEMU properties |
| QEMU customization | None | `<qemu:commandline>` injection from `VfioPassthrough::qemu_properties` |
| Package management | APT-only | `PackageManager` enum (Apt/Dnf/Pacman/Zypper) |
| Cleanup | `virsh` CLI fallback | `virt` crate only, gated on `libvirt` feature |

### API Surface (stable)

```rust
// Core types
BenchScaleConfig        // Unified config (timeouts, network, storage, monitoring)
LibvirtBackend          // VM lifecycle (create, delete, wait_for_ip, discover_interface)
VfioPassthrough         // GPU passthrough config (BDF, attach_mode, qemu_properties)
SshClient               // Pure-Rust SSH (connect, connect_with_key, exec_stdout, push/fetch)
DiskManager             // CoW overlay management
CloudInitBuilder        // Cloud-init YAML generation

// Key traits
Backend                 // VM lifecycle abstraction (libvirt, docker-hardened-catalog)
GpuLifecycle            // VFIO bind/unbind, IOMMU group discovery
```

### For Upstream Teams

- **agentReagents** depends on `benchscale` via path (`path = "../benchScale"`).
  Both should move to a shared workspace or versioned dependency for CI.
- **BenchScaleConfig** is the single source of truth for timeouts, SSH, storage.
  New consumers should construct it from YAML/env, not hardcode values.
- **SshClient** is usable standalone for any SSH automation in the ecosystem.

---

## 2. agentReagents — Infra Readiness

### What It Is

Manifest-driven VM image builder. Reads a YAML template, creates a builder VM
via benchScale, runs cloud-init + post-boot steps, verifies the result, and
saves a reusable template image.

### Evolution Completed

| Area | Before | After |
|------|--------|-------|
| Build states | `InstallingCosmic`, `InstallingRustDesk` | `InstallingDesktop`, `InstallingServices` (generic) |
| Build API | `build_cosmic_desktop()` (deprecated) | `build()` — manifest-driven |
| Verification | 1061-line monolith | 3-module split: `types.rs`, `package.rs`, `mod.rs` |
| Desktop check | Cosmic-specific grep | Distro-agnostic (xorg/wayland/gdm/sddm/lightdm, dpkg+rpm) |
| Network discovery | Hardcoded `enp1s0` | Dynamic `ip -o -4 route show default` |
| SSH | Mixed CLI + russh | russh-first with CLI fallback |
| Cosmic references | Scattered in production code | Removed — all generic |

### Template Manifest (YAML)

Templates drive everything. Key fields for GPU sovereignty:

```yaml
pci_passthrough:
  - bdf: "0000:4d:00.0"
    attach_mode: cold        # cold | hot_managed | hot_unmanaged
    rom_bar: false
    managed: false
    qemu_properties:
      x-no-mmap: "on"       # injected as <qemu:commandline>
```

### For Upstream Teams

- **GPU sovereignty via VM isolation**: agentReagents + benchScale is the proven
  path for running proprietary NVIDIA drivers (470/535) inside a VM to extract
  firmware init sequences. The `reagent-nvidia470-titanv.yaml` template is the
  reference for this workflow.
- **Template library**: `templates/` contains validated manifests for multiple
  GPU + driver combinations. These are reusable by any team needing VM-based
  GPU experimentation.
- **Verification module**: `verification/package.rs` implements 4-method package
  verification (dpkg-query → dpkg -l → apt-cache → reverse-dependency). Reusable
  for any VM validation pipeline.

---

## 3. Composition Patterns for NUCLEUS

### Three-Tier Validation (proven by hotSpring)

```
Python baseline → Rust proof → NUCLEUS IPC composition
```

Each tier validates the next. The same tolerance-driven, exit-code-gated
methodology that proves Rust matches Python now proves IPC-composed NUCLEUS
patterns match direct Rust execution.

### Socket Layout

All primals communicate via JSON-RPC 2.0 over Unix domain sockets:

```
/run/biomeos/                    # biomeOS runtime
/run/coralreef/                  # coralReef fleet (ember, glowplug)
/run/ecoPrimals/registry.sock    # agentReagents registry
$XDG_RUNTIME_DIR/biomeos/        # user-local biomeOS
$XDG_RUNTIME_DIR/coralreef/      # user-local coralReef
```

Discovery cascade (from `niche.rs`):
1. `$BIOMEOS_SOCKET_DIR` / `$CORALREEF_RUN_DIR` env override
2. `$XDG_RUNTIME_DIR/{biomeos,coralreef}/`
3. `/run/{biomeos,coralreef}/`

### Registration Protocol

Springs register with biomeOS via two JSON-RPC calls:

```json
{"method": "lifecycle.register", "params": {"name": "hotspring", "pid": 1234, ...}}
{"method": "capability.register", "params": {"capabilities": ["physics.md", "compute.gpu", ...]}}
```

### Capability Routing (by_domain)

All functional calls route through capability domains, not primal names:

```rust
ctx.by_domain("compute")       // → toadStool
ctx.by_domain("crypto")        // → bearDog
ctx.by_domain("shader")        // → coralReef / coral-glowplug
ctx.by_domain("dag")           // → rhizoCrypt
ctx.by_domain("ledger")        // → loamSpine
ctx.by_domain("attribution")   // → sweetGrass
```

`NucleusContext::detect()` discovers alive primals at runtime via socket probing
and `health.liveness` / `capability.list` RPC calls. Named accessors
(`toadstool()`, `beardog()`, etc.) have been removed — all code uses `by_domain()`.

### Deploy Graphs

NUCLEUS deployment via biomeOS uses TOML deploy graphs:

```
biomeos deploy --graph graphs/hotspring_qcd_deploy.toml
```

7 domain-specific graphs exist in `hotSpring/graphs/`. These define primal
composition (Tower/Node/Nest/Full), bonding policy, and spawn order.

### Neural API Integration

hotSpring resolves Neural API sockets via:

```rust
fn resolve_neural_api_socket(family_id: &str) -> PathBuf {
    // $NEURAL_API_SOCKET env override, or
    // {socket_dir}/neural-api-{family_id}.sock
}
```

Springs bind to Neural API for LLM/AI tool integration via the `mcp_tools.rs`
module (5 MCP tool definitions for physics/compute operations).

---

## 4. Hardware Interaction Lessons

### GPU Sovereignty Stack

```
Application (hotSpring physics)
  └─ toadStool / barraCuda (WGSL shaders, GPU dispatch)
      └─ coralReef (sovereign driver stack)
          ├─ coral-glowplug (fleet orchestrator, VFIO management)
          ├─ coral-ember (per-device lifecycle, firmware intermediary)
          └─ coral-driver (BAR0 mmap, MMIO, Falcon boot, SovereignInit)
```

### Key Hardware Findings

| Finding | Implication for Primals |
|---------|----------------------|
| `reset_method=none` preserves GPU VRAM across driver swap | coralReef must set this before VFIO rebind |
| GV100 PMU firmware absent from linux-firmware | SEC2 ACR path blocked; DRM firmware-agnostic path is the solution |
| K80 PCIe link dead after VFIO unbind (PLX D3cold) | Physical power cycle required; automated via ember keepalive |
| RTX 5060 requires QMD v5.0 + SM120 ISA | coralReef sovereign compiler needs per-generation profiles |
| nvidia-470 extracts firmware init sequences | agentReagents VM isolation is the capture method |
| Ember fork-isolation prevents host lockup | All MMIO must go through forked child process |
| PRAMIN access can lock GPU hard | coral-driver must use guarded_sysfs_read with timeout |

### For coralReef Team

- `VfioPassthrough` in benchScale now has full `qemu_properties` support for
  QEMU commandline injection. Use `x-no-mmap: "on"` for NVIDIA GPUs that need
  it.
- `AttachMode::Cold` is the only mode that supports `<qemu:commandline>` injection.
- The diesel engine architecture (glowplug → ember → driver) is validated across
  3 GPU generations. The hierarchical model scales.

---

## 5. Gaps for Upstream Teams

### primalPsing Audit Items

1. **benchScale `config_legacy` module**: Still exists for backward compat but
   all live code uses `BenchScaleConfig`. Safe to remove the module if no external
   consumers remain.
2. **benchScale `PciPassthroughDevice`**: Legacy type in `config_legacy`. All
   internal code uses `VfioPassthrough`. Check if any external consumer references
   it via `benchscale::PciPassthroughDevice`.
3. **agentReagents verification**: Package verification is currently APT-centric
   (dpkg-query, dpkg -l, apt-cache). RPM verification paths exist but are less
   tested. Fedora/RHEL manifests should be validated.
4. **hotSpring `PRIMAL_ALIASES`**: One alias remains
   (`("coralreef", &["coral-glowplug"])`). This maps the service name to the
   socket name. Verify this is still the canonical naming.

### For Springs Teams (Absorption Targets)

1. **Verification module pattern**: `verification/{types,package,mod}.rs` is a
   clean pattern for any spring that needs to verify VM builds. Extract to a
   shared crate if multiple springs need it.
2. **Suite inference**: `ImageBuilder::infer_suite()` maps base image filenames
   to codenames (noble, jammy, bookworm, etc.). Extend for new distros.
3. **BenchScaleConfig**: The config system supports YAML files, env overrides,
   and runtime discovery. Springs should use this rather than rolling their own
   config for VM operations.

### For biomeOS Team

1. **Registration protocol** is stable. Springs send `lifecycle.register` +
   `capability.register` on startup. biomeOS should validate the capability
   schema.
2. **Socket directory discovery** cascade is implemented identically in hotSpring
   and would benefit from a shared `biomeos-discovery` crate.
3. **Neural API socket naming** (`neural-api-{family_id}.sock`) should be
   documented in the Neural API spec as the canonical convention.

---

## 6. What's Archive-Ready

### Already Archived
- `experiments/archive/` — experiments 001-143
- `scripts/archive/` — superseded deploy, capture, and raw sysfs scripts
- `barracuda/src/bin/_fossilized/` — 30 historical binaries (not in Cargo.toml)

### Should Be Archived
- `wateringHole/warm_handoff.sh` — marked DEPRECATED, supersedes by coralctl
- `scripts/boot/plx-keepalive.sh` — marked DEPRECATED, superseded by coral-ember
- `scripts/lab/titanv_nvidia470_warm_handoff.sh` — marked DEPRECATED, prefers
  benchScale VM path

### Clean (Not Archivable)
- `scripts/livepatch/` — active livepatch module source (`.c`, `Makefile`, `Kbuild`)
  plus build artifacts (`.ko`, `.mod.c`, `.cmd`, `Module.symvers`). Build artifacts
  should be gitignored, not archived.
- `data/firmware/` — untracked binary firmware extracts. Should remain untracked
  (gitignored). Not for archiving — these are operational artifacts.

---

## 7. Ecosystem State Summary

```
ecoPrimals/
├── infra/
│   ├── benchScale     ✅ MATURE — pure Rust VM lifecycle, BenchScaleConfig, russh
│   ├── agentReagents  ✅ MATURE — manifest-driven image builder, generic states
│   └── wateringHole   📋 GUIDANCE — ecosystem standards, handoffs, fossil record
│
├── primals/           (13 primals — toadStool thru songBird)
│   └── biomeOS        🔧 INTEGRATION — Neural API, registry, deploy graphs
│
└── springs/           (8 springs — hotSpring thru wetSpring)
    └── hotSpring      ✅ L6 CERTIFIED — 189 experiments, NUCLEUS composition proven
```

benchScale and agentReagents are ready to be recognized as mature infra
components alongside wateringHole. They provide the VM-based experimentation
layer that hotSpring's sovereign GPU work depends on.

---

*This handoff was generated during the hotSpring deep-debt evolution sprint
(May 2026). All changes have been committed and pushed to benchScale, agentReagents,
and hotSpring repositories.*
