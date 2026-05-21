# scripts/ — hotSpring Operational Scripts

## Safety Policy

**All GPU driver transitions (bind, unbind, swap) MUST go through `toadstool device`
(toadStool daemon).** Raw sysfs writes to `driver_override`, `bind`, `unbind`,
or `drivers_probe` are prohibited in new scripts.

## Active Scripts

| Script | Description | Safe to run? |
|--------|-------------|-------------|
| `regenerate-all.sh` | Full project regeneration | Yes |
| `boot/*.sh` | Boot-time setup scripts | Yes |
| ~~`archive/k80_warm_catch.sh`~~ | **ARCHIVED** → `toadstool device warm-catch <BDF> --memory-type gddr5` | Replaced by pure Rust |
| ~~`archive/titanv_warm_handoff.sh`~~ | **ARCHIVED** → `toadstool device warm-catch <BDF> --memory-type hbm2` | Replaced by pure Rust |
| ~~`archive/patch_nouveau_teardown.py`~~ | **ARCHIVED** → `toadstool_cylinder::tools::elf_patcher` | Replaced by pure Rust |

| Script | Description | Safe to run? |
|--------|-------------|-------------|
| `../tools/nucleus_composition_lib.sh` | NUCLEUS composition library (repo root `tools/`, from primalSpring Phase 46). 41 functions: discovery, transport, DAG, ledger, braids, petalTongue, sensor streams. Sourced by `hotspring_composition.sh`. | Yes |

| `archive/titan-v-module-swap.sh` | **Archived.** Superseded by benchScale VM isolation + pure Rust warm-catch. | — |

## Deployment

Deployment is handled by **plasmidBin** (ecoBin deployment):

```bash
cd infra/plasmidBin
./fetch.sh --primal toadstool
sudo install -m 755 primals/toadstool /usr/local/bin/toadstool
```

Or via the local install script:

```bash
sudo ./scripts/boot/install-glowplug.sh  # Uses plasmidBin, falls back to cargo build
```

Systemd units are at `scripts/boot/toadstool-ember.service` and
`scripts/boot/toadstool-glowplug.service`.

## Archived Scripts (scripts/archive/) — Fossil Record

**Archival started 2026-03-25; supplemented through May 2026.** All scripts in `archive/` are superseded by `toadstool device`
commands. They are preserved as fossil record of the evolution from manual
scripting to daemon-managed GPU lifecycle.

| Archived Script | Replaced By |
|-----------------|-------------|
| `build_nvidia_oracle.sh` | `agentReagents` VM image build (oracle module coexistence) |
| `distill_oracle_recipe.sh` | `coralctl oracle distill` (mmiotrace distillation) |
| `read_bar0_regs.py` | `toadstool device mmio read <BDF> <offset>` |
| `read_bar0_deep.py` | `toadstool device probe <BDF>` |
| `test_pramin.py` | `toadstool device vram-probe <BDF>` |
| `capture_gr_registers.py` | `toadstool device snapshot save <BDF>` |
| `compare_gr_state.py` | `toadstool device snapshot diff <BDF>` |
| `compare_snapshots.py` | `toadstool device snapshot diff <BDF>` |
| `exp070_backend_matrix.sh` | `toadstool device experiment sweep <BDF>` |
| `bind-titanv-vfio.sh` | `toadstool device swap <BDF> vfio` |
| `unbind-titanv-vfio.sh` | `toadstool device swap <BDF> unbound` |
| `rebind_titanv_*.sh` | `toadstool device swap <BDF> <target>` |
| `warm_and_test.sh` | `toadstool device warm-catch <BDF>` |
| `k80_nouveau_post.sh` | `toadstool device warm-catch <BDF> --memory-type gddr5` |
| `k80_warm_catch.sh` | `toadstool device warm-catch <BDF> --memory-type gddr5` |
| `titanv_warm_handoff.sh` | `toadstool device warm-catch <BDF> --memory-type hbm2` |
| `patch_nouveau_teardown.py` | `toadstool_cylinder::tools::elf_patcher::KmodPatcher` (pure Rust) |

## Adding New Scripts

1. Driver transitions: use `toadstool device swap <BDF> <target>`
2. Register reads: BAR0 mmap or VFIO test harness (safe, no writes)
3. Power management: `gpu-ctl d0 <BDF>` (power pinning only)
