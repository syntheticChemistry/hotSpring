# Workspace Migration Handoff — 2026-03-28

## What Happened

The `ecoPrimals` workspace was reorganized from a flat top-level layout to the
canonical structure defined in `infra/wateringHole/WORKSPACE_LAYOUT.md`:

```
ecoPrimals/
  primals/       barraCuda, bearDog, bingoCube, biomeOS, coralReef, loamSpine,
                 nestGate, petalTongue, rhizoCrypt, songBird, squirrel,
                 sweetGrass, toadStool
  springs/       airSpring, groundSpring, healthSpring, hotSpring, ludoSpring,
                 neuralSpring, primalSpring, wetSpring
  gardens/       (empty — esotericWebb, blueFish not yet cloned)
  infra/         agentReagents, benchScale, wateringHole, whitePaper
  sort-after/    envytools, fossil, lammps, results, rustChip
```

Old `phase1/`, `phase2/`, `primalTools/`, and `__cmake_systeminformation/`
directories were emptied and removed. No symlinks — clean breaks.

## What Was Fixed

### Cargo.toml Cross-Repo Path Dependencies

All `path = "../../barraCuda/..."`, `path = "../../phase1/toadStool/..."`,
`path = "../../phase2/rhizoCrypt/..."`, and `path = "../../primalTools/bingoCube/..."`
references were updated to use the new `primals/` prefix with correct depth.

Affected springs: hotSpring, airSpring, groundSpring, wetSpring, ludoSpring,
neuralSpring, primalSpring.

### hotSpring Script Paths

| File | Change |
|------|--------|
| `scripts/deploy_glowplug.sh` | `CORALREEF` → `primals/coralReef` |
| `scripts/deploy_ember_first_time.sh` | `CORALREEF` → `primals/coralReef` |
| `scripts/boot/coralreef-sudoers` | All paths updated to `springs/hotSpring/` and `primals/coralReef/` |
| `scripts/archive/exp084_b1b4_test.sh` | `CORAL_DIR`, `HOTSPRING_DIR` updated |
| `scripts/archive/bind-titanv-vfio.sh` | coralReef cd path updated |
| `scripts/archive/capture_nouveau_mmiotrace_v2.sh` | `TRACE_OUT` path updated |
| Several experiment .md files | Path references in code blocks updated |

### Sudoers (ACTION REQUIRED)

The file `scripts/boot/coralreef-sudoers` was updated but the deployed copy
at `/etc/sudoers.d/coralreef` still has old paths. Redeploy:

```bash
pkexec cp scripts/boot/coralreef-sudoers /etc/sudoers.d/coralreef
```

## GPU Cracking Resume Context

### Hardware State

- **Titan V #1** (`4b:00.0`): `vfio-pci`, health_policy=active, operational
- **Titan V #2**: removed from fleet (was `4a:00.0`)
- **K80 die #1** (`4c:00.0`): `vfio-pci`, health_policy=passive, sovereign target
- **K80 die #2** (`4d:00.0`): `vfio-pci`, health_policy=passive, oracle (swappable to nouveau)
- **RTX 5070** (`41:00.0`): `nvidia`, display GPU, untouched
- **USB controller** (`47:00.3`): shares root complex `0000:40` with K80s

### coralReef Services

- `coral-glowplug.service` and `coral-ember.service` are deployed and running
- Binary paths are at `/usr/local/bin/` (system-installed, not referencing source tree)
- Config at `/etc/coralreef/glowplug.toml`
- Source at `primals/coralReef/`

### K80 Oracle Plan (Next Steps)

1. **Patch nouveau for GK210**: kernel source was downloaded. Need to add
   `case 0x0f2` to `drivers/gpu/drm/nouveau/nvkm/engine/device/base.c`
   mapping to `nvf1_chipset` (GK110B). Build as out-of-tree module.
   Consider using `agentReagents` VM for isolation.

2. **Capture init sequence**: Bind patched nouveau to K80 die #2 (`4d:00.0`)
   with `mmiotrace` enabled. Capture full BAR0 register writes.

3. **Distill InitRecipe**: Diff cold vs warm BAR0 snapshots. Extract ordered
   register write sequence. Save captured firmware binaries.

4. **Sovereign replay**: Apply captured sequence to K80 die #1 via
   `coral-driver` `apply_bar0()` + PIO firmware load.

5. **Compute dispatch**: Once FECS boots: PFIFO channel setup + compute
   shader dispatch on K80.

### Key Discovery: PCIe Root Complex Sharing

K80 GPUs and USB controller share root complex `0000:40`. BAR0 reads on
uninitialized K80s cause PCIe completion timeouts that cascade to USB,
freezing input. The `health_policy=passive` in glowplug.toml prevents this.
Never run active health checks on K80s until they are fully initialized.

### Infrastructure Tools

- `infra/agentReagents/`: template-driven VM image builder (YAML manifests,
  cloud-init). Requires `libvirt` (not yet installed). Intended for isolating
  hazardous kernel/driver operations in VMs.
- `infra/benchScale/`: VM provisioner (libvirt, CloudInit). Sibling to
  agentReagents. Validation tool for substrate testing.
