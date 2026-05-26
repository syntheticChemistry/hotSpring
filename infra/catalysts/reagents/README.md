# Reagent Library

Versioned reagent captures — firmware "chemical agents" extracted from vendor
drivers while they are loaded. Each directory contains a `manifest.json` plus
artifact files (firmware blobs, BAR0 snapshots, mmiotrace recipes).

## Directory Layout

```
reagents/
  gv100_nvidia47025602_k6.17.9/
    manifest.json           # ReagentManifest
    bar0_snapshot.json      # Domain-scoped alive registers
    bar0_replay.json        # GrInitSequence
    patch_set.json          # PatchSet recipe
    firmware/
      fecs_inst.bin         # from linux-firmware or extraction
      fecs_data.bin
      gpccs_inst.bin
      gpccs_data.bin
      pmu_dmem.bin          # from Exp 168
      acr_boot_sequence.json
    mmiotrace/
      nvidia535_recipe.json         # full distilled recipe
      nvidia535_recipe_acr_subset.json  # ACR-domain subset
```

## Capture Pipeline

Reagents are captured via `sovereign.reagent_capture` RPC or by running
`toadstool device oracle reagent-capture --bdf <BDF>`.

## Fossil Scripts

Two Python scripts are preserved as fossil record — **DO NOT EXECUTE**:

- `acr_sovereign_boot.py` — v1 ACR boot script using the flawed ENGCTL
  "HS-unlock" approach that irreversibly kills falcon CPU execution.
  Superseded by Rust `exp224_pmu_acr_catalyst` (barracuda). See Exp 223.
- `post_reboot_acr_boot.py` — Post-reboot companion to the above.
  Same flawed approach. Superseded.

The `gv100_acr_catalyst.json` recipe contains `hs_unlock` steps from the
v1 approach — these are historical/flawed (ENGCTL is an irreversible
engine reset, not an unlock). The correct boot path uses the Boot Falcon
(NVDEC/SEC2) for ACR, which is what toadStool's `sovereign.init` implements.

## Large Files

Large binary artifacts (frozen .ko modules, full mmiotrace logs) live in
`/var/lib/toadstool/reagents/` at runtime and are NOT committed to the repo.
Only small JSON manifests, recipes, and metadata are tracked here.
