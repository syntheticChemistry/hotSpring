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

## Large Files

Large binary artifacts (frozen .ko modules, full mmiotrace logs) live in
`/var/lib/toadstool/reagents/` at runtime and are NOT committed to the repo.
Only small JSON manifests, recipes, and metadata are tracked here.
