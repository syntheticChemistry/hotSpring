# hotSpring Scripts

## Safety Policy

**All GPU driver transitions (bind, unbind, swap) MUST go through `coralctl`
(GlowPlug/Ember).** Raw sysfs writes to `driver_override`, `bind`, `unbind`,
or `drivers_probe` are prohibited in new scripts.

Why: Raw sysfs writes bypass Ember's safety mechanisms:
- **Atomic rollback** — if bind fails, Ember re-binds the previous driver
- **D-state watchdog** — process-isolated sysfs writes with 10s timeout
- **driver_override cleanup** — always cleared after bind attempts
- **Pre-swap validation** — module existence check, power state, config space
- **IOMMU group handling** — symmetric bind/release for multi-device groups

Gap 13 (Exp 089c) demonstrated that raw sysfs scripts can cause D-state hangs
requiring forced power-off.

## Active Scripts

| Script | Purpose | Safe? |
|--------|---------|-------|
| `build_nvidia_oracle.sh` | Build nvidia_oracle.ko from source | Yes (build only) |
| `ci-coverage-gate.sh` | CI coverage gate (thresholds / reporting) | Yes |
| `clone-repos.sh` | Clone project repositories | Yes |
| `coverage.sh` | Local coverage workflow | Yes |
| `distill_oracle_recipe.sh` | Oracle recipe distillation via toadStool hw-learn | Read-only |
| `harvest-ecobin.sh` | Harvest ecobin artifacts | Yes |
| `regenerate-all.sh` | Full project regeneration | Yes |
| `boot/*.sh` | Boot-time setup scripts | Yes |

Other non-archived helpers in `scripts/` (e.g. `build-container.sh`, `build-guidestone.sh`, `download-data.sh`, `prepare-usb.sh`, `setup-envs.sh`, `validate-guidestone-multi.sh`) are also active as needed for builds and lab setup.

All Python lab analysis scripts and the `titan_timing_attack.sh` experiment script
have been archived — their functionality is now available via `coralctl` subcommands
(see archive table below).

### Deployment

Deploy scripts (`deploy_glowplug.sh`, `deploy_ember_first_time.sh`, etc.) have been
archived — deployment is now handled by `coralctl deploy` or the systemd service
units shipped in the `coral-glowplug` crate (`coral-glowplug.service`,
`coral-ember.service`).

## Archived Scripts (scripts/archive/) — Fossil Record

**Archived 2026-03-25.** All scripts in `archive/` are superseded by `coralctl`
commands. They are preserved as fossil record of the evolution from manual
scripting to daemon-managed GPU lifecycle.

| Archived Script | Replaced By |
|-----------------|-------------|
| `read_bar0_regs.py` | `coralctl mmio read <BDF> <offset>` |
| `read_bar0_deep.py` | `coralctl probe <BDF>` |
| `test_pramin.py` | `coralctl vram-probe <BDF>` |
| `capture_gr_registers.py` | `coralctl snapshot save <BDF>` |
| `compare_gr_state.py` | `coralctl snapshot diff <BDF>` |
| `compare_snapshots.py` | `coralctl snapshot diff <BDF>` |
| `exp070_backend_matrix.sh` | `coralctl experiment sweep <BDF>` |
| `exp086_run_matrix.sh` | `coralctl experiment sweep <BDF>` |
| `exp086_falcon_profiler.py` | `coralctl probe <BDF>` |
| `exp086_analyze.py` | `coralctl journal stats` |
| `exp089_sec2_cmdq_probe.py` | Absorbed into coral-driver diagnostic module |
| `cross_card_oracle.py` | `coralctl experiment sweep` + journal |
| `bind-titanv-vfio.sh` | `coralctl swap <BDF> vfio` |
| `unbind-titanv-vfio.sh` | `coralctl swap <BDF> unbound` |
| `rebind_titanv_*.sh` | `coralctl swap <BDF> <target>` |
| `vfio-bind-quick.sh` | `coralctl swap <BDF> vfio` |
| `setup_dual_titanv.sh` | `glowplug.toml` config + systemd |
| `warm_and_test.sh` | `coralctl warm-fecs <BDF>` |
| `capture_nouveau_mmiotrace*.sh` | `coralctl swap <BDF> nouveau --trace` |
| `capture_mmiotrace_oracle.sh` | `coralctl experiment sweep --trace` |
| `exp084_b1b4_test.sh` | Absorbed into coral-driver test suite |
| `exp089b_warm_swap_test.sh` | `coralctl warm-fecs <BDF>` |
| `capture_multi_backend.sh` | `coralctl swap <BDF> <target> --trace` |
| `titan_timing_attack.sh` | `coralctl warm-fecs <BDF>` (Exp 127 complete) |
| `bar0_read.py` | `coralctl mmio read <BDF> <offset>` |
| `parse_mmiotrace.py` | `coralctl trace-parse <file>` |
| `replay_devinit.py` | `coralctl devinit replay <BDF>` |
| `generate_titanv_recipe.py` | `coralctl trace-parse --recipe-json <file>` |
| `extract_devinit.py` | `coralctl devinit replay <BDF>` |
| `apply_recipe.py` | `coralctl oracle apply <BDF> <recipe>` |
| `run_reagent_capture.sh` | Historical — preserved under `scripts/archive/` (agentReagents VM capture) |

## Adding New Scripts

1. Driver transitions: use `coralctl swap <BDF> <target>`
2. Register reads: BAR0 mmap or VFIO test harness (safe, no writes)
3. Power management: `gpu-ctl d0 <BDF>` (power pinning only)
4. Never write directly to `driver_override`, `bind`, `unbind`, or `remove`
