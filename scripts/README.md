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
| `deploy_glowplug.sh` | Deploy GlowPlug systemd service | Yes |
| `deploy_ember_first_time.sh` | First-time Ember setup | Yes |
| `clone-repos.sh` | Clone project repositories | Yes |
| `capture_multi_backend.sh` | Multi-backend register capture | Read-only |
| `distill_oracle_recipe.sh` | Oracle recipe distillation | Read-only |
| `regenerate-all.sh` | Full project regeneration | Yes |
| `boot/*.sh` | Boot-time setup scripts | Yes |

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

## Adding New Scripts

1. Driver transitions: use `coralctl swap <BDF> <target>`
2. Register reads: BAR0 mmap or VFIO test harness (safe, no writes)
3. Power management: `gpu-ctl d0 <BDF>` (power pinning only)
4. Never write directly to `driver_override`, `bind`, `unbind`, or `remove`
