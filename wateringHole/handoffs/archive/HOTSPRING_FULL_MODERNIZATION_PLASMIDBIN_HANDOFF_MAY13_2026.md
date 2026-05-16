# hotSpring Full Modernization Handoff — May 13, 2026

**Audience:** upstream teams — primalSpring, toadStool, barraCuda, coralReef  
**Scope:** Full modernization sprint completed May 13, 2026 (plasmidBin deployment path, coral ecosystem migration, NUCLEUS and capability routing updates).

---

## Summary

- **Full hotSpring modernization:** Removed all coral-gpu dead code; migrated coral-ember / coral-glowplug / coralctl references to toadStool equivalents throughout the codebase and tooling.
- **Compute trio deployment:** Deployed the compute trio (toadStool + barraCuda + coralReef) via **plasmidBin** ecoBins (no source-build requirement on target hosts for routine rollout).
- **NUCLEUS socket discovery:** Modernized to accept both `{name}-{family}.sock` and `{name}.sock` naming patterns so mixed primal conventions coexist during transition.
- **Capability-based routing:** Upgraded to align with `provided_capabilities.type` and `domains` arrays as emitted/consumed by current upstream surfaces.
- **toadstool-ember systemd:** Service hardened with automatic correction of socket permission issues where applicable.

---

## Validation Results

| Area | Result |
|------|--------|
| **Library tests** | **1043 / 1043** pass (0 failures) |
| **Fleet integration** | **5 / 5** pass |
| **Compute trio pipeline** | **10 / 15** pass |

**Compute trio highlights**

- All **three primals** report **ALIVE** via NUCLEUS discovery.
- **7 / 9** barrier shaders compile successfully through coralReef IPC.
- **2 shader failures:** attributed to **upstream coralReef** limitations (subgroup ops; WGSL type error on specific shader source).
- **2 dispatch submit failures:** attributed to **toadStool IPC protocol gap** — responses omit `job_id`, which hotSpring’s trio pipeline expects for async result polling.

---

## Files Changed (Summary)

| Category | Detail |
|----------|--------|
| **Deleted** | `bin_helpers/coral_sovereign/` (**5 files**, ~40 KB); **3 broken bins** (~50 KB) |
| **Updated** | ~**30** Rust sources (socket paths, documentation comments, environment variables) |
| **Updated** | **6** shell scripts (`gpu-ctl`, `hw-test`, `install-glowplug.sh`, and related tooling) |
| **Created** | `upgrade-toadstool.sh` — plasmidBin-oriented daemon upgrade path |
| **Updated** | `scripts/README.md` — full rewrite describing modern deployment |

---

## Upstream Gaps for primalSpring Audit

### toadStool

| Status | Item |
|--------|------|
| **GAP** | `compute.dispatch.submit` IPC does **not** return a `job_id` field — hotSpring’s compute trio pipeline expects it for async result polling. |
| **GAP** | Socket ownership/mode defaults (**root:root**, **0600**) — prefer group-accessible defaults or a configurable **UMask** / equivalent so non-root consumers can integrate safely. |
| **RESOLVED** | `device.list`, `device.warm_catch`, `health.liveness`, `capability.list` verified working from hotSpring’s integration angle. |

### coralReef

| Status | Item |
|--------|------|
| **GAP** | `shader.compile.wgsl` does **not** support subgroup operations (**Discriminant(20)**). |
| **GAP** | WGSL parse failure on `deformed_wavefunction_f64.wgsl` — math function **type mismatch**. |
| **RESOLVED** | **7 / 9** barrier shaders compile correctly through IPC. |

### barraCuda

| Status | Item |
|--------|------|
| **RESOLVED** | Fully clear of thermal niche compute responsibilities on the hotSpring side. |
| **Note** | Socket at `/run/user/1000/biomeos/math.sock` — NUCLEUS discovery confirmed working with current hotSpring logic. |

### NUCLEUS Discovery

- **Convention drift:** Primals use mixed socket naming (`{name}-{family}.sock`, `{name}.sock`, capability symlinks). hotSpring now tolerates both primary patterns; **upstream standardization** remains desirable.
- **System-level daemons (root):** Discovery paths should include **`/run/{primal}/biomeos/`** where relevant so root-managed services remain visible consistently.

---

## Deployment Infrastructure

- All three primals are deployed from **plasmidBin ecoBins** (**zero `cargo build`** on routine deployment hosts).
- **toadstool-ember** runs as a **systemd** service using the **plasmidBin** binary.
- **`upgrade-toadstool.sh`** implements pull / fetch / install / restart for the daemon lifecycle.
- **`install-glowplug.sh`** prefers **plasmidBin** as primary install source with **cargo** as fallback.

---

## Next Steps

1. Warm boot / cold boot **sovereign** experiments on **Titan V** and **K80**.
2. **toadStool** IPC alignment: **`job_id`** in dispatch responses and async dispatch contract clarity.
3. **coralReef:** subgroup operation support in `shader.compile.wgsl` (and related WGSL front-end).
4. **Cross-primal:** standardize socket naming and documented discovery roots (including `/run/{primal}/biomeos/` for root services).

---

*End of handoff — May 13, 2026.*
