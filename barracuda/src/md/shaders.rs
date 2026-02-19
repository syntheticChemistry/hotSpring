// SPDX-License-Identifier: AGPL-3.0-only

//! MD WGSL shader sources for production GPU simulation.
//!
//! Production simulation (`md/simulation.rs`) uses GPU-resident dispatch
//! with these local shader sources. Force shaders are compiled via
//! [`crate::gpu::GpuF64::create_pipeline_f64`] for driver-aware NVK patching.
//!
//! Barracuda ops (`barracuda::ops::md::*`) provide the same physics through
//! a Tensor API used by validation binaries. The cell-list force shader
//! remains hotSpring-local (barracuda's cell-list path is CPU-only).
//!
//! VV, half-kick, Berendsen, and KE shaders use no `exp()`/`log()` on f64
//! and compile safely on all drivers via [`crate::gpu::GpuF64::create_pipeline`].

/// Patches the barracuda math_f64 preamble for GPUs without full f64 support.
///
/// Strips the gamma section (`lanczos_core_f64` + `gamma_f64`) and replaces
/// large exponent values (1e308 → 1e38) to avoid overflow in WGSL.
///
/// # Upstream precision evolution (Feb 16 2026, commit `0c477306`)
///
/// ToadStool's math_f64.wgsl now uses the `(zero + literal)` pattern for full
/// f64 constant precision — `let zero = x - x; let c = zero + 0.33333...;`
/// instead of `let c = f64(0.33333...);` which truncates through f32.
/// This fixed `log_f64()` from ~1e-3 to ~1e-15 precision and improved all
/// transcendentals. The patched output inherits these improvements.
///
/// hotSpring MD shaders use native WGSL builtins (SHADER_F64) and do not
/// need this preamble. It is only used by `celllist_diag` and `f64_builtin_test`
/// for software-vs-native comparison testing.
pub fn patch_math_f64_preamble(preamble: &str) -> String {
    let mut result = String::with_capacity(preamble.len());
    let mut skip = false;
    for line in preamble.lines() {
        if line.contains("GAMMA FUNCTION")
            || line.starts_with("fn gamma_f64")
            || line.starts_with("fn lanczos_core_f64")
        {
            skip = true;
        }
        if skip {
            // Stop at the next section header (e.g. "// ==== ERROR FUNCTION")
            if line.starts_with("// ====") || line.starts_with("// ==========") {
                skip = false;
                result.push_str("// (gamma_f64 stripped)\n");
                // Output the section header so structure is preserved
                result.push_str(line);
                result.push('\n');
            }
            continue;
        }
        let patched = line.replace("1e308", "1e38").replace("-1e308", "-1e38");
        result.push_str(&patched);
        result.push('\n');
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// Yukawa All-Pairs Force Kernel (f64)
// ═══════════════════════════════════════════════════════════════════
//
// Computes pairwise Yukawa forces with PBC minimum-image convention.
// Each thread handles one particle, loops over all others.
// O(N²) — suitable for N ≤ ~5,000.  Cell-list version planned for N > 5k.
//
// Physics:
//   U(r) = prefactor * exp(-kappa * r) / r
//   F = -dU/dr = prefactor * exp(-kappa*r) * (1 + kappa*r) / r² * r_hat
//
// The shader also accumulates per-particle potential energy (half-counted).

pub const SHADER_YUKAWA_FORCE: &str = include_str!("shaders/yukawa_force_f64.wgsl");

// ═══════════════════════════════════════════════════════════════════
// Velocity-Verlet Half-Kick + Drift + PBC Wrap (f64)
// ═══════════════════════════════════════════════════════════════════

pub const SHADER_VV_KICK_DRIFT: &str = include_str!("shaders/vv_kick_drift_f64.wgsl");

// ═══════════════════════════════════════════════════════════════════
// Velocity-Verlet Second Half-Kick (f64)
// ═══════════════════════════════════════════════════════════════════
//
// After forces are recomputed with new positions, apply the second
// half-kick: v += 0.5 * dt * a_new

pub const SHADER_VV_HALF_KICK: &str = r"
// VV second half-kick (f64)
//
// Bindings:
//   0: velocities   [N*3] f64, read-write  — updated in-place
//   1: forces       [N*3] f64, read        — NEW forces after drift
//   2: params       [4]   f64, read        — [n, dt, mass, _]

@group(0) @binding(0) var<storage, read_write> velocities: array<f64>;
@group(0) @binding(1) var<storage, read> forces: array<f64>;
@group(0) @binding(2) var<storage, read> params: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let dt   = params[1];
    let mass = params[2];

    let inv_m   = 1.0 / mass;
    let half_dt = 0.5 * dt;

    velocities[i * 3u]      = velocities[i * 3u]      + half_dt * forces[i * 3u]      * inv_m;
    velocities[i * 3u + 1u] = velocities[i * 3u + 1u] + half_dt * forces[i * 3u + 1u] * inv_m;
    velocities[i * 3u + 2u] = velocities[i * 3u + 2u] + half_dt * forces[i * 3u + 2u] * inv_m;
}
";

// ═══════════════════════════════════════════════════════════════════
// Berendsen Thermostat (f64)
// ═══════════════════════════════════════════════════════════════════
//
// Rescales velocities: v *= sqrt(1 + (dt/tau) * (T_target/T_current - 1))
// Applied once per step during equilibration.

pub const SHADER_BERENDSEN: &str = r"
// Berendsen velocity rescaling (f64)
//
// Bindings:
//   0: velocities [N*3] f64, read-write
//   1: params     [4]   f64, read  — [n, scale_factor, _, _]

@group(0) @binding(0) var<storage, read_write> velocities: array<f64>;
@group(0) @binding(1) var<storage, read> params: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let scale = params[1];

    velocities[i * 3u]      = velocities[i * 3u]      * scale;
    velocities[i * 3u + 1u] = velocities[i * 3u + 1u] * scale;
    velocities[i * 3u + 2u] = velocities[i * 3u + 2u] * scale;
}
";

// ═══════════════════════════════════════════════════════════════════
// Kinetic Energy Reduction (f64)
// ═══════════════════════════════════════════════════════════════════
//
// Computes per-particle KE = 0.5 * m * v² for temperature calculation.

pub const SHADER_KINETIC_ENERGY: &str = r"
// Per-particle kinetic energy (f64)
//
// Bindings:
//   0: velocities [N*3] f64, read
//   1: ke_buf     [N]   f64, write
//   2: params     [4]   f64, read  — [n, mass, _, _]

@group(0) @binding(0) var<storage, read> velocities: array<f64>;
@group(0) @binding(1) var<storage, read_write> ke_buf: array<f64>;
@group(0) @binding(2) var<storage, read> params: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let mass = params[1];
    let vx = velocities[i * 3u];
    let vy = velocities[i * 3u + 1u];
    let vz = velocities[i * 3u + 2u];

    ke_buf[i] = 0.5 * mass * (vx * vx + vy * vy + vz * vz);
}
";

// ═══════════════════════════════════════════════════════════════════
// Yukawa Cell-List Force Kernel (f64)
// ═══════════════════════════════════════════════════════════════════

pub const SHADER_YUKAWA_FORCE_CELLLIST: &str =
    include_str!("shaders/yukawa_force_celllist_f64.wgsl");

// ═══════════════════════════════════════════════════════════════════
// Yukawa Cell-List Force Kernel v2 (f64) — flat neighbor loop
// ═══════════════════════════════════════════════════════════════════

pub const SHADER_YUKAWA_FORCE_CELLLIST_V2: &str =
    include_str!("shaders/yukawa_force_celllist_v2_f64.wgsl");

// ═══════════════════════════════════════════════════════════════════
// RDF Histogram Kernel (f64)
// ═══════════════════════════════════════════════════════════════════

pub const SHADER_RDF_HISTOGRAM: &str = include_str!("shaders/rdf_histogram_f64.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    const SHADER_CONSTANTS: &[(&str, &str)] = &[
        ("SHADER_YUKAWA_FORCE", SHADER_YUKAWA_FORCE),
        ("SHADER_VV_KICK_DRIFT", SHADER_VV_KICK_DRIFT),
        ("SHADER_VV_HALF_KICK", SHADER_VV_HALF_KICK),
        ("SHADER_BERENDSEN", SHADER_BERENDSEN),
        ("SHADER_KINETIC_ENERGY", SHADER_KINETIC_ENERGY),
        ("SHADER_YUKAWA_FORCE_CELLLIST", SHADER_YUKAWA_FORCE_CELLLIST),
        (
            "SHADER_YUKAWA_FORCE_CELLLIST_V2",
            SHADER_YUKAWA_FORCE_CELLLIST_V2,
        ),
        ("SHADER_RDF_HISTOGRAM", SHADER_RDF_HISTOGRAM),
    ];

    #[test]
    fn each_shader_constant_non_empty() {
        for (name, shader) in SHADER_CONSTANTS {
            assert!(!shader.is_empty(), "{name} must not be empty");
            assert!(shader.len() > 100, "{name} should be substantial");
        }
    }

    #[test]
    fn each_shader_has_compute_and_workgroup_size() {
        for (name, shader) in SHADER_CONSTANTS {
            assert!(shader.contains("@compute"), "{name} must contain @compute");
            assert!(
                shader.contains("@workgroup_size"),
                "{name} must contain @workgroup_size"
            );
        }
    }

    #[test]
    fn each_shader_has_binding_declarations() {
        for (name, shader) in SHADER_CONSTANTS {
            assert!(
                shader.contains("@group("),
                "{name} must contain @group binding"
            );
            assert!(
                shader.contains("@binding("),
                "{name} must contain @binding declaration"
            );
        }
    }

    #[test]
    fn patch_math_f64_preamble_removes_gamma_section() {
        let preamble = r"
// GAMMA FUNCTION
fn gamma_f64(x: f64) -> f64 {
    return 1e308;
}

// ============================================================================
// ERROR FUNCTION (erf)
// ============================================================================
fn erf_f64(x: f64) -> f64 { return 0.0; }
";
        let patched = patch_math_f64_preamble(preamble);
        assert!(!patched.contains("fn gamma_f64"));
        assert!(patched.contains("(gamma_f64 stripped)"));
        assert!(patched.contains("fn erf_f64"));
    }

    #[test]
    fn patch_math_f64_preamble_replaces_1e308_overflow() {
        let preamble = "let x = 1e308;\nlet y = -1e308;";
        let patched = patch_math_f64_preamble(preamble);
        assert!(!patched.contains("1e308"));
        assert!(!patched.contains("-1e308"));
        assert!(patched.contains("1e38"));
        assert!(patched.contains("-1e38"));
    }

    #[test]
    fn patch_math_f64_preamble_preserves_unrelated_content() {
        let preamble = "let a = 1.0;\nlet b = 2.0;";
        let patched = patch_math_f64_preamble(preamble);
        assert!(patched.contains("let a = 1.0"));
        assert!(patched.contains("let b = 2.0"));
    }
}
