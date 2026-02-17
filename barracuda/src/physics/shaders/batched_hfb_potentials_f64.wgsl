// SPDX-License-Identifier: AGPL-3.0-only
// Batched HFB Potentials (f64) — GPU Shader
//
// Computes Skyrme + Coulomb + effective-mass potentials for a batch of nuclei.
// Pre-computed rho^alpha and rho^(alpha-1) passed from CPU for full f64 precision.
// Coulomb exchange uses Newton-refined cube root for f64 accuracy.

struct PotentialDims {
    nr: u32,
    batch_size: u32,
}

@group(0) @binding(0) var<uniform> dims: PotentialDims;
@group(0) @binding(1) var<storage, read> rho_p_batch: array<f64>;
@group(0) @binding(2) var<storage, read> rho_n_batch: array<f64>;
@group(0) @binding(3) var<storage, read_write> u_total_batch: array<f64>;
@group(0) @binding(4) var<storage, read_write> u_total_n_batch: array<f64>;
@group(0) @binding(5) var<storage, read> r_grid: array<f64>;
@group(0) @binding(6) var<storage, read_write> f_q_p_batch: array<f64>;
@group(0) @binding(7) var<storage, read_write> f_q_n_batch: array<f64>;
@group(0) @binding(8) var<storage, read_write> charge_enclosed: array<f64>;
@group(0) @binding(9) var<storage, read_write> phi_outer: array<f64>;
@group(0) @binding(10) var<storage, read> sky_params: array<f64>;
// Pre-computed on CPU with full f64 precision:
@group(0) @binding(11) var<storage, read> rho_alpha_batch: array<f64>;
@group(0) @binding(12) var<storage, read> rho_alpha_m1_batch: array<f64>;

const E2: f64 = 1.4399764;
const PI_VAL: f64 = 3.141592653589793;

// Newton-refined cube root in f64 (f32 seed + 2 Newton iterations)
// TODO(evolution): Replace with barracuda::shaders::math::math_f64.wgsl canonical cbrt_f64
// via ShaderTemplate::math_f64_subset(["cbrt_f64"]) preamble injection.
fn cbrt_f64(x: f64) -> f64 {
    if (x <= f64(0.0)) { return f64(0.0); }
    var y = f64(pow(f32(x), f32(0.333333343)));
    y = (f64(2.0) * y + x / (y * y)) / f64(3.0);
    y = (f64(2.0) * y + x / (y * y)) / f64(3.0);
    return y;
}

// ─── Skyrme potential ────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn compute_skyrme(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let batch_idx = global_id.y;
    let nr = dims.nr;

    if (k >= nr || batch_idx >= dims.batch_size) {
        return;
    }

    let t0 = sky_params[0];
    let t3 = sky_params[1];
    let x0 = sky_params[2];
    let x3 = sky_params[3];
    let alpha = sky_params[4];

    let idx = batch_idx * nr + k;
    let rp = rho_p_batch[idx];
    let rn = rho_n_batch[idx];
    let rho = rp + rn;

    // Pre-computed on CPU with full f64 precision
    let rho_alpha = rho_alpha_batch[idx];
    let rho_alpha_m1 = rho_alpha_m1_batch[idx];
    let sum_rho2 = rp * rp + rn * rn;

    // Proton Skyrme
    let u_t0_p = t0 * ((f64(1.0) + x0 / f64(2.0)) * rho - (f64(0.5) + x0) * rp);
    let u_t3_p = (t3 / f64(12.0))
        * ((f64(1.0) + x3 / f64(2.0)) * (alpha + f64(2.0)) * rho_alpha * rho
            - (f64(0.5) + x3) * (alpha * rho_alpha_m1 * sum_rho2
                + f64(2.0) * rho_alpha * rp));
    u_total_batch[idx] = u_t0_p + u_t3_p;

    // Neutron Skyrme
    let u_t0_n = t0 * ((f64(1.0) + x0 / f64(2.0)) * rho - (f64(0.5) + x0) * rn);
    let u_t3_n = (t3 / f64(12.0))
        * ((f64(1.0) + x3 / f64(2.0)) * (alpha + f64(2.0)) * rho_alpha * rho
            - (f64(0.5) + x3) * (alpha * rho_alpha_m1 * sum_rho2
                + f64(2.0) * rho_alpha * rn));
    u_total_n_batch[idx] = u_t0_n + u_t3_n;
}

// ─── Coulomb: forward cumulative sum (charge enclosed) ───────────
@compute @workgroup_size(256, 1, 1)
fn compute_coulomb_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let nr = dims.nr;
    let dr = sky_params[5];

    if (batch_idx >= dims.batch_size) {
        return;
    }

    let base = batch_idx * nr;
    var cumsum: f64 = f64(0.0);
    for (var k = 0u; k < nr; k++) {
        let rk = r_grid[base + k];
        cumsum = cumsum + rho_p_batch[base + k] * f64(4.0) * PI_VAL * rk * rk * dr;
        charge_enclosed[base + k] = cumsum;
    }
}

// ─── Coulomb: backward cumulative sum (outer potential) ──────────
@compute @workgroup_size(256, 1, 1)
fn compute_coulomb_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let nr = dims.nr;
    let dr = sky_params[5];

    if (batch_idx >= dims.batch_size) {
        return;
    }

    let base = batch_idx * nr;
    var cumsum_rev: f64 = f64(0.0);
    for (var k_rev = 0u; k_rev < nr; k_rev++) {
        let k = nr - 1u - k_rev;
        let rk = r_grid[base + k];
        cumsum_rev = cumsum_rev + rho_p_batch[base + k] * f64(4.0) * PI_VAL * rk * dr;
        phi_outer[base + k] = cumsum_rev;
    }
}

// ─── Finalize proton potential: add Coulomb to Skyrme ────────────
@compute @workgroup_size(256, 1, 1)
fn finalize_proton_potential(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let batch_idx = global_id.y;
    let nr = dims.nr;

    if (k >= nr || batch_idx >= dims.batch_size) {
        return;
    }

    let idx = batch_idx * nr + k;
    let rk = max(r_grid[idx], f64(1e-10));
    let rp = max(rho_p_batch[idx], f64(0.0));

    let v_c = E2 * (charge_enclosed[idx] / rk + phi_outer[idx]);
    let coeff = -E2 * cbrt_f64(f64(3.0) / PI_VAL);
    let v_cx = coeff * cbrt_f64(rp);

    u_total_batch[idx] = u_total_batch[idx] + v_c + v_cx;
}

// ─── Effective mass function f_q(r) ──────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn compute_f_q(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let batch_idx = global_id.y;
    let nr = dims.nr;

    if (k >= nr || batch_idx >= dims.batch_size) {
        return;
    }

    let idx = batch_idx * nr + k;
    let c0t = sky_params[6];
    let c1n = sky_params[7];
    let hbar2_2m = sky_params[8];

    let rp = rho_p_batch[idx];
    let rn = rho_n_batch[idx];

    let f_p = max(hbar2_2m + c0t * (rp + rn) - c1n * rp, hbar2_2m * f64(0.3));
    f_q_p_batch[idx] = f_p;

    let f_n = max(hbar2_2m + c0t * (rp + rn) - c1n * rn, hbar2_2m * f64(0.3));
    f_q_n_batch[idx] = f_n;
}
