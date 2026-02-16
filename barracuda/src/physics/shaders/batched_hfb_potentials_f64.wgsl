// Batched HFB Potentials (f64) — GPU Shader
//
// Computes Skyrme potential, Coulomb direct, and Coulomb exchange for all
// nuclei in a batch simultaneously. Each thread handles one grid point
// for one nucleus.
//
// This replaces the CPU-side skyrme_potential(), coulomb_direct(), and
// coulomb_exchange() calls in hfb.rs.
//
// Memory layout:
//   rho_p_batch: [batch_size × nr] f64 — proton densities
//   rho_n_batch: [batch_size × nr] f64 — neutron densities
//   u_total_batch: [batch_size × nr] f64 — output total potential
//   r_grid: [nr] f64 — radial grid (shared across batch)
//   f_q_batch: [batch_size × nr] f64 — effective mass function f_q(r)
//
// Physics:
//   U_sky(r) = t0 * [(1+x0/2)*rho - (1/2+x0)*rho_q]
//            + (t3/12) * [derivative terms]
//   V_C(r)   = e² * [Z_enc(r)/r + integral_r^inf rho_p(r')*r' dr']
//   V_Cx(r)  = -e² * (3/π)^{1/3} * rho_p^{1/3}
//
// Constants (matching hfb.rs):
//   E2 = 1.4399764 (e² in MeV·fm)
//   PI = 3.141592653589793

// Parameters passed via storage buffer (f64 native, no bitcast needed)
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
// Skyrme parameters as f64 storage: [t0, t3, x0, x3, alpha, dr, c0t, c1n, hbar2_2m]
@group(0) @binding(10) var<storage, read> sky_params: array<f64>;

const E2: f64 = 1.4399764;  // e² in MeV·fm
const PI_VAL: f64 = 3.141592653589793;

// WGSL pow/exp/log are f32-only; route through f32 for transcendentals
fn pow_f64(base: f64, exponent: f64) -> f64 {
    return f64(pow(f32(base), f32(exponent)));
}
fn cbrt_f64(x: f64) -> f64 {
    return f64(pow(f32(x), f32(1.0 / 3.0)));
}

// ─── Skyrme potential ────────────────────────────────────────────
// Thread per (k, batch): computes U_sky(r_k) for one nucleus
// Dispatch: (ceil(nr/256), batch_size, 1)
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
    let rho_safe = max(rho, f64(1e-20));

    // Proton potential
    let u_t0_p = t0 * ((f64(1.0) + x0 / f64(2.0)) * rho - (f64(0.5) + x0) * rp);
    let rho_alpha = pow_f64(rho_safe, alpha);
    var rho_alpha_m1: f64;
    if (rho > f64(1e-15)) {
        rho_alpha_m1 = pow_f64(rho_safe, alpha - f64(1.0));
    } else {
        rho_alpha_m1 = f64(0.0);
    }
    let sum_rho2 = rp * rp + rn * rn;
    let u_t3_p = (t3 / f64(12.0))
        * ((f64(1.0) + x3 / f64(2.0)) * (alpha + f64(2.0)) * rho_alpha * rho
            - (f64(0.5) + x3) * (alpha * rho_alpha_m1 * sum_rho2
                + f64(2.0) * rho_alpha * rp));

    // Store proton Skyrme potential (Coulomb added in separate pass)
    u_total_batch[idx] = u_t0_p + u_t3_p;

    // Neutron potential
    let u_t0_n = t0 * ((f64(1.0) + x0 / f64(2.0)) * rho - (f64(0.5) + x0) * rn);
    let u_t3_n = (t3 / f64(12.0))
        * ((f64(1.0) + x3 / f64(2.0)) * (alpha + f64(2.0)) * rho_alpha * rho
            - (f64(0.5) + x3) * (alpha * rho_alpha_m1 * sum_rho2
                + f64(2.0) * rho_alpha * rn));
    u_total_n_batch[idx] = u_t0_n + u_t3_n;
}

// ─── Coulomb: forward cumulative sum (charge enclosed) ───────────
// One thread per batch element, sequential over grid points
// Dispatch: (batch_size, 1, 1)
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
// One thread per batch element, sequential backward
// Dispatch: (batch_size, 1, 1)
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
// Dispatch: (ceil(nr/256), batch_size, 1)
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

    // Coulomb direct
    let v_c = E2 * (charge_enclosed[idx] / rk + phi_outer[idx]);

    // Coulomb exchange (Slater LDA)
    let coeff = -E2 * cbrt_f64(f64(3.0) / PI_VAL);
    let v_cx = coeff * cbrt_f64(rp);

    // Add to proton potential (already contains Skyrme from compute_skyrme)
    u_total_batch[idx] = u_total_batch[idx] + v_c + v_cx;
}

// ─── Effective mass function f_q(r) ──────────────────────────────
// Dispatch: (ceil(nr/256), batch_size, 1)
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

    // Proton f_q
    let f_p = max(hbar2_2m + c0t * (rp + rn) - c1n * rp, hbar2_2m * f64(0.3));
    f_q_p_batch[idx] = f_p;

    // Neutron f_q
    let f_n = max(hbar2_2m + c0t * (rp + rn) - c1n * rn, hbar2_2m * f64(0.3));
    f_q_n_batch[idx] = f_n;
}
