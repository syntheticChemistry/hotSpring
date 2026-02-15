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

struct PotentialParams {
    nr: u32,
    batch_size: u32,
    // Skyrme parameters (Naga f64 uniform — encoded as u32 pairs)
    t0_lo: u32, t0_hi: u32,
    t3_lo: u32, t3_hi: u32,
    x0_lo: u32, x0_hi: u32,
    x3_lo: u32, x3_hi: u32,
    alpha_lo: u32, alpha_hi: u32,
    dr_lo: u32, dr_hi: u32,
    // Effective mass coefficients (precomputed on CPU)
    c0t_lo: u32, c0t_hi: u32,   // 0.25*(t1*(1+x1/2) + t2*(1+x2/2))
    c1n_lo: u32, c1n_hi: u32,   // 0.25*(t1*(1/2+x1) - t2*(1/2+x2))
    hbar2_2m_lo: u32, hbar2_2m_hi: u32,
}

fn decode_f64(lo: u32, hi: u32) -> f64 {
    return bitcast<f64>(vec2<u32>(lo, hi));
}

@group(0) @binding(0) var<uniform> params: PotentialParams;
@group(0) @binding(1) var<storage, read> rho_p_batch: array<f64>;
@group(0) @binding(2) var<storage, read> rho_n_batch: array<f64>;
@group(0) @binding(3) var<storage, read_write> u_total_batch: array<f64>;  // proton potential
@group(0) @binding(4) var<storage, read_write> u_total_n_batch: array<f64>; // neutron potential
@group(0) @binding(5) var<storage, read> r_grid: array<f64>;
@group(0) @binding(6) var<storage, read_write> f_q_p_batch: array<f64>; // f_q for protons
@group(0) @binding(7) var<storage, read_write> f_q_n_batch: array<f64>; // f_q for neutrons
// Coulomb scratch: charge_enclosed and phi_outer
@group(0) @binding(8) var<storage, read_write> charge_enclosed: array<f64>; // [batch_size × nr]
@group(0) @binding(9) var<storage, read_write> phi_outer: array<f64>;       // [batch_size × nr]

const E2: f64 = 1.4399764;  // e² in MeV·fm
const PI_VAL: f64 = 3.141592653589793;

// ─── Skyrme potential ────────────────────────────────────────────
// Thread per (k, batch): computes U_sky(r_k) for one nucleus
// Dispatch: (ceil(nr/256), batch_size, 1)
@compute @workgroup_size(256, 1, 1)
fn compute_skyrme(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let batch_idx = global_id.y;
    let nr = params.nr;

    if (k >= nr || batch_idx >= params.batch_size) {
        return;
    }

    let t0 = decode_f64(params.t0_lo, params.t0_hi);
    let t3 = decode_f64(params.t3_lo, params.t3_hi);
    let x0 = decode_f64(params.x0_lo, params.x0_hi);
    let x3 = decode_f64(params.x3_lo, params.x3_hi);
    let alpha = decode_f64(params.alpha_lo, params.alpha_hi);

    let idx = batch_idx * nr + k;
    let rp = rho_p_batch[idx];
    let rn = rho_n_batch[idx];
    let rho = rp + rn;
    let rho_safe = max(rho, f64(1e-20));

    // Proton potential
    let u_t0_p = t0 * ((f64(1.0) + x0 / f64(2.0)) * rho - (f64(0.5) + x0) * rp);
    let rho_alpha = pow(rho_safe, alpha);
    var rho_alpha_m1: f64;
    if (rho > f64(1e-15)) {
        rho_alpha_m1 = pow(rho_safe, alpha - f64(1.0));
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
    let nr = params.nr;
    let dr = decode_f64(params.dr_lo, params.dr_hi);

    if (batch_idx >= params.batch_size) {
        return;
    }

    let base = batch_idx * nr;
    var cumsum: f64 = f64(0.0);
    for (var k = 0u; k < nr; k++) {
        let rk = r_grid[k];
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
    let nr = params.nr;
    let dr = decode_f64(params.dr_lo, params.dr_hi);

    if (batch_idx >= params.batch_size) {
        return;
    }

    let base = batch_idx * nr;
    var cumsum_rev: f64 = f64(0.0);
    for (var k_rev = 0u; k_rev < nr; k_rev++) {
        let k = nr - 1u - k_rev;
        let rk = r_grid[k];
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
    let nr = params.nr;

    if (k >= nr || batch_idx >= params.batch_size) {
        return;
    }

    let idx = batch_idx * nr + k;
    let rk = max(r_grid[k], f64(1e-10));
    let rp = max(rho_p_batch[idx], f64(0.0));

    // Coulomb direct
    let v_c = E2 * (charge_enclosed[idx] / rk + phi_outer[idx]);

    // Coulomb exchange (Slater LDA)
    let coeff = -E2 * pow(f64(3.0) / PI_VAL, f64(1.0) / f64(3.0));
    let v_cx = coeff * pow(rp, f64(1.0) / f64(3.0));

    // Add to proton potential (already contains Skyrme from compute_skyrme)
    u_total_batch[idx] = u_total_batch[idx] + v_c + v_cx;
}

// ─── Effective mass function f_q(r) ──────────────────────────────
// Dispatch: (ceil(nr/256), batch_size, 1)
@compute @workgroup_size(256, 1, 1)
fn compute_f_q(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let batch_idx = global_id.y;
    let nr = params.nr;

    if (k >= nr || batch_idx >= params.batch_size) {
        return;
    }

    let idx = batch_idx * nr + k;
    let c0t = decode_f64(params.c0t_lo, params.c0t_hi);
    let c1n = decode_f64(params.c1n_lo, params.c1n_hi);
    let hbar2_2m = decode_f64(params.hbar2_2m_lo, params.hbar2_2m_hi);

    let rp = rho_p_batch[idx];
    let rn = rho_n_batch[idx];

    // Proton f_q
    let f_p = max(hbar2_2m + c0t * (rp + rn) - c1n * rp, hbar2_2m * f64(0.3));
    f_q_p_batch[idx] = f_p;

    // Neutron f_q
    let f_n = max(hbar2_2m + c0t * (rp + rn) - c1n * rn, hbar2_2m * f64(0.3));
    f_q_n_batch[idx] = f_n;
}
