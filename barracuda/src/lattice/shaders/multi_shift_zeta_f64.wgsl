// Multi-shift CG zeta recurrence — scalar kernel.
//
// Computes the shifted CG parameters (zeta_next, alpha_s, beta_s) for each
// pole in the rational approximation. Runs with workgroup_size(1) since it
// processes a small number of shifts (typically 4-16 poles).
//
// Inputs (uniform):
//   alpha     — base CG step size (scalar)
//   rz_new    — new residual norm squared (scalar)
//   rz_old    — previous residual norm squared (scalar)
//   n_shifts  — number of active shifts
//
// Inputs/Outputs (storage):
//   shifts      — sigma_i values (read-only, n_shifts)
//   zeta_curr   — current zeta per shift (read/write)
//   zeta_prev   — previous zeta per shift (read/write)
//   beta_prev   — previous beta per shift (read/write)
//   alpha_shift — output alpha_s per shift (write)
//   beta_shift  — output beta_s per shift (write)
//   active      — per-shift active flag (read/write, 0 or 1)

struct Params {
    alpha: f64,
    alpha_prev: f64,
    rz_new: f64,
    rz_old: f64,
    sigma_0: f64,
    n_shifts: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> shifts: array<f64>;
@group(0) @binding(2) var<storage, read_write> zeta_curr: array<f64>;
@group(0) @binding(3) var<storage, read_write> zeta_prev: array<f64>;
@group(0) @binding(4) var<storage, read_write> beta_prev: array<f64>;
@group(0) @binding(5) var<storage, read_write> alpha_shift: array<f64>;
@group(0) @binding(6) var<storage, read_write> beta_shift: array<f64>;
@group(0) @binding(7) var<storage, read_write> active: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if s >= params.n_shifts || active[s] == 0u {
        return;
    }

    let ds = shifts[s] - params.sigma_0;
    let zc = zeta_curr[s];
    let zp = zeta_prev[s];
    let bp = beta_prev[s];

    // Zeta recurrence denominator
    var denom = 1.0 + params.alpha * ds;
    if abs(bp) > 1e-30 {
        denom = denom + params.alpha * params.alpha_prev * (1.0 - zp / zc) / bp;
    }

    if abs(denom) < 1e-30 {
        active[s] = 0u;
        alpha_shift[s] = 0.0;
        beta_shift[s] = 0.0;
        return;
    }

    let zeta_next = zc / denom;
    let a_s = params.alpha * zeta_next / zc;

    var b_s = 0.0;
    if abs(params.rz_old) > 1e-30 {
        b_s = (zeta_next / zc) * (zeta_next / zc) * (params.rz_new / params.rz_old);
    }

    alpha_shift[s] = a_s;
    beta_shift[s] = b_s;

    zeta_prev[s] = zc;
    zeta_curr[s] = zeta_next;
    beta_prev[s] = b_s;
}
