// Deformed HO Wavefunction Evaluation on 2D Cylindrical Grid (f64)
//
// Evaluates Nilsson-basis wavefunctions ψ(ρ,z) = φ_nz(z/b_z) × R_{n⊥,|Λ|}(ρ/b⊥)
// for axially-deformed nuclear structure (L3 HFB).
//
// Grid: cylindrical (ρ, z), row-major [n_rho × n_z]
//   index(i_rho, i_z) = i_rho * n_z + i_z
//
// Each thread computes ONE grid point for ONE state.
// Dispatch: (ceil(n_grid/256), n_states, 1) or similar.
//
// Hermite: H_n(ξ) via recurrence, ξ = z/b_z
// Laguerre: L_n^α(η) via recurrence, η = (ρ/b⊥)²
//
// Deep Debt: pure WGSL, f64, no recursion, self-contained.

// ═══════════════════════════════════════════════════════════════════
// Parameters
// ═══════════════════════════════════════════════════════════════════

struct WfParams {
    n_rho: u32,       // grid points in ρ direction
    n_z: u32,         // grid points in z direction
    n_states: u32,    // total basis states
    _pad0: u32,
    d_rho: f64,       // grid spacing in ρ
    d_z: f64,         // grid spacing in z
    z_min: f64,       // z grid starts at z_min + 0.5*d_z
    b_z: f64,         // HO length parameter along z
    b_perp: f64,      // HO length parameter perpendicular
    rho_max: f64,     // max ρ (for reference)
}

// Per-state quantum numbers (SoA layout for GPU efficiency)
struct StateParams {
    n_z: u32,         // z oscillator quanta
    n_perp: u32,      // perpendicular oscillator quanta
    abs_lambda: u32,  // |Λ|
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: WfParams;
@group(0) @binding(1) var<storage, read> state_params: array<StateParams>;
// Output: wavefunctions[state_idx * n_grid + grid_idx]
@group(0) @binding(2) var<storage, read_write> wavefunctions: array<f64>;
// Output: normalization accumulators (one per state, for renormalization pass)
@group(0) @binding(3) var<storage, read_write> norm_accum: array<f64>;

const PI: f64 = 3.14159265358979323846;
const WG_SIZE: u32 = 256u;

// ═══════════════════════════════════════════════════════════════════
// Math helpers (inlined, no recursion)
// ═══════════════════════════════════════════════════════════════════

// Hermite polynomial H_n(x) via three-term recurrence
// H_0(x) = 1, H_1(x) = 2x, H_{n+1}(x) = 2x·H_n(x) - 2n·H_{n-1}(x)
fn hermite(n: u32, x: f64) -> f64 {
    if (n == 0u) { return f64(1.0); }
    if (n == 1u) { return f64(2.0) * x; }
    var h_prev = f64(1.0);
    var h_curr = f64(2.0) * x;
    for (var k = 2u; k <= n; k++) {
        let h_next = f64(2.0) * x * h_curr - f64(2.0) * f64(k - 1u) * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }
    return h_curr;
}

// Generalized Laguerre polynomial L_n^α(x) via recurrence
// L_0^α = 1, L_1^α = 1+α-x
// (k+1)·L_{k+1}^α = (2k+1+α-x)·L_k^α - (k+α)·L_{k-1}^α
fn laguerre(n: u32, alpha: f64, x: f64) -> f64 {
    if (n == 0u) { return f64(1.0); }
    var l_prev = f64(1.0);
    var l_curr = f64(1.0) + alpha - x;
    for (var k = 1u; k < n; k++) {
        let kf = f64(k);
        let l_next = ((f64(2.0) * kf + f64(1.0) + alpha - x) * l_curr
                      - (kf + alpha) * l_prev) / (kf + f64(1.0));
        l_prev = l_curr;
        l_curr = l_next;
    }
    return l_curr;
}

// Factorial for small n (max ~16 shells → max n ~ 16)
fn factorial(n: u32) -> f64 {
    var result = f64(1.0);
    for (var k = 2u; k <= n; k++) {
        result = result * f64(k);
    }
    return result;
}

// Gamma function approximation for half-integer + integer arguments
// For our use: gamma(n + alpha + 1) where n is integer, alpha is integer
// This is just (n + alpha)!
fn gamma_int(n: u32) -> f64 {
    // gamma(n) = (n-1)! for positive integers
    if (n <= 1u) { return f64(1.0); }
    return factorial(n - 1u);
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Evaluate wavefunctions on grid
// Each thread: one (grid_point, state) pair
// Dispatch: (ceil(n_grid / 256), n_states, 1)
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(256, 1, 1)
fn evaluate_wavefunctions(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let grid_idx = gid.x;
    let state_idx = gid.y;
    let n_grid = params.n_rho * params.n_z;

    if (grid_idx >= n_grid || state_idx >= params.n_states) {
        return;
    }

    // Decode grid indices
    let i_rho = grid_idx / params.n_z;
    let i_z = grid_idx % params.n_z;

    // Grid coordinates: ρ starts at d_rho (not 0), z centered around 0
    let rho = f64(i_rho + 1u) * params.d_rho;
    let z = params.z_min + (f64(i_z) + f64(0.5)) * params.d_z;

    // State quantum numbers
    let sp = state_params[state_idx];
    let n_z_val = sp.n_z;
    let n_perp_val = sp.n_perp;
    let abs_lambda = sp.abs_lambda;

    // ── z part: Hermite oscillator ──
    let xi = z / params.b_z;
    let h_n = hermite(n_z_val, xi);
    // Normalization: 1 / sqrt(b_z * sqrt(pi) * 2^n * n!)
    let norm_z = f64(1.0) / sqrt(params.b_z * sqrt(PI) * f64(1u << n_z_val) * factorial(n_z_val));
    let phi_z = norm_z * h_n * exp(-xi * xi / f64(2.0));

    // ── ρ part: 2D oscillator (Laguerre) ──
    let eta = (rho / params.b_perp) * (rho / params.b_perp);  // (ρ/b⊥)²
    let alpha = f64(abs_lambda);

    let n_fact = factorial(n_perp_val);
    // gamma(n_perp + |lambda| + 1) = (n_perp + |lambda|)!
    let gamma_val = factorial(n_perp_val + abs_lambda);
    let norm_rho = sqrt(n_fact / (PI * params.b_perp * params.b_perp * gamma_val));

    let lag = laguerre(n_perp_val, alpha, eta);
    let phi_rho = norm_rho * pow(rho / params.b_perp, alpha) * exp(-eta / f64(2.0)) * lag;

    let psi = phi_z * phi_rho;

    // Store: wavefunctions[state * n_grid + grid_point]
    let out_idx = state_idx * n_grid + grid_idx;
    wavefunctions[out_idx] = psi;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Compute norm² for each state (for grid renormalization)
// Each thread accumulates partial norm² for a chunk of grid points.
// Uses workgroup reduction, then atomicAdd to global accumulator.
// Dispatch: (ceil(n_grid / WG_SIZE), n_states, 1)
// ═══════════════════════════════════════════════════════════════════

var<workgroup> shared_norm: array<f64, 256>;

@compute @workgroup_size(256, 1, 1)
fn compute_norms(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let grid_idx = gid.x;
    let state_idx = gid.y;
    let n_grid = params.n_rho * params.n_z;

    var local_norm2 = f64(0.0);

    if (grid_idx < n_grid && state_idx < params.n_states) {
        let i_rho = grid_idx / params.n_z;
        let i_z = grid_idx % params.n_z;
        let rho = f64(i_rho + 1u) * params.d_rho;

        // Volume element: 2π ρ dρ dz
        let dv = f64(2.0) * PI * rho * params.d_rho * params.d_z;

        let psi = wavefunctions[state_idx * n_grid + grid_idx];
        local_norm2 = psi * psi * dv;
    }

    // Workgroup tree reduction
    shared_norm[lid.x] = local_norm2;
    workgroupBarrier();

    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_norm[lid.x] = shared_norm[lid.x] + shared_norm[lid.x + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes partial sum to global accumulator
    // NOTE: f64 atomics not available in WGSL — use output buffer indexed by
    // (state_idx, workgroup_x) and do final reduction on CPU or second pass
    if (lid.x == 0u) {
        let wg_x = gid.x / WG_SIZE;
        let n_wg_x = (n_grid + WG_SIZE - 1u) / WG_SIZE;
        norm_accum[state_idx * n_wg_x + wg_x] = shared_norm[0];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Renormalize wavefunctions given final norms
// norm_accum[state_idx] now contains the total norm² (after CPU/GPU final reduction)
// Dispatch: (ceil(n_grid / 256), n_states, 1)
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(256, 1, 1)
fn renormalize_wavefunctions(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let grid_idx = gid.x;
    let state_idx = gid.y;
    let n_grid = params.n_rho * params.n_z;

    if (grid_idx >= n_grid || state_idx >= params.n_states) {
        return;
    }

    let norm2 = norm_accum[state_idx];
    if (norm2 > f64(1e-30)) {
        let scale = f64(1.0) / sqrt(norm2);
        let idx = state_idx * n_grid + grid_idx;
        wavefunctions[idx] = wavefunctions[idx] * scale;
    }
}
