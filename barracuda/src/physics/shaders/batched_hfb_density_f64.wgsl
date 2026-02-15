// Batched HFB Density + BCS + Energy (f64) — GPU Shader
//
// Three kernels that complete the post-eigensolve SCF step:
// 1. BCS occupations: compute v² from eigenvalues and chemical potential
// 2. Density from eigenstates: rho(r) = sum_i deg_i * v²_i * |phi_i(r)|² / (4π)
// 3. Density mixing: rho_new = alpha * rho_computed + (1-alpha) * rho_old
//
// These replace the CPU-side bcs_occupations(), density_from_eigenstates(),
// and density mixing in hfb_gpu.rs.

struct DensityParams {
    n_states: u32,
    nr: u32,
    batch_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: DensityParams;

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: BCS Occupations
// ═══════════════════════════════════════════════════════════════════
// Computes v²[i] = 0.5 * (1 - (ε_i - λ) / sqrt((ε_i - λ)² + Δ²))
// where λ is the chemical potential (pre-computed via Brent on CPU)
//
// Thread per (i, batch): one occupation per eigenvalue
// Dispatch: (ceil(n_states/256), batch_size, 1)

@group(1) @binding(0) var<storage, read> eigenvalues: array<f64>;  // [batch × n_states]
@group(1) @binding(1) var<storage, read> lambda_batch: array<f64>;  // [batch] chemical potentials
@group(1) @binding(2) var<storage, read> delta_batch: array<f64>;   // [batch] pairing gaps
@group(1) @binding(3) var<storage, read_write> v2_batch: array<f64>; // [batch × n_states] output

@compute @workgroup_size(256, 1, 1)
fn compute_bcs_v2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let batch_idx = global_id.y;
    let ns = params.n_states;

    if (i >= ns || batch_idx >= params.batch_size) {
        return;
    }

    let idx = batch_idx * ns + i;
    let eps = eigenvalues[idx];
    let lam = lambda_batch[batch_idx];
    let delta = delta_batch[batch_idx];

    let ek = eps - lam;
    let big_ek = sqrt(ek * ek + delta * delta);

    var v2: f64;
    if (big_ek < f64(1e-15)) {
        v2 = f64(0.5);
    } else {
        v2 = f64(0.5) * (f64(1.0) - ek / big_ek);
    }

    v2_batch[idx] = v2;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Density from eigenstates
// ═══════════════════════════════════════════════════════════════════
// rho(r_k) = sum_i deg_i * v²_i * |phi_i(r_k)|² / (4π)
// where phi_i(r_k) = sum_j C[j,i] * wf_j(r_k)
//
// Thread per (k, batch): computes rho at one grid point for one nucleus
// Dispatch: (ceil(nr/256), batch_size, 1)

@group(2) @binding(0) var<storage, read> eigenvectors: array<f64>;  // [batch × ns × ns]
@group(2) @binding(1) var<storage, read> v2_in: array<f64>;          // [batch × ns]
@group(2) @binding(2) var<storage, read> degs: array<f64>;           // [ns] degeneracies (shared)
@group(2) @binding(3) var<storage, read> wf: array<f64>;             // [ns × nr] wavefunctions (shared)
@group(2) @binding(4) var<storage, read_write> rho_out: array<f64>;  // [batch × nr] output

const PI_4: f64 = 12.566370614359172;  // 4π

@compute @workgroup_size(256, 1, 1)
fn compute_density(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let batch_idx = global_id.y;
    let ns = params.n_states;
    let nr = params.nr;

    if (k >= nr || batch_idx >= params.batch_size) {
        return;
    }

    let v2_base = batch_idx * ns;
    let vec_base = batch_idx * ns * ns;

    var rho_sum: f64 = f64(0.0);

    for (var i = 0u; i < ns; i++) {
        let deg_v2 = degs[i] * v2_in[v2_base + i];
        if (deg_v2 < f64(1e-12)) {
            continue;
        }

        // phi_i(r_k) = sum_j C[j,i] * wf_j(r_k)
        var phi: f64 = f64(0.0);
        for (var j = 0u; j < ns; j++) {
            let c = eigenvectors[vec_base + j * ns + i]; // V[j, i] row-major
            phi = phi + c * wf[j * nr + k];
        }

        rho_sum = rho_sum + deg_v2 * phi * phi / PI_4;
    }

    // Floor at 1e-15
    rho_out[batch_idx * nr + k] = max(rho_sum, f64(1e-15));
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Density mixing
// ═══════════════════════════════════════════════════════════════════
// rho_mixed[k] = alpha * rho_new[k] + (1-alpha) * rho_old[k]
// Clamp to 1e-15 minimum
//
// Dispatch: (ceil(batch_size * nr / 256), 1, 1)

struct MixParams {
    total_size: u32,  // batch_size * nr
    _pad1: u32,
    alpha_lo: u32, alpha_hi: u32,
}

fn decode_mix_f64(lo: u32, hi: u32) -> f64 {
    return bitcast<f64>(vec2<u32>(lo, hi));
}

@group(3) @binding(0) var<uniform> mix_params: MixParams;
@group(3) @binding(1) var<storage, read> rho_new: array<f64>;
@group(3) @binding(2) var<storage, read_write> rho_old: array<f64>;  // in-place update

@compute @workgroup_size(256)
fn mix_density(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= mix_params.total_size) {
        return;
    }

    let alpha = decode_mix_f64(mix_params.alpha_lo, mix_params.alpha_hi);
    let mixed = alpha * rho_new[idx] + (f64(1.0) - alpha) * rho_old[idx];
    rho_old[idx] = max(mixed, f64(1e-15));
}
