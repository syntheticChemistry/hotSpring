// Batched HFB Hamiltonian Construction (f64) — GPU Shader
//
// Builds the full Hamiltonian matrix H = T_eff + V for all nuclei in a batch.
// Each thread computes one matrix element H[i,j] for one nucleus by performing
// a radial numerical integration (trapezoidal rule).
//
// This replaces the CPU build_hamiltonian() + build_t_eff() in hfb.rs.
//
// Memory layout:
//   wf: [n_states × nr] f64 — basis wavefunctions (shared across batch)
//   dwf: [n_states × nr] f64 — wavefunction derivatives (shared across batch)
//   u_total_batch: [batch_size × nr] f64 — total potential per nucleus
//   f_q_batch: [batch_size × nr] f64 — effective mass function per nucleus
//   r_grid: [nr] f64 — radial grid
//   lj_same: [n_states × n_states] u32 — 1 if states i,j share (l,j), 0 otherwise
//   H_batch: [batch_size × n_states × n_states] f64 — output Hamiltonians
//
// Algorithm:
//   For T_eff matrix element (same (l,j) block only):
//     T_eff[i,j] = integral( f_q(r) * (dwf_i * dwf_j * r² + l(l+1) * wf_i * wf_j) dr )
//
//   For potential matrix element:
//     Diagonal: V[i,i] = integral( wf_i² * U_total * r² dr )
//     Off-diagonal (same (l,j)): V[i,j] = integral( wf_i * wf_j * U_total * r² dr )
//
//   H[i,j] = T_eff[i,j] + V[i,j]

struct HamiltonianParams {
    n_states: u32,
    nr: u32,
    batch_size: u32,
    dr_lo: u32, dr_hi: u32,
    _pad: u32,
}

fn decode_f64_h(lo: u32, hi: u32) -> f64 {
    return bitcast<f64>(vec2<u32>(lo, hi));
}

@group(0) @binding(0) var<uniform> params: HamiltonianParams;
@group(0) @binding(1) var<storage, read> wf: array<f64>;        // [n_states × nr]
@group(0) @binding(2) var<storage, read> dwf: array<f64>;       // [n_states × nr]
@group(0) @binding(3) var<storage, read> u_total_batch: array<f64>;  // [batch × nr]
@group(0) @binding(4) var<storage, read> f_q_batch: array<f64>;      // [batch × nr]
@group(0) @binding(5) var<storage, read> r_grid: array<f64>;         // [nr]
@group(0) @binding(6) var<storage, read> lj_same: array<u32>;        // [n_states × n_states]
@group(0) @binding(7) var<storage, read> ll1_values: array<f64>;     // [n_states] — l(l+1) per state
@group(0) @binding(8) var<storage, read_write> H_batch: array<f64>;  // [batch × ns × ns]

// Compute one matrix element H[batch_idx][i][j]
// Thread index encodes (i, j) pair; batch_idx from z dimension
// Dispatch: (ceil(n_states/16), ceil(n_states/16), batch_size)
@compute @workgroup_size(16, 16, 1)
fn build_hamiltonian(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let j_idx = global_id.x;
    let i_idx = global_id.y;
    let batch_idx = wg_id.z;

    let ns = params.n_states;
    let nr = params.nr;
    let dr = decode_f64_h(params.dr_lo, params.dr_hi);

    if (i_idx >= ns || j_idx >= ns || batch_idx >= params.batch_size) {
        return;
    }

    // Only compute upper triangle + diagonal (symmetric)
    if (j_idx < i_idx) {
        return;
    }

    let pot_base = batch_idx * nr;
    let h_base = batch_idx * ns * ns;
    let same_lj = lj_same[i_idx * ns + j_idx];

    var h_val: f64 = f64(0.0);

    // T_eff contribution (only within same (l,j) block)
    if (same_lj == 1u) {
        let ll1 = ll1_values[i_idx]; // l(l+1), same for i and j in same block
        var t_eff_sum: f64 = f64(0.0);
        for (var k = 0u; k < nr; k++) {
            let rk = r_grid[k];
            let fq = f_q_batch[pot_base + k];
            let wf_i = wf[i_idx * nr + k];
            let wf_j = wf[j_idx * nr + k];
            let dwf_i = dwf[i_idx * nr + k];
            let dwf_j = dwf[j_idx * nr + k];

            // T_eff integrand: f_q * (dwf_i * dwf_j * r² + l(l+1) * wf_i * wf_j)
            t_eff_sum = t_eff_sum + fq * (dwf_i * dwf_j * rk * rk + ll1 * wf_i * wf_j);
        }
        h_val = h_val + t_eff_sum * dr;
    }

    // Potential contribution: V[i,j] = integral(wf_i * wf_j * U * r² dr)
    // Diagonal always; off-diagonal only if same (l,j)
    if (i_idx == j_idx || same_lj == 1u) {
        var v_sum: f64 = f64(0.0);
        for (var k = 0u; k < nr; k++) {
            let rk = r_grid[k];
            let u_k = u_total_batch[pot_base + k];
            let wf_i = wf[i_idx * nr + k];
            let wf_j = wf[j_idx * nr + k];
            v_sum = v_sum + wf_i * wf_j * u_k * rk * rk;
        }
        h_val = h_val + v_sum * dr;
    }

    // Write to symmetric matrix
    H_batch[h_base + i_idx * ns + j_idx] = h_val;
    if (i_idx != j_idx) {
        H_batch[h_base + j_idx * ns + i_idx] = h_val;
    }
}
