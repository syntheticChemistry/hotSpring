// SPDX-License-Identifier: AGPL-3.0-only
// Block Hamiltonian Assembly for Deformed HFB on 2D Grid (f64)
//
// Builds block-diagonal Hamiltonian matrices H[bi,bj] for each Omega block.
// Each matrix element is:
//   H[bi,bj] = T_ij (kinetic, diagonal) + <ψ_i|V|ψ_j> (potential, grid integral)
//
// The potential matrix element integral is the bottleneck:
//   <ψ_i|V|ψ_j> = Σ_{grid} ψ_i(ρ,z) · V(ρ,z) · ψ_j(ρ,z) · dV
//
// This is a weighted dot product over the 2D grid — perfect GPU workload.
//
// Strategy:
//   Kernel 1: compute_potential_matrix_element — one thread per (bi, bj) pair,
//     loops over grid. For 20k-50k grid points × ~50² pairs, the grid loop
//     dominates and each thread gets substantial work.
//   Kernel 2: build_block_hamiltonian — adds kinetic diagonal to potential.
//   Kernel 3: batch version — process all Omega blocks for one nucleus in a
//     single dispatch using block offsets.
//
// Grid: row-major [n_rho × n_z]
// Deep Debt: pure WGSL, f64, self-contained.

// ═══════════════════════════════════════════════════════════════════
// Parameters
// ═══════════════════════════════════════════════════════════════════

struct HamiltonianParams {
    n_rho: u32,
    n_z: u32,
    block_size: u32,       // number of states in this Omega block
    n_blocks: u32,         // number of Omega blocks to process
    d_rho: f64,
    d_z: f64,
}

@group(0) @binding(0) var<uniform> params: HamiltonianParams;
// Wavefunctions: [n_states_total × n_grid] f64, row-major per state
@group(0) @binding(1) var<storage, read> wavefunctions: array<f64>;
// Mean-field potential V(ρ,z): [n_grid] f64
@group(0) @binding(2) var<storage, read> v_potential: array<f64>;
// Block indices: maps (block_index, local_index) → global state index
// Layout: block_offsets[block * max_block_size + local_idx] = global_state_idx
@group(0) @binding(3) var<storage, read> block_indices: array<u32>;
// Kinetic energies: T_i for each state (HO energy)
@group(0) @binding(4) var<storage, read> kinetic_energies: array<f64>;
// Block sizes: actual size of each Omega block (may differ)
@group(0) @binding(5) var<storage, read> block_sizes: array<u32>;
// Block offsets into output H matrix: where each block's H starts
@group(0) @binding(6) var<storage, read> block_h_offsets: array<u32>;
// Output: H matrices, packed contiguously [block0: bs0×bs0, block1: bs1×bs1, ...]
@group(0) @binding(7) var<storage, read_write> h_matrices: array<f64>;

const PI: f64 = 3.14159265358979323846;

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Potential matrix elements via grid integration
//
// One thread per (block_idx, bi, bj) triple.
// Thread computes: <ψ_i|V|ψ_j> = Σ_grid ψ_i · V · ψ_j · dV
//
// For upper-triangle only (symmetric): bi <= bj.
// Dispatch: (ceil(max_pairs / 256), n_blocks, 1)
//   where max_pairs = max_block_size * (max_block_size + 1) / 2
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(256, 1, 1)
fn compute_potential_matrix_elements(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let pair_idx = gid.x;      // flattened (bi, bj) pair index (upper triangle)
    let block_idx = gid.y;     // which Omega block

    if (block_idx >= params.n_blocks) { return; }

    let bs = block_sizes[block_idx];
    let n_pairs = bs * (bs + 1u) / 2u;

    if (pair_idx >= n_pairs) { return; }

    // Decode (bi, bj) from upper-triangle linear index
    // pair_idx = bi * bs - bi*(bi-1)/2 + (bj - bi)
    // Solve: find bi such that bi*bs - bi*(bi-1)/2 <= pair_idx
    var bi = 0u;
    var offset = 0u;
    for (var trial = 0u; trial < bs; trial++) {
        let row_count = bs - trial; // number of elements in row trial
        if (offset + row_count > pair_idx) {
            bi = trial;
            break;
        }
        offset += row_count;
    }
    let bj = bi + (pair_idx - offset);

    let n_grid = params.n_rho * params.n_z;

    // Global state indices
    let max_bs = params.block_size; // max block size (for indexing block_indices)
    let i_global = block_indices[block_idx * max_bs + bi];
    let j_global = block_indices[block_idx * max_bs + bj];

    // Compute <ψ_i|V|ψ_j> = Σ ψ_i(k) · V(k) · ψ_j(k) · dV(k)
    var integral = f64(0.0);

    for (var k = 0u; k < n_grid; k++) {
        let i_rho = k / params.n_z;
        let rho_coord = f64(i_rho + 1u) * params.d_rho;
        let dv = f64(2.0) * PI * rho_coord * params.d_rho * params.d_z;

        let psi_i = wavefunctions[i_global * n_grid + k];
        let psi_j = wavefunctions[j_global * n_grid + k];

        integral += psi_i * v_potential[k] * psi_j * dv;
    }

    // Add kinetic energy (diagonal only)
    var h_ij = integral;
    if (bi == bj) {
        h_ij += kinetic_energies[i_global];
    }

    // Write to H matrix (symmetric: both H[bi,bj] and H[bj,bi])
    let h_base = block_h_offsets[block_idx];
    h_matrices[h_base + bi * bs + bj] = h_ij;
    if (bi != bj) {
        h_matrices[h_base + bj * bs + bi] = h_ij;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Potential matrix elements — WORKGROUP REDUCTION variant
//
// For large grids (>10k points), the per-thread grid loop in kernel 1
// is efficient but could benefit from shared-memory reduction.
// This kernel uses one workgroup per (bi, bj) pair and reduces over
// the grid dimension using shared memory.
//
// Dispatch: (n_workgroups_per_pair, n_pairs, n_blocks)
//   where n_workgroups_per_pair = ceil(n_grid / 256)
//   Output: partial sums [n_blocks × n_pairs × n_wg_per_pair]
//   Then a final reduction pass sums the partial results.
// ═══════════════════════════════════════════════════════════════════

var<workgroup> shared_sum: array<f64, 256>;

@compute @workgroup_size(256, 1, 1)
fn compute_potential_matrix_elements_reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let grid_chunk_idx = gid.x;  // which grid point this thread handles
    let pair_idx = wg_id.y;      // which (bi, bj) pair
    let block_idx = wg_id.z;     // which Omega block

    if (block_idx >= params.n_blocks) { return; }

    let bs = block_sizes[block_idx];
    let n_pairs = bs * (bs + 1u) / 2u;
    if (pair_idx >= n_pairs) {
        shared_sum[lid.x] = f64(0.0);
        workgroupBarrier();
        return;
    }

    // Decode (bi, bj)
    var bi = 0u;
    var tri_offset = 0u;
    for (var trial = 0u; trial < bs; trial++) {
        let row_count = bs - trial;
        if (tri_offset + row_count > pair_idx) {
            bi = trial;
            break;
        }
        tri_offset += row_count;
    }
    let bj = bi + (pair_idx - tri_offset);

    let n_grid = params.n_rho * params.n_z;
    let max_bs = params.block_size;
    let i_global = block_indices[block_idx * max_bs + bi];
    let j_global = block_indices[block_idx * max_bs + bj];

    // Each thread processes one grid point
    var local_sum = f64(0.0);
    if (grid_chunk_idx < n_grid) {
        let k = grid_chunk_idx;
        let i_rho = k / params.n_z;
        let rho_coord = f64(i_rho + 1u) * params.d_rho;
        let dv = f64(2.0) * PI * rho_coord * params.d_rho * params.d_z;

        let psi_i = wavefunctions[i_global * n_grid + k];
        let psi_j = wavefunctions[j_global * n_grid + k];
        local_sum = psi_i * v_potential[k] * psi_j * dv;
    }

    // Workgroup tree reduction
    shared_sum[lid.x] = local_sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum[lid.x] += shared_sum[lid.x + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes the workgroup partial sum
    if (lid.x == 0u) {
        let n_wg = (n_grid + 255u) / 256u;
        // Output index: [block_idx][pair_idx][wg_x]
        h_matrices[block_idx * n_pairs * n_wg + pair_idx * n_wg + wg_id.x] = shared_sum[0];
    }
}
