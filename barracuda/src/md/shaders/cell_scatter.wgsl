// SPDX-License-Identifier: AGPL-3.0-only
//
// GPU Cell-List Pass 3/3: Particle Index Scatter
//
// Each particle writes its original index into the sorted position within
// its cell using atomic write cursors. After this pass:
//
//   sorted_indices[cell_start[c] + k] = original particle index
//
// for all particles in cell c, in arbitrary order within the cell.
//
// Dispatch: (ceil(N / 64), 1, 1)
//
// Bindings:
//   0: params          uniform   — [n_particles, n_cells, 0, 0]
//   1: cell_ids        [N]  u32, read   — from pass 1
//   2: cell_start      [Nc] u32, read   — from pass 2
//   3: write_cursors   [Nc] atomic<u32>, read_write (zero-initialized)
//   4: sorted_indices  [N]  u32, write  — output

struct ScatterParams {
    n_particles: u32,
    n_cells:     u32,
    _pad0:       u32,
    _pad1:       u32,
}

@group(0) @binding(0) var<uniform>             params:         ScatterParams;
@group(0) @binding(1) var<storage, read>       cell_ids:       array<u32>;
@group(0) @binding(2) var<storage, read>       cell_start:     array<u32>;
@group(0) @binding(3) var<storage, read_write> write_cursors:  array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> sorted_indices: array<u32>;

@compute @workgroup_size(64)
fn cell_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_particles) { return; }

    let c   = cell_ids[i];
    let pos = atomicAdd(&write_cursors[c], 1u);
    sorted_indices[cell_start[c] + pos] = i;
}
