// SPDX-License-Identifier: AGPL-3.0-only
//
// GPU Cell-List Pass 2/3: Exclusive Prefix Sum
//
// Computes cell_start[c] = sum(cell_counts[0..c]) so that particles in cell c
// occupy sorted_indices[cell_start[c] .. cell_start[c] + cell_counts[c]).
//
// Sequential scan (workgroup_size=1) is optimal for Nc < 1000. At typical
// MD parameters (box_side=10, rc=2 → 5×5×5 = 125 cells), this runs in
// sub-microsecond time.
//
// Dispatch: (1, 1, 1)
//
// Bindings:
//   0: counts  [Nc] u32, read  — per-cell particle counts (from pass 1)
//   1: starts  [Nc] u32, write — exclusive prefix sum output
//   2: params  uniform         — [size, 0, 0, 0]

struct ScanParams {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read>       counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> starts: array<u32>;
@group(0) @binding(2) var<uniform>             params: ScanParams;

@compute @workgroup_size(1)
fn exclusive_scan(@builtin(global_invocation_id) gid: vec3<u32>) {
    var sum = 0u;
    for (var i = 0u; i < params.size; i++) {
        starts[i] = sum;
        sum += counts[i];
    }
}
