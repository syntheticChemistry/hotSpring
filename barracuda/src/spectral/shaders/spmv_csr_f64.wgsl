// SPDX-License-Identifier: AGPL-3.0-only

struct Params {
    n: u32,
    nnz: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> col_idx: array<u32>;
@group(0) @binding(3) var<storage, read> vals: array<f64>;
@group(0) @binding(4) var<storage, read> x_vec: array<f64>;
@group(0) @binding(5) var<storage, read_write> y_vec: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.n {
        return;
    }

    let start = row_ptr[row];
    let end = row_ptr[row + 1u];

    var sum: f64 = f64(0.0);
    for (var j = start; j < end; j = j + 1u) {
        sum = sum + vals[j] * x_vec[col_idx[j]];
    }

    y_vec[row] = sum;
}
