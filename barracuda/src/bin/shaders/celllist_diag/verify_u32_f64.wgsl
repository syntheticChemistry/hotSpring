// SPDX-License-Identifier: AGPL-3.0-only

@group(0) @binding(0) var<storage, read> cell_start_in: array<u32>;
@group(0) @binding(1) var<storage, read> cell_count_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f64>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Read first 27 cell_start and cell_count values, output as f64
    for (var c = 0u; c < 27u; c = c + 1u) {
        out[c * 2u]      = f64(cell_start_in[c]);
        out[c * 2u + 1u] = f64(cell_count_in[c]);
    }
}
