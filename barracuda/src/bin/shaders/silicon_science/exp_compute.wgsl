// SPDX-License-Identifier: AGPL-3.0-or-later

@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let x = f32(idx) * 0.01 - 5.0;
    out[idx] = exp(x);
}
