// SPDX-License-Identifier: AGPL-3.0-or-later

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var exp_table: texture_2d<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let x = f32(idx) * 0.01 - 5.0;
    let u = clamp((x + 5.0) / 10.0, 0.0, 1.0);
    let texel = min(u32(u * 1023.0), 1023u);
    out[idx] = textureLoad(exp_table, vec2<u32>(texel, 0u), 0).x;
}
