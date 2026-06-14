// SPDX-License-Identifier: AGPL-3.0-or-later
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var tex: texture_2d<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var sum: f32 = 0.0;
    let base_y = gid.x / 1024u;
    let base_x = gid.x % 1024u;
    for (var i = 0u; i < 64u; i = i + 1u) {
        let coord = vec2<i32>(i32((base_x + i * 16u) % 1024u), i32(base_y % 1024u));
        sum = sum + textureLoad(tex, coord, 0).r;
    }
    out[gid.x] = sum;
}
