// SPDX-License-Identifier: AGPL-3.0-or-later
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var a: f32 = f32(gid.x + 1u) * 1.0001;
    var b: f32 = 0.9999;
    for (var i = 0u; i < 512u; i = i + 1u) {
        a = fma(a, b, a);
        a = fma(a, b, a);
        a = fma(a, b, a);
        a = fma(a, b, a);
    }
    out[gid.x] = a;
}
