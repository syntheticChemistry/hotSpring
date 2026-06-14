// SPDX-License-Identifier: AGPL-3.0-or-later

@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    var sum: f32 = 0.0;
    let base = f32(idx) * 0.01;
    for (var i = 0u; i < 64u; i = i + 1u) {
        let a = base + f32(i) * 0.001;
        let b = base + f32(i + 1u) * 0.001;
        sum = sum + fma(a, b, -a * b);
    }
    if idx == 0u { out[0] = sum; }
}
