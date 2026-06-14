// SPDX-License-Identifier: AGPL-3.0-or-later
@group(0) @binding(0) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    var sum: f64 = 0.0lf;
    var c: f64 = 0.0lf;
    for (var i = 0u; i < 1024u; i = i + 1u) {
        let y = 1.0lf - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    out[0] = sum;
}
