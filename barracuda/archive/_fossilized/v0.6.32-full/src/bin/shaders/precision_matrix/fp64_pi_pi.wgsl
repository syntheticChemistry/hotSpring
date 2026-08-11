// SPDX-License-Identifier: AGPL-3.0-or-later
@group(0) @binding(0) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let pi: f64 = 3.14159265358979323846lf;
    out[0] = pi * pi;
}
