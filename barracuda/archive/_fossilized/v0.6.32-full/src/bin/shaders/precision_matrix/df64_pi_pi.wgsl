// SPDX-License-Identifier: AGPL-3.0-or-later
@group(0) @binding(0) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let pi = Df64(3.14159274, -8.74227766e-8);
    out[0] = df64_to_f64(df64_mul(pi, pi));
}
