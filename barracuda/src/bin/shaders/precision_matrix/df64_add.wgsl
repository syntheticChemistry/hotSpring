// SPDX-License-Identifier: AGPL-3.0-or-later
@group(0) @binding(0) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    out[0] = df64_to_f64(df64_add(Df64(1.0, 0.0), Df64(1.0, 0.0)));
}
