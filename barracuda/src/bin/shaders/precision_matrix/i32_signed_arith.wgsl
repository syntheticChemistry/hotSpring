// SPDX-License-Identifier: AGPL-3.0-or-later
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let a: i32 = -1000000;
    let b: i32 = 999999;
    let c: i32 = a + b;
    out[0] = bitcast<u32>(c);
}
