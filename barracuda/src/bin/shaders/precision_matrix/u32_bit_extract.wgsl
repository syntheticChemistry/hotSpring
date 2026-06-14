// SPDX-License-Identifier: AGPL-3.0-or-later
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let packed: u32 = 0xB4E4B4E4u;
    let int2_at_0  = packed & 3u;
    let int2_at_2  = (packed >> 2u) & 3u;
    let int2_at_4  = (packed >> 4u) & 3u;
    let int4_at_0  = packed & 15u;
    let int4_at_4  = (packed >> 4u) & 15u;
    out[0] = int2_at_0 + int2_at_2 + int2_at_4 + int4_at_0 + int4_at_4;
}
