// SPDX-License-Identifier: AGPL-3.0-or-later
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let a: u32 = 0xDEu;
    let b: u32 = 0xADu;
    let c: u32 = 0xBEu;
    let d: u32 = 0xEFu;
    let packed = (d << 24u) | (c << 16u) | (b << 8u) | a;
    let unpacked_a = packed & 0xFFu;
    let unpacked_d = (packed >> 24u) & 0xFFu;
    out[0] = unpacked_a + (unpacked_d << 8u);
}
