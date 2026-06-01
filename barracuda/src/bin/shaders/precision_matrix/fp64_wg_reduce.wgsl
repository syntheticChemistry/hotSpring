// SPDX-License-Identifier: AGPL-3.0-or-later
@group(0) @binding(0) var<storage, read_write> out: array<f64>;
var<workgroup> wg_data: array<f64, 4>;
@compute @workgroup_size(4)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    wg_data[lid.x] = f64(lid.x + 1u);
    workgroupBarrier();
    if lid.x == 0u {
        out[0] = wg_data[0] + wg_data[1] + wg_data[2] + wg_data[3];
    }
}
