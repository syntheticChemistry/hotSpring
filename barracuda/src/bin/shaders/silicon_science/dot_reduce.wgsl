// SPDX-License-Identifier: AGPL-3.0-or-later

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    wg_data[lid.x] = f32(lid.x + 1u) * f32(lid.x + 1u);
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if lid.x < s { wg_data[lid.x] = wg_data[lid.x] + wg_data[lid.x + s]; }
        workgroupBarrier();
    }
    if lid.x == 0u { out[0] = wg_data[0]; }
}
