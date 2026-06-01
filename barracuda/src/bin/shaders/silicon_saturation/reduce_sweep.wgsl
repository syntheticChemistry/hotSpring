// SPDX-License-Identifier: AGPL-3.0-or-later
var<workgroup> wg_data: array<f32, 1024>;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1024)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    wg_data[lid.x] = f32(gid.x + 1u) * 0.001;
    workgroupBarrier();
    // Full tree reduction
    for (var s = 512u; s > 0u; s = s >> 1u) {
        if lid.x < s {
            wg_data[lid.x] = wg_data[lid.x] + wg_data[lid.x + s];
        }
        workgroupBarrier();
    }
    if lid.x == 0u { out[gid.x / 1024u] = wg_data[0]; }
}
