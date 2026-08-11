// SPDX-License-Identifier: AGPL-3.0-or-later
@group(0) @binding(0) var<storage, read_write> out: array<f64>;
var<workgroup> wg_hi: array<f32, 256>;
var<workgroup> wg_lo: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    wg_hi[lid.x] = 1.0; wg_lo[lid.x] = 0.0;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if lid.x < s {
            let a = Df64(wg_hi[lid.x], wg_lo[lid.x]);
            let b = Df64(wg_hi[lid.x + s], wg_lo[lid.x + s]);
            let r = df64_add(a, b);
            wg_hi[lid.x] = r.hi; wg_lo[lid.x] = r.lo;
        }
        workgroupBarrier();
    }
    if lid.x == 0u { out[0] = df64_to_f64(Df64(wg_hi[0], wg_lo[0])); }
}
