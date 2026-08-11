// SPDX-License-Identifier: AGPL-3.0-or-later
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var hi: f32 = f32(gid.x + 1u) * 1.0001;
    var lo: f32 = 0.0;
    let b_hi: f32 = 0.9999;
    let b_lo: f32 = 0.00001;
    for (var i = 0u; i < 256u; i = i + 1u) {
        // Dekker two_prod: p = hi*b_hi, e = fma(hi, b_hi, -p)
        let p = hi * b_hi;
        let e = fma(hi, b_hi, -p);
        // Accumulate: hi = p, lo = e + lo*b_hi + hi*b_lo
        lo = fma(lo, b_hi, fma(hi, b_lo, e));
        hi = p;
        // two_sum for renormalization
        let s = hi + lo;
        lo = lo - (s - hi);
        hi = s;
    }
    out[gid.x * 2u] = hi;
    out[gid.x * 2u + 1u] = lo;
}
