// SPDX-License-Identifier: AGPL-3.0-or-later

struct Df64 { hi: f32, lo: f32, }
fn two_prod(a: f32, b: f32) -> Df64 {
    let p = a * b;
    let e = fma(a, b, -p);
    return Df64(p, e);
}
fn two_sum(a: f32, b: f32) -> Df64 {
    let s = a + b; let v = s - a;
    return Df64(s, (a - (s - v)) + (b - v));
}
fn df64_add(a: Df64, b: Df64) -> Df64 {
    let s = two_sum(a.hi, b.hi);
    return two_sum(s.hi, s.lo + a.lo + b.lo);
}
fn df64_mul(a: Df64, b: Df64) -> Df64 {
    let p = two_prod(a.hi, b.hi);
    return two_sum(p.hi, p.lo + a.hi * b.lo + a.lo * b.hi);
}

@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let v = f32(idx + 1u) * 0.001;
    var acc = Df64(0.0, 0.0);
    let val = Df64(v, 0.0);
    for (var i = 0u; i < 64u; i = i + 1u) {
        acc = df64_add(acc, df64_mul(val, val));
    }
    if idx == 0u {
        out[0] = acc.hi;
        out[1] = acc.lo;
    }
}
