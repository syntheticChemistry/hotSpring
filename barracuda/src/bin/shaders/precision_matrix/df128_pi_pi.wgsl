// SPDX-License-Identifier: AGPL-3.0-or-later
struct Df128 { hi: f64, lo: f64, }
fn two_sum_64(a: f64, b: f64) -> Df128 {
    let s = a + b;
    let v = s - a;
    return Df128(s, (a - (s - v)) + (b - v));
}
fn two_prod_64(a: f64, b: f64) -> Df128 {
    let p = a * b;
    let e = fma(a, b, -p);
    return Df128(p, e);
}
fn df128_mul(a: Df128, b: Df128) -> Df128 {
    let p = two_prod_64(a.hi, b.hi);
    return two_sum_64(p.hi, p.lo + a.hi * b.lo + a.lo * b.hi);
}

@group(0) @binding(0) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let pi = Df128(3.14159265358979323846lf, 1.2246467991473532e-16lf);
    let r = df128_mul(pi, pi);
    out[0] = r.hi;
    out[1] = r.lo;
}
