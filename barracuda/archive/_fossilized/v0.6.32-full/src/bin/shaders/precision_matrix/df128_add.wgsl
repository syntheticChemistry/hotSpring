// SPDX-License-Identifier: AGPL-3.0-or-later
struct Df128 { hi: f64, lo: f64, }
fn two_sum_64(a: f64, b: f64) -> Df128 {
    let s = a + b;
    let v = s - a;
    return Df128(s, (a - (s - v)) + (b - v));
}
fn df128_add(a: Df128, b: Df128) -> Df128 {
    let s = two_sum_64(a.hi, b.hi);
    return two_sum_64(s.hi, s.lo + a.lo + b.lo);
}

@group(0) @binding(0) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let one = Df128(1.0lf, 0.0lf);
    let two = df128_add(one, one);
    out[0] = two.hi;
    out[1] = two.lo;
}
