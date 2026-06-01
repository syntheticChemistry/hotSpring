// SPDX-License-Identifier: AGPL-3.0-or-later
struct Df64 { hi: f32, lo: f32, }
fn two_sum(a: f32, b: f32) -> Df64 {
    let s = a + b; let v = s - a;
    return Df64(s, (a - (s - v)) + (b - v));
}
fn two_prod(a: f32, b: f32) -> Df64 {
    let p = a * b; let e = fma(a, b, -p);
    return Df64(p, e);
}
fn df64_add(a: Df64, b: Df64) -> Df64 {
    let s = two_sum(a.hi, b.hi);
    return two_sum(s.hi, s.lo + a.lo + b.lo);
}
fn df64_mul(a: Df64, b: Df64) -> Df64 {
    let p = two_prod(a.hi, b.hi);
    return two_sum(p.hi, p.lo + a.hi * b.lo + a.lo * b.hi);
}
fn df64_from_f64(v: f64) -> Df64 {
    let hi = f32(v); return Df64(hi, f32(v - f64(hi)));
}
fn df64_to_f64(v: Df64) -> f64 { return f64(v.hi) + f64(v.lo); }
