struct Df64 { hi: f32, lo: f32, }

fn two_sum(a: f32, b: f32) -> Df64 {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return Df64(s, e);
}

fn two_prod(a: f32, b: f32) -> Df64 {
    let p = a * b;
    let e = fma(a, b, -p);
    return Df64(p, e);
}

fn df64_add(a: Df64, b: Df64) -> Df64 {
    let s = two_sum(a.hi, b.hi);
    let e = a.lo + b.lo;
    let v = two_sum(s.hi, s.lo + e);
    return v;
}

fn df64_mul(a: Df64, b: Df64) -> Df64 {
    let p = two_prod(a.hi, b.hi);
    let lo = p.lo + fma(a.hi, b.lo, a.lo * b.hi);
    let r = two_sum(p.hi, lo);
    return r;
}
