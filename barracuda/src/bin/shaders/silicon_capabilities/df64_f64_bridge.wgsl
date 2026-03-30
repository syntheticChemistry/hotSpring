fn df64_from_f64(v: f64) -> Df64 {
    let hi = f32(v);
    let lo = f32(v - f64(hi));
    return Df64(hi, lo);
}

fn df64_to_f64(v: Df64) -> f64 {
    return f64(v.hi) + f64(v.lo);
}
