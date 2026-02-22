// SPDX-License-Identifier: AGPL-3.0-only
// Complex64 WGSL library: f64-precision complex arithmetic
// Prepend to any shader needing complex number operations.

struct Complex64 {
    re: f64,
    im: f64,
}

fn c64_new(re: f64, im: f64) -> Complex64 {
    return Complex64(re, im);
}

fn c64_zero() -> Complex64 {
    return Complex64(f64(0.0), f64(0.0));
}

fn c64_one() -> Complex64 {
    return Complex64(f64(1.0), f64(0.0));
}

fn c64_add(a: Complex64, b: Complex64) -> Complex64 {
    return Complex64(a.re + b.re, a.im + b.im);
}

fn c64_sub(a: Complex64, b: Complex64) -> Complex64 {
    return Complex64(a.re - b.re, a.im - b.im);
}

fn c64_mul(a: Complex64, b: Complex64) -> Complex64 {
    return Complex64(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

fn c64_conj(a: Complex64) -> Complex64 {
    return Complex64(a.re, -a.im);
}

fn c64_scale(a: Complex64, s: f64) -> Complex64 {
    return Complex64(a.re * s, a.im * s);
}

fn c64_abs_sq(a: Complex64) -> f64 {
    return a.re * a.re + a.im * a.im;
}

fn c64_abs(a: Complex64) -> f64 {
    return sqrt(c64_abs_sq(a));
}

fn c64_inv(a: Complex64) -> Complex64 {
    let d = c64_abs_sq(a);
    return Complex64(a.re / d, -a.im / d);
}

fn c64_div(a: Complex64, b: Complex64) -> Complex64 {
    return c64_mul(a, c64_inv(b));
}

fn c64_exp(a: Complex64) -> Complex64 {
    let r = exp(a.re);
    return Complex64(r * cos(a.im), r * sin(a.im));
}
