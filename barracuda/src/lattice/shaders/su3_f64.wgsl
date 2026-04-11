// SPDX-License-Identifier: AGPL-3.0-only
// SU(3) WGSL library: buffer I/O + math operations for lattice QCD.
// Depends on complex_f64.wgsl (prepend before this file).
//
// Math operations (su3_mul, su3_adjoint, su3_trace, etc.) are defined in
// su3_math_f64.wgsl. Shaders that need both I/O and math should prepend
// complex_f64.wgsl → su3_math_f64.wgsl → su3_f64.wgsl.
//
// This file adds buffer load/store on top of the pure-math library.

// SU(3) link variable: 3x3 complex matrix stored as 18 f64 values
// Layout: row-major [row0_col0.re, row0_col0.im, row0_col1.re, ...]

fn su3_load(data: ptr<storage, array<f64>>, offset: u32) -> array<Complex64, 9> {
    var m: array<Complex64, 9>;
    for (var i = 0u; i < 9u; i++) {
        let base = offset + i * 2u;
        m[i] = c64_new(data[base], data[base + 1u]);
    }
    return m;
}

fn su3_store(data: ptr<storage, array<f64>, read_write>, offset: u32, m: array<Complex64, 9>) {
    for (var i = 0u; i < 9u; i++) {
        let base = offset + i * 2u;
        data[base] = m[i].re;
        data[base + 1u] = m[i].im;
    }
}
