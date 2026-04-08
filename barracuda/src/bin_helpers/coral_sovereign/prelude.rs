// SPDX-License-Identifier: AGPL-3.0-or-later

//! barraCuda WGSL library fragments for composite shader builds in the sovereign path.

/// Library sources for shader concatenation.
/// Composite shaders use barraCuda's vec2<f64> representation (not hotSpring's Complex64 struct).
pub(crate) const LIB_COMPLEX_VEC2: &str = include_str!(
    "../../../../../../primals/barraCuda/crates/barracuda/src/shaders/math/complex_f64.wgsl"
);
pub(crate) const LIB_SU3_VEC2: &str = include_str!(
    "../../../../../../primals/barraCuda/crates/barracuda/src/shaders/math/su3.wgsl"
);
pub(crate) const LIB_LCG_F64: &str = include_str!(
    "../../../../../../primals/barraCuda/crates/barracuda/src/shaders/math/lcg_f64.wgsl"
);
pub(crate) const LIB_SU3_EXTENDED: &str = include_str!(
    "../../../../../../primals/barraCuda/crates/barracuda/src/shaders/math/su3_extended_f64.wgsl"
);

/// Strip c64_exp/c64_phase (use exp_f64/sin_f64/cos_f64 polyfills not available standalone).
pub(crate) fn complex_no_transcendentals() -> String {
    LIB_COMPLEX_VEC2
        .lines()
        .take_while(|l| !l.contains("fn c64_exp"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build a vec2<f64> preamble that suppresses coral-reef's auto-prepend.
/// Coral-reef checks for "struct Complex64" and "fn su3_identity" to skip injection.
/// We add a guard comment so coral-reef sees the marker but our code uses vec2<f64>.
pub(crate) fn vec2_preamble_with_su3() -> String {
    let c = complex_no_transcendentals();
    // "struct Complex64" in this comment suppresses coral-reef's Complex64 auto-prepend.
    // "fn xorshift32" guard not needed — consumer shaders don't use PRNG directly.
    format!(
        "// [guard] struct Complex64 — suppressed, using vec2<f64> convention\n{c}\n{LIB_SU3_VEC2}"
    )
}
