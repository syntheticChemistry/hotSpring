// SPDX-License-Identifier: AGPL-3.0-or-later
// Momentum update: P[i] += dt * F[i] for SU(3) algebra elements (DF64 path).
//
// Runs on FP32 cores via DF64 arithmetic, freeing FP64 units for reductions.
// Buffers remain f64 — conversion at load/store boundary only.
//
// Prepend: su3_df64_preamble (provides Df64, df64_from_f64, df64_to_f64, etc.)

struct MomParams {
    n_links: u32,
    _pad0: u32,
    dt: f64,
}

@group(0) @binding(0) var<uniform> params: MomParams;
@group(0) @binding(1) var<storage, read> force: array<f64>;
@group(0) @binding(2) var<storage, read_write> momenta: array<f64>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_links { return; }

    let base = idx * 18u;
    let dt = df64_from_f64(params.dt);

    for (var i = 0u; i < 18u; i++) {
        let p = df64_from_f64(momenta[base + i]);
        let f = df64_from_f64(force[base + i]);
        let result = df64_add(p, df64_mul(dt, f));
        momenta[base + i] = df64_to_f64(result);
    }
}
