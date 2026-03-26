// SPDX-License-Identifier: AGPL-3.0-only
//
// Multi-shift CG search direction update: p[i] = zeta * r[i] + beta * p[i]
//
// Used per shifted system in the multi-shift CG solver. Combines the
// shared residual r (scaled by zeta) with the per-shift search direction
// p (scaled by beta) in a single fused kernel.

struct Params {
    n: u32,
    pad0: u32,
    zeta: f64,
    beta: f64,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> r: array<f64>;
@group(0) @binding(2) var<storage, read_write> p: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    let i = idx;
    if i >= params.n { return; }
    p[i] = params.zeta * r[i] + params.beta * p[i];
}
