// SPDX-License-Identifier: AGPL-3.0-only
// wilson_plaquette_df64.wgsl — Hybrid Wilson plaquette (DF64 core streaming)
//
// Prepend: complex_f64.wgsl + su3.wgsl + df64_core.wgsl + df64_transcendentals.wgsl + su3_df64.wgsl
//
// HYBRID PRECISION:
//   DF64 (FP32 cores): 4 SU(3) matmuls per plaquette × 6 planes = 24 matmuls/site
//   f64  (FP64 units): Re Tr / 3 scalar output
//
// Buffer layout: UNCHANGED from wilson_plaquette_f64.wgsl (neighbor-buffer indexing).
//   params: PlaqParams { volume }
//   links[V × 4 × 18]: f64 gauge links
//   nbr[V × 8]: u32 neighbor table
//   out[V]: f64 per-site plaquette sum
//
// hotSpring core-streaming discovery (Feb 2026). DF64 expansion: toadStool S60.

struct PlaqParams {
    volume: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: PlaqParams;
@group(0) @binding(1) var<storage, read> links: array<f64>;
@group(0) @binding(2) var<storage, read> nbr: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f64>;

fn load_su3_df64_at(base: u32) -> array<Cdf64, 9> {
    var m: array<Cdf64, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        m[i] = cdf64_from_f64(links[base + i * 2u], links[base + i * 2u + 1u]);
    }
    return m;
}

fn plaquette_re_tr_df64(site: u32, mu: u32, nu: u32) -> f64 {
    let fwd_mu = nbr[site * 8u + mu * 2u];
    let fwd_nu = nbr[site * 8u + nu * 2u];

    let u_mu_x   = load_su3_df64_at((site   * 4u + mu) * 18u);
    let u_nu_xmu = load_su3_df64_at((fwd_mu * 4u + nu) * 18u);
    let u_mu_xnu = load_su3_df64_at((fwd_nu * 4u + mu) * 18u);
    let u_nu_x   = load_su3_df64_at((site   * 4u + nu) * 18u);

    // P = U_mu(x) * U_nu(x+mu) * U_mu†(x+nu) * U_nu†(x)
    let step1 = su3_mul_df64(u_mu_x, u_nu_xmu);
    let step2 = su3_mul_df64(step1, su3_adjoint_df64(u_mu_xnu));
    let step3 = su3_mul_df64(step2, su3_adjoint_df64(u_nu_x));

    let re_tr = su3_re_trace_df64(step3);
    return df64_to_f64(re_tr) / 3.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    let site = idx;
    if site >= params.volume { return; }

    var plaq_sum: f64 = 0.0;
    for (var mu = 0u; mu < 4u; mu++) {
        for (var nu = mu + 1u; nu < 4u; nu++) {
            plaq_sum += plaquette_re_tr_df64(site, mu, nu);
        }
    }
    out[site] = plaq_sum;
}
