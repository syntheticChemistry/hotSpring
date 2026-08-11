// SPDX-License-Identifier: AGPL-3.0-or-later
//
// su3_plaquette_reduce_f64.wgsl — Native f64 plaquette with workgroup reduction
//
// Fused kernel: per-site plaquette (native f64 matmuls) + workgroup partial sum.
// Each workgroup of 64 threads computes per-site plaquette and reduces to
// a single partial sum. Final scalar = sum of partial sums (host or second pass).
//
// This shader is used in the Concurrent precision strategy for the accept/reject
// Hamiltonian and observable measurement, while DF64 handles the bulk force work.
//
// Buffer layout:
//   params: PlaqParams { volume, n_workgroups (for output indexing) }
//   links[V × 4 × 18]: f64 gauge links (read)
//   nbr[V × 8]: u32 neighbor table (read)
//   partial_sums[n_workgroups]: f64 workgroup partial sums (output)

struct PlaqReduceParams {
    volume: u32,
    n_workgroups: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: PlaqReduceParams;
@group(0) @binding(1) var<storage, read> links: array<f64>;
@group(0) @binding(2) var<storage, read> nbr: array<u32>;
@group(0) @binding(3) var<storage, read_write> partial_sums: array<f64>;

var<workgroup> shared_data: array<f64, 64>;

// Native f64 plaquette: full precision SU(3) matrix multiplication.
fn plaquette_re_tr(site: u32, mu: u32, nu: u32) -> f64 {
    let fwd_mu = nbr[site * 8u + mu * 2u];
    let fwd_nu = nbr[site * 8u + nu * 2u];

    let oa = (site * 4u + mu) * 18u;
    let ob = (fwd_mu * 4u + nu) * 18u;
    let oc = (fwd_nu * 4u + mu) * 18u;
    let od = (site * 4u + nu) * 18u;

    // step1 = U_mu(x) * U_nu(x+mu)
    var s1: array<f64, 18>;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            var re: f64 = f64(0.0);
            var im: f64 = f64(0.0);
            for (var k = 0u; k < 3u; k++) {
                let ar = links[oa + (i*3u+k)*2u];
                let ai = links[oa + (i*3u+k)*2u + 1u];
                let br = links[ob + (k*3u+j)*2u];
                let bi = links[ob + (k*3u+j)*2u + 1u];
                re += ar*br - ai*bi;
                im += ar*bi + ai*br;
            }
            s1[(i*3u+j)*2u] = re;
            s1[(i*3u+j)*2u + 1u] = im;
        }
    }

    // step2 = step1 * U_mu†(x+nu)
    var s2: array<f64, 18>;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            var re: f64 = f64(0.0);
            var im: f64 = f64(0.0);
            for (var k = 0u; k < 3u; k++) {
                let ar = s1[(i*3u+k)*2u];
                let ai = s1[(i*3u+k)*2u + 1u];
                let br = links[oc + (j*3u+k)*2u];
                let bi = -links[oc + (j*3u+k)*2u + 1u];
                re += ar*br - ai*bi;
                im += ar*bi + ai*br;
            }
            s2[(i*3u+j)*2u] = re;
            s2[(i*3u+j)*2u + 1u] = im;
        }
    }

    // step3 = Re Tr(step2 * U_nu†(x))
    var trace_re: f64 = f64(0.0);
    for (var i = 0u; i < 3u; i++) {
        var re: f64 = f64(0.0);
        for (var k = 0u; k < 3u; k++) {
            let ar = s2[(i*3u+k)*2u];
            let ai = s2[(i*3u+k)*2u + 1u];
            let br = links[od + (i*3u+k)*2u];
            let bi = -links[od + (i*3u+k)*2u + 1u];
            re += ar*br - ai*bi;
        }
        trace_re += re;
    }

    return trace_re / f64(3.0);
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let site = gid.x + gid.y * nwg.x * 64u;

    // Per-site plaquette sum (6 orientations)
    var plaq_sum: f64 = f64(0.0);
    if (site < params.volume) {
        for (var mu = 0u; mu < 4u; mu++) {
            for (var nu = mu + 1u; nu < 4u; nu++) {
                plaq_sum += plaquette_re_tr(site, mu, nu);
            }
        }
    }

    // Workgroup reduction via shared memory
    shared_data[tid] = plaq_sum;
    workgroupBarrier();

    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let wg_linear = wg_id.x + wg_id.y * nwg.x;
        partial_sums[wg_linear] = shared_data[0];
    }
}
