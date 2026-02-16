// SPDX-License-Identifier: AGPL-3.0-only

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> counts: array<f64>;
@group(0) @binding(2) var<storage, read> params: array<f64>;

fn pbc_delta_ac(delta: f64, box_size: f64) -> f64 {
    return delta - box_size * round_f64(delta / box_size);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let cutoff_sq = params[3];
    let box_x = params[4];
    let box_y = params[5];
    let box_z = params[6];

    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];

    var pc = xi - xi;  // 0.0
    var pe = xi - xi;

    for (var j = 0u; j < n; j = j + 1u) {
        if (i == j) { continue; }
        let xj = positions[j * 3u];
        let yj = positions[j * 3u + 1u];
        let zj = positions[j * 3u + 2u];
        var ddx = pbc_delta_ac(xj - xi, box_x);
        var ddy = pbc_delta_ac(yj - yi, box_y);
        var ddz = pbc_delta_ac(zj - zi, box_z);
        let r_sq = ddx * ddx + ddy * ddy + ddz * ddz;
        if (r_sq > cutoff_sq) { continue; }
        pc = pc + 1.0;
        let r = sqrt_f64(r_sq);
        let kappa = params[1];
        let prefactor = params[2];
        pe = pe + 0.5 * prefactor * exp_f64(-kappa * r) / r;
    }

    counts[i * 2u] = pc;
    counts[i * 2u + 1u] = pe;
}
