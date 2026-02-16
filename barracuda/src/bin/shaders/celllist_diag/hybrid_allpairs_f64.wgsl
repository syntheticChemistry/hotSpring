// SPDX-License-Identifier: AGPL-3.0-only

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;
@group(0) @binding(4) var<storage, read> cell_start: array<u32>;
@group(0) @binding(5) var<storage, read> cell_count: array<u32>;

fn pbc_delta_h(delta: f64, box_size: f64) -> f64 {
    return delta - box_size * round_f64(delta / box_size);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let kappa     = params[1];
    let prefactor = params[2];
    let cutoff_sq = params[3];
    let box_x     = params[4];
    let box_y     = params[5];
    let box_z     = params[6];
    let eps       = params[7];

    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];

    var fx = xi - xi;
    var fy = xi - xi;
    var fz = xi - xi;
    var pe = xi - xi;

    // Touch cell_start/cell_count to keep bindings alive (compiler optimization workaround)
    let _cs0 = cell_start[0u];
    let _cc0 = cell_count[0u];

    // ALL-PAIRS loop (ignoring cell_start/cell_count)
    for (var j = 0u; j < n; j = j + 1u) {
        if (i == j) { continue; }
        let xj = positions[j * 3u];
        let yj = positions[j * 3u + 1u];
        let zj = positions[j * 3u + 2u];

        var ddx = pbc_delta_h(xj - xi, box_x);
        var ddy = pbc_delta_h(yj - yi, box_y);
        var ddz = pbc_delta_h(zj - zi, box_z);

        let r_sq = ddx * ddx + ddy * ddy + ddz * ddz;
        if (r_sq > cutoff_sq) { continue; }

        let r = sqrt_f64(r_sq + eps);
        let screening = exp_f64(-kappa * r);
        let force_mag = prefactor * screening * (1.0 + kappa * r) / r_sq;
        let inv_r = 1.0 / r;

        fx = fx - force_mag * ddx * inv_r;
        fy = fy - force_mag * ddy * inv_r;
        fz = fz - force_mag * ddz * inv_r;
        pe = pe + 0.5 * prefactor * screening * inv_r;
    }

    forces[i * 3u]      = fx;
    forces[i * 3u + 1u] = fy;
    forces[i * 3u + 2u] = fz;
    pe_buf[i] = pe;
}
