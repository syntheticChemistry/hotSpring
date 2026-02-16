// SPDX-License-Identifier: AGPL-3.0-only

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;
@group(0) @binding(4) var<storage, read> cell_start_arr: array<u32>;
@group(0) @binding(5) var<storage, read> cell_count_arr: array<u32>;

fn pbc_v3(delta: f64, box_size: f64) -> f64 {
    var d = delta;
    let half = box_size / 2.0;
    if (d > half)  { d = d - box_size; }
    if (d < -half) { d = d + box_size; }
    return d;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n_part = u32(params[0]);
    if (idx >= n_part) { return; }

    let kappa      = params[1];
    let prefactor  = params[2];
    let box_x      = params[4];
    let box_y      = params[5];
    let box_z      = params[6];
    let softening  = params[7];

    // Touch cell buffers to keep bindings alive
    let _cs = cell_start_arr[0u];
    let _cc = cell_count_arr[0u];

    let px = positions[idx * 3u];
    let py = positions[idx * 3u + 1u];
    let pz = positions[idx * 3u + 2u];

    var acc_fx = px - px;
    var acc_fy = px - px;
    var acc_fz = px - px;
    var acc_pe = px - px;

    // ALL-PAIRS loop (same as hybrid) but with cell-list bindings
    // NO cutoff check — accumulate ALL pairs
    for (var j = 0u; j < n_part; j = j + 1u) {
        if (idx == j) { continue; }

        let qx = positions[j * 3u];
        let qy = positions[j * 3u + 1u];
        let qz = positions[j * 3u + 2u];

        let rx = pbc_v3(qx - px, box_x);
        let ry = pbc_v3(qy - py, box_y);
        let rz = pbc_v3(qz - pz, box_z);

        let r_sq = rx * rx + ry * ry + rz * rz;
        // NO cutoff check!

        let r = sqrt_f64(r_sq + softening);
        let scr = exp_f64(-kappa * r);
        let fmag = prefactor * scr * (1.0 + kappa * r) / r_sq;
        let ir = 1.0 / r;

        acc_fx = acc_fx - fmag * rx * ir;
        acc_fy = acc_fy - fmag * ry * ir;
        acc_fz = acc_fz - fmag * rz * ir;
        acc_pe = acc_pe + 0.5 * prefactor * scr * ir;
    }

    forces[idx * 3u]      = acc_fx;
    forces[idx * 3u + 1u] = acc_fy;
    forces[idx * 3u + 2u] = acc_fz;
    pe_buf[idx] = acc_pe;
}
