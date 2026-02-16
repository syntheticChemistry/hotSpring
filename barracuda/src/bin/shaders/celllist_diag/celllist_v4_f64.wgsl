// SPDX-License-Identifier: AGPL-3.0-only

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;
@group(0) @binding(4) var<storage, read> cell_data: array<f64>;

fn pbc_v4(delta: f64, box_size: f64) -> f64 {
    var d = delta;
    let half = box_size / 2.0;
    if (d > half)  { d = d - box_size; }
    if (d < -half) { d = d + box_size; }
    return d;
}

fn wrap_cell_v4(c: i32, n: i32) -> i32 {
    return ((c % n) + n) % n;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n_part = u32(params[0]);
    if (idx >= n_part) { return; }

    let kappa      = params[1];
    let prefactor  = params[2];
    let cutoff_sq  = params[3];
    let box_x      = params[4];
    let box_y      = params[5];
    let box_z      = params[6];
    let softening  = params[7];
    let ncx        = i32(params[8]);
    let ncy        = i32(params[9]);
    let ncz        = i32(params[10]);
    let csx        = params[11];
    let csy        = params[12];
    let csz        = params[13];

    let px = positions[idx * 3u];
    let py = positions[idx * 3u + 1u];
    let pz = positions[idx * 3u + 2u];

    let my_cx = i32(px / csx);
    let my_cy = i32(py / csy);
    let my_cz = i32(pz / csz);

    var acc_fx = px - px;
    var acc_fy = px - px;
    var acc_fz = px - px;
    var acc_pe = px - px;

    // cell_data layout: [start_0, count_0, start_1, count_1, ...]
    // All stored as f64, cast to u32 when needed
    for (var neigh = 0u; neigh < 27u; neigh = neigh + 1u) {
        let off_x = i32(neigh % 3u) - 1;
        let off_y = i32((neigh / 3u) % 3u) - 1;
        let off_z = i32(neigh / 9u) - 1;

        let nb_cx = wrap_cell_v4(my_cx + off_x, ncx);
        let nb_cy = wrap_cell_v4(my_cy + off_y, ncy);
        let nb_cz = wrap_cell_v4(my_cz + off_z, ncz);
        let cell_linear = u32(nb_cx + nb_cy * ncx + nb_cz * ncx * ncy);

        let start = u32(cell_data[cell_linear * 2u]);
        let cnt   = u32(cell_data[cell_linear * 2u + 1u]);

        for (var k = 0u; k < cnt; k = k + 1u) {
            let j = start + k;
            if (idx == j) { continue; }

            let qx = positions[j * 3u];
            let qy = positions[j * 3u + 1u];
            let qz = positions[j * 3u + 2u];

            let rx = pbc_v4(qx - px, box_x);
            let ry = pbc_v4(qy - py, box_y);
            let rz = pbc_v4(qz - pz, box_z);

            let r_sq = rx * rx + ry * ry + rz * rz;
            if (r_sq > cutoff_sq) { continue; }

            let r = sqrt_f64(r_sq + softening);
            let scr = exp_f64(-kappa * r);
            let fmag = prefactor * scr * (1.0 + kappa * r) / r_sq;
            let ir = 1.0 / r;

            acc_fx = acc_fx - fmag * rx * ir;
            acc_fy = acc_fy - fmag * ry * ir;
            acc_fz = acc_fz - fmag * rz * ir;
            acc_pe = acc_pe + 0.5 * prefactor * scr * ir;
        }
    }

    forces[idx * 3u]      = acc_fx;
    forces[idx * 3u + 1u] = acc_fy;
    forces[idx * 3u + 2u] = acc_fz;
    pe_buf[idx] = acc_pe;
}
