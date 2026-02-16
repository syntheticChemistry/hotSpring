// SPDX-License-Identifier: AGPL-3.0-only

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> j_trace: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;
@group(0) @binding(4) var<storage, read> cell_data: array<f64>;

fn wrap_v6(c: i32, n: i32) -> i32 {
    return ((c % n) + n) % n;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_part = u32(params[0]);
    let ncx = i32(params[8]);
    let ncy = i32(params[9]);
    let ncz = i32(params[10]);
    let csx = params[11];
    let csy = params[12];
    let csz = params[13];

    let px = positions[0u];
    let py = positions[1u];
    let pz = positions[2u];

    let my_cx = i32(px / csx);
    let my_cy = i32(py / csy);
    let my_cz = i32(pz / csz);

    var out_idx = 0u;

    // Write header: [my_cx, my_cy, my_cz, n_part]
    j_trace[0u] = f64(my_cx);
    j_trace[1u] = f64(my_cy);
    j_trace[2u] = f64(my_cz);
    j_trace[3u] = f64(n_part);
    out_idx = 4u;

    for (var neigh = 0u; neigh < 27u; neigh = neigh + 1u) {
        let off_x = i32(neigh % 3u) - 1;
        let off_y = i32((neigh / 3u) % 3u) - 1;
        let off_z = i32(neigh / 9u) - 1;

        let nb_cx = wrap_v6(my_cx + off_x, ncx);
        let nb_cy = wrap_v6(my_cy + off_y, ncy);
        let nb_cz = wrap_v6(my_cz + off_z, ncz);
        let cell_linear = u32(nb_cx + nb_cy * ncx + nb_cz * ncx * ncy);

        let start = u32(cell_data[cell_linear * 2u]);
        let cnt   = u32(cell_data[cell_linear * 2u + 1u]);

        // Record: [neigh, off_x, off_y, off_z, cell_linear, start, cnt]
        j_trace[out_idx]      = f64(neigh);
        j_trace[out_idx + 1u] = f64(off_x);
        j_trace[out_idx + 2u] = f64(off_y);
        j_trace[out_idx + 3u] = f64(off_z);
        j_trace[out_idx + 4u] = f64(cell_linear);
        j_trace[out_idx + 5u] = f64(start);
        j_trace[out_idx + 6u] = f64(cnt);
        out_idx = out_idx + 7u;

        for (var k = 0u; k < cnt; k = k + 1u) {
            let j = start + k;
            // Record each j visited
            j_trace[out_idx] = f64(j);
            out_idx = out_idx + 1u;
        }
    }
    // Final marker
    j_trace[out_idx] = px - px - f64(999);
    pe_buf[0u] = f64(out_idx);
}
