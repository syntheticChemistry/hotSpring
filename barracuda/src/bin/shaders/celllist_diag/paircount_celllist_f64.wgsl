// SPDX-License-Identifier: AGPL-3.0-only

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> debug_out: array<f64>;
@group(0) @binding(2) var<storage, read> params: array<f64>;
@group(0) @binding(3) var<storage, read> cell_start_buf: array<u32>;
@group(0) @binding(4) var<storage, read> cell_count_buf: array<u32>;

fn cell_idx_d(cx: i32, cy: i32, cz: i32, nx: i32, ny: i32, nz: i32) -> u32 {
    let wx = ((cx % nx) + nx) % nx;
    let wy = ((cy % ny) + ny) % ny;
    let wz = ((cz % nz) + nz) % nz;
    return u32(wx + wy * nx + wz * nx * ny);
}

fn pbc_delta_d(delta: f64, box_size: f64) -> f64 {
    return delta - box_size * round_f64(delta / box_size);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let cutoff_sq = params[3];
    let box_x     = params[4];
    let box_y     = params[5];
    let box_z     = params[6];
    let nx        = i32(params[8]);
    let ny        = i32(params[9]);
    let nz        = i32(params[10]);
    let cell_sx   = params[11];
    let cell_sy   = params[12];
    let cell_sz   = params[13];

    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];

    let ci_x = i32(xi / cell_sx);
    let ci_y = i32(yi / cell_sy);
    let ci_z = i32(zi / cell_sz);

    var pair_count = xi - xi;  // 0.0 as f64
    var total_checked = xi - xi;
    var cells_visited = xi - xi;

    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let c_idx = cell_idx_d(ci_x + dx, ci_y + dy, ci_z + dz, nx, ny, nz);
                let start = cell_start_buf[c_idx];
                let count = cell_count_buf[c_idx];
                cells_visited = cells_visited + 1.0;
                total_checked = total_checked + f64(count);

                for (var jj = 0u; jj < count; jj = jj + 1u) {
                    let j = start + jj;
                    if (i == j) { continue; }

                    let xj = positions[j * 3u];
                    let yj = positions[j * 3u + 1u];
                    let zj = positions[j * 3u + 2u];

                    var ddx = pbc_delta_d(xj - xi, box_x);
                    var ddy = pbc_delta_d(yj - yi, box_y);
                    var ddz = pbc_delta_d(zj - zi, box_z);

                    let r_sq = ddx * ddx + ddy * ddy + ddz * ddz;
                    if (r_sq > cutoff_sq) { continue; }
                    pair_count = pair_count + 1.0;
                }
            }
        }
    }

    // Output: [cell_x, cell_y, cell_z, cells_visited, total_checked, pair_count, nx_read, cell_sx_read]
    debug_out[i * 8u]      = f64(ci_x);
    debug_out[i * 8u + 1u] = f64(ci_y);
    debug_out[i * 8u + 2u] = f64(ci_z);
    debug_out[i * 8u + 3u] = cells_visited;
    debug_out[i * 8u + 4u] = total_checked;
    debug_out[i * 8u + 5u] = pair_count;
    debug_out[i * 8u + 6u] = f64(nx);
    debug_out[i * 8u + 7u] = cell_sx;
}
