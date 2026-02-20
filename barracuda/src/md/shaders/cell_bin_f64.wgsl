// SPDX-License-Identifier: AGPL-3.0-only
//
// GPU Cell-List Pass 1/3: Atomic Particle Binning
//
// Assigns each particle to its spatial cell and atomically increments the
// per-cell count. Positions are read as f64 but cast to f32 for cell
// assignment (f32 precision is sufficient for spatial binning).
//
// Dispatch: (ceil(N / 64), 1, 1)
//
// Bindings:
//   0: params       uniform  — [n, mx, my, mz, box_lx, box_ly, box_lz, cell_size] (u32/f32-bits)
//   1: positions    [N*3]    f64, read
//   2: cell_counts  [Nc]     atomic<u32>, read_write (zero-initialized before dispatch)
//   3: cell_ids     [N]      u32, write

struct CellBinParams {
    n_particles: u32,
    mx:          u32,
    my:          u32,
    mz:          u32,
    box_lx:      f32,
    box_ly:      f32,
    box_lz:      f32,
    cell_size:   f32,
}

@group(0) @binding(0) var<uniform>             params:      CellBinParams;
@group(0) @binding(1) var<storage, read>       positions:   array<f64>;
@group(0) @binding(2) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cell_ids:    array<u32>;

@compute @workgroup_size(64)
fn cell_bin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_particles) { return; }

    let px = f32(positions[i * 3u]);
    let py = f32(positions[i * 3u + 1u]);
    let pz = f32(positions[i * 3u + 2u]);

    // Periodic wrap into [0, L)
    let wx = ((px % params.box_lx) + params.box_lx) % params.box_lx;
    let wy = ((py % params.box_ly) + params.box_ly) % params.box_ly;
    let wz = ((pz % params.box_lz) + params.box_lz) % params.box_lz;

    let cx = min(u32(wx / params.cell_size), params.mx - 1u);
    let cy = min(u32(wy / params.cell_size), params.my - 1u);
    let cz = min(u32(wz / params.cell_size), params.mz - 1u);

    let cell_id = cx + params.mx * cy + params.mx * params.my * cz;

    cell_ids[i] = cell_id;
    atomicAdd(&cell_counts[cell_id], 1u);
}
