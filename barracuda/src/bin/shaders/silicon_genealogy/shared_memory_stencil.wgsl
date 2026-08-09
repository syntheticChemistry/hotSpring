// Workgroup shared memory experiment — tile-local stencil with neighbor reuse.
//
// Without shared memory: each thread loads its 8 neighbors from VRAM (8 reads/thread).
// With shared memory: workgroup cooperatively loads a tile + halo into shared,
// then each thread reads neighbors from shared (10-100× faster than VRAM).
//
// For a 4×4×4×4 = 256 thread workgroup arranged as a spatial tile:
// - Interior threads: ALL neighbors in shared memory (zero VRAM reads!)
// - Boundary threads: some neighbors from VRAM halo
//
// Expected: 3-10× speedup for stencil-heavy operations (force, Laplacian).

struct Params {
    volume: u32,
    tile_l: u32,       // tile size per dimension (e.g., 4 → 4⁴=256 workgroup)
    lattice_l: u32,    // full lattice size per dimension
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>>;

// Shared memory tile: tile_l⁴ interior + 1-deep halo on each face
// For tile_l=4: padded = 6⁴ = 1296 vec4<f32> = 20.7 KB
// WGSL workgroup memory limit: typically 16-48 KB per workgroup
var<workgroup> tile: array<vec4<f32>, 1296>;

// Convert 4D coordinates to linear index within padded tile
fn tile_idx(x: u32, y: u32, z: u32, t: u32, pl: u32) -> u32 {
    return x + pl * (y + pl * (z + pl * t));
}

// Convert global site to 4D coordinates
fn site_to_4d(site: u32, l: u32) -> vec4<u32> {
    let x = site % l;
    let y = (site / l) % l;
    let z = (site / (l * l)) % l;
    let t = site / (l * l * l);
    return vec4<u32>(x, y, z, t);
}

// Global linear index from 4D with periodic boundary
fn global_idx(x: i32, y: i32, z: i32, t: i32, l: i32) -> u32 {
    let xp = ((x % l) + l) % l;
    let yp = ((y % l) + l) % l;
    let zp = ((z % l) + l) % l;
    let tp = ((t % l) + l) % l;
    return u32(xp + l * (yp + l * (zp + l * tp)));
}

@compute @workgroup_size(256)
fn stencil_shared(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tl = params.tile_l;
    let pl = tl + 2u; // padded tile dimension (with halo)
    let ll = params.lattice_l;

    // This thread's position within the tile (4D from linear local_id)
    let local_id = lid.x;
    let lx = local_id % tl;
    let ly = (local_id / tl) % tl;
    let lz = (local_id / (tl * tl)) % tl;
    let lt = local_id / (tl * tl * tl);

    // This workgroup's origin in global lattice (4D)
    let wg_id = wid.x;
    let wx = (wg_id % (ll / tl)) * tl;
    let wy = ((wg_id / (ll / tl)) % (ll / tl)) * tl;
    let wz = ((wg_id / ((ll / tl) * (ll / tl))) % (ll / tl)) * tl;
    let wt = (wg_id / ((ll / tl) * (ll / tl) * (ll / tl))) * tl;

    // Global coordinates of this thread
    let gx = i32(wx + lx);
    let gy = i32(wy + ly);
    let gz = i32(wz + lz);
    let gt = i32(wt + lt);
    let l = i32(ll);

    // Phase 1: Cooperatively load tile + halo into shared memory
    // Each thread loads its own interior point
    let interior_tidx = tile_idx(lx + 1u, ly + 1u, lz + 1u, lt + 1u, pl);
    let g_idx = global_idx(gx, gy, gz, gt, l);
    tile[interior_tidx] = field[g_idx];

    // Boundary threads also load halo points
    // Face halos: threads on faces load the adjacent exterior point
    if lx == 0u {
        tile[tile_idx(0u, ly + 1u, lz + 1u, lt + 1u, pl)] = field[global_idx(gx - 1, gy, gz, gt, l)];
    }
    if lx == tl - 1u {
        tile[tile_idx(tl + 1u, ly + 1u, lz + 1u, lt + 1u, pl)] = field[global_idx(gx + 1, gy, gz, gt, l)];
    }
    if ly == 0u {
        tile[tile_idx(lx + 1u, 0u, lz + 1u, lt + 1u, pl)] = field[global_idx(gx, gy - 1, gz, gt, l)];
    }
    if ly == tl - 1u {
        tile[tile_idx(lx + 1u, tl + 1u, lz + 1u, lt + 1u, pl)] = field[global_idx(gx, gy + 1, gz, gt, l)];
    }
    if lz == 0u {
        tile[tile_idx(lx + 1u, ly + 1u, 0u, lt + 1u, pl)] = field[global_idx(gx, gy, gz - 1, gt, l)];
    }
    if lz == tl - 1u {
        tile[tile_idx(lx + 1u, ly + 1u, tl + 1u, lt + 1u, pl)] = field[global_idx(gx, gy, gz + 1, gt, l)];
    }
    if lt == 0u {
        tile[tile_idx(lx + 1u, ly + 1u, lz + 1u, 0u, pl)] = field[global_idx(gx, gy, gz, gt - 1, l)];
    }
    if lt == tl - 1u {
        tile[tile_idx(lx + 1u, ly + 1u, lz + 1u, tl + 1u, pl)] = field[global_idx(gx, gy, gz, gt + 1, l)];
    }

    // Synchronize — all threads must finish loading before any reads from shared
    workgroupBarrier();

    // Phase 2: Compute stencil from shared memory (ZERO VRAM reads!)
    let cx = lx + 1u;
    let cy = ly + 1u;
    let cz = lz + 1u;
    let ct = lt + 1u;

    var acc = vec4<f32>(0.0);
    acc += tile[tile_idx(cx - 1u, cy, cz, ct, pl)];
    acc += tile[tile_idx(cx + 1u, cy, cz, ct, pl)];
    acc += tile[tile_idx(cx, cy - 1u, cz, ct, pl)];
    acc += tile[tile_idx(cx, cy + 1u, cz, ct, pl)];
    acc += tile[tile_idx(cx, cy, cz - 1u, ct, pl)];
    acc += tile[tile_idx(cx, cy, cz + 1u, ct, pl)];
    acc += tile[tile_idx(cx, cy, cz, ct - 1u, pl)];
    acc += tile[tile_idx(cx, cy, cz, ct + 1u, pl)];

    let center = tile[tile_idx(cx, cy, cz, ct, pl)];
    result[g_idx] = acc * 0.125 - center;
}

// Baseline: no shared memory, direct VRAM reads
@compute @workgroup_size(256)
fn stencil_global(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let site = gid.x;
    if site >= params.volume { return; }

    let ll = params.lattice_l;
    let l = i32(ll);
    let coords = site_to_4d(site, ll);
    let x = i32(coords.x);
    let y = i32(coords.y);
    let z = i32(coords.z);
    let t = i32(coords.w);

    var acc = vec4<f32>(0.0);
    acc += field[global_idx(x - 1, y, z, t, l)];
    acc += field[global_idx(x + 1, y, z, t, l)];
    acc += field[global_idx(x, y - 1, z, t, l)];
    acc += field[global_idx(x, y + 1, z, t, l)];
    acc += field[global_idx(x, y, z - 1, t, l)];
    acc += field[global_idx(x, y, z + 1, t, l)];
    acc += field[global_idx(x, y, z, t - 1, l)];
    acc += field[global_idx(x, y, z, t + 1, l)];

    let center = field[site];
    result[site] = acc * 0.125 - center;
}
