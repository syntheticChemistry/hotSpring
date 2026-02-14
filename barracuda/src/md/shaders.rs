//! f64 WGSL shaders for GPU molecular dynamics
//!
//! All shaders use SHADER_F64 and the math_f64 library (exp_f64, sqrt_f64, etc.).
//! Prepend math_f64 preamble via `ShaderTemplate::with_math_f64()` before compilation.

// ═══════════════════════════════════════════════════════════════════
// Yukawa All-Pairs Force Kernel (f64)
// ═══════════════════════════════════════════════════════════════════
//
// Computes pairwise Yukawa forces with PBC minimum-image convention.
// Each thread handles one particle, loops over all others.
// O(N²) — suitable for N ≤ ~5,000.  Cell-list version planned for N > 5k.
//
// Physics:
//   U(r) = prefactor * exp(-kappa * r) / r
//   F = -dU/dr = prefactor * exp(-kappa*r) * (1 + kappa*r) / r² * r_hat
//
// The shader also accumulates per-particle potential energy (half-counted).

pub const SHADER_YUKAWA_FORCE: &str = r#"
// Yukawa All-Pairs Force (f64) with PBC + potential energy
//
// Bindings:
//   0: positions  [N*3] f64, read     — (x,y,z) per particle
//   1: forces     [N*3] f64, write    — (fx,fy,fz) per particle
//   2: pe_buf     [N]   f64, write    — per-particle PE (half-counted)
//   3: params     [12]  f64, read     — simulation parameters

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;

// params layout:
//   [0] = n_particles (as f64, cast to u32)
//   [1] = kappa (screening parameter, reduced units)
//   [2] = prefactor (coupling: Gamma * a_ws in reduced = Gamma for OCP convention)
//   [3] = cutoff_sq (rc² in reduced units)
//   [4] = box_x (box side in reduced units)
//   [5] = box_y
//   [6] = box_z
//   [7] = epsilon (softening, typically 0 or 1e-30)

fn pbc_delta(delta: f64, box_size: f64) -> f64 {
    return delta - box_size * round_f64(delta / box_size);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];

    let kappa    = params[1];
    let prefactor = params[2];
    let cutoff_sq = params[3];
    let box_x    = params[4];
    let box_y    = params[5];
    let box_z    = params[6];
    let eps      = params[7];

    // Accumulate force and PE
    var fx = xi - xi;  // 0.0 as f64
    var fy = xi - xi;
    var fz = xi - xi;
    var pe = xi - xi;

    for (var j = 0u; j < n; j = j + 1u) {
        if (i == j) { continue; }

        let xj = positions[j * 3u];
        let yj = positions[j * 3u + 1u];
        let zj = positions[j * 3u + 2u];

        // PBC minimum image
        var dx = pbc_delta(xj - xi, box_x);
        var dy = pbc_delta(yj - yi, box_y);
        var dz = pbc_delta(zj - zi, box_z);

        let r_sq = dx * dx + dy * dy + dz * dz;

        if (r_sq > cutoff_sq) { continue; }

        let r = sqrt_f64(r_sq + eps);

        // Yukawa force: F = prefactor * exp(-kappa*r) * (1 + kappa*r) / r^2
        let screening = exp_f64(-kappa * r);
        let force_mag = prefactor * screening * (1.0 + kappa * r) / r_sq;

        // Force on particle i due to j: repulsive → push AWAY from j
        // F_i = -prefactor * exp(-κr) * (1+κr)/r² * r̂_ij
        // r̂_ij = (dx,dy,dz)/r points from i to j, so negate for repulsion
        let inv_r = 1.0 / r;
        fx = fx - force_mag * dx * inv_r;
        fy = fy - force_mag * dy * inv_r;
        fz = fz - force_mag * dz * inv_r;

        // PE: U = prefactor * exp(-kappa*r) / r  (half-count: each pair once)
        pe = pe + 0.5 * prefactor * screening * inv_r;
    }

    forces[i * 3u]      = fx;
    forces[i * 3u + 1u] = fy;
    forces[i * 3u + 2u] = fz;
    pe_buf[i] = pe;
}
"#;

// ═══════════════════════════════════════════════════════════════════
// Velocity-Verlet Half-Kick + Drift + PBC Wrap (f64)
// ═══════════════════════════════════════════════════════════════════
//
// Fused kernel: performs VV half-kick, drift, and PBC position wrapping
// in a single dispatch. This is steps 1-3 of the VV algorithm:
//   1. v += 0.5 * dt * a       (half-kick)
//   2. x += dt * v             (drift)
//   3. x = pbc_wrap(x)         (wrap into box)
//
// After this, forces are recomputed, then another half-kick is applied.

pub const SHADER_VV_KICK_DRIFT: &str = r#"
// VV half-kick + drift + PBC wrap (f64)
//
// Bindings:
//   0: positions    [N*3] f64, read-write  — updated in-place
//   1: velocities   [N*3] f64, read-write  — updated in-place
//   2: forces       [N*3] f64, read        — current forces
//   3: params       [8]   f64, read        — [n, dt, mass, _, box_x, box_y, box_z, _]

@group(0) @binding(0) var<storage, read_write> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> velocities: array<f64>;
@group(0) @binding(2) var<storage, read> forces: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;

// params: [n_particles, dt, mass, _, box_x, box_y, box_z, _]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let dt    = params[1];
    let mass  = params[2];
    let box_x = params[4];
    let box_y = params[5];
    let box_z = params[6];

    let inv_m = 1.0 / mass;
    let half_dt = 0.5 * dt;

    // Load
    var vx = velocities[i * 3u];
    var vy = velocities[i * 3u + 1u];
    var vz = velocities[i * 3u + 2u];

    let ax = forces[i * 3u]      * inv_m;
    let ay = forces[i * 3u + 1u] * inv_m;
    let az = forces[i * 3u + 2u] * inv_m;

    // Half-kick: v += 0.5 * dt * a
    vx = vx + half_dt * ax;
    vy = vy + half_dt * ay;
    vz = vz + half_dt * az;

    // Drift: x += dt * v
    var px = positions[i * 3u]      + dt * vx;
    var py = positions[i * 3u + 1u] + dt * vy;
    var pz = positions[i * 3u + 2u] + dt * vz;

    // PBC wrap: keep x in [0, box)
    px = px - box_x * floor_f64(px / box_x);
    py = py - box_y * floor_f64(py / box_y);
    pz = pz - box_z * floor_f64(pz / box_z);

    // Store
    positions[i * 3u]      = px;
    positions[i * 3u + 1u] = py;
    positions[i * 3u + 2u] = pz;
    velocities[i * 3u]      = vx;
    velocities[i * 3u + 1u] = vy;
    velocities[i * 3u + 2u] = vz;
}
"#;

// ═══════════════════════════════════════════════════════════════════
// Velocity-Verlet Second Half-Kick (f64)
// ═══════════════════════════════════════════════════════════════════
//
// After forces are recomputed with new positions, apply the second
// half-kick: v += 0.5 * dt * a_new

pub const SHADER_VV_HALF_KICK: &str = r#"
// VV second half-kick (f64)
//
// Bindings:
//   0: velocities   [N*3] f64, read-write  — updated in-place
//   1: forces       [N*3] f64, read        — NEW forces after drift
//   2: params       [4]   f64, read        — [n, dt, mass, _]

@group(0) @binding(0) var<storage, read_write> velocities: array<f64>;
@group(0) @binding(1) var<storage, read> forces: array<f64>;
@group(0) @binding(2) var<storage, read> params: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let dt   = params[1];
    let mass = params[2];

    let inv_m   = 1.0 / mass;
    let half_dt = 0.5 * dt;

    velocities[i * 3u]      = velocities[i * 3u]      + half_dt * forces[i * 3u]      * inv_m;
    velocities[i * 3u + 1u] = velocities[i * 3u + 1u] + half_dt * forces[i * 3u + 1u] * inv_m;
    velocities[i * 3u + 2u] = velocities[i * 3u + 2u] + half_dt * forces[i * 3u + 2u] * inv_m;
}
"#;

// ═══════════════════════════════════════════════════════════════════
// Berendsen Thermostat (f64)
// ═══════════════════════════════════════════════════════════════════
//
// Rescales velocities: v *= sqrt(1 + (dt/tau) * (T_target/T_current - 1))
// Applied once per step during equilibration.

pub const SHADER_BERENDSEN: &str = r#"
// Berendsen velocity rescaling (f64)
//
// Bindings:
//   0: velocities [N*3] f64, read-write
//   1: params     [4]   f64, read  — [n, scale_factor, _, _]

@group(0) @binding(0) var<storage, read_write> velocities: array<f64>;
@group(0) @binding(1) var<storage, read> params: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let scale = params[1];

    velocities[i * 3u]      = velocities[i * 3u]      * scale;
    velocities[i * 3u + 1u] = velocities[i * 3u + 1u] * scale;
    velocities[i * 3u + 2u] = velocities[i * 3u + 2u] * scale;
}
"#;

// ═══════════════════════════════════════════════════════════════════
// Kinetic Energy Reduction (f64)
// ═══════════════════════════════════════════════════════════════════
//
// Computes per-particle KE = 0.5 * m * v² for temperature calculation.

pub const SHADER_KINETIC_ENERGY: &str = r#"
// Per-particle kinetic energy (f64)
//
// Bindings:
//   0: velocities [N*3] f64, read
//   1: ke_buf     [N]   f64, write
//   2: params     [4]   f64, read  — [n, mass, _, _]

@group(0) @binding(0) var<storage, read> velocities: array<f64>;
@group(0) @binding(1) var<storage, read_write> ke_buf: array<f64>;
@group(0) @binding(2) var<storage, read> params: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let mass = params[1];
    let vx = velocities[i * 3u];
    let vy = velocities[i * 3u + 1u];
    let vz = velocities[i * 3u + 2u];

    ke_buf[i] = 0.5 * mass * (vx * vx + vy * vy + vz * vz);
}
"#;

// ═══════════════════════════════════════════════════════════════════
// Yukawa Cell-List Force Kernel (f64)
// ═══════════════════════════════════════════════════════════════════
//
// Same physics as SHADER_YUKAWA_FORCE but uses a cell list for O(N) scaling.
// Particles are pre-sorted by cell index on CPU. Each thread loops over
// 27 neighbor cells using cell_start/cell_count arrays.
//
// Requires: particles sorted by cell index, cell_start[] and cell_count[]
// uploaded to GPU.

pub const SHADER_YUKAWA_FORCE_CELLLIST: &str = r#"
// Yukawa Cell-List Force (f64) with PBC + potential energy
//
// Bindings:
//   0: positions    [N*3]       f64, read  — sorted by cell index
//   1: forces       [N*3]       f64, write
//   2: pe_buf       [N]         f64, write — per-particle PE (half-counted)
//   3: params       [16]        f64, read  — simulation parameters
//   4: cell_start   [n_cells_total] u32, read — first particle index in each cell
//   5: cell_count   [n_cells_total] u32, read — number of particles in each cell

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;
@group(0) @binding(4) var<storage, read> cell_start: array<u32>;
@group(0) @binding(5) var<storage, read> cell_count: array<u32>;

// params layout:
//   [0]  = n_particles
//   [1]  = kappa
//   [2]  = prefactor (1.0 in reduced units)
//   [3]  = cutoff_sq
//   [4]  = box_x
//   [5]  = box_y
//   [6]  = box_z
//   [7]  = epsilon (softening)
//   [8]  = n_cells_x
//   [9]  = n_cells_y
//   [10] = n_cells_z
//   [11] = cell_size_x
//   [12] = cell_size_y
//   [13] = cell_size_z
//   [14] = n_cells_total

fn pbc_delta_cl(delta: f64, box_size: f64) -> f64 {
    return delta - box_size * round_f64(delta / box_size);
}

// Map 3D cell coordinates to linear index
// Uses branch-based wrapping instead of modular arithmetic because
// WGSL i32 remainder (%) has inconsistent behavior for negative operands
// across GPU drivers (Naga/NVIDIA), causing cell 0 to be revisited.
fn cell_idx(cx: i32, cy: i32, cz: i32, nx: i32, ny: i32, nz: i32) -> u32 {
    var wx = cx;
    if (wx < 0)  { wx = wx + nx; }
    if (wx >= nx) { wx = wx - nx; }
    var wy = cy;
    if (wy < 0)  { wy = wy + ny; }
    if (wy >= ny) { wy = wy - ny; }
    var wz = cz;
    if (wz < 0)  { wz = wz + nz; }
    if (wz >= nz) { wz = wz - nz; }
    return u32(wx + wy * nx + wz * nx * ny);
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
    let nx        = i32(params[8]);
    let ny        = i32(params[9]);
    let nz        = i32(params[10]);
    let cell_sx   = params[11];
    let cell_sy   = params[12];
    let cell_sz   = params[13];

    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];

    // Which cell is particle i in?
    let ci_x = i32(xi / cell_sx);
    let ci_y = i32(yi / cell_sy);
    let ci_z = i32(zi / cell_sz);

    var fx = xi - xi;
    var fy = xi - xi;
    var fz = xi - xi;
    var pe = xi - xi;

    // Loop over 27 neighbor cells (including self)
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let c_idx = cell_idx(ci_x + dx, ci_y + dy, ci_z + dz, nx, ny, nz);
                let start = cell_start[c_idx];
                let count = cell_count[c_idx];

                for (var jj = 0u; jj < count; jj = jj + 1u) {
                    let j = start + jj;
                    if (i == j) { continue; }

                    let xj = positions[j * 3u];
                    let yj = positions[j * 3u + 1u];
                    let zj = positions[j * 3u + 2u];

                    var ddx = pbc_delta_cl(xj - xi, box_x);
                    var ddy = pbc_delta_cl(yj - yi, box_y);
                    var ddz = pbc_delta_cl(zj - zi, box_z);

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
            }
        }
    }

    forces[i * 3u]      = fx;
    forces[i * 3u + 1u] = fy;
    forces[i * 3u + 2u] = fz;
    pe_buf[i] = pe;
}
"#;

// ═══════════════════════════════════════════════════════════════════
// Yukawa Cell-List Force Kernel v2 (f64) — flat neighbor loop
// ═══════════════════════════════════════════════════════════════════
//
// Identical physics to SHADER_YUKAWA_FORCE_CELLLIST but uses a flat
// single loop over 27 neighbor offsets (precomputed) instead of
// 3 nested for-loops. This tests whether the Naga/SPIR-V compilation
// of deeply nested i32 loops was causing the force computation bug.

pub const SHADER_YUKAWA_FORCE_CELLLIST_V2: &str = r#"
// Yukawa Cell-List Force v2 (f64) — flat loop, same bindings as v1
//
// Bindings: same as SHADER_YUKAWA_FORCE_CELLLIST

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;
@group(0) @binding(4) var<storage, read> cell_start: array<u32>;
@group(0) @binding(5) var<storage, read> cell_count: array<u32>;

fn pbc_wrap(delta: f64, box_size: f64) -> f64 {
    var d = delta;
    let half = box_size / 2.0;
    if (d > half)  { d = d - box_size; }
    if (d < -half) { d = d + box_size; }
    return d;
}

fn wrap_cell(c: i32, n: i32) -> i32 {
    var w = c;
    if (w < 0)  { w = w + n; }
    if (w >= n) { w = w - n; }
    return w;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n_particles = u32(params[0]);
    if (idx >= n_particles) { return; }

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

    // Determine this particle's cell
    let my_cx = i32(px / csx);
    let my_cy = i32(py / csy);
    let my_cz = i32(pz / csz);

    // Accumulate force and PE as f64 zeros
    var acc_fx = px - px;
    var acc_fy = px - px;
    var acc_fz = px - px;
    var acc_pe = px - px;

    // Flat loop over 27 neighbor offsets
    for (var neigh = 0u; neigh < 27u; neigh = neigh + 1u) {
        // Decode neighbor offset from flat index
        let off_x = i32(neigh % 3u) - 1;
        let off_y = i32((neigh / 3u) % 3u) - 1;
        let off_z = i32(neigh / 9u) - 1;

        let nb_cx = wrap_cell(my_cx + off_x, ncx);
        let nb_cy = wrap_cell(my_cy + off_y, ncy);
        let nb_cz = wrap_cell(my_cz + off_z, ncz);
        let cell_linear = u32(nb_cx + nb_cy * ncx + nb_cz * ncx * ncy);

        let start = cell_start[cell_linear];
        let cnt   = cell_count[cell_linear];

        for (var k = 0u; k < cnt; k = k + 1u) {
            let j = start + k;
            if (idx == j) { continue; }

            let qx = positions[j * 3u];
            let qy = positions[j * 3u + 1u];
            let qz = positions[j * 3u + 2u];

            let rx = pbc_wrap(qx - px, box_x);
            let ry = pbc_wrap(qy - py, box_y);
            let rz = pbc_wrap(qz - pz, box_z);

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
"#;

// ═══════════════════════════════════════════════════════════════════
// RDF Histogram Kernel (f64)
// ═══════════════════════════════════════════════════════════════════
//
// Computes all-pairs distances with PBC and bins into a histogram.
// Uses atomicAdd on u32 bins, then normalized on CPU.

pub const SHADER_RDF_HISTOGRAM: &str = r#"
// RDF pair distance histogram (f64 positions, u32 bins)
//
// Bindings:
//   0: positions   [N*3] f64, read
//   1: histogram   [n_bins] atomic<u32>, read-write
//   2: params      [8]   f64, read  — [n, n_bins, dr, _, box_x, box_y, box_z, _]

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> params: array<f64>;

fn pbc_delta_rdf(delta: f64, box_size: f64) -> f64 {
    return delta - box_size * round_f64(delta / box_size);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n      = u32(params[0]);
    let n_bins = u32(params[1]);
    let dr     = params[2];
    let box_x  = params[4];
    let box_y  = params[5];
    let box_z  = params[6];

    if (i >= n) { return; }

    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];

    // Only count pairs where j > i to avoid double-counting
    for (var j = i + 1u; j < n; j = j + 1u) {
        let dx = pbc_delta_rdf(positions[j * 3u]      - xi, box_x);
        let dy = pbc_delta_rdf(positions[j * 3u + 1u] - yi, box_y);
        let dz = pbc_delta_rdf(positions[j * 3u + 2u] - zi, box_z);

        let r = sqrt_f64(dx * dx + dy * dy + dz * dz);
        let bin = u32(r / dr);

        if (bin < n_bins) {
            atomicAdd(&histogram[bin], 1u);
        }
    }
}
"#;
