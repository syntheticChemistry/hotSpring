// SPDX-License-Identifier: AGPL-3.0-only
//
// Yukawa All-Pairs Force (f64) with PBC + potential energy
// NVK-safe: compiled via ShaderTemplate::for_device_auto() which patches
// exp() → exp_f64() on NVK/nouveau drivers.
//
// Canonical source: barracuda::ops::md::forces::yukawa_f64.wgsl
// hotSpring local copy for NVK workaround (toadstool absorption pending).
//
// Bindings:
//   0: positions  [N*3] f64, read     — (x,y,z) per particle
//   1: forces     [N*3] f64, write    — (fx,fy,fz) per particle
//   2: pe_buf     [N]   f64, write    — per-particle PE (half-counted)
//   3: params     [8]   f64, read     — simulation parameters

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;

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

    let kappa     = params[1];
    let prefactor = params[2];
    let cutoff_sq = params[3];
    let box_x     = params[4];
    let box_y     = params[5];
    let box_z     = params[6];
    let eps       = params[7];

    var fx = xi - xi;
    var fy = xi - xi;
    var fz = xi - xi;
    var pe = xi - xi;

    for (var j = 0u; j < n; j = j + 1u) {
        if (i == j) { continue; }

        let xj = positions[j * 3u];
        let yj = positions[j * 3u + 1u];
        let zj = positions[j * 3u + 2u];

        var dx = pbc_delta(xj - xi, box_x);
        var dy = pbc_delta(yj - yi, box_y);
        var dz = pbc_delta(zj - zi, box_z);

        let r_sq = dx * dx + dy * dy + dz * dz;

        if (r_sq > cutoff_sq) { continue; }

        let r = sqrt(r_sq + eps);

        // exp() auto-patched to exp_f64() on NVK by for_device_auto()
        let screening = exp(-kappa * r);
        let force_mag = prefactor * screening * (1.0 + kappa * r) / r_sq;

        let inv_r = 1.0 / r;
        fx = fx - force_mag * dx * inv_r;
        fy = fy - force_mag * dy * inv_r;
        fz = fz - force_mag * dz * inv_r;

        pe = pe + 0.5 * prefactor * screening * inv_r;
    }

    forces[i * 3u]      = fx;
    forces[i * 3u + 1u] = fy;
    forces[i * 3u + 2u] = fz;
    pe_buf[i] = pe;
}
