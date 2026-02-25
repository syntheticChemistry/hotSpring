// SPDX-License-Identifier: AGPL-3.0-only
// GPU-resident Gaussian random fermion field generation via PCG hash PRNG.
//
// Each thread generates one site's fermion field: 3 colors × (re, im) = 6 f64.
// Used for pseudofermion heat bath: η ~ N(0,1), then φ = D†η.
//
// PRNG: PCG hash on (site_idx ⊕ seed ⊕ traj_id) — counter-based,
// no inter-thread state, trivially parallel.
//
// Uses sqrt_f64/log_f64 polyfills auto-injected by ShaderTemplate.

struct Params {
    volume: u32,
    traj_id: u32,
    seed_lo: u32,
    seed_hi: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> fermion: array<f64>;

// pcg_hash, hash_u32, uniform_f64 provided by prng_pcg_f64.wgsl (prepended in Rust).

fn box_muller_cos(u1: f64, u2: f64) -> f64 {
    var safe = u1;
    if safe < f64(1e-20) { safe = f64(1e-20); }
    let r = sqrt_f64(f64(-2.0) * log_f64(safe));
    let theta = f32(f64(6.283185307179586) * u2);
    return r * f64(cos(theta));
}

fn gaussian(site_idx: u32, pair: u32) -> f64 {
    let u1 = uniform_f64(site_idx, pair * 2u);
    let u2 = uniform_f64(site_idx, pair * 2u + 1u);
    return box_muller_cos(u1, u2);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    if idx >= params.volume { return; }

    let base = idx * 6u;
    // 3 colors × (re, im) = 3 Box-Muller pairs
    fermion[base + 0u] = gaussian(idx, 0u);  // color 0 re
    fermion[base + 1u] = gaussian(idx, 1u);  // color 0 im
    fermion[base + 2u] = gaussian(idx, 2u);  // color 1 re
    fermion[base + 3u] = gaussian(idx, 3u);  // color 1 im
    fermion[base + 4u] = gaussian(idx, 4u);  // color 2 re
    fermion[base + 5u] = gaussian(idx, 5u);  // color 2 im
}
