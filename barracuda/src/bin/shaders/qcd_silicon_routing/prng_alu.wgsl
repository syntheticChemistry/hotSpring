// Box-Muller via ALU: software log/cos/sqrt (the current production path)
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
struct Params { volume: u32, seed: u32, }
@group(0) @binding(1) var<uniform> params: Params;

fn pcg_hash(inp: u32) -> u32 {
    var s = inp * 747796405u + 2891336453u;
    var w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn uniform01(idx: u32, seq: u32) -> f32 {
    let h = pcg_hash(pcg_hash(idx ^ params.seed) ^ seq);
    return f32(h) / 4294967295.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.volume { return; }

    // 3 color components × 2 (re, im) = 3 Box-Muller pairs
    var total: f32 = 0.0;
    for (var pair = 0u; pair < 3u; pair++) {
        let u1 = max(uniform01(idx, pair * 2u), 1e-10);
        let u2 = uniform01(idx, pair * 2u + 1u);
        // ALU transcendentals
        let r = sqrt(-2.0 * log(u1));
        let theta = 6.283185 * u2;
        total += r * cos(theta);
        total += r * sin(theta);
    }
    out[idx] = total;
}
