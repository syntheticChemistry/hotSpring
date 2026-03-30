// Compound kernel: SU(3) force (ALU) + Box-Muller PRNG (TMU) in same thread
// Demonstrates multi-unit composition — ALU and TMU run in parallel
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
struct Params { volume: u32, seed: u32, }
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var log_table: texture_2d<f32>;
@group(0) @binding(3) var trig_table: texture_2d<f32>;

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

    // ── ALU work: SU(3) force proxy (same as force_alu) ──
    var force_acc: f32 = f32(idx + 1u) * 1e-5;
    for (var staple = 0u; staple < 6u; staple++) {
        for (var mat = 0u; mat < 3u; mat++) {
            for (var i = 0u; i < 9u; i++) {
                force_acc = fma(force_acc, 0.9999, f32(i) * 0.0001);
                force_acc = fma(force_acc, 0.9998, f32(staple) * 0.0001);
            }
        }
    }

    // ── TMU work: Box-Muller PRNG via texture lookup (interleaved with ALU) ──
    var prng_acc: f32 = 0.0;
    for (var pair = 0u; pair < 3u; pair++) {
        let u1 = max(uniform01(idx, pair * 2u), 1e-10);
        let u2 = uniform01(idx, pair * 2u + 1u);
        let log_idx = u32(u1 * 4095.0);
        let neg2log = textureLoad(log_table, vec2<u32>(log_idx, 0u), 0).r;
        let r = sqrt(neg2log);
        let trig_idx = u32(u2 * 4095.0);
        let cs = textureLoad(trig_table, vec2<u32>(trig_idx, 0u), 0);
        prng_acc += r * cs.r;
    }

    out[idx] = force_acc + prng_acc;
}
