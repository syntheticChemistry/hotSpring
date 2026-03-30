// Box-Muller via TMU: texture lookup for log/cos/sin, ALU only for mul/add
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

    var total: f32 = 0.0;
    for (var pair = 0u; pair < 3u; pair++) {
        let u1 = max(uniform01(idx, pair * 2u), 1e-10);
        let u2 = uniform01(idx, pair * 2u + 1u);

        // TMU lookup: log_table[x] = -2 * log(x/4095)  (pre-negated, pre-doubled)
        let log_idx = u32(u1 * 4095.0);
        let neg2log = textureLoad(log_table, vec2<u32>(log_idx, 0u), 0).r;
        let r = sqrt(neg2log);

        // TMU lookup: trig_table[x] = (cos(2pi*x/4095), sin(2pi*x/4095))
        let trig_idx = u32(u2 * 4095.0);
        let cs = textureLoad(trig_table, vec2<u32>(trig_idx, 0u), 0);
        total += r * cs.r;
        total += r * cs.g;
    }
    out[idx] = total;
}
