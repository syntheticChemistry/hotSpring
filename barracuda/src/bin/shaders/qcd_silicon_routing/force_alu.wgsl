// SU(3) gauge force proxy — pure ALU FMA chain
// Proxy for staple sum: 6 directions × 3 matmuls × 3×3 complex = heavy FMA
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
struct ForceParams { volume: u32, pad0: u32, }
@group(0) @binding(1) var<uniform> params: ForceParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.volume { return; }

    // 6 staple directions × 3 SU(3) multiplies × 3 rows × 3 cols × 2 FMA (re,im)
    var acc: f32 = f32(idx + 1u) * 1e-5;
    for (var staple = 0u; staple < 6u; staple++) {
        for (var mat = 0u; mat < 3u; mat++) {
            for (var i = 0u; i < 9u; i++) {
                acc = fma(acc, 0.9999, f32(i) * 0.0001);
                acc = fma(acc, 0.9998, f32(staple) * 0.0001);
            }
        }
    }
    out[idx] = acc;
}
