@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.x;
    var sum: f32 = 0.0;
    var idx = gid.x;
    for (var i = 0u; i < 64u; i = i + 1u) {
        sum = sum + input[idx % size];
        idx = idx + 256u;
    }
    out[gid.x] = sum;
}
