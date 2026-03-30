struct Params { size: u32, _pad1: u32, _pad2: u32, _pad3: u32, }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_hi: array<f32, 256>;
var<workgroup> shared_lo: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let tid = lid.x;
    if (gid.x < params.size) {
        shared_hi[tid] = input[gid.x];
        shared_lo[tid] = 0.0;
    } else {
        shared_hi[tid] = 0.0;
        shared_lo[tid] = 0.0;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let a = Df64(shared_hi[tid], shared_lo[tid]);
            let b = Df64(shared_hi[tid + stride], shared_lo[tid + stride]);
            let s = df64_add(a, b);
            shared_hi[tid] = s.hi;
            shared_lo[tid] = s.lo;
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[0] = shared_hi[0];
        output[1] = shared_lo[0];
    }
}
