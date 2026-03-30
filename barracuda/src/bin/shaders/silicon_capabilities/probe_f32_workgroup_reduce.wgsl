struct Params { size: u32, _pad1: u32, _pad2: u32, _pad3: u32, }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let tid = lid.x;
    if (gid.x < params.size) {
        wg_data[tid] = input[gid.x];
    } else {
        wg_data[tid] = 0.0;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            wg_data[tid] = wg_data[tid] + wg_data[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[wg.x] = wg_data[0];
    }
}
