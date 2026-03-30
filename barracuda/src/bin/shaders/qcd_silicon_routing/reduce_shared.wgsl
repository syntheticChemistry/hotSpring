// Tree reduce via workgroup shared memory (current production path)
var<workgroup> wg_data: array<f32, 256>;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
struct ReduceParams { size: u32, pad0: u32, pad1: u32, pad2: u32, }
@group(0) @binding(2) var<uniform> params: ReduceParams;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let g = gid.x + gid.y * nwg.x * 256u;
    wg_data[lid.x] = select(0.0, input[g], g < params.size);
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if lid.x < s { wg_data[lid.x] += wg_data[lid.x + s]; }
        workgroupBarrier();
    }
    if lid.x == 0u { output[wgid.x + wgid.y * nwg.x] = wg_data[0]; }
}
