@group(0) @binding(0) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    out[0] = gid.x;
    out[1] = gid.y;
    out[2] = nwg.x;
    out[3] = nwg.y;
    out[4] = wid.x;
    out[5] = lid.x;
}
