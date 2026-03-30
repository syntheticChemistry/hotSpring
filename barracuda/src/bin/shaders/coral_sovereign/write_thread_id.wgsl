@group(0) @binding(0)
var<storage, read_write> out: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < 64u {
        out[gid.x] = gid.x * 3u + 7u;
    }
}
