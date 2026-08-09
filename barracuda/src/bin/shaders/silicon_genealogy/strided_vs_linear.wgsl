// Strided access pattern mimicking lattice neighbor lookups.
// Compares linear sequential access vs 4-direction strided access.

struct Params {
    n_elements: u32,
    stride_x: u32,
    stride_y: u32,
    stride_z: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size(256)
fn linear_access(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_elements { return; }
    dst[idx] = src[idx];
}

@compute @workgroup_size(256)
fn strided_access(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_elements { return; }

    let fwd_x = (idx + params.stride_x) % params.n_elements;
    let fwd_y = (idx + params.stride_y) % params.n_elements;
    let fwd_z = (idx + params.stride_z) % params.n_elements;
    let bwd_x = (idx + params.n_elements - params.stride_x) % params.n_elements;

    let v0 = src[idx];
    let v1 = src[fwd_x];
    let v2 = src[fwd_y];
    let v3 = src[fwd_z];
    let v4 = src[bwd_x];

    dst[idx] = v0 + v1 + v2 + v3 + v4;
}
