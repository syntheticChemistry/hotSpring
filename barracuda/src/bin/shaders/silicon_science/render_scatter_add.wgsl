// ROP additive blend: vertex+fragment shaders for force scatter-add.
// Each lattice link emits a point primitive; the fragment shader writes
// the weighted force contribution. Hardware additive blending accumulates
// all contributions without atomics or barriers.
//
// This replaces atomicAdd(i32) with hardware blend: result = src + dst.
// The ROP unit performs the addition at fill rate — no shader intervention.

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) force_re: f32,
    @location(1) force_im: f32,
}

struct Params {
    inv_width: f32,
    inv_height: f32,
    n_links: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> force_data: array<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;

    let link_idx = vid;
    let component_pair = link_idx / params.n_links;
    let link = link_idx % params.n_links;

    let x = f32(link % 256u) * params.inv_width * 2.0 - 1.0;
    let y = f32(link / 256u) * params.inv_height * 2.0 - 1.0;

    out.pos = vec4<f32>(x, y, 0.0, 1.0);

    let data_idx = link * 18u + component_pair;
    out.force_re = force_data[data_idx * 2u];
    out.force_im = force_data[data_idx * 2u + 1u];

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.force_re, in.force_im, 0.0, 1.0);
}
