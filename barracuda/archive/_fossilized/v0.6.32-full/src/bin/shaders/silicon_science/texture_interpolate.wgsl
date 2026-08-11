// Texture sampling for field interpolation: use TMU hardware bilinear filtering
// to interpolate lattice fields at sub-site positions.
//
// The TMU performs bilinear interpolation in a single clock cycle using dedicated
// filter hardware. For multigrid prolongation/restriction or fermion field
// interpolation between lattice refinement levels, this is free computation.
//
// QCD application: multigrid prolongation operator, field smearing,
// APE/stout smearing interpolation, gradient flow intermediate steps.

struct Params {
    grid_dim: u32,
    n_samples: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var field_texture: texture_2d<f32>;
@group(0) @binding(2) var field_sampler: sampler;
@group(0) @binding(3) var<storage, read> sample_coords: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> interpolated: array<f32>;

@compute @workgroup_size(64)
fn interpolate_field(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_samples { return; }

    let uv = sample_coords[idx];

    // Hardware bilinear interpolation — single TMU cycle
    let val = textureSampleLevel(field_texture, field_sampler, uv, 0.0);

    interpolated[idx * 4u] = val.r;
    interpolated[idx * 4u + 1u] = val.g;
    interpolated[idx * 4u + 2u] = val.b;
    interpolated[idx * 4u + 3u] = val.a;
}

// Mipmap-based multigrid: sample from different LODs = different grid resolutions
@compute @workgroup_size(64)
fn multigrid_restrict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_samples { return; }

    let uv = sample_coords[idx];

    // LOD 0 = fine grid, LOD 1 = 2x coarser, LOD 2 = 4x coarser...
    // Hardware generates averaged (restricted) values via mipmap filtering
    let fine = textureSampleLevel(field_texture, field_sampler, uv, 0.0);
    let coarse_2x = textureSampleLevel(field_texture, field_sampler, uv, 1.0);
    let coarse_4x = textureSampleLevel(field_texture, field_sampler, uv, 2.0);

    // Multigrid residual: fine - prolongated(coarse)
    let residual = fine - coarse_2x;

    interpolated[idx * 4u] = residual.r;
    interpolated[idx * 4u + 1u] = residual.g;
    interpolated[idx * 4u + 2u] = residual.b;
    interpolated[idx * 4u + 3u] = coarse_4x.r; // store 4x coarse for next level
}
