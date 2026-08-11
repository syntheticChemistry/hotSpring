// Depth buffer distance field: render lattice sites as point sprites,
// use depth test (Less) to compute nearest-site distance fields at
// hardware speed. The depth buffer performs min-reduction at fill rate.
//
// Each site renders as a screen-space quad. Fragment depth = euclidean
// distance from pixel center to site. After all sites, depth buffer
// contains the distance to nearest site at every pixel — a Voronoi diagram.

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) site_pos: vec3<f32>,
}

struct Params {
    grid_dim: f32,
    n_sites: u32,
    viewport_size: f32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> site_positions: array<vec4<f32>>;

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32
) -> VertexOutput {
    var out: VertexOutput;

    let site = site_positions[iid];

    // Quad vertices (2 triangles, 6 vertices per instance)
    var quad: array<vec2<f32>, 6>;
    quad[0] = vec2<f32>(-1.0, -1.0);
    quad[1] = vec2<f32>( 1.0, -1.0);
    quad[2] = vec2<f32>(-1.0,  1.0);
    quad[3] = vec2<f32>(-1.0,  1.0);
    quad[4] = vec2<f32>( 1.0, -1.0);
    quad[5] = vec2<f32>( 1.0,  1.0);

    let sprite_size = 2.0 / params.grid_dim;
    let ndc_center = site.xy / params.grid_dim * 2.0 - 1.0;
    let offset = quad[vid] * sprite_size;

    out.pos = vec4<f32>(ndc_center + offset, 0.5, 1.0);
    out.site_pos = site.xyz;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @builtin(frag_depth) f32 {
    // Pixel world-space position from fragment position
    let pixel_x = in.pos.x / params.viewport_size * params.grid_dim;
    let pixel_y = in.pos.y / params.viewport_size * params.grid_dim;

    let dx = pixel_x - in.site_pos.x;
    let dy = pixel_y - in.site_pos.y;
    let dist = sqrt(dx * dx + dy * dy) / params.grid_dim;

    // Depth = normalized distance (0=on site, 1=far)
    return clamp(dist, 0.0, 1.0);
}
