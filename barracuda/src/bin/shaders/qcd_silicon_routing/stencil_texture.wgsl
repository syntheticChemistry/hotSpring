// SU(3) matvec stencil — TMU textureLoad path
// Gauge links stored in Rgba32Float texture: 5 texels per SU(3) matrix (18 floats / 4 = 4.5, pad to 5)
// Texture layout: width = n_links * 5, height = 1
@group(0) @binding(0) var link_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> psi: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
struct StencilParams { volume: u32, pad0: u32, pad1: u32, pad2: u32, }
@group(0) @binding(3) var<uniform> params: StencilParams;

fn load_link_elem(link_idx: u32, elem: u32) -> f32 {
    // Each SU(3) = 5 texels of Rgba32Float (20 channels, first 18 used)
    let texel = link_idx * 5u + elem / 4u;
    let channel = elem % 4u;
    let v = textureLoad(link_tex, vec2<u32>(texel, 0u), 0);
    // Branch-free channel select
    return v[channel];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if site >= params.volume { return; }

    var result = array<f32, 6>();

    for (var mu = 0u; mu < 4u; mu++) {
        let fwd = (site + mu * 317u + 1u) % params.volume;
        let bwd = (site + params.volume - mu * 317u - 1u) % params.volume;

        let fwd_link = site * 4u + mu;
        let bwd_link = bwd * 4u + mu;
        let fp = fwd * 6u;
        let bp = bwd * 6u;

        // Forward: U(x,mu) * psi(x+mu) — read links via TMU
        for (var row = 0u; row < 3u; row++) {
            var re: f32 = 0.0;
            var im: f32 = 0.0;
            for (var col = 0u; col < 3u; col++) {
                let elem = (row * 3u + col) * 2u;
                let lr = load_link_elem(fwd_link, elem);
                let li = load_link_elem(fwd_link, elem + 1u);
                re += lr * psi[fp + col * 2u] - li * psi[fp + col * 2u + 1u];
                im += lr * psi[fp + col * 2u + 1u] + li * psi[fp + col * 2u];
            }
            result[row * 2u] += re;
            result[row * 2u + 1u] += im;
        }

        // Backward: U†(x-mu,mu) * psi(x-mu) — read links via TMU
        for (var row = 0u; row < 3u; row++) {
            var re: f32 = 0.0;
            var im: f32 = 0.0;
            for (var col = 0u; col < 3u; col++) {
                let elem = (col * 3u + row) * 2u;
                let lr = load_link_elem(bwd_link, elem);
                let li = load_link_elem(bwd_link, elem + 1u);
                re += lr * psi[bp + col * 2u] + li * psi[bp + col * 2u + 1u];
                im += lr * psi[bp + col * 2u + 1u] - li * psi[bp + col * 2u];
            }
            result[row * 2u] -= re;
            result[row * 2u + 1u] -= im;
        }
    }

    let base = site * 6u;
    for (var i = 0u; i < 6u; i++) { out[base + i] = result[i]; }
}
