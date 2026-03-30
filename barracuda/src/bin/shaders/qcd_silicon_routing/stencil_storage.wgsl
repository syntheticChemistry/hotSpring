// SU(3) matvec stencil — storage buffer path (current production)
// Proxy: each thread reads 8 neighbor SU(3) matrices (18 f32 each) + 8 color vectors (6 f32)
@group(0) @binding(0) var<storage, read> links: array<f32>;
@group(0) @binding(1) var<storage, read> psi: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
struct StencilParams { volume: u32, pad0: u32, pad1: u32, pad2: u32, }
@group(0) @binding(3) var<uniform> params: StencilParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if site >= params.volume { return; }

    var result = array<f32, 6>();
    // 4 directions × fwd + bwd = 8 neighbor reads
    for (var mu = 0u; mu < 4u; mu++) {
        // Neighbor index (wrapping; proxy uses modular addressing)
        let fwd = (site + mu * 317u + 1u) % params.volume;
        let bwd = (site + params.volume - mu * 317u - 1u) % params.volume;

        let fl = (site * 4u + mu) * 18u;
        let fp = fwd * 6u;
        let bl = (bwd * 4u + mu) * 18u;
        let bp = bwd * 6u;

        // Forward: U(x,mu) * psi(x+mu) — 3×3 complex matvec
        for (var row = 0u; row < 3u; row++) {
            var re: f32 = 0.0;
            var im: f32 = 0.0;
            for (var col = 0u; col < 3u; col++) {
                let li = fl + (row * 3u + col) * 2u;
                let pi = fp + col * 2u;
                re += links[li] * psi[pi] - links[li + 1u] * psi[pi + 1u];
                im += links[li] * psi[pi + 1u] + links[li + 1u] * psi[pi];
            }
            result[row * 2u] += re;
            result[row * 2u + 1u] += im;
        }

        // Backward: U†(x-mu,mu) * psi(x-mu)
        for (var row = 0u; row < 3u; row++) {
            var re: f32 = 0.0;
            var im: f32 = 0.0;
            for (var col = 0u; col < 3u; col++) {
                let li = bl + (col * 3u + row) * 2u;
                let pi = bp + col * 2u;
                re += links[li] * psi[pi] + links[li + 1u] * psi[pi + 1u];
                im += links[li] * psi[pi + 1u] - links[li + 1u] * psi[pi];
            }
            result[row * 2u] -= re;
            result[row * 2u + 1u] -= im;
        }
    }

    let base = site * 6u;
    for (var i = 0u; i < 6u; i++) { out[base + i] = result[i]; }
}
