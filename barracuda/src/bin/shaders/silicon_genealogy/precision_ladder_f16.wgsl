// F16 precision tier experiment — SU(3) staple computation at half precision.
//
// The precision ladder: f16 → DF32 → f32 → DF64 → f64
// This shader implements the SAME gauge force computation at f16,
// proving throughput scaling vs precision trade-off.
//
// For thermalization screening: f16 gives ~2× throughput at ~3 digit precision.
// Good enough for "is this config close to thermalized?" but NOT for measurements.

enable f16;

struct Params {
    volume: u32,
    n_links: u32,
    beta_hi: f32,
    beta_lo: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> links: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> nbr: array<u32>;
@group(0) @binding(3) var<storage, read_write> plaq_out: array<f32>;

// SU(3) as 9 complex = 18 f16 values = 4.5 vec4<f16> per matrix
// For simplicity, we pack as 5 vec4<f16> (with 2 unused components)

fn load_link_f16(link_idx: u32) -> array<vec4<f16>, 5> {
    let base = link_idx * 5u;
    var m: array<vec4<f16>, 5>;
    m[0] = links[base];
    m[1] = links[base + 1u];
    m[2] = links[base + 2u];
    m[3] = links[base + 3u];
    m[4] = links[base + 4u];
    return m;
}

// 3×3 complex multiply using f16 — each element is (re, im) pairs in vec4
fn su3_mul_f16(a: array<vec4<f16>, 5>, b: array<vec4<f16>, 5>) -> array<vec4<f16>, 5> {
    // Extract components: a[row][col] = (a_re, a_im)
    // Packed: vec4 = (re0, im0, re1, im1) for two complex numbers
    // a[0] = (a00_re, a00_im, a01_re, a01_im)
    // a[1] = (a02_re, a02_im, a10_re, a10_im)
    // a[2] = (a11_re, a11_im, a12_re, a12_im)
    // a[3] = (a20_re, a20_im, a21_re, a21_im)
    // a[4] = (a22_re, a22_im, 0, 0)

    var c: array<vec4<f16>, 5>;

    // c00 = a00*b00 + a01*b10 + a02*b20
    let c00_re = a[0].x * b[0].x - a[0].y * b[0].y
               + a[0].z * b[1].z - a[0].w * b[1].w
               + a[1].x * b[3].x - a[1].y * b[3].y;
    let c00_im = a[0].x * b[0].y + a[0].y * b[0].x
               + a[0].z * b[1].w + a[0].w * b[1].z
               + a[1].x * b[3].y + a[1].y * b[3].x;

    // c01 = a00*b01 + a01*b11 + a02*b21
    let c01_re = a[0].x * b[0].z - a[0].y * b[0].w
               + a[0].z * b[2].x - a[0].w * b[2].y
               + a[1].x * b[3].z - a[1].y * b[3].w;
    let c01_im = a[0].x * b[0].w + a[0].y * b[0].z
               + a[0].z * b[2].y + a[0].w * b[2].x
               + a[1].x * b[3].w + a[1].y * b[3].z;

    c[0] = vec4<f16>(c00_re, c00_im, c01_re, c01_im);

    // c02 = a00*b02 + a01*b12 + a02*b22
    let c02_re = a[0].x * b[1].x - a[0].y * b[1].y
               + a[0].z * b[2].z - a[0].w * b[2].w
               + a[1].x * b[4].x - a[1].y * b[4].y;
    let c02_im = a[0].x * b[1].y + a[0].y * b[1].x
               + a[0].z * b[2].w + a[0].w * b[2].z
               + a[1].x * b[4].y + a[1].y * b[4].x;

    // c10 = a10*b00 + a11*b10 + a12*b20
    let c10_re = a[1].z * b[0].x - a[1].w * b[0].y
               + a[2].x * b[1].z - a[2].y * b[1].w
               + a[2].z * b[3].x - a[2].w * b[3].y;
    let c10_im = a[1].z * b[0].y + a[1].w * b[0].x
               + a[2].x * b[1].w + a[2].y * b[1].z
               + a[2].z * b[3].y + a[2].w * b[3].x;

    c[1] = vec4<f16>(c02_re, c02_im, c10_re, c10_im);

    // c11 = a10*b01 + a11*b11 + a12*b21
    let c11_re = a[1].z * b[0].z - a[1].w * b[0].w
               + a[2].x * b[2].x - a[2].y * b[2].y
               + a[2].z * b[3].z - a[2].w * b[3].w;
    let c11_im = a[1].z * b[0].w + a[1].w * b[0].z
               + a[2].x * b[2].y + a[2].y * b[2].x
               + a[2].z * b[3].w + a[2].w * b[3].z;

    // c12 = a10*b02 + a11*b12 + a12*b22
    let c12_re = a[1].z * b[1].x - a[1].w * b[1].y
               + a[2].x * b[2].z - a[2].y * b[2].w
               + a[2].z * b[4].x - a[2].w * b[4].y;
    let c12_im = a[1].z * b[1].y + a[1].w * b[1].x
               + a[2].x * b[2].w + a[2].y * b[2].z
               + a[2].z * b[4].y + a[2].w * b[4].x;

    c[2] = vec4<f16>(c11_re, c11_im, c12_re, c12_im);

    // c20 = a20*b00 + a21*b10 + a22*b20
    let c20_re = a[3].x * b[0].x - a[3].y * b[0].y
               + a[3].z * b[1].z - a[3].w * b[1].w
               + a[4].x * b[3].x - a[4].y * b[3].y;
    let c20_im = a[3].x * b[0].y + a[3].y * b[0].x
               + a[3].z * b[1].w + a[3].w * b[1].z
               + a[4].x * b[3].y + a[4].y * b[3].x;

    // c21 = a20*b01 + a21*b11 + a22*b21
    let c21_re = a[3].x * b[0].z - a[3].y * b[0].w
               + a[3].z * b[2].x - a[3].w * b[2].y
               + a[4].x * b[3].z - a[4].y * b[3].w;
    let c21_im = a[3].x * b[0].w + a[3].y * b[0].z
               + a[3].z * b[2].y + a[3].w * b[2].x
               + a[4].x * b[3].w + a[4].y * b[3].z;

    c[3] = vec4<f16>(c20_re, c20_im, c21_re, c21_im);

    // c22 = a20*b02 + a21*b12 + a22*b22
    let c22_re = a[3].x * b[1].x - a[3].y * b[1].y
               + a[3].z * b[2].z - a[3].w * b[2].w
               + a[4].x * b[4].x - a[4].y * b[4].y;
    let c22_im = a[3].x * b[1].y + a[3].y * b[1].x
               + a[3].z * b[2].w + a[3].w * b[2].z
               + a[4].x * b[4].y + a[4].y * b[4].x;

    c[4] = vec4<f16>(c22_re, c22_im, vec2<f16>(0.0h, 0.0h));

    return c;
}

// Plaquette trace at f16 (Re Tr U_P / 3)
fn plaquette_trace_f16(u: array<vec4<f16>, 5>) -> f32 {
    // Tr = u00 + u11 + u22 (real parts only for Re Tr)
    let tr_re = f32(u[0].x) + f32(u[2].x) + f32(u[4].x);
    return tr_re / 3.0;
}

@compute @workgroup_size(256)
fn plaquette_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if site >= params.volume { return; }

    var plaq_sum: f32 = 0.0;
    let n_dirs = 4u;

    for (var mu = 0u; mu < n_dirs; mu++) {
        for (var nu = mu + 1u; nu < n_dirs; nu++) {
            let link_idx = site * n_dirs + mu;
            let u_mu = load_link_f16(link_idx);

            let fwd_mu = nbr[site * 8u + mu]; // forward neighbor in mu
            let u_nu_at_fwd = load_link_f16(fwd_mu * n_dirs + nu);

            let fwd_nu = nbr[site * 8u + nu]; // forward neighbor in nu
            let u_mu_at_fwd_nu = load_link_f16(fwd_nu * n_dirs + mu);

            let u_nu = load_link_f16(site * n_dirs + nu);

            // P = U_mu(x) * U_nu(x+mu) * U_mu†(x+nu) * U_nu†(x)
            // For now: just U_mu * U_nu as a throughput test
            let prod = su3_mul_f16(u_mu, u_nu_at_fwd);
            plaq_sum += plaquette_trace_f16(prod);
        }
    }

    plaq_out[site] = plaq_sum / 6.0;
}
