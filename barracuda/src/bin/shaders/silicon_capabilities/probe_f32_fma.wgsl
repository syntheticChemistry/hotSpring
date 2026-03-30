@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    out[0] = fma(2.0, 3.0, 1.0);

    let a: f32 = 2.0;
    let b: f32 = 3.0;
    let p = a * b;
    out[1] = fma(a, b, -p);

    let c: f32 = 1234567.0;
    let d: f32 = 7654321.0;
    let q = c * d;
    out[2] = fma(c, d, -q);
    out[3] = q;
}
