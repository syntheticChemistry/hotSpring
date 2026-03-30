@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let one = Df64(1.0, 0.0);
    let sum = df64_add(one, one);
    out[0] = sum.hi;
    out[1] = sum.lo;

    let three = Df64(3.0, 0.0);
    let prod = df64_mul(sum, three);
    out[2] = prod.hi;
    out[3] = prod.lo;

    let pi_hi: f32 = 3.1415927;
    let pi_lo: f32 = -8.742278e-8;
    let pi = Df64(pi_hi, pi_lo);
    let pi_sq = df64_mul(pi, pi);
    out[4] = pi_sq.hi;
    out[5] = pi_sq.lo;
}
