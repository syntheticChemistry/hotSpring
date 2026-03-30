@group(0) @binding(0) var<storage> a: array<f64>;
@group(0) @binding(1) var<storage> b: array<f64>;
@group(0) @binding(2) var<storage, read_write> out: array<f64>;

@compute @workgroup_size(1)
fn main() {
    out[0] = a[0] / b[0];
}
