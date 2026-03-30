@group(0) @binding(0) var<storage, read_write> out: array<f64>;

@compute @workgroup_size(1)
fn main() {
    out[0] = f64(3.14);
}
