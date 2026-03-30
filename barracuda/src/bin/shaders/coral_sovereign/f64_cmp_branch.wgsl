@group(0) @binding(0) var<storage> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(1)
fn main() {
    let val = input[0];
    if (abs(val) > f64(1e-30)) {
        output[0] = f64(1.0);
    } else {
        output[0] = f64(0.0);
    }
}
