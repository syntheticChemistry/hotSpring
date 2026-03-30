@group(0) @binding(0) var<storage> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(1)
fn main() {
    output[0] = input[0];
}
