struct Params { n: u32, pad0: u32, alpha: f64, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f64>;
@group(0) @binding(2) var<storage, read_write> y: array<f64>;

@compute @workgroup_size(1)
fn main() {
    y[0] = y[0] + params.alpha * x[0];
}
