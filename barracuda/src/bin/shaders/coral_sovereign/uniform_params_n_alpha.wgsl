struct Params { n: u32, pad0: u32, alpha: f64, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f64>;

@compute @workgroup_size(1)
fn main() {
    out[0] = f64(params.n);
    out[1] = params.alpha;
}
