struct P { val: f64, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> out: array<f64>;

@compute @workgroup_size(1)
fn main() {
    out[0] = p.val;
}
