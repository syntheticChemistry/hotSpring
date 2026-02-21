// SPDX-License-Identifier: AGPL-3.0-only

//! Compressed Sparse Row matrix format and SpMV.
//!
//! CSR is the standard format for GPU SpMV kernels (P1 for Kachkovskiy GPU
//! promotion). Includes the WGSL compute shader for direct GPU absorption.

/// Sparse symmetric matrix in Compressed Sparse Row format.
///
/// This is the standard format for GPU SpMV kernels (P1 for Kachkovskiy GPU
/// promotion). The CSR layout maps directly to a WGSL compute shader.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    pub n: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Sparse matrix-vector product: y = A * x.
    ///
    /// This is the P1 primitive for GPU promotion — the inner loop of Lanczos.
    /// CPU version; GPU SpMV available via `WGSL_SPMV_CSR_F64` shader.
    pub fn spmv(&self, x: &[f64], y: &mut [f64]) {
        for (i, yi) in y.iter_mut().enumerate().take(self.n) {
            let mut sum = 0.0;
            for j in self.row_ptr[i]..self.row_ptr[i + 1] {
                sum += self.values[j] * x[self.col_idx[j]];
            }
            *yi = sum;
        }
    }

    /// Number of non-zero entries.
    pub const fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// WGSL compute shader for CSR sparse matrix-vector product y = A*x (f64).
///
/// Direct GPU port of [`CsrMatrix::spmv()`]: one thread per matrix row,
/// f64 multiply-accumulate in the inner loop. The binding layout is
/// documented for direct absorption by toadstool/barracuda.
///
/// ## Binding layout
///
/// | Binding | Type | Content |
/// |---------|------|---------|
/// | 0 | uniform | `{ n: u32, nnz: u32, pad: u32, pad: u32 }` |
/// | 1 | storage, read | `row_ptr: array<u32>` (n+1 entries) |
/// | 2 | storage, read | `col_idx: array<u32>` (nnz entries) |
/// | 3 | storage, read | `values: array<f64>` (nnz entries) |
/// | 4 | storage, read | `x: array<f64>` (n entries, input) |
/// | 5 | storage, read_write | `y: array<f64>` (n entries, output) |
///
/// ## Dispatch
///
/// `ceil(n / 64)` workgroups of 64 threads.
///
/// ## Provenance
///
/// GPU promotion of CPU [`CsrMatrix::spmv()`] for Kachkovskiy spectral
/// theory GPU Lanczos and Bazavov lattice QCD GPU Dirac.
pub const WGSL_SPMV_CSR_F64: &str = r"
struct Params {
    n: u32,
    nnz: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> col_idx: array<u32>;
@group(0) @binding(3) var<storage, read> vals: array<f64>;
@group(0) @binding(4) var<storage, read> x_vec: array<f64>;
@group(0) @binding(5) var<storage, read_write> y_vec: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.n {
        return;
    }

    let start = row_ptr[row];
    let end = row_ptr[row + 1u];

    var sum: f64 = f64(0.0);
    for (var j = start; j < end; j = j + 1u) {
        sum = sum + vals[j] * x_vec[col_idx[j]];
    }

    y_vec[row] = sum;
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_spmv_identity() {
        // 3×3 identity in CSR
        let mat = CsrMatrix {
            n: 3,
            row_ptr: vec![0, 1, 2, 3],
            col_idx: vec![0, 1, 2],
            values: vec![1.0, 1.0, 1.0],
        };
        let x = vec![3.0, 5.0, 7.0];
        let mut y = vec![0.0; 3];
        mat.spmv(&x, &mut y);
        assert!((y[0] - 3.0).abs() < 1e-14);
        assert!((y[1] - 5.0).abs() < 1e-14);
        assert!((y[2] - 7.0).abs() < 1e-14);
    }

    #[test]
    fn csr_spmv_tridiag() {
        // 4×4 tridiagonal: d=2, e=-1 → same as 1D tight-binding with V=2
        let mat = CsrMatrix {
            n: 4,
            row_ptr: vec![0, 2, 5, 8, 10],
            col_idx: vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
            values: vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        };
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let mut y = vec![0.0; 4];
        mat.spmv(&x, &mut y);
        assert!((y[0] - 2.0).abs() < 1e-14);
        assert!((y[1] - -1.0).abs() < 1e-14);
        assert!((y[2] - 0.0).abs() < 1e-14);
        assert!((y[3] - 0.0).abs() < 1e-14);
    }
}
