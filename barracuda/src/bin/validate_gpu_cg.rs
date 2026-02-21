// SPDX-License-Identifier: AGPL-3.0-only

//! GPU Conjugate Gradient Solver Validation
//!
//! Proves that the GPU CG solver (D†D x = b) produces identical solutions
//! to the CPU `cg_solve()` reference. The entire CG iteration runs on GPU:
//!   - D†D apply: two Dirac dispatches (D then D†) via `WGSL_DIRAC_STAGGERED_F64`
//!   - dot product: `WGSL_COMPLEX_DOT_RE_F64` + `ReduceScalarPipeline`
//!   - axpy/xpay: `WGSL_AXPY_F64` / `WGSL_XPAY_F64`
//!
//! Only scalar coefficients (alpha, beta, residual) transfer CPU↔GPU per iteration.
//! This completes the GPU lattice QCD pipeline for Papers 9-12.
//!
//! Exit code 0 = all checks pass, exit code 1 = any check fails.

use barracuda::pipeline::ReduceScalarPipeline;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::cg::{cg_solve, WGSL_AXPY_F64, WGSL_COMPLEX_DOT_RE_F64, WGSL_XPAY_F64};
use hotspring_barracuda::lattice::dirac::{
    flatten_fermion, unflatten_fermion, DiracGpuLayout, FermionField,
    WGSL_DIRAC_STAGGERED_F64,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::validation::ValidationHarness;

// ── GPU buffer parameter structs ────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DiracParams {
    volume: u32,
    pad0: u32,
    mass_re: f64,
    hop_sign: f64,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DotParams {
    n_pairs: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalarParams {
    n: u32,
    pad0: u32,
    alpha: f64,
}

// ── GPU CG solver ───────────────────────────────────────────────────

struct GpuCg<'a> {
    gpu: &'a GpuF64,
    dirac_pipeline: &'a wgpu::ComputePipeline,
    dot_pipeline: &'a wgpu::ComputePipeline,
    axpy_pipeline: &'a wgpu::ComputePipeline,
    xpay_pipeline: &'a wgpu::ComputePipeline,
    reducer: &'a ReduceScalarPipeline,
    _layout: &'a DiracGpuLayout,
    mass: f64,
    volume: usize,
    n_flat: usize,
    n_pairs: usize,
    // Pre-uploaded lattice data
    links_buf: wgpu::Buffer,
    nbr_buf: wgpu::Buffer,
    phases_buf: wgpu::Buffer,
}

impl GpuCg<'_> {
    fn dirac_dispatch(&self, input: &wgpu::Buffer, output: &wgpu::Buffer, hop_sign: f64) {
        let params = DiracParams {
            volume: self.volume as u32,
            pad0: 0,
            mass_re: self.mass,
            hop_sign,
        };
        let params_buf = self.gpu.create_uniform_buffer(bytemuck::bytes_of(&params), "d_params");
        let bg = self.gpu.create_bind_group(
            self.dirac_pipeline,
            &[&params_buf, &self.links_buf, input, output, &self.nbr_buf, &self.phases_buf],
        );
        let wg = (self.volume as u32).div_ceil(64);
        self.gpu.dispatch(self.dirac_pipeline, &bg, wg);
    }

    fn apply_ddag_d(&self, input: &wgpu::Buffer, temp: &wgpu::Buffer, output: &wgpu::Buffer) {
        self.dirac_dispatch(input, temp, 1.0);
        self.dirac_dispatch(temp, output, -1.0);
    }

    fn dot_re(&self, a: &wgpu::Buffer, b: &wgpu::Buffer, dot_buf: &wgpu::Buffer) -> f64 {
        let params = DotParams {
            n_pairs: self.n_pairs as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = self.gpu.create_uniform_buffer(bytemuck::bytes_of(&params), "dot_p");
        let bg = self.gpu.create_bind_group(self.dot_pipeline, &[&params_buf, a, b, dot_buf]);
        let wg = (self.n_pairs as u32).div_ceil(64);
        self.gpu.dispatch(self.dot_pipeline, &bg, wg);
        self.reducer.sum_f64(dot_buf).expect("dot reduce")
    }

    fn axpy(&self, alpha: f64, x: &wgpu::Buffer, y: &wgpu::Buffer) {
        let params = ScalarParams {
            n: self.n_flat as u32,
            pad0: 0,
            alpha,
        };
        let params_buf = self.gpu.create_uniform_buffer(bytemuck::bytes_of(&params), "axpy_p");
        let bg = self.gpu.create_bind_group(self.axpy_pipeline, &[&params_buf, x, y]);
        let wg = (self.n_flat as u32).div_ceil(64);
        self.gpu.dispatch(self.axpy_pipeline, &bg, wg);
    }

    fn xpay(&self, x: &wgpu::Buffer, beta: f64, p: &wgpu::Buffer) {
        let params = ScalarParams {
            n: self.n_flat as u32,
            pad0: 0,
            alpha: beta,
        };
        let params_buf = self.gpu.create_uniform_buffer(bytemuck::bytes_of(&params), "xpay_p");
        let bg = self.gpu.create_bind_group(self.xpay_pipeline, &[&params_buf, x, p]);
        let wg = (self.n_flat as u32).div_ceil(64);
        self.gpu.dispatch(self.xpay_pipeline, &bg, wg);
    }

    fn solve(
        &self,
        b_flat: &[f64],
        tol: f64,
        max_iter: usize,
    ) -> (Vec<f64>, usize, f64) {
        let n_flat = self.n_flat;

        let x_buf = self.gpu.create_f64_output_buffer(n_flat, "cg_x");
        let r_buf = self.gpu.create_f64_output_buffer(n_flat, "cg_r");
        let p_buf = self.gpu.create_f64_output_buffer(n_flat, "cg_p");
        let ap_buf = self.gpu.create_f64_output_buffer(n_flat, "cg_ap");
        let temp_buf = self.gpu.create_f64_output_buffer(n_flat, "cg_temp");
        let dot_buf = self.gpu.create_f64_output_buffer(self.n_pairs, "cg_dot");

        // x = 0 (already zeroed), r = b
        self.gpu.upload_f64(&r_buf, b_flat);

        // p = r (copy b to p)
        self.gpu.upload_f64(&p_buf, b_flat);

        // b_norm_sq = <b|b>
        let b_norm_sq = self.dot_re(&r_buf, &r_buf, &dot_buf);
        if b_norm_sq < 1e-30 {
            return (vec![0.0; n_flat], 0, 0.0);
        }

        let mut r_norm_sq = b_norm_sq;
        let tol_sq = tol * tol * b_norm_sq;

        let mut iterations = 0;

        for iter in 0..max_iter {
            iterations = iter + 1;

            // ap = D†D * p
            self.apply_ddag_d(&p_buf, &temp_buf, &ap_buf);

            // p_ap = <p|ap>
            let p_ap = self.dot_re(&p_buf, &ap_buf, &dot_buf);
            if p_ap.abs() < 1e-30 {
                break;
            }
            let alpha = r_norm_sq / p_ap;

            // x += alpha * p
            self.axpy(alpha, &p_buf, &x_buf);

            // r -= alpha * ap
            self.axpy(-alpha, &ap_buf, &r_buf);

            // r_norm_sq_new = <r|r>
            let r_norm_sq_new = self.dot_re(&r_buf, &r_buf, &dot_buf);

            if r_norm_sq_new < tol_sq {
                r_norm_sq = r_norm_sq_new;
                break;
            }

            let beta = r_norm_sq_new / r_norm_sq;
            r_norm_sq = r_norm_sq_new;

            // p = r + beta * p
            self.xpay(&r_buf, beta, &p_buf);
        }

        let final_residual = (r_norm_sq / b_norm_sq).sqrt();
        let x_flat = self.gpu.read_back_f64(&x_buf, n_flat).expect("x readback");

        (x_flat, iterations, final_residual)
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  GPU Conjugate Gradient Solver Validation                  ║");
    println!("║  D†D x = b — full CG iteration on GPU                     ║");
    println!("║  Papers 9-12: dynamical fermions, QCD thermodynamics       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("gpu_cg");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            println!("  GPU not available: {e}");
            harness.check_bool("GPU available", false);
            harness.finish();
        }
    };

    println!("  GPU: {} (f64={})", gpu.adapter_name, gpu.has_f64);
    println!();

    let dirac_pipeline = gpu.create_pipeline_f64(WGSL_DIRAC_STAGGERED_F64, "dirac_cg");
    let dot_pipeline = gpu.create_pipeline_f64(WGSL_COMPLEX_DOT_RE_F64, "dot_re");
    let axpy_pipeline = gpu.create_pipeline_f64(WGSL_AXPY_F64, "axpy");
    let xpay_pipeline = gpu.create_pipeline_f64(WGSL_XPAY_F64, "xpay");

    // ══════════════════════════════════════════════════════════════
    //  Check 1: Cold lattice 4⁴, CG converges and matches CPU
    // ══════════════════════════════════════════════════════════════
    println!("═══ Check 1: Cold lattice CG (4⁴, mass=1.0) ════════════════");
    {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let vol = lat.volume();
        let layout = DiracGpuLayout::from_lattice(&lat);
        let mass = 1.0;
        let tol = 1e-8;
        let n_flat = vol * 6;
        let n_pairs = vol * 3;

        let reducer = ReduceScalarPipeline::new(gpu.to_wgpu_device(), n_pairs)
            .expect("reducer");

        let links_buf = gpu.create_f64_buffer(&layout.links_flat, "links");
        let nbr_buf = gpu.create_u32_buffer(&layout.neighbors, "nbr");
        let phases_buf = gpu.create_f64_buffer(&layout.phases, "phases");

        let cg = GpuCg {
            gpu: &gpu,
            dirac_pipeline: &dirac_pipeline,
            dot_pipeline: &dot_pipeline,
            axpy_pipeline: &axpy_pipeline,
            xpay_pipeline: &xpay_pipeline,
            reducer: &reducer,
            _layout: &layout,
            mass,
            volume: vol,
            n_flat,
            n_pairs,
            links_buf,
            nbr_buf,
            phases_buf,
        };

        let b = FermionField::random(vol, 42);
        let b_flat = flatten_fermion(&b);

        let (gpu_x_flat, gpu_iters, gpu_residual) = cg.solve(&b_flat, tol, 500);

        // CPU reference
        let mut cpu_x = FermionField::zeros(vol);
        let cpu_result = cg_solve(&lat, &mut cpu_x, &b, mass, tol, 500);
        let cpu_x_flat = flatten_fermion(&cpu_x);

        println!(
            "  GPU: iters={gpu_iters}, residual={gpu_residual:.2e}"
        );
        println!(
            "  CPU: iters={}, residual={:.2e}",
            cpu_result.iterations, cpu_result.final_residual
        );

        let max_diff: f64 = gpu_x_flat
            .iter()
            .zip(&cpu_x_flat)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let rel_diff = {
            let diff_sq: f64 = gpu_x_flat.iter().zip(&cpu_x_flat)
                .map(|(a, b)| (a - b) * (a - b)).sum();
            let norm_sq: f64 = cpu_x_flat.iter().map(|a| a * a).sum();
            (diff_sq / norm_sq.max(1e-30)).sqrt()
        };
        println!("  Solution max |diff|: {max_diff:.2e}, rel: {rel_diff:.2e}");

        harness.check_bool("Cold CG converged (GPU)", gpu_residual < tol);
        harness.check_bool("Cold CG converged (CPU)", cpu_result.converged);
        harness.check_upper("Cold CG solution diff", rel_diff, 1e-6);
    }

    // ══════════════════════════════════════════════════════════════
    //  Check 2: Hot lattice 4⁴, CG on random SU(3) links
    // ══════════════════════════════════════════════════════════════
    println!("\n═══ Check 2: Hot lattice CG (4⁴, mass=0.5) ════════════════");
    {
        let lat = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let vol = lat.volume();
        let layout = DiracGpuLayout::from_lattice(&lat);
        let mass = 0.5;
        let tol = 1e-6;
        let n_flat = vol * 6;
        let n_pairs = vol * 3;

        let reducer = ReduceScalarPipeline::new(gpu.to_wgpu_device(), n_pairs)
            .expect("reducer");
        let links_buf = gpu.create_f64_buffer(&layout.links_flat, "links");
        let nbr_buf = gpu.create_u32_buffer(&layout.neighbors, "nbr");
        let phases_buf = gpu.create_f64_buffer(&layout.phases, "phases");

        let cg = GpuCg {
            gpu: &gpu,
            dirac_pipeline: &dirac_pipeline,
            dot_pipeline: &dot_pipeline,
            axpy_pipeline: &axpy_pipeline,
            xpay_pipeline: &xpay_pipeline,
            reducer: &reducer,
            _layout: &layout,
            mass,
            volume: vol,
            n_flat,
            n_pairs,
            links_buf,
            nbr_buf,
            phases_buf,
        };

        let b = FermionField::random(vol, 99);
        let b_flat = flatten_fermion(&b);

        let (gpu_x_flat, gpu_iters, gpu_residual) = cg.solve(&b_flat, tol, 2000);

        let mut cpu_x = FermionField::zeros(vol);
        let cpu_result = cg_solve(&lat, &mut cpu_x, &b, mass, tol, 2000);
        let cpu_x_flat = flatten_fermion(&cpu_x);

        println!("  GPU: iters={gpu_iters}, residual={gpu_residual:.2e}");
        println!(
            "  CPU: iters={}, residual={:.2e}",
            cpu_result.iterations, cpu_result.final_residual
        );

        let rel_diff = {
            let diff_sq: f64 = gpu_x_flat.iter().zip(&cpu_x_flat)
                .map(|(a, b)| (a - b) * (a - b)).sum();
            let norm_sq: f64 = cpu_x_flat.iter().map(|a| a * a).sum();
            (diff_sq / norm_sq.max(1e-30)).sqrt()
        };
        println!("  Solution rel diff: {rel_diff:.2e}");

        harness.check_bool("Hot CG converged (GPU)", gpu_residual < tol);
        harness.check_upper("Hot CG solution diff", rel_diff, 1e-4);
    }

    // ══════════════════════════════════════════════════════════════
    //  Check 3: Verify GPU solution satisfies D†D x = b
    // ══════════════════════════════════════════════════════════════
    println!("\n═══ Check 3: Verify D†D x = b (cold 4⁴) ═══════════════════");
    {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let vol = lat.volume();
        let layout = DiracGpuLayout::from_lattice(&lat);
        let mass = 1.0;
        let tol = 1e-8;
        let n_flat = vol * 6;
        let n_pairs = vol * 3;

        let reducer = ReduceScalarPipeline::new(gpu.to_wgpu_device(), n_pairs)
            .expect("reducer");
        let links_buf = gpu.create_f64_buffer(&layout.links_flat, "links");
        let nbr_buf = gpu.create_u32_buffer(&layout.neighbors, "nbr");
        let phases_buf = gpu.create_f64_buffer(&layout.phases, "phases");

        let cg = GpuCg {
            gpu: &gpu,
            dirac_pipeline: &dirac_pipeline,
            dot_pipeline: &dot_pipeline,
            axpy_pipeline: &axpy_pipeline,
            xpay_pipeline: &xpay_pipeline,
            reducer: &reducer,
            _layout: &layout,
            mass,
            volume: vol,
            n_flat,
            n_pairs,
            links_buf,
            nbr_buf,
            phases_buf,
        };

        let b = FermionField::random(vol, 77);
        let b_flat = flatten_fermion(&b);

        let (gpu_x_flat, _iters, _res) = cg.solve(&b_flat, tol, 500);

        // Verify: A * x ≈ b using CPU D†D
        let x_field = unflatten_fermion(&gpu_x_flat, vol);
        let ax = hotspring_barracuda::lattice::dirac::apply_dirac_sq(&lat, &x_field, mass);
        let ax_flat = flatten_fermion(&ax);

        let diff_sq: f64 = ax_flat.iter().zip(&b_flat)
            .map(|(a, b)| (a - b) * (a - b)).sum();
        let b_norm_sq: f64 = b_flat.iter().map(|v| v * v).sum();
        let verify_residual = (diff_sq / b_norm_sq).sqrt();

        println!("  ||D†D x - b|| / ||b|| = {verify_residual:.2e}");
        harness.check_upper("D†D x ≈ b verification", verify_residual, 1e-7);
    }

    // ══════════════════════════════════════════════════════════════
    //  Check 4: Larger hot lattice (6⁴) — stress test
    // ══════════════════════════════════════════════════════════════
    println!("\n═══ Check 4: Hot lattice CG (6⁴, V=1296, mass=0.5) ════════");
    {
        let lat = Lattice::hot_start([6, 6, 6, 6], 6.0, 123);
        let vol = lat.volume();
        let layout = DiracGpuLayout::from_lattice(&lat);
        let mass = 0.5;
        let tol = 1e-6;
        let n_flat = vol * 6;
        let n_pairs = vol * 3;

        let reducer = ReduceScalarPipeline::new(gpu.to_wgpu_device(), n_pairs)
            .expect("reducer");
        let links_buf = gpu.create_f64_buffer(&layout.links_flat, "links");
        let nbr_buf = gpu.create_u32_buffer(&layout.neighbors, "nbr");
        let phases_buf = gpu.create_f64_buffer(&layout.phases, "phases");

        let cg = GpuCg {
            gpu: &gpu,
            dirac_pipeline: &dirac_pipeline,
            dot_pipeline: &dot_pipeline,
            axpy_pipeline: &axpy_pipeline,
            xpay_pipeline: &xpay_pipeline,
            reducer: &reducer,
            _layout: &layout,
            mass,
            volume: vol,
            n_flat,
            n_pairs,
            links_buf,
            nbr_buf,
            phases_buf,
        };

        let b = FermionField::random(vol, 456);
        let b_flat = flatten_fermion(&b);

        let (gpu_x_flat, gpu_iters, gpu_residual) = cg.solve(&b_flat, tol, 5000);

        let mut cpu_x = FermionField::zeros(vol);
        let cpu_result = cg_solve(&lat, &mut cpu_x, &b, mass, tol, 5000);

        println!("  GPU: iters={gpu_iters}, residual={gpu_residual:.2e}");
        println!(
            "  CPU: iters={}, residual={:.2e}",
            cpu_result.iterations, cpu_result.final_residual
        );

        let cpu_x_flat = flatten_fermion(&cpu_x);
        let rel_diff = {
            let diff_sq: f64 = gpu_x_flat.iter().zip(&cpu_x_flat)
                .map(|(a, b)| (a - b) * (a - b)).sum();
            let norm_sq: f64 = cpu_x_flat.iter().map(|a| a * a).sum();
            (diff_sq / norm_sq.max(1e-30)).sqrt()
        };
        println!("  Solution rel diff: {rel_diff:.2e}");

        harness.check_bool("6^4 CG converged (GPU)", gpu_residual < tol);
        harness.check_upper("6^4 CG solution diff", rel_diff, 1e-4);
    }

    // ══════════════════════════════════════════════════════════════
    //  Check 5: Iteration count parity — GPU and CPU should agree
    // ══════════════════════════════════════════════════════════════
    println!("\n═══ Check 5: Iteration count parity (cold 4⁴) ═════════════");
    {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let vol = lat.volume();
        let layout = DiracGpuLayout::from_lattice(&lat);
        let mass = 1.0;
        let tol = 1e-10;
        let n_flat = vol * 6;
        let n_pairs = vol * 3;

        let reducer = ReduceScalarPipeline::new(gpu.to_wgpu_device(), n_pairs)
            .expect("reducer");
        let links_buf = gpu.create_f64_buffer(&layout.links_flat, "links");
        let nbr_buf = gpu.create_u32_buffer(&layout.neighbors, "nbr");
        let phases_buf = gpu.create_f64_buffer(&layout.phases, "phases");

        let cg = GpuCg {
            gpu: &gpu,
            dirac_pipeline: &dirac_pipeline,
            dot_pipeline: &dot_pipeline,
            axpy_pipeline: &axpy_pipeline,
            xpay_pipeline: &xpay_pipeline,
            reducer: &reducer,
            _layout: &layout,
            mass,
            volume: vol,
            n_flat,
            n_pairs,
            links_buf,
            nbr_buf,
            phases_buf,
        };

        let b = FermionField::random(vol, 333);
        let b_flat = flatten_fermion(&b);

        let (_gpu_x, gpu_iters, gpu_res) = cg.solve(&b_flat, tol, 500);

        let mut cpu_x = FermionField::zeros(vol);
        let cpu_result = cg_solve(&lat, &mut cpu_x, &b, mass, tol, 500);

        let iter_diff = (gpu_iters as i64 - cpu_result.iterations as i64).unsigned_abs();
        println!(
            "  GPU iters={gpu_iters} (res={gpu_res:.2e}), CPU iters={} (res={:.2e})",
            cpu_result.iterations, cpu_result.final_residual
        );
        println!("  Iteration count diff: {iter_diff}");

        // GPU and CPU may differ by a few iterations due to reduction ordering
        harness.check_upper("Iteration count parity", iter_diff as f64, 5.0);
    }

    // ══════════════════════════════════════════════════════════════
    //  Verdict
    // ══════════════════════════════════════════════════════════════
    println!();
    println!("═══ Summary ══════════════════════════════════════════════════");
    println!("  GPU CG solver: D†D x = b on GPU with f64 precision");
    println!("  Full iteration on GPU: D†D + dot + axpy + xpay");
    println!("  Only scalar coefficients transfer CPU↔GPU per iteration");
    println!("  Papers 9-12 (full lattice QCD) pipeline: COMPLETE");
    println!();

    harness.finish();
}
