// SPDX-License-Identifier: AGPL-3.0-or-later

//! CG iteration encoding for true multi-shift solver.

use super::TrueMultiShiftPipelines;
use super::ms_cg_buffers::CgBuffers;
use crate::lattice::gpu_hmc::GpuF64;
use crate::lattice::gpu_hmc::dynamical::GpuDynHmcPipelines;
use crate::lattice::gpu_hmc::resident_cg_buffers::encode_reduce_chain;

impl CgBuffers {
    /// Encode one batch of true multi-shift CG iterations.
    pub fn encode_iteration_batch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        dyn_pipelines: &GpuDynHmcPipelines,
        ms_pipelines: &TrueMultiShiftPipelines,
        ms_x_bgs: &[wgpu::BindGroup],
        batch: usize,
    ) {
        for _ in 0..batch {
            GpuF64::encode_pass(
                enc,
                &dyn_pipelines.dirac_pipeline,
                &self.dirac_d_bg,
                self.wg_dirac,
            );
            GpuF64::encode_pass(
                enc,
                &dyn_pipelines.dirac_pipeline,
                &self.dirac_ddag_bg,
                self.wg_dirac,
            );

            GpuF64::encode_pass(
                enc,
                &dyn_pipelines.dot_pipeline,
                &self.dot_pap_bg,
                self.wg_dot,
            );
            encode_reduce_chain(enc, &ms_pipelines.base.reduce_pipeline, &self.reduce_to_pap);
            GpuF64::encode_pass(
                enc,
                &dyn_pipelines.dot_pipeline,
                &self.dot_pp_bg,
                self.wg_dot,
            );
            encode_reduce_chain(enc, &ms_pipelines.base.reduce_pipeline, &self.reduce_to_pp);
            GpuF64::encode_pass(
                enc,
                &ms_pipelines.shifted_alpha_pipeline,
                &self.compute_alpha_bg,
                1,
            );

            GpuF64::encode_pass(enc, &ms_pipelines.zeta_pipeline, &self.zeta_bg, 1);

            for ms_bg in ms_x_bgs.iter().take(self.n_shifts) {
                GpuF64::encode_pass(enc, &ms_pipelines.ms_x_pipeline, ms_bg, self.wg_vec);
            }

            GpuF64::encode_pass(
                enc,
                &ms_pipelines.shifted_xr_pipeline,
                &self.update_xr_bg,
                self.wg_vec,
            );

            GpuF64::encode_pass(
                enc,
                &dyn_pipelines.dot_pipeline,
                &self.dot_rr_bg,
                self.wg_dot,
            );
            encode_reduce_chain(
                enc,
                &ms_pipelines.base.reduce_pipeline,
                &self.reduce_to_rz_new,
            );

            GpuF64::encode_pass(
                enc,
                &ms_pipelines.base.compute_beta_pipeline,
                &self.compute_beta_bg,
                1,
            );

            GpuF64::encode_pass(
                enc,
                &ms_pipelines.base.update_p_pipeline,
                &self.update_p_bg,
                self.wg_vec,
            );

            for ms_p_bg in self.ms_p_bgs.iter().take(self.n_shifts) {
                GpuF64::encode_pass(enc, &ms_pipelines.ms_p_pipeline, ms_p_bg, self.wg_vec);
            }

            enc.copy_buffer_to_buffer(&self.alpha_buf, 0, &self.alpha_prev_buf, 0, 8);
            enc.copy_buffer_to_buffer(&self.beta_buf, 0, &self.beta_prev_buf, 0, 8);
        }
    }
}
