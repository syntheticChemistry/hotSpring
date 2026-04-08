// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sovereign pipeline validation: coralReef compile + DRM dispatch test kernels.

mod prelude;

pub mod basic;
pub mod f64_ops;
pub mod shader_inventory;

pub use basic::{test_buffer_read_probe, test_write_constant, test_write_constant_inner, test_write_thread_id};
pub use f64_ops::{
    test_axpy_f64, test_axpy_minimal, test_axpy_with_num_workgroups, test_cg_compute_alpha,
    test_complex_dot_re, test_f64_add_3buf, test_f64_cmp_branch, test_f64_copy, test_f64_div_3buf,
    test_f64_literal_write, test_nwg_idx_debug, test_num_workgroups, test_uniform_read,
};
pub use shader_inventory::test_qcd_shader_compilation;
