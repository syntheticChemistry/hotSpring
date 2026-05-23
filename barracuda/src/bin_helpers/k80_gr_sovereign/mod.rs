// SPDX-License-Identifier: AGPL-3.0-or-later

//! Helpers extracted from `exp184_k80_gr_sovereign`: register map, firmware loading,
//! and ember MMIO RPC wrappers for K80 sovereign GR engine boot.

pub mod ember_mmio;
pub mod firmware;
pub mod registers;

pub use ember_mmio::{connect_ember, csdata_batch_ops, r32, upload_dmem, upload_imem, w32};
pub use firmware::{extract_arg, load_fw_words, load_mmio_init, validate_csdata_layout};
pub use registers::*;
