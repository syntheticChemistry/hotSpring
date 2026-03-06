// SPDX-License-Identifier: AGPL-3.0-only

#![allow(missing_docs)]

//! NPU checkpoint save/load: ESN weights and Nautilus shell JSON serialization.
//!
//! Shell config uses a sibling path convention: `weights.json` → `weights.nautilus.json`.

use crate::md::reservoir::{Activation, ExportedWeights, MultiHeadNpu};
use barracuda::nautilus::NautilusBrain;

/// Derive the Nautilus shell path from an ESN weights path.
/// E.g. `weights.json` → `weights.nautilus.json`, `model.bin` → `model.nautilus.json`.
#[must_use]
pub fn nautilus_shell_path_from_weights(weights_path: &str) -> String {
    weights_path
        .replace(".bin", ".nautilus.json")
        .replace(".json", ".nautilus.json")
}

/// Save ESN weights to JSON. Creates parent directories if needed.
/// Returns `true` on success.
pub fn save_esn_weights(npu: &mut MultiHeadNpu, path: &str) -> bool {
    let base = npu.base_mut();
    let weights = ExportedWeights {
        w_in: base.export_w_in(),
        w_res: base.export_w_res(),
        w_out: base.export_w_out(),
        input_size: base.input_size(),
        reservoir_size: base.reservoir_size(),
        output_size: base.output_size(),
        leak_rate: base.leak_rate(),
        activation: Activation::default(),
    };
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, serde_json::to_string(&weights).unwrap_or_default()).is_ok()
}

/// Load ESN weights from JSON. Returns `None` on parse or read error.
#[must_use]
pub fn load_esn_weights(path: &str) -> Option<ExportedWeights> {
    let json = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&json).ok()
}

/// Save Nautilus shell to JSON. Creates parent directories if needed.
/// Returns `true` on success.
pub fn save_nautilus_shell(brain: &NautilusBrain, path: &str) -> bool {
    match brain.to_json() {
        Ok(json) => {
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::write(path, json).is_ok()
        }
        Err(e) => {
            eprintln!("  [Nautilus] Failed to serialize shell: {e}");
            false
        }
    }
}

/// Load Nautilus shell from JSON. Returns `None` on parse or read error.
#[must_use]
pub fn load_nautilus_shell(path: &str) -> Option<NautilusBrain> {
    let json = std::fs::read_to_string(path).ok()?;
    NautilusBrain::from_json(&json).ok()
}
