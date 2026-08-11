// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU-only fallback when no GPU is available.

use hotspring_barracuda::bench::{generate_test_sequence, generate_training_data};
use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use hotspring_barracuda::validation::ValidationHarness;

use super::{INPUT_SIZE, SEQUENCE_LENGTH};

pub fn run(harness: &mut ValidationHarness) {
    println!();
    println!("═══ CPU-Only Mode: NPU Capability Envelope ═══");

    let reservoir_sizes = [16, 50, 100, 200];
    for &rs in &reservoir_sizes {
        let config = EsnConfig {
            input_size: INPUT_SIZE,
            reservoir_size: rs,
            output_size: 1,
            spectral_radius: 0.95,
            connectivity: 0.2,
            leak_rate: 0.3,
            regularization: 1e-4,
            seed: 42,
            ..Default::default()
        };
        let (train_seqs, train_targets) = generate_training_data(8, SEQUENCE_LENGTH, INPUT_SIZE);
        let test_seq = generate_test_sequence(999, SEQUENCE_LENGTH, INPUT_SIZE);

        let mut esn = EchoStateNetwork::new(config);
        esn.train(&train_seqs, &train_targets);
        let exported = esn.export_weights().expect("export");

        let cpu_pred = esn.predict(&test_seq).unwrap_or_default();
        let mut npu = NpuSimulator::from_exported(&exported);
        let npu_pred = npu.predict(&test_seq);

        let diff = (cpu_pred[0] - npu_pred[0]).abs();
        let rel = diff / cpu_pred[0].abs().max(1e-10);
        println!(
            "  RS={rs}: CPU={:.6}, NPU={:.6}, rel_diff={rel:.4e}",
            cpu_pred[0], npu_pred[0]
        );
        harness.check_bool(&format!("cpu_npu_parity_rs{rs}"), rel < 0.05);
    }
}
