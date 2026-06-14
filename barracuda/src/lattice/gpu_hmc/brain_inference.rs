// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::md::reservoir::npu::{ExportedWeights, MultiHeadNpu};

use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuBackend {
    Hardware,
    Software,
}

#[derive(Debug, Clone)]
pub struct NpuInferenceMetrics {
    pub backend: NpuBackend,
    pub latency_us: u64,
    pub energy_uj: f64,
    pub observation_count: usize,
}

#[derive(Debug, Clone)]
pub struct NpuRunStats {
    pub inferences: usize,
    pub suggestions_applied: usize,
    pub mean_latency_us: f64,
    pub mean_energy_uj: f64,
    pub backend: NpuBackend,
    pub cross_gpu_agreements: usize,
    pub cross_gpu_disagreements: usize,
    pub total_observations: usize,
}

impl NpuRunStats {
    pub(crate) fn new(backend: NpuBackend) -> Self {
        Self {
            inferences: 0,
            suggestions_applied: 0,
            mean_latency_us: 0.0,
            mean_energy_uj: 0.0,
            backend,
            cross_gpu_agreements: 0,
            cross_gpu_disagreements: 0,
            total_observations: 0,
        }
    }

    pub(crate) fn update_inference(&mut self, metrics: &NpuInferenceMetrics) {
        self.inferences += 1;
        let n = self.inferences as f64;
        self.mean_latency_us += (metrics.latency_us as f64 - self.mean_latency_us) / n;
        self.mean_energy_uj += (metrics.energy_uj - self.mean_energy_uj) / n;
    }

    pub fn display_table(&self, traj_range: &str) -> String {
        let backend_label = match self.backend {
            NpuBackend::Hardware => "Akida (int8)",
            NpuBackend::Software => "CPU (f32)",
        };
        let agree_rate = if self.cross_gpu_agreements + self.cross_gpu_disagreements > 0 {
            self.cross_gpu_agreements as f64
                / (self.cross_gpu_agreements + self.cross_gpu_disagreements) as f64
        } else {
            1.0
        };
        format!(
            "\
╭─ NPU Brain ({traj_range}) ─────────────────────────────────────────╮
│ Backend             │ {:<50} │
│ Mean latency        │ {:<50} │
│ Mean energy/infer   │ {:<50} │
│ Inferences          │ {:<50} │
│ Observations (2/tr) │ {:<50} │
│ Suggestions applied │ {:<50} │
│ Cross-GPU agreement │ {:<50} │
╰─────────────────────┴────────────────────────────────────────────────╯",
            backend_label,
            format!("{:.0} us", self.mean_latency_us),
            format!("{:.2} uJ", self.mean_energy_uj),
            self.inferences,
            self.total_observations,
            self.suggestions_applied,
            format!(
                "{:.0}% ({}/{})",
                agree_rate * 100.0,
                self.cross_gpu_agreements,
                self.cross_gpu_agreements + self.cross_gpu_disagreements
            ),
        )
    }
}

pub struct NpuInference {
    sw_npu: MultiHeadNpu,
    #[cfg(feature = "npu-hw")]
    hw_npu: Option<crate::md::npu_hw::NpuHardware>,
    cpu_tdp_per_core_w: f64,
}

impl NpuInference {
    pub fn new(weights: &ExportedWeights) -> Self {
        let sw_npu = MultiHeadNpu::from_exported(weights);

        #[cfg(feature = "npu-hw")]
        let hw_npu = {
            use crate::md::npu_hw::NpuHardware;
            match NpuHardware::discover() {
                Some(info) => {
                    eprintln!(
                        "[brain] Akida NPU PRIMARY: {} NPUs, {} MB SRAM, PCIe {}",
                        info.npu_count, info.memory_mb, info.pcie_address
                    );
                    Some(NpuHardware::from_exported(weights, info))
                }
                None => {
                    eprintln!("[brain] No Akida NPU — falling back to software ESN");
                    None
                }
            }
        };

        Self {
            sw_npu,
            #[cfg(feature = "npu-hw")]
            hw_npu,
            cpu_tdp_per_core_w: 4.0,
        }
    }

    pub fn active_backend(&self) -> NpuBackend {
        #[cfg(feature = "npu-hw")]
        if self.hw_npu.is_some() {
            return NpuBackend::Hardware;
        }
        NpuBackend::Software
    }

    pub fn infer(&mut self, input_sequence: &[Vec<f64>]) -> (Vec<f64>, NpuInferenceMetrics) {
        #[cfg(feature = "npu-hw")]
        if let Some(ref mut hw) = self.hw_npu {
            let start = Instant::now();
            let outputs = hw.predict(input_sequence);
            let latency_us = start.elapsed().as_micros() as u64;
            let energy_uj = 300.0 * latency_us as f64 / 1_000.0;
            return (
                outputs,
                NpuInferenceMetrics {
                    backend: NpuBackend::Hardware,
                    latency_us,
                    energy_uj,
                    observation_count: input_sequence.len(),
                },
            );
        }

        let start = Instant::now();
        let outputs = self.sw_npu.predict_all_heads(input_sequence);
        let latency_us = start.elapsed().as_micros() as u64;
        let energy_uj = latency_us as f64 * self.cpu_tdp_per_core_w / 1_000.0;
        (
            outputs,
            NpuInferenceMetrics {
                backend: NpuBackend::Software,
                latency_us,
                energy_uj,
                observation_count: input_sequence.len(),
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct CrossGpuAgreement {
    pub traj_idx: usize,
    pub both_accepted: bool,
    pub both_rejected: bool,
    pub delta_h_spread: f64,
    pub plaquette_spread: f64,
}

impl CrossGpuAgreement {
    pub fn acceptance_agrees(&self) -> bool {
        self.both_accepted || self.both_rejected
    }
}
