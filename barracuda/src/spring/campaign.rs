// SPDX-License-Identifier: AGPL-3.0-or-later

//! Production campaign orchestration.
//!
//! Manages multi-volume, multi-beta, multi-seed campaign grids with
//! checkpoint/resume, provenance, and result aggregation.

use serde::{Deserialize, Serialize};

/// A single configuration point in the campaign grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignConfig {
    pub dims: [u32; 4],
    pub beta: f64,
    pub seed: u32,
    pub n_warmup: usize,
    pub n_production: usize,
    pub dt: f64,
    pub n_md_steps: usize,
    /// Hot start epsilon (0 = cold start, >0 = random perturbation magnitude).
    #[serde(default)]
    pub epsilon: f64,
    /// Whether warmup count is adaptive (runtime convergence detection).
    #[serde(default)]
    pub adaptive_warmup: bool,
}

/// Result of a completed configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignResult {
    pub config: CampaignConfig,
    pub mean_plaquette: f64,
    pub plaquette_std: f64,
    pub acceptance_rate: f64,
    pub measurements: Vec<f64>,
}

/// Campaign grid definition.
pub struct CampaignGrid {
    pub configs: Vec<CampaignConfig>,
    pub output_dir: std::path::PathBuf,
}

impl CampaignGrid {
    /// Generate the standard arXiv campaign grid:
    /// 3 volumes × 3 β × 5 seeds = 45 configurations.
    #[must_use]
    pub fn arxiv_standard(output_dir: std::path::PathBuf) -> Self {
        let volumes: &[[u32; 4]] = &[
            [16, 16, 16, 16],
            [24, 24, 24, 24],
            [32, 32, 32, 32],
        ];
        let betas = [5.70, 5.90, 6.00];
        let seeds = [42, 137, 271, 503, 719];

        let mut configs = Vec::with_capacity(45);
        for &dims in volumes {
            let n_warmup = match dims[0] {
                16 => 200,
                24 => 500,
                _ => 1000,
            };
            for &beta in &betas {
                for &seed in &seeds {
                    configs.push(CampaignConfig {
                        dims,
                        beta,
                        seed,
                        n_warmup,
                        n_production: 200,
                        dt: 0.01,
                        n_md_steps: 20,
                        epsilon: 0.0,
                        adaptive_warmup: false,
                    });
                }
            }
        }
        Self { configs, output_dir }
    }

    /// Generate the v3 "climate shift" campaign grid:
    /// Unified protocol — one dt, one start type, adaptive warmup.
    ///
    /// 3 volumes x 3 betas x 5 seeds = 45 configurations.
    /// Protocol: dt=0.005, n_md=20 (tau=0.1), hot start epsilon=0.2,
    /// adaptive warmup (convergence-detected), 500 production trajectories.
    #[must_use]
    pub fn arxiv_v3(output_dir: std::path::PathBuf) -> Self {
        let volumes: &[[u32; 4]] = &[
            [16, 16, 16, 16],
            [24, 24, 24, 24],
            [32, 32, 32, 32],
        ];
        let betas = [5.90, 6.00, 6.20];
        let seeds = [42, 137, 271, 503, 719];

        let mut configs = Vec::with_capacity(45);
        for &dims in volumes {
            for &beta in &betas {
                for &seed in &seeds {
                    configs.push(CampaignConfig {
                        dims,
                        beta,
                        seed,
                        n_warmup: 2000,
                        n_production: 500,
                        dt: 0.005,
                        n_md_steps: 20,
                        epsilon: 0.2,
                        adaptive_warmup: true,
                    });
                }
            }
        }
        Self { configs, output_dir }
    }

    /// Generate a cross-validation subset (1 seed per grid point).
    /// Used for NVIDIA overnight cross-check runs.
    #[must_use]
    pub fn arxiv_v3_xval(output_dir: std::path::PathBuf) -> Self {
        let volumes: &[[u32; 4]] = &[
            [16, 16, 16, 16],
            [24, 24, 24, 24],
            [32, 32, 32, 32],
        ];
        let betas = [5.90, 6.00, 6.20];
        let seed = 42;

        let mut configs = Vec::with_capacity(9);
        for &dims in volumes {
            for &beta in &betas {
                configs.push(CampaignConfig {
                    dims,
                    beta,
                    seed,
                    n_warmup: 2000,
                    n_production: 500,
                    dt: 0.005,
                    n_md_steps: 20,
                    epsilon: 0.2,
                    adaptive_warmup: true,
                });
            }
        }
        Self { configs, output_dir }
    }

    /// Check which configurations have already been completed (JSON exists).
    #[must_use]
    pub fn remaining(&self) -> Vec<&CampaignConfig> {
        self.configs
            .iter()
            .filter(|c| {
                let filename = format!(
                    "su3_{}x{}x{}x{}_b{:.2}_s{}.json",
                    c.dims[0], c.dims[1], c.dims[2], c.dims[3], c.beta, c.seed
                );
                !self.output_dir.join(filename).exists()
            })
            .collect()
    }
}
