// SPDX-License-Identifier: AGPL-3.0-or-later

//! UniBin CLI — clap subcommands for the eukaryotic hotSpring binary.

use clap::{Parser, Subcommand};

/// hotSpring UniBin — physics validation, certification, and IPC server.
#[derive(Parser)]
#[command(
    name = "hotspring",
    version,
    about = "Eukaryotic physics validation primal — certification, validation, and IPC server"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Available subcommands.
#[derive(Subcommand)]
pub enum Commands {
    /// Run composition certification (absorbed guideStone, L0-L5).
    Certify {
        /// Maximum certification layer (0-5, default 5).
        #[arg(long, value_name = "N")]
        layer: Option<u8>,
        /// Run only Layer 0 (bare structural validation, no primals needed).
        #[arg(long, default_value_t = false)]
        bare: bool,
    },
    /// Run validation scenarios (absorbed experiments).
    Validate {
        /// Filter by track (e.g. nuclear-physics, lattice-qcd, gpu-compute).
        #[arg(long)]
        track: Option<String>,
        /// Run a single scenario by ID.
        #[arg(long)]
        scenario: Option<String>,
        /// Filter by tier: rust (structural), live (IPC), both.
        #[arg(long)]
        tier: Option<String>,
        /// List all available scenarios without running them.
        #[arg(long, default_value_t = false)]
        list: bool,
    },
    /// Show composition health and capability discovery status.
    Status,
    /// Show version information.
    Version,
}
