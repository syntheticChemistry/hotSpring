// SPDX-License-Identifier: AGPL-3.0-only

#![deny(clippy::expect_used, clippy::unwrap_used)]

//! hotSpring Forge — local hardware discovery and cross-substrate dispatch.
//!
//! Forge discovers what compute substrates exist on THIS machine at runtime
//! and routes workloads to the best capable substrate. It leans on
//! toadstool/barracuda for GPU discovery and device management, and adds
//! NPU probing and cross-substrate orchestration locally.
//!
//! # Design Principle
//!
//! hotSpring is a biome. Toadstool (barracuda) is the fungus — it lives in
//! every biome. We lean on it for what it already provides (GPU enumeration,
//! shader dispatch, buffer management), and evolve new capabilities locally
//! (NPU probing, cross-substrate routing). Toadstool absorbs what works,
//! then all springs benefit.
//!
//! Springs don't reference each other. neuralSpring doesn't import hotSpring.
//! But both lean on toadstool independently — hotSpring evolves f64 shaders,
//! neuralSpring evolves ML shaders, and toadstool absorbs both.
//!
//! # Architecture
//!
//! ```text
//!    ┌─────────────────────────────┐
//!    │  probe (barracuda + local)  │  wgpu adapters + /dev + /proc
//!    └──────────┬──────────────────┘
//!               │ Vec<Substrate>
//!    ┌──────────▼──────────────────┐
//!    │       inventory             │  unified view of all substrates
//!    └──────────┬──────────────────┘
//!               │ &Substrate
//!    ┌──────────▼──────────────────┐
//!    │       dispatch              │  capability-based routing
//!    └─────────────────────────────┘
//! ```

pub mod bridge;
pub mod dispatch;
pub mod inventory;
pub mod pipeline;
pub mod probe;
pub mod substrate;
