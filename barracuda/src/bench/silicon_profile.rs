// SPDX-License-Identifier: AGPL-3.0-or-later

//! Silicon Profile: measured personality of every functional unit on a GPU.
//!
//! Each GPU card contains multiple independent silicon blocks (ALU, TMU, ROP,
//! memory controller, L2 cache, tensor cores, shared memory / LDS). Traditional
//! HPC codes use only one — the FP64 ALU — and leave the rest idle. ecoPrimals
//! routes workload phases to the cheapest silicon that can handle them, filling
//! alternative units first and reserving FP64 for precision-critical work.
//!
//! ## Tier routing philosophy
//!
//! ```text
//! TIER 0  TMU          lookup tables, PRNG transcendentals, stencil access
//! TIER 1  Tensor cores  SU(3) matmul, preconditioner (NVIDIA only via SASS)
//! TIER 2  FP32 ALU      DF64 Dekker pairs — bulk compute
//! TIER 3  ROP / Atomics scatter-add for force accumulation
//! TIER 4  Subgroup       warp/wavefront intrinsics for reductions
//! TIER 5  Shared memory  workgroup-level communication, halo exchange
//! TIER 6  FP64 ALU      LAST — Metropolis test, observable accumulation
//! ```
//!
//! The `SiliconProfile` captures what we know about each unit (both from spec
//! sheets and from measured benchmarks) so the tier router has concrete numbers.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Every distinct silicon functional unit we can address on a GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SiliconUnit {
    /// Shader ALU — FP32 FMAC pipeline (bulk compute, DF64 host)
    Fp32Alu,
    /// Shader ALU — native FP64 pipeline (1:2 on HPC, 1:64 on gaming)
    Fp64Alu,
    /// Texture Mapping Unit — hardware interpolation, cache-backed lookup
    Tmu,
    /// Render Output Pipeline — blend, atomicAdd, scatter-write
    Rop,
    /// Tensor / Matrix Cores — FP16/TF32/FP64 DMMA tiles (NVIDIA)
    TensorCore,
    /// Memory controller — VRAM bandwidth (sequential, coalesced)
    MemoryBandwidth,
    /// L2 cache (+ Infinity Cache on AMD RDNA)
    CacheHierarchy,
    /// Workgroup shared memory / Local Data Share
    SharedMemory,
    /// Subgroup (warp/wavefront) intrinsics — shuffle, ballot, reduce
    SubgroupIntrinsics,
}

impl std::fmt::Display for SiliconUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fp32Alu => write!(f, "FP32 ALU"),
            Self::Fp64Alu => write!(f, "FP64 ALU"),
            Self::Tmu => write!(f, "TMU"),
            Self::Rop => write!(f, "ROP"),
            Self::TensorCore => write!(f, "Tensor"),
            Self::MemoryBandwidth => write!(f, "Mem BW"),
            Self::CacheHierarchy => write!(f, "Cache"),
            Self::SharedMemory => write!(f, "LDS/Shared"),
            Self::SubgroupIntrinsics => write!(f, "Subgroup"),
        }
    }
}

/// Throughput measurement for a single silicon unit.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UnitThroughput {
    /// Theoretical peak from vendor spec sheet (TFLOPS, GT/s, GB/s — unit-specific).
    pub theoretical_peak: f64,
    /// Measured peak from micro-benchmark saturation experiment.
    pub measured_peak: f64,
    /// Efficiency: measured / theoretical (0.0–1.0+). >1.0 means spec undercount.
    pub efficiency: f64,
    /// Human-readable unit for the throughput value (e.g. "TFLOPS", "GT/s", "GB/s").
    pub unit: String,
    /// GPU idle power (Watts) measured before this unit's benchmark.
    #[serde(default)]
    pub idle_watts: f64,
    /// GPU average power (Watts) during saturation of this unit.
    #[serde(default)]
    pub loaded_watts: f64,
    /// Marginal power cost: `loaded_watts - idle_watts`. 0.0 if no power data.
    #[serde(default)]
    pub delta_watts: f64,
    /// Energy efficiency: `measured_peak / delta_watts`. 0.0 if no power data.
    #[serde(default)]
    pub ops_per_watt: f64,
}

/// Composition multiplier: measured speedup when two units run simultaneously.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompositionEntry {
    pub unit_a: SiliconUnit,
    pub unit_b: SiliconUnit,
    /// Time for A alone + B alone (serial estimate).
    pub serial_ms: f64,
    /// Time for A+B in same dispatch (compound).
    pub compound_ms: f64,
    /// Multiplier: serial / compound. >1.0 means they truly run in parallel.
    pub multiplier: f64,
    /// GPU idle power before composition benchmark (Watts). 0.0 if not measured.
    #[serde(default)]
    pub idle_watts: f64,
    /// GPU average power during compound dispatch (Watts). 0.0 if not measured.
    #[serde(default)]
    pub compound_watts: f64,
    /// Marginal power of running both units together (Watts). 0.0 if not measured.
    #[serde(default)]
    pub delta_watts: f64,
}

/// QCD kernel classification — each maps to a preferred tier ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QcdKernel {
    /// Box-Muller PRNG: exp()/cos()/sin() → TMU candidate
    Prng,
    /// Gauge force / staple chain: SU(3) multiply-accumulate → FP32/Tensor
    GaugeForce,
    /// Dirac operator: stencil across 8 neighbors → TMU stencil + ALU
    DiracOperator,
    /// CG dot product: global reduction → Subgroup + Shared Memory
    CgDotProduct,
    /// CG vector update (axpy): bandwidth-bound → Memory BW
    CgAxpy,
    /// Force accumulation: scatter-add from 8 neighbors → ROP atomics
    ForceAccumulation,
    /// Metropolis ΔH: single scalar comparison → FP64 ALU
    MetropolisTest,
    /// Plaquette / observable: precision accumulation → FP64 ALU
    ObservableAccumulation,
    /// Link / momentum update: SU(3) exponentiation → FP32 ALU (DF64)
    LinkUpdate,
    /// Gradient flow integration: link smearing → FP32 ALU (DF64)
    GradientFlow,
}

impl std::fmt::Display for QcdKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Prng => write!(f, "PRNG (Box-Muller)"),
            Self::GaugeForce => write!(f, "Gauge Force"),
            Self::DiracOperator => write!(f, "Dirac Operator"),
            Self::CgDotProduct => write!(f, "CG Dot Product"),
            Self::CgAxpy => write!(f, "CG axpy"),
            Self::ForceAccumulation => write!(f, "Force Accumulation"),
            Self::MetropolisTest => write!(f, "Metropolis ΔH"),
            Self::ObservableAccumulation => write!(f, "Observable Accum."),
            Self::LinkUpdate => write!(f, "Link Update"),
            Self::GradientFlow => write!(f, "Gradient Flow"),
        }
    }
}

/// Tier routing: preferred silicon ordering for a QCD kernel class.
///
/// The router tries units left-to-right. The first unit with measured
/// throughput > 0 and an available shader path gets the work.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TierRoute {
    pub kernel: QcdKernel,
    pub preferred_units: Vec<SiliconUnit>,
    pub rationale: String,
}

/// Full silicon personality for a single GPU adapter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SiliconProfile {
    /// Adapter name (e.g. "NVIDIA GeForce RTX 3090").
    pub adapter_name: String,
    /// Vendor classification.
    pub vendor: GpuVendorTag,
    /// VRAM in bytes.
    pub vram_bytes: u64,
    /// Boost clock in GHz.
    pub boost_ghz: f64,
    /// Per-unit throughput: theoretical + measured.
    pub units: BTreeMap<SiliconUnit, UnitThroughput>,
    /// Measured composition multipliers (pairs of units running simultaneously).
    pub compositions: Vec<CompositionEntry>,
    /// DF64 TFLOPS (Dekker FP32-pair arithmetic on FP32 ALU).
    pub df64_tflops: f64,
    /// L2 cache size in bytes.
    pub l2_bytes: u64,
    /// AMD Infinity Cache size in bytes (0 for non-AMD).
    pub infinity_cache_bytes: u64,
    /// TMU count (from spec sheet).
    pub tmu_count: u32,
    /// ROP count (from spec sheet).
    pub rop_count: u32,
    /// Subgroup (warp/wavefront) size, 0 if unknown.
    pub subgroup_size: u32,
    /// ISO-8601 timestamp of when this profile was last measured.
    pub measured_at: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuVendorTag {
    Nvidia,
    Amd,
    Intel,
    Software,
    Unknown,
}

impl std::fmt::Display for GpuVendorTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nvidia => write!(f, "NVIDIA"),
            Self::Amd => write!(f, "AMD"),
            Self::Intel => write!(f, "Intel"),
            Self::Software => write!(f, "Software"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ── Tier routing table ──────────────────────────────────────────────

impl SiliconProfile {
    /// The canonical tier routing table for QCD kernels.
    ///
    /// Each kernel lists preferred silicon units from cheapest to most
    /// expensive. The production dispatcher walks this list and picks
    /// the first unit whose shader path is available and whose measured
    /// throughput is nonzero.
    #[must_use]
    pub fn qcd_tier_routes(&self) -> Vec<TierRoute> {
        vec![
            TierRoute {
                kernel: QcdKernel::Prng,
                preferred_units: vec![SiliconUnit::Tmu, SiliconUnit::Fp32Alu],
                rationale: "exp()/cos()/sin() → TMU lookup tables free ALU for physics".into(),
            },
            TierRoute {
                kernel: QcdKernel::GaugeForce,
                preferred_units: vec![
                    SiliconUnit::TensorCore,
                    SiliconUnit::Fp32Alu,
                    SiliconUnit::Fp64Alu,
                ],
                rationale: "SU(3)×SU(3) = 3×3 complex matmul → tensor tile or DF64 FMA".into(),
            },
            TierRoute {
                kernel: QcdKernel::DiracOperator,
                preferred_units: vec![SiliconUnit::Tmu, SiliconUnit::Fp32Alu, SiliconUnit::Fp64Alu],
                rationale:
                    "8-neighbor stencil → TMU texture cache for link loads, ALU for spinor math"
                        .into(),
            },
            TierRoute {
                kernel: QcdKernel::CgDotProduct,
                preferred_units: vec![
                    SiliconUnit::SubgroupIntrinsics,
                    SiliconUnit::SharedMemory,
                    SiliconUnit::Fp32Alu,
                ],
                rationale: "Global reduction → shuffle-reduce (no shared mem), then tree-reduce"
                    .into(),
            },
            TierRoute {
                kernel: QcdKernel::CgAxpy,
                preferred_units: vec![SiliconUnit::MemoryBandwidth, SiliconUnit::Fp32Alu],
                rationale: "Stream a*x+y is pure bandwidth — limited by VRAM controller, not ALU"
                    .into(),
            },
            TierRoute {
                kernel: QcdKernel::ForceAccumulation,
                preferred_units: vec![
                    SiliconUnit::Rop,
                    SiliconUnit::SharedMemory,
                    SiliconUnit::Fp32Alu,
                ],
                rationale: "8 neighbors scatter-add force → ROP atomicAdd (AMD 6× faster)".into(),
            },
            TierRoute {
                kernel: QcdKernel::MetropolisTest,
                preferred_units: vec![SiliconUnit::Fp64Alu, SiliconUnit::Fp32Alu],
                rationale: "Single scalar ΔH comparison — needs full precision, trivial cost"
                    .into(),
            },
            TierRoute {
                kernel: QcdKernel::ObservableAccumulation,
                preferred_units: vec![SiliconUnit::Fp64Alu, SiliconUnit::Fp32Alu],
                rationale: "Plaquette/Polyakov accumulation — precision-critical, low FLOP".into(),
            },
            TierRoute {
                kernel: QcdKernel::LinkUpdate,
                preferred_units: vec![SiliconUnit::Fp32Alu, SiliconUnit::TensorCore],
                rationale: "SU(3) exp(iH·dt) via Cayley-Hamilton — FMA-heavy, DF64 is natural"
                    .into(),
            },
            TierRoute {
                kernel: QcdKernel::GradientFlow,
                preferred_units: vec![SiliconUnit::Fp32Alu, SiliconUnit::Tmu, SiliconUnit::Fp64Alu],
                rationale: "Link smearing with stencil access — same pattern as force + stencil"
                    .into(),
            },
        ]
    }

    /// Look up measured throughput for a unit. Returns 0.0 if not yet measured.
    #[must_use]
    pub fn measured(&self, unit: SiliconUnit) -> f64 {
        self.units.get(&unit).map_or(0.0, |u| u.measured_peak)
    }

    /// Look up theoretical peak for a unit. Returns 0.0 if not specified.
    #[must_use]
    pub fn theoretical(&self, unit: SiliconUnit) -> f64 {
        self.units.get(&unit).map_or(0.0, |u| u.theoretical_peak)
    }

    /// Efficiency ratio for a unit (measured / theoretical).
    #[must_use]
    pub fn efficiency(&self, unit: SiliconUnit) -> f64 {
        self.units.get(&unit).map_or(0.0, |u| u.efficiency)
    }

    /// Best composition multiplier involving a given unit.
    #[must_use]
    pub fn best_composition_with(&self, unit: SiliconUnit) -> Option<&CompositionEntry> {
        self.compositions
            .iter()
            .filter(|c| c.unit_a == unit || c.unit_b == unit)
            .max_by(|a, b| {
                a.multiplier
                    .partial_cmp(&b.multiplier)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Total compound throughput estimate: sum of all measured peaks weighted
    /// by the best-known composition multiplier. This is the theoretical
    /// ceiling if every unit runs on a different sub-problem simultaneously.
    #[must_use]
    pub fn compound_ceiling_tflops(&self) -> f64 {
        let fp32 = self.measured(SiliconUnit::Fp32Alu);
        let tmu = self.measured(SiliconUnit::Tmu);
        let tensor = self.measured(SiliconUnit::TensorCore);
        let rop = self.measured(SiliconUnit::Rop);
        fp32 + tmu + tensor + rop
    }

    /// Select the best available silicon unit for a QCD kernel based on
    /// measured data. Falls back through the tier list.
    #[must_use]
    pub fn route_kernel(&self, kernel: QcdKernel) -> SiliconUnit {
        let routes = self.qcd_tier_routes();
        let route = routes.iter().find(|r| r.kernel == kernel);
        if let Some(r) = route {
            for unit in &r.preferred_units {
                if self.measured(*unit) > 0.0 {
                    return *unit;
                }
            }
            *r.preferred_units.last().unwrap_or(&SiliconUnit::Fp32Alu)
        } else {
            SiliconUnit::Fp32Alu
        }
    }

    /// Pretty-print the full profile.
    pub fn print_summary(&self) {
        let has_energy = self.units.values().any(|u| u.delta_watts > 0.1);

        println!(
            "╔════════════════════════════════════════════════════════════════════════════════╗"
        );
        println!("║  Silicon Profile: {:<60} ║", self.adapter_name);
        println!(
            "║  Vendor: {:<6}  VRAM: {:>5} MB  Clock: {:.2} GHz{:<30} ║",
            self.vendor,
            self.vram_bytes / (1024 * 1024),
            self.boost_ghz,
            "",
        );
        println!(
            "╠════════════════════════════════════════════════════════════════════════════════╣"
        );

        if has_energy {
            println!(
                "║  {:<14} {:>10} {:>10} {:>7}  {:<10} {:>7} {:>12}      ║",
                "Unit", "Theoretic", "Measured", "Eff%", "Units", "ΔW", "Ops/W"
            );
            println!(
                "╠════════════════════════════════════════════════════════════════════════════════╣"
            );
            for (unit, t) in &self.units {
                let eff_pct = t.efficiency * 100.0;
                let dw = if t.delta_watts > 0.1 {
                    format!("{:.0}W", t.delta_watts)
                } else {
                    "—".into()
                };
                let opw = if t.ops_per_watt > 0.0 {
                    format!("{:.1}", t.ops_per_watt)
                } else {
                    "—".into()
                };
                println!(
                    "║  {:<14} {:>10.2} {:>10.2} {:>6.1}%  {:<10} {:>7} {:>12}      ║",
                    unit, t.theoretical_peak, t.measured_peak, eff_pct, t.unit, dw, opw,
                );
            }
        } else {
            println!(
                "║  {:<14} {:>10} {:>10} {:>7}  {:<10}                          ║",
                "Unit", "Theoretic", "Measured", "Eff%", "Units"
            );
            println!(
                "╠════════════════════════════════════════════════════════════════════════════════╣"
            );
            for (unit, t) in &self.units {
                let eff_pct = t.efficiency * 100.0;
                println!(
                    "║  {:<14} {:>10.2} {:>10.2} {:>6.1}%  {:<10}                          ║",
                    unit, t.theoretical_peak, t.measured_peak, eff_pct, t.unit,
                );
            }
        }

        if !self.compositions.is_empty() {
            println!(
                "╠════════════════════════════════════════════════════════════════════════════════╣"
            );
            println!(
                "║  Composition Multipliers                                                     ║"
            );
            for c in &self.compositions {
                let energy_note = if c.delta_watts > 0.1 {
                    format!("  ΔW={:.0}W", c.delta_watts)
                } else {
                    String::new()
                };
                println!(
                    "║    {} + {} → {:.2}x  ({:.1}ms serial, {:.1}ms compound){:<18} ║",
                    c.unit_a, c.unit_b, c.multiplier, c.serial_ms, c.compound_ms, energy_note,
                );
            }
        }

        println!(
            "╠════════════════════════════════════════════════════════════════════════════════╣"
        );
        println!(
            "║  Tier Routing Table                                                          ║"
        );
        for route in self.qcd_tier_routes() {
            let chosen = self.route_kernel(route.kernel);
            let units_str: Vec<String> = route
                .preferred_units
                .iter()
                .map(|u| {
                    if *u == chosen {
                        format!("[{u}]")
                    } else {
                        format!("{u}")
                    }
                })
                .collect();
            println!(
                "║  {:<22} → {:<53} ║",
                format!("{}", route.kernel),
                units_str.join(" → "),
            );
        }

        println!(
            "╠════════════════════════════════════════════════════════════════════════════════╣"
        );
        println!(
            "║  Compound ceiling: {:.2} TFLOPS (all measured units parallel){:<18} ║",
            self.compound_ceiling_tflops(),
            ""
        );
        println!("║  Measured at: {:<64} ║", self.measured_at);
        println!(
            "╚════════════════════════════════════════════════════════════════════════════════╝"
        );
    }
}

// Re-export spec-sheet helpers from `silicon_spec_sheets` for a stable `bench::silicon_profile` API.
pub use super::silicon_spec_sheets::{classify_vendor, from_spec_sheet};

// ── Persistence ─────────────────────────────────────────────────────

/// Default directory for saved profiles: `$WORKSPACE/profiles/silicon/`
fn default_profile_dir() -> PathBuf {
    let workspace = std::env::var("HOTSPRING_ROOT").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(workspace).join("profiles").join("silicon")
}

/// Sanitize adapter name into a filesystem-safe filename.
fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .to_lowercase()
}

impl SiliconProfile {
    /// Save this profile to disk as JSON.
    ///
    /// # Errors
    /// Returns IO error if the directory cannot be created or the file cannot be written.
    pub fn save(&self, dir: Option<&Path>) -> std::io::Result<PathBuf> {
        let dir = dir.map_or_else(default_profile_dir, Path::to_path_buf);
        std::fs::create_dir_all(&dir)?;
        let filename = format!("{}.json", sanitize_name(&self.adapter_name));
        let path = dir.join(filename);
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(&path, json)?;
        Ok(path)
    }

    /// Load a profile from a JSON file.
    ///
    /// # Errors
    /// Returns IO error if the file cannot be read or parsed.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(std::io::Error::other)
    }

    /// Try to find a saved profile for this adapter name.
    #[must_use]
    pub fn find_saved(adapter_name: &str, dir: Option<&Path>) -> Option<PathBuf> {
        let dir = dir.map_or_else(default_profile_dir, Path::to_path_buf);
        let filename = format!("{}.json", sanitize_name(adapter_name));
        let path = dir.join(filename);
        if path.exists() { Some(path) } else { None }
    }

    /// Update the measured throughput for a unit and recompute efficiency.
    pub fn set_measured(&mut self, unit: SiliconUnit, measured: f64) {
        if let Some(entry) = self.units.get_mut(&unit) {
            entry.measured_peak = measured;
            entry.efficiency = if entry.theoretical_peak > 0.0 {
                measured / entry.theoretical_peak
            } else {
                0.0
            };
        }
    }

    /// Record idle/loaded GPU power for a unit and compute energy efficiency.
    ///
    /// `idle_w` is the total GPU board power at rest (no dispatches).
    /// `loaded_w` is the average GPU power during a saturation benchmark
    /// that exercises only this unit. The difference is the marginal energy
    /// cost, and `ops_per_watt = measured_peak / delta_watts`.
    pub fn set_measured_energy(&mut self, unit: SiliconUnit, idle_w: f64, loaded_w: f64) {
        if let Some(entry) = self.units.get_mut(&unit) {
            entry.idle_watts = idle_w;
            entry.loaded_watts = loaded_w;
            entry.delta_watts = (loaded_w - idle_w).max(0.0);
            entry.ops_per_watt = if entry.delta_watts > 0.1 {
                entry.measured_peak / entry.delta_watts
            } else {
                0.0
            };
        }
    }

    /// Record a composition measurement (timing only, no energy data).
    pub fn add_composition(
        &mut self,
        unit_a: SiliconUnit,
        unit_b: SiliconUnit,
        serial_ms: f64,
        compound_ms: f64,
    ) {
        let multiplier = if compound_ms > 0.0 {
            serial_ms / compound_ms
        } else {
            0.0
        };
        self.compositions.push(CompositionEntry {
            unit_a,
            unit_b,
            serial_ms,
            compound_ms,
            multiplier,
            idle_watts: 0.0,
            compound_watts: 0.0,
            delta_watts: 0.0,
        });
    }

    /// Record a composition measurement with energy data.
    pub fn add_composition_with_energy(
        &mut self,
        unit_a: SiliconUnit,
        unit_b: SiliconUnit,
        serial_ms: f64,
        compound_ms: f64,
        idle_w: f64,
        compound_w: f64,
    ) {
        let multiplier = if compound_ms > 0.0 {
            serial_ms / compound_ms
        } else {
            0.0
        };
        let delta = (compound_w - idle_w).max(0.0);
        self.compositions.push(CompositionEntry {
            unit_a,
            unit_b,
            serial_ms,
            compound_ms,
            multiplier,
            idle_watts: idle_w,
            compound_watts: compound_w,
            delta_watts: delta,
        });
    }

    /// Stamp the current time.
    pub fn stamp_now(&mut self) {
        use std::time::SystemTime;
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.measured_at = format!("{now}");
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

    use super::*;

    #[test]
    fn spec_sheet_3090_populated() {
        let p = from_spec_sheet("NVIDIA GeForce RTX 3090", 0x10DE);
        assert_eq!(p.vendor, GpuVendorTag::Nvidia);
        assert!(p.theoretical(SiliconUnit::Fp32Alu) > 30.0);
        assert!(p.theoretical(SiliconUnit::Tmu) > 500.0);
        assert!(p.theoretical(SiliconUnit::TensorCore) > 60.0);
        assert_eq!(p.measured(SiliconUnit::Fp32Alu), 0.0);
        assert_eq!(p.tmu_count, 328);
        assert_eq!(p.rop_count, 112);
    }

    #[test]
    fn spec_sheet_6950_populated() {
        let p = from_spec_sheet("AMD Radeon RX 6950 XT", 0x1002);
        assert_eq!(p.vendor, GpuVendorTag::Amd);
        assert!(p.theoretical(SiliconUnit::Fp32Alu) > 20.0);
        assert!(p.infinity_cache_bytes > 0);
        assert_eq!(p.theoretical(SiliconUnit::TensorCore), 0.0);
    }

    #[test]
    fn route_kernel_fallback_to_last_tier() {
        let p = from_spec_sheet("unknown card", 0x0000);
        // No measured data → falls through to last preferred unit in the tier list.
        // GaugeForce: [TensorCore, Fp32Alu, Fp64Alu] → Fp64Alu
        assert_eq!(p.route_kernel(QcdKernel::GaugeForce), SiliconUnit::Fp64Alu);
        // CgAxpy: [MemoryBandwidth, Fp32Alu] → Fp32Alu
        assert_eq!(p.route_kernel(QcdKernel::CgAxpy), SiliconUnit::Fp32Alu);
    }

    #[test]
    fn set_measured_updates_efficiency() {
        let mut p = from_spec_sheet("NVIDIA GeForce RTX 3090", 0x10DE);
        p.set_measured(SiliconUnit::Fp32Alu, 28.0);
        let eff = p.efficiency(SiliconUnit::Fp32Alu);
        assert!(eff > 0.7 && eff < 0.9, "efficiency should be ~78%: {eff}");
    }

    #[test]
    fn composition_entry() {
        let mut p = from_spec_sheet("NVIDIA GeForce RTX 3090", 0x10DE);
        p.add_composition(SiliconUnit::Fp32Alu, SiliconUnit::Tmu, 10.0, 6.0);
        assert_eq!(p.compositions.len(), 1);
        let c = &p.compositions[0];
        assert!((c.multiplier - 10.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn serde_round_trip() {
        let p = from_spec_sheet("NVIDIA GeForce RTX 3090", 0x10DE);
        let json = serde_json::to_string(&p).expect("serialize");
        let back: SiliconProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.adapter_name, p.adapter_name);
        assert_eq!(back.units.len(), p.units.len());
    }

    #[test]
    fn serde_backward_compat_no_energy_fields() {
        let json = r#"{
            "theoretical_peak": 35.6,
            "measured_peak": 8.5,
            "efficiency": 0.24,
            "unit": "TFLOPS"
        }"#;
        let t: UnitThroughput = serde_json::from_str(json).expect("old JSON without energy");
        assert_eq!(t.idle_watts, 0.0);
        assert_eq!(t.loaded_watts, 0.0);
        assert_eq!(t.delta_watts, 0.0);
        assert_eq!(t.ops_per_watt, 0.0);
        assert!((t.measured_peak - 8.5).abs() < 1e-6);
    }

    #[test]
    fn set_measured_energy_computes_ops_per_watt() {
        let mut p = from_spec_sheet("NVIDIA GeForce RTX 3090", 0x10DE);
        p.set_measured(SiliconUnit::Fp32Alu, 8.5);
        p.set_measured_energy(SiliconUnit::Fp32Alu, 25.0, 145.0);
        let entry = p.units.get(&SiliconUnit::Fp32Alu).unwrap();
        assert!((entry.idle_watts - 25.0).abs() < 1e-6);
        assert!((entry.loaded_watts - 145.0).abs() < 1e-6);
        assert!((entry.delta_watts - 120.0).abs() < 1e-6);
        let expected_opw = 8.5 / 120.0;
        assert!((entry.ops_per_watt - expected_opw).abs() < 1e-6);
    }

    #[test]
    fn set_measured_energy_clamps_negative_delta() {
        let mut p = from_spec_sheet("NVIDIA GeForce RTX 3090", 0x10DE);
        p.set_measured(SiliconUnit::Tmu, 364.0);
        p.set_measured_energy(SiliconUnit::Tmu, 30.0, 25.0);
        let entry = p.units.get(&SiliconUnit::Tmu).unwrap();
        assert_eq!(entry.delta_watts, 0.0);
        assert_eq!(entry.ops_per_watt, 0.0);
    }

    #[test]
    fn sanitize_name_safe() {
        assert_eq!(
            sanitize_name("NVIDIA GeForce RTX 3090"),
            "nvidia_geforce_rtx_3090"
        );
        assert_eq!(
            sanitize_name("AMD Radeon RX 6950 XT"),
            "amd_radeon_rx_6950_xt"
        );
    }

    #[test]
    fn tier_routes_cover_all_kernels() {
        let p = from_spec_sheet("NVIDIA GeForce RTX 3090", 0x10DE);
        let routes = p.qcd_tier_routes();
        let kernels: Vec<QcdKernel> = routes.iter().map(|r| r.kernel).collect();
        assert!(kernels.contains(&QcdKernel::Prng));
        assert!(kernels.contains(&QcdKernel::GaugeForce));
        assert!(kernels.contains(&QcdKernel::DiracOperator));
        assert!(kernels.contains(&QcdKernel::CgDotProduct));
        assert!(kernels.contains(&QcdKernel::CgAxpy));
        assert!(kernels.contains(&QcdKernel::ForceAccumulation));
        assert!(kernels.contains(&QcdKernel::MetropolisTest));
        assert!(kernels.contains(&QcdKernel::ObservableAccumulation));
        assert!(kernels.contains(&QcdKernel::LinkUpdate));
        assert!(kernels.contains(&QcdKernel::GradientFlow));
    }
}
