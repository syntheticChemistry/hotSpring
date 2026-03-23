// SPDX-License-Identifier: AGPL-3.0-only
#![allow(missing_docs)]
//! Vendor-agnostic register map abstraction for GPU reverse engineering.
//!
//! Each GPU architecture provides a `RegisterMap` implementation that
//! describes its BAR0 MMIO register layout. The dump/diff binaries use
//! this trait to work across NVIDIA, AMD, and future vendors without
//! code duplication.

mod amd_gfx906;
mod nv_gv100;

pub use amd_gfx906::AmdGfx906Map;
pub use nv_gv100::NvGv100Map;

/// A single named register definition.
#[derive(Debug, Clone)]
pub struct RegDef {
    pub offset: u32,
    pub name: &'static str,
    pub group: &'static str,
}

/// Vendor-agnostic register map trait.
///
/// Each GPU architecture implements this to describe its BAR0 register
/// layout, thermal decoding, and boot identification.
pub trait RegisterMap {
    /// GPU vendor name (e.g. `"nvidia"`, `"amd"`).
    fn vendor(&self) -> &str;

    /// Architecture name (e.g. `"GV100"`, `"GFX906"`).
    fn arch(&self) -> &str;

    /// Ordered list of registers to dump.
    fn registers(&self) -> &[RegDef];

    /// Decode a raw thermal register value to degrees Celsius.
    fn decode_temp_c(&self, raw: u32) -> Option<u32>;

    /// Decode the boot/identity register to a human-readable string.
    fn decode_boot_id(&self, raw: u32) -> String;

    /// Offset of the thermal register (for automatic temp decoding in dumps).
    fn thermal_offset(&self) -> Option<u32>;
}

/// Detect the appropriate register map from PCI vendor ID.
///
/// Falls back to NVIDIA GV100 for `0x10de`, AMD GFX906 for `0x1002`.
/// Returns `None` for unknown vendors.
pub fn detect_register_map(vendor_id: u16) -> Option<Box<dyn RegisterMap>> {
    match vendor_id {
        0x10de => Some(Box::new(NvGv100Map)),
        0x1002 => Some(Box::new(AmdGfx906Map)),
        _ => None,
    }
}

/// Unified JSON schema for register dumps across vendors.
///
/// Used by the dump binary to produce a single JSON format regardless
/// of GPU vendor.
#[derive(serde::Serialize)]
pub struct RegisterDump {
    pub vendor: String,
    pub arch: String,
    pub pci_id: String,
    pub bdf: String,
    pub timestamp: String,
    pub registers: Vec<RegisterEntry>,
}

/// A single register entry in a dump.
#[derive(serde::Serialize)]
pub struct RegisterEntry {
    pub offset: String,
    pub name: String,
    pub group: String,
    pub value: String,
    pub raw_offset: u32,
    pub raw_value: u32,
}
