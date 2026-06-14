// SPDX-License-Identifier: AGPL-3.0-or-later

use std::path::Path;

use super::registers::{FECS_DMEM_FW_OFFSET, GPCCS_DMEM_FW_OFFSET};

pub fn extract_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

pub fn load_fw_bytes(dir: &Path, name: &str) -> Vec<u8> {
    let path = dir.join(name);
    std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("FATAL: cannot read {}: {e}", path.display());
        std::process::exit(1);
    })
}

pub fn load_fw_words(dir: &Path, name: &str) -> Vec<u32> {
    let bytes = load_fw_bytes(dir, name);
    bytes
        .chunks(4)
        .map(|c| {
            u32::from_le_bytes([
                *c.first().unwrap_or(&0),
                *c.get(1).unwrap_or(&0),
                *c.get(2).unwrap_or(&0),
                *c.get(3).unwrap_or(&0),
            ])
        })
        .collect()
}

pub fn load_mmio_init(dir: &Path, name: &str) -> Vec<(u32, u32)> {
    let bytes = load_fw_bytes(dir, name);
    bytes
        .chunks(8)
        .filter(|c| c.len() == 8)
        .map(|c| {
            let addr = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let val = u32::from_le_bytes([c[4], c[5], c[6], c[7]]);
            (addr, val)
        })
        .collect()
}

/// Verify CSDATA blobs fit below the firmware DMEM offsets; exit on overflow.
pub fn validate_csdata_layout(hub_csdata: &[u32], gpccs_csdata: &[u32]) {
    let hub_bytes = (hub_csdata.len() as u32) * 4;
    let gpccs_bytes = (gpccs_csdata.len() as u32) * 4;
    if hub_bytes > FECS_DMEM_FW_OFFSET {
        eprintln!("FATAL: hub_csdata ({hub_bytes}B) overflows FECS DMEM fw offset");
        std::process::exit(1);
    }
    if gpccs_bytes > GPCCS_DMEM_FW_OFFSET {
        eprintln!("FATAL: gpccs_csdata ({gpccs_bytes}B) overflows GPCCS DMEM fw offset");
        std::process::exit(1);
    }
    println!(
        "    Layout OK: hub ends {hub_bytes:#06x} < fw@{FECS_DMEM_FW_OFFSET:#06x}  \
              gpccs ends {gpccs_bytes:#06x} < fw@{GPCCS_DMEM_FW_OFFSET:#06x}  ✓"
    );
}
