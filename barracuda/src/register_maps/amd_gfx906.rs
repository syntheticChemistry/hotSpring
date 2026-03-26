// SPDX-License-Identifier: AGPL-3.0-only

use super::{RegDef, RegisterMap};

pub struct AmdGfx906Map;

const REGISTERS: &[RegDef] = &[
    // SMC — indirect thermal / power access
    RegDef {
        offset: 0x0000_01AC,
        name: "SMC_IND_INDEX_11",
        group: "SMC",
    },
    RegDef {
        offset: 0x0000_01B0,
        name: "SMC_IND_DATA_11",
        group: "SMC",
    },
    RegDef {
        offset: 0x0000_0394,
        name: "SRBM_SOFT_RESET",
        group: "SRBM",
    },
    // MMHUB
    RegDef {
        offset: 0x0000_0600,
        name: "MMHUB_VM_BASE",
        group: "MMHUB",
    },
    RegDef {
        offset: 0x0000_06A0,
        name: "MMMC_VM_FB_OFFSET",
        group: "MMHUB",
    },
    // SRBM (status)
    RegDef {
        offset: 0x0000_0E50,
        name: "SRBM_STATUS",
        group: "SRBM",
    },
    RegDef {
        offset: 0x0000_0E54,
        name: "SRBM_STATUS2",
        group: "SRBM",
    },
    // UMC / MC
    RegDef {
        offset: 0x0000_2023,
        name: "MC_VM_FB_LOCATION_BASE",
        group: "MC",
    },
    RegDef {
        offset: 0x0000_2024,
        name: "MC_VM_FB_LOCATION_TOP",
        group: "MC",
    },
    RegDef {
        offset: 0x0000_2025,
        name: "MC_VM_AGP_TOP",
        group: "MC",
    },
    RegDef {
        offset: 0x0000_2026,
        name: "MC_VM_AGP_BOT",
        group: "MC",
    },
    RegDef {
        offset: 0x0000_2027,
        name: "MC_VM_AGP_BASE",
        group: "MC",
    },
    // GC — address / backend config
    RegDef {
        offset: 0x0000_263E,
        name: "GB_ADDR_CONFIG",
        group: "GC",
    },
    RegDef {
        offset: 0x0000_2640,
        name: "GB_BACKEND_MAP",
        group: "GC",
    },
    // SDMA
    RegDef {
        offset: 0x0000_4D00,
        name: "SDMA0_STATUS_REG",
        group: "SDMA0",
    },
    RegDef {
        offset: 0x0000_5900,
        name: "SDMA1_STATUS_REG",
        group: "SDMA1",
    },
    // GRBM
    RegDef {
        offset: 0x0000_8008,
        name: "GRBM_STATUS2",
        group: "GRBM",
    },
    RegDef {
        offset: 0x0000_8010,
        name: "GRBM_STATUS",
        group: "GRBM",
    },
    RegDef {
        offset: 0x0000_8020,
        name: "GRBM_SOFT_RESET",
        group: "GRBM",
    },
    RegDef {
        offset: 0x0000_8090,
        name: "GRBM_STATUS_SE0",
        group: "GRBM",
    },
    RegDef {
        offset: 0x0000_8094,
        name: "GRBM_STATUS_SE1",
        group: "GRBM",
    },
    RegDef {
        offset: 0x0000_8098,
        name: "GRBM_STATUS_SE2",
        group: "GRBM",
    },
    RegDef {
        offset: 0x0000_809C,
        name: "GRBM_STATUS_SE3",
        group: "GRBM",
    },
    // CP
    RegDef {
        offset: 0x0000_8680,
        name: "CP_STAT",
        group: "CP",
    },
    RegDef {
        offset: 0x0000_8684,
        name: "CP_ME_CNTL",
        group: "CP",
    },
    RegDef {
        offset: 0x0000_86C4,
        name: "CP_RB_BUFSZ",
        group: "CP",
    },
    // RLC
    RegDef {
        offset: 0x0000_EC10,
        name: "RLC_CNTL",
        group: "RLC",
    },
    RegDef {
        offset: 0x0000_EC14,
        name: "RLC_STAT",
        group: "RLC",
    },
    RegDef {
        offset: 0x0000_EC18,
        name: "RLC_GPM_STAT",
        group: "RLC",
    },
];

impl RegisterMap for AmdGfx906Map {
    fn vendor(&self) -> &'static str {
        "amd"
    }

    fn arch(&self) -> &'static str {
        "GFX906"
    }

    fn registers(&self) -> &[RegDef] {
        REGISTERS
    }

    fn decode_temp_c(&self, raw: u32) -> Option<u32> {
        Some(raw & 0x1FF)
    }

    fn decode_boot_id(&self, raw: u32) -> String {
        format!("GFX{:X}", (raw >> 16) & 0xFFF)
    }

    fn thermal_offset(&self) -> Option<u32> {
        Some(0xC0300E04)
    }
}
