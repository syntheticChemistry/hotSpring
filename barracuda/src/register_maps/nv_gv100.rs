// SPDX-License-Identifier: AGPL-3.0-or-later

use super::{RegDef, RegisterMap};

/// BAR0 register map for NVIDIA GV100 (Titan V / V100).
pub struct NvGv100Map;

const REGISTERS: &[RegDef] = &[
    // PMC
    RegDef {
        offset: 0x000000,
        name: "BOOT0",
        group: "PMC",
    },
    RegDef {
        offset: 0x000004,
        name: "BOOT1",
        group: "PMC",
    },
    RegDef {
        offset: 0x000200,
        name: "PMC_ENABLE",
        group: "PMC",
    },
    RegDef {
        offset: 0x000204,
        name: "PMC_DEVICE_ENABLE",
        group: "PMC",
    },
    // PBUS
    RegDef {
        offset: 0x001C00,
        name: "PBUS_EXT_CG",
        group: "PBUS",
    },
    RegDef {
        offset: 0x001C04,
        name: "PBUS_EXT_CG1",
        group: "PBUS",
    },
    // PFIFO
    RegDef {
        offset: 0x002004,
        name: "PFIFO_PBDMA_MAP",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002100,
        name: "PFIFO_INTR",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002140,
        name: "PFIFO_INTR_EN",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002200,
        name: "PFIFO_ENABLE",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002254,
        name: "PFIFO_FB_TIMEOUT",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002270,
        name: "RUNLIST_BASE",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002274,
        name: "RUNLIST_SUBMIT",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002280,
        name: "RUNLIST0_BASE",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002284,
        name: "RUNLIST0_INFO",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x00228C,
        name: "RUNLIST1_INFO",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002390,
        name: "PBDMA_RUNL_MAP_0",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002394,
        name: "PBDMA_RUNL_MAP_1",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002398,
        name: "PBDMA_RUNL_MAP_2",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x00239C,
        name: "PBDMA_RUNL_MAP_3",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002504,
        name: "SCHED_EN",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002508,
        name: "SCHED_STATUS",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x00252C,
        name: "BIND_ERROR",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002630,
        name: "SCHED_DISABLE",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002634,
        name: "PREEMPT",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002638,
        name: "PREEMPT_PENDING",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002640,
        name: "ENGN0_STATUS",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002A00,
        name: "RUNLIST_ACK",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002A04,
        name: "PBDMA_INTR_EN",
        group: "PFIFO",
    },
    // PBDMA idle
    RegDef {
        offset: 0x003080,
        name: "PBDMA0_IDLE",
        group: "PBDMA_IDLE",
    },
    RegDef {
        offset: 0x003084,
        name: "PBDMA1_IDLE",
        group: "PBDMA_IDLE",
    },
    RegDef {
        offset: 0x003088,
        name: "PBDMA2_IDLE",
        group: "PBDMA_IDLE",
    },
    RegDef {
        offset: 0x00308C,
        name: "PBDMA3_IDLE",
        group: "PBDMA_IDLE",
    },
    // PBDMA0
    RegDef {
        offset: 0x040040,
        name: "PBDMA0_GP_BASE_LO",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x040044,
        name: "PBDMA0_GP_BASE_HI",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x040048,
        name: "PBDMA0_GP_FETCH",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x04004C,
        name: "PBDMA0_GP_STATE",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x040050,
        name: "PBDMA0_GP_PUT_HI",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x040054,
        name: "PBDMA0_GP_PUT_LO",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x0400A8,
        name: "PBDMA0_TARGET",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x0400AC,
        name: "PBDMA0_SET_CHANNEL_INFO",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x0400B0,
        name: "PBDMA0_CHANNEL_STATE",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x0400C0,
        name: "PBDMA0_SIGNATURE",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x0400D0,
        name: "PBDMA0_USERD_LO",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x0400D4,
        name: "PBDMA0_USERD_HI",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x040108,
        name: "PBDMA0_INTR",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x04010C,
        name: "PBDMA0_INTR_EN",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x040148,
        name: "PBDMA0_HCE",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x04014C,
        name: "PBDMA0_HCE_EN",
        group: "PBDMA0",
    },
    // PBDMA2
    RegDef {
        offset: 0x044040,
        name: "PBDMA2_GP_BASE_LO",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x044044,
        name: "PBDMA2_GP_BASE_HI",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x044048,
        name: "PBDMA2_GP_FETCH",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x04404C,
        name: "PBDMA2_GP_STATE",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x044054,
        name: "PBDMA2_GP_PUT_LO",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x0440A8,
        name: "PBDMA2_TARGET",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x0440AC,
        name: "PBDMA2_SET_CHANNEL_INFO",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x0440B0,
        name: "PBDMA2_CHANNEL_STATE",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x0440C0,
        name: "PBDMA2_SIGNATURE",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x0440D0,
        name: "PBDMA2_USERD_LO",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x0440D4,
        name: "PBDMA2_USERD_HI",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x044108,
        name: "PBDMA2_INTR",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x04410C,
        name: "PBDMA2_INTR_EN",
        group: "PBDMA2",
    },
    // PRIV ring
    RegDef {
        offset: 0x012070,
        name: "PRIV_RING_INTR",
        group: "PRIV",
    },
    // Thermal
    RegDef {
        offset: 0x020460,
        name: "GPU_TEMP",
        group: "THERM",
    },
    // PTOP
    RegDef {
        offset: 0x022700,
        name: "PTOP_INFO_0",
        group: "PTOP",
    },
    RegDef {
        offset: 0x022704,
        name: "PTOP_INFO_1",
        group: "PTOP",
    },
    RegDef {
        offset: 0x022708,
        name: "PTOP_INFO_2",
        group: "PTOP",
    },
    RegDef {
        offset: 0x02270C,
        name: "PTOP_INFO_3",
        group: "PTOP",
    },
    // SEC2 falcon
    RegDef {
        offset: 0x087100,
        name: "SEC2_CPUCTL",
        group: "SEC2",
    },
    RegDef {
        offset: 0x087104,
        name: "SEC2_BOOTVEC",
        group: "SEC2",
    },
    RegDef {
        offset: 0x087240,
        name: "SEC2_SCTL",
        group: "SEC2",
    },
    // FECS falcon
    RegDef {
        offset: 0x409100,
        name: "FECS_CPUCTL",
        group: "FECS",
    },
    RegDef {
        offset: 0x409104,
        name: "FECS_BOOTVEC",
        group: "FECS",
    },
    RegDef {
        offset: 0x409110,
        name: "FECS_PC",
        group: "FECS",
    },
    RegDef {
        offset: 0x409240,
        name: "FECS_SCTL",
        group: "FECS",
    },
    // PMU falcon
    RegDef {
        offset: 0x10A100,
        name: "PMU_CPUCTL",
        group: "PMU",
    },
    RegDef {
        offset: 0x10A104,
        name: "PMU_BOOTVEC",
        group: "PMU",
    },
    RegDef {
        offset: 0x10A240,
        name: "PMU_SCTL",
        group: "PMU",
    },
    // MMU
    RegDef {
        offset: 0x100A2C,
        name: "MMU_FAULT_STATUS",
        group: "MMU",
    },
    RegDef {
        offset: 0x100A30,
        name: "MMU_FAULT_ADDR_LO",
        group: "MMU",
    },
    RegDef {
        offset: 0x100A34,
        name: "MMU_FAULT_ADDR_HI",
        group: "MMU",
    },
    RegDef {
        offset: 0x100A38,
        name: "MMU_FAULT_INST_LO",
        group: "MMU",
    },
    RegDef {
        offset: 0x100A3C,
        name: "MMU_FAULT_INST_HI",
        group: "MMU",
    },
    RegDef {
        offset: 0x100A40,
        name: "MMU_FAULT_INFO",
        group: "MMU",
    },
    RegDef {
        offset: 0x100C80,
        name: "MMU_PRI_CTRL",
        group: "MMU",
    },
    RegDef {
        offset: 0x100CBC,
        name: "MMU_TLB_FLUSH",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E24,
        name: "MMU_FAULT_BUF0_LO",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E28,
        name: "MMU_FAULT_BUF0_HI",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E2C,
        name: "MMU_FAULT_BUF0_SIZE",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E30,
        name: "MMU_FAULT_BUF0_GET",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E34,
        name: "MMU_FAULT_BUF0_PUT",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E44,
        name: "MMU_FAULT_BUF1_LO",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E48,
        name: "MMU_FAULT_BUF1_HI",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E4C,
        name: "MMU_FAULT_BUF1_SIZE",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E50,
        name: "MMU_FAULT_BUF1_GET",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E54,
        name: "MMU_FAULT_BUF1_PUT",
        group: "MMU",
    },
    // BAR
    RegDef {
        offset: 0x001704,
        name: "BAR1_BLOCK",
        group: "BAR",
    },
    RegDef {
        offset: 0x001710,
        name: "BIND_STATUS",
        group: "BAR",
    },
    RegDef {
        offset: 0x001714,
        name: "BAR2_BLOCK",
        group: "BAR",
    },
    // PFB/FBHUB
    RegDef {
        offset: 0x100000,
        name: "PFB_BOOT0",
        group: "PFB",
    },
    RegDef {
        offset: 0x100200,
        name: "PFB_CFG",
        group: "PFB",
    },
    RegDef {
        offset: 0x100204,
        name: "PFB_SIZE",
        group: "PFB",
    },
    RegDef {
        offset: 0x100800,
        name: "FBHUB_0",
        group: "FBHUB",
    },
    RegDef {
        offset: 0x100804,
        name: "FBHUB_4",
        group: "FBHUB",
    },
    // PMU base registers
    RegDef {
        offset: 0x10A000,
        name: "PMU_FALCON_HWCFG",
        group: "PMU",
    },
    RegDef {
        offset: 0x10A040,
        name: "PMU_FALCON_DMACTL",
        group: "PMU",
    },
    RegDef {
        offset: 0x10A044,
        name: "PMU_FALCON_DMATRFBASE",
        group: "PMU",
    },
    // PCLOCK
    RegDef {
        offset: 0x137000,
        name: "PCLOCK_BASE",
        group: "PCLOCK",
    },
    RegDef {
        offset: 0x137050,
        name: "NVPLL",
        group: "PCLOCK",
    },
    RegDef {
        offset: 0x137100,
        name: "MEMPLL",
        group: "PCLOCK",
    },
    // PRAMIN
    RegDef {
        offset: 0x700000,
        name: "NV_PRAMIN_0",
        group: "PRAMIN",
    },
    RegDef {
        offset: 0x700004,
        name: "NV_PRAMIN_4",
        group: "PRAMIN",
    },
    // PROM
    RegDef {
        offset: 0x300000,
        name: "PROM_0",
        group: "PROM",
    },
    RegDef {
        offset: 0x300004,
        name: "PROM_4",
        group: "PROM",
    },
    // PCCSR
    RegDef {
        offset: 0x800000,
        name: "PCCSR_INST_0",
        group: "PCCSR",
    },
    RegDef {
        offset: 0x800004,
        name: "PCCSR_CHAN_0",
        group: "PCCSR",
    },
    RegDef {
        offset: 0x800008,
        name: "PCCSR_INST_1",
        group: "PCCSR",
    },
    RegDef {
        offset: 0x80000C,
        name: "PCCSR_CHAN_1",
        group: "PCCSR",
    },
    // USERMODE
    RegDef {
        offset: 0x810000,
        name: "USERMODE_CFG",
        group: "USERMODE",
    },
    RegDef {
        offset: 0x810004,
        name: "USERMODE_4",
        group: "USERMODE",
    },
    RegDef {
        offset: 0x810010,
        name: "USERMODE_TIME_LO",
        group: "USERMODE",
    },
    RegDef {
        offset: 0x810014,
        name: "USERMODE_TIME_HI",
        group: "USERMODE",
    },
    RegDef {
        offset: 0x810090,
        name: "USERMODE_NOTIFY_CHAN_PENDING",
        group: "USERMODE",
    },
    // GPCCS scan
    RegDef {
        offset: 0x400000,
        name: "GPC0_BOOT0",
        group: "GPCCS",
    },
    RegDef {
        offset: 0x418000,
        name: "GPC0_GPCCS_BOOT0",
        group: "GPCCS",
    },
    RegDef {
        offset: 0x418100,
        name: "GPC0_GPCCS_CPUCTL",
        group: "GPCCS",
    },
    RegDef {
        offset: 0x418110,
        name: "GPC0_GPCCS_PC",
        group: "GPCCS",
    },
    RegDef {
        offset: 0x420000,
        name: "GPC1_BOOT0",
        group: "GPCCS",
    },
    RegDef {
        offset: 0x428000,
        name: "GPC1_GPCCS_BOOT0",
        group: "GPCCS",
    },
    RegDef {
        offset: 0x428100,
        name: "GPC1_GPCCS_CPUCTL",
        group: "GPCCS",
    },
];

impl RegisterMap for NvGv100Map {
    fn vendor(&self) -> &str {
        "nvidia"
    }

    fn arch(&self) -> &str {
        "GV100"
    }

    fn registers(&self) -> &[RegDef] {
        REGISTERS
    }

    fn decode_temp_c(&self, raw: u32) -> Option<u32> {
        Some((raw >> 8) & 0xFF)
    }

    fn decode_boot_id(&self, raw: u32) -> String {
        let chipset = (raw >> 20) & 0x1FF;
        let impl_id = raw & 0xFF;
        format!("NV{chipset:03X} impl={impl_id:#x}")
    }

    fn thermal_offset(&self) -> Option<u32> {
        Some(0x020460)
    }
}
