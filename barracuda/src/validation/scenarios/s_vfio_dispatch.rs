// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: VFIO Sovereign Dispatch — validates in-process GPU dispatch
//! via the sovereign VFIO path, bypassing wgpu and Vulkan entirely.
//!
//! **NOTE (May 2026):** `coral-gpu` was excised from coralReef Sprint 9.
//! The `sovereign-dispatch` feature gate keeps this scenario inert until
//! toadStool Phase C provides the equivalent dispatch API.
//!
//! Exercises:
//! - VFIO GPU detection via sysfs probing
//! - Warm VFIO open: `from_vfio_warm_with_sm()` (Titan V) / `from_vfio_warm_legacy()` (K80)
//! - WGSL → native binary compilation via coral-reef
//! - Buffer alloc → upload → dispatch → readback on real hardware
//!
//! Target hardware (biomeGate compute trio):
//!   - Titan V (GV100, SM70) at BDF 02:00.0
//!   - Tesla K80 (GK210, SM37) at BDF 4b:00.0

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "vfio-dispatch",
        track: Track::GpuCompute,
        tier: Tier::Live,
        provenance_crate: "validate_vfio_sovereign",
        provenance_date: "2026-05-12",
        description: "VFIO sovereign dispatch: VFIO → compile → dispatch → readback (awaits toadStool Phase C)",
    },
    run,
};

#[cfg(feature = "sovereign-dispatch")]
const WRITE_CONSTANT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = 42u;
}
"#;

#[allow(dead_code)]
#[derive(Clone)]
struct VfioTarget {
    name: String,
    bdf: String,
    sm: u32,
    use_legacy: bool,
}

fn discover_vfio_targets() -> Vec<VfioTarget> {
    let mut targets = Vec::new();

    let titan_bdf =
        std::env::var("HOTSPRING_TITAN_V_BDF").unwrap_or_else(|_| "0000:02:00.0".to_string());
    targets.push(VfioTarget {
        name: "titan-v".into(),
        bdf: titan_bdf,
        sm: 70,
        use_legacy: false,
    });

    let k80_bdf = std::env::var("HOTSPRING_K80_BDF").map_or_else(
        |_| "0000:4b:00.0".to_string(),
        |b| {
            if b.contains(':') && b.len() > 7 {
                b
            } else {
                format!("0000:{b}")
            }
        },
    );
    targets.push(VfioTarget {
        name: "k80-die0".into(),
        bdf: k80_bdf,
        sm: 37,
        use_legacy: true,
    });

    targets
}

pub fn run(v: &mut ValidationHarness) {
    let vfio_driver = std::path::Path::new("/sys/bus/pci/drivers/vfio-pci");
    let vfio_loaded = vfio_driver.exists();
    v.check_bool("vfio:driver_present", vfio_loaded);
    if !vfio_loaded {
        return;
    }

    let targets = discover_vfio_targets();
    for target in &targets {
        let prefix = format!("vfio:{}:", target.name);

        let device_sysfs = format!("/sys/bus/pci/devices/{}", target.bdf);
        let sysfs_present = std::path::Path::new(&device_sysfs).exists();
        v.check_bool(&format!("{prefix}sysfs_present"), sysfs_present);
        if !sysfs_present {
            continue;
        }

        let driver_link = format!("{device_sysfs}/driver");
        let bound_to_vfio = std::fs::read_link(&driver_link)
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .is_some_and(|name| name == "vfio-pci");
        v.check_bool(&format!("{prefix}bound_to_vfio"), bound_to_vfio);
        if !bound_to_vfio {
            continue;
        }

        #[cfg(feature = "sovereign-dispatch")]
        {
            validate_vfio_gpu(v, &prefix, target);
        }

        #[cfg(not(feature = "sovereign-dispatch"))]
        {
            v.check_bool(&format!("{prefix}sovereign_dispatch_feature"), false);
        }
    }
}

#[cfg(feature = "sovereign-dispatch")]
fn validate_vfio_gpu(v: &mut ValidationHarness, prefix: &str, target: &VfioTarget) {
    use coral_gpu::GpuContext;

    let ctx_result = if target.use_legacy {
        GpuContext::from_vfio_warm_legacy(target.bdf, target.sm)
    } else {
        GpuContext::from_vfio_warm_with_sm(target.bdf, target.sm)
    };
    let ctx_ok = ctx_result.is_ok();
    v.check_bool(&format!("{prefix}vfio_open"), ctx_ok);

    let mut ctx = match ctx_result {
        Ok(c) => c,
        Err(_) => return,
    };

    let target_arch = ctx.target();
    v.check_bool(
        &format!("{prefix}target_detected"),
        format!("{target_arch:?}").contains("Nvidia"),
    );

    let compile_result = ctx.compile_wgsl(WRITE_CONSTANT_WGSL);
    let compile_ok = compile_result.is_ok();
    v.check_bool(&format!("{prefix}wgsl_compile"), compile_ok);

    let kernel = match compile_result {
        Ok(k) => k,
        Err(_) => return,
    };

    v.check_bool(
        &format!("{prefix}binary_nonzero"),
        !kernel.binary.is_empty(),
    );

    let buf = match ctx.alloc(4096) {
        Ok(b) => b,
        Err(_) => {
            v.check_bool(&format!("{prefix}alloc"), false);
            return;
        }
    };
    v.check_bool(&format!("{prefix}alloc"), true);

    let sentinel: u32 = 0xDEAD_BEEF;
    let mut init_data = vec![0u8; 4096];
    for chunk in init_data[..16].chunks_exact_mut(4) {
        chunk.copy_from_slice(&sentinel.to_le_bytes());
    }
    let upload_ok = ctx.upload(buf, &init_data).is_ok();
    v.check_bool(&format!("{prefix}upload"), upload_ok);
    if !upload_ok {
        return;
    }

    let dispatch_ok = ctx.dispatch(&kernel, &[buf], [1, 1, 1]).is_ok();
    v.check_bool(&format!("{prefix}dispatch"), dispatch_ok);
    if !dispatch_ok {
        return;
    }

    match ctx.readback(buf, 16) {
        Ok(data) => {
            let vals: &[u32] = bytemuck::cast_slice(&data[..4]);
            v.check_bool(&format!("{prefix}readback_42"), vals[0] == 42);
        }
        Err(_) => {
            v.check_bool(&format!("{prefix}readback"), false);
        }
    }
}
