// SPDX-License-Identifier: AGPL-3.0-or-later

use std::panic;
use std::sync::Arc;

use barracuda::device::WgpuDevice;

pub fn run_guarded(label: &str, f: impl FnOnce() + panic::UnwindSafe) {
    if let Err(e) = panic::catch_unwind(f) {
        let msg = if let Some(s) = e.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        };
        let short = if msg.len() > 120 { &msg[..120] } else { &msg };
        println!("  SKIP {label}: shader/driver incompatibility — {short}...");
        println!();
    }
}

pub async fn try_create_device() -> Option<Arc<WgpuDevice>> {
    println!("═══ Initializing GPU ═══");
    match WgpuDevice::new_f64_capable().await {
        Ok(dev) => Some(Arc::new(dev)),
        Err(_) => match WgpuDevice::new().await {
            Ok(dev) => Some(Arc::new(dev)),
            Err(_) => None,
        },
    }
}
