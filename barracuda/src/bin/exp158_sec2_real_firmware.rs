// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 158: SEC2 Real Firmware Upload
//!
//! Uploads NVIDIA's signed SEC2/ACR firmware (from linux-firmware) to the Titan V's
//! SEC2 falcon. Replaces the NOP/halt probe from exp154 with the actual GV100
//! firmware stack: ACR bootloader + SEC2 HS image + FECS real firmware.
//!
//! ## Usage
//!
//! ```text
//! sudo ./target/release/exp158_sec2_real_firmware --bdf 0000:03:00.0
//! ```

use std::path::{Path, PathBuf};
use std::time::Duration;

use hotspring_barracuda::fleet_client::{EmberClient, FleetDiscovery};
use hotspring_barracuda::validation::ValidationHarness;

const SEC2_BASE: u32 = 0x087000;
const SEC2_CPUCTL: u32 = SEC2_BASE + 0x100;
const FECS_BASE: u32 = 0x409000;
const PMU_BASE: u32 = 0x10a000;
const PMU_CPUCTL: u32 = PMU_BASE + 0x100;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  SEC2 Real Firmware Upload — Experiment 158                 ║");
    println!("║  Pipeline: ACR BL → SEC2 HS → FECS firmware                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut harness = ValidationHarness::new("exp158_sec2_real_firmware");

    let args: Vec<String> = std::env::args().collect();
    let bdf = extract_arg(&args, "--bdf").unwrap_or_else(|| "0000:03:00.0".to_string());
    let fw_dir = extract_arg(&args, "--firmware-dir")
        .unwrap_or_else(|| "/lib/firmware/nvidia/gv100".to_string());

    println!("  Target BDF: {bdf}");
    println!("  Firmware dir: {fw_dir}");

    let ember = match connect_ember(&bdf) {
        Some(e) => e,
        None => {
            eprintln!("ERROR: cannot connect to ember for {bdf}");
            harness.check_bool("ember reachable", false);
            harness.finish();
        }
    };
    harness.check_bool("ember reachable", true);

    // Phase 1: Load firmware
    println!("\n━━━ Phase 1: Load Firmware ━━━\n");
    let fw = match load_firmware(&fw_dir) {
        Some(fw) => fw,
        None => {
            eprintln!("ERROR: failed to load firmware");
            harness.check_bool("firmware loaded", false);
            harness.finish();
        }
    };
    harness.check_bool("firmware loaded", true);
    println!("  ACR BL code: {} bytes (offset {:#x})", fw.acr_bl_code.len(), fw.acr_bl_code_offset);
    println!("  SEC2 image: {} bytes, code: {} bytes", fw.sec2_image.len(), fw.sec2_code.len());
    println!("  SEC2 data: {} bytes", fw.sec2_data.len());
    println!("  SEC2 sig: {} bytes", fw.sec2_sig.len());
    println!("  FECS inst: {} bytes, data: {} bytes", fw.fecs_inst.len(), fw.fecs_data.len());

    // Phase 2: Pre-flight
    println!("\n━━━ Phase 2: Pre-flight ━━━\n");
    let boot0 = mmio_rd(&ember, &bdf, 0);
    let pmc_en = mmio_rd(&ember, &bdf, 0x200);
    println!("  BOOT0: {boot0:#010x}");
    println!("  PMC_ENABLE: {pmc_en:#010x}");

    let warm_pmc = 0x4200_1120_u32;
    if pmc_en != warm_pmc {
        println!("  Setting PMC_ENABLE → {warm_pmc:#010x}");
        let _ = ember.mmio_write(&bdf, 0x200, warm_pmc);
        std::thread::sleep(Duration::from_millis(10));
    }

    // Phase 3: ACR bootloader → SEC2 IMEM (PIO)
    println!("\n━━━ Phase 3: ACR Bootloader → SEC2 IMEM ━━━\n");
    println!("  Uploading {} bytes to SEC2 IMEM...", fw.acr_bl_code.len());

    match ember.falcon_upload_imem(&bdf, SEC2_BASE, 0, &fw.acr_bl_code, 0, false) {
        Ok(r) => {
            println!("  ACR BL upload: ok={}, bytes={:?}", r.ok, r.bytes);
            harness.check_bool("ACR BL IMEM upload", r.ok);
        }
        Err(e) => {
            eprintln!("  ACR BL upload failed: {e}");
            harness.check_bool("ACR BL IMEM upload", false);
        }
    }

    // Phase 4: SEC2 firmware code → IMEM (after bootloader)
    println!("\n━━━ Phase 4: SEC2 Code → IMEM ━━━\n");
    let bl_words = fw.acr_bl_code.len() / 4;
    let code_imem_start = (bl_words as u32) * 4; // Start after bootloader
    println!("  SEC2 code: {} bytes → IMEM offset {code_imem_start:#x}", fw.sec2_code.len());

    let chunk_size = 4096;
    let mut uploaded = 0;
    let mut upload_ok = true;
    for chunk in fw.sec2_code.chunks(chunk_size) {
        let imem_addr = code_imem_start + uploaded as u32;
        match ember.falcon_upload_imem(&bdf, SEC2_BASE, imem_addr, chunk, 0, true) {
            Ok(r) if r.ok => uploaded += chunk.len(),
            Ok(r) => {
                eprintln!("  IMEM chunk at {imem_addr:#x} ok=false");
                let _ = r;
                upload_ok = false;
                break;
            }
            Err(e) => {
                eprintln!("  IMEM error at {imem_addr:#x}: {e}");
                upload_ok = false;
                break;
            }
        }
    }
    println!("  Uploaded: {uploaded}/{} bytes", fw.sec2_code.len());
    harness.check_bool("SEC2 code upload", upload_ok && uploaded == fw.sec2_code.len());

    // Phase 5: SEC2 data → DMEM
    println!("\n━━━ Phase 5: SEC2 Data → DMEM ━━━\n");
    if !fw.sec2_data.is_empty() {
        println!("  Uploading {} bytes to SEC2 DMEM...", fw.sec2_data.len());
        match ember.falcon_upload_dmem(&bdf, SEC2_BASE, 0, &fw.sec2_data) {
            Ok(r) => {
                println!("  DMEM upload: ok={}, bytes={:?}", r.ok, r.bytes);
                harness.check_bool("SEC2 DMEM upload", r.ok);
            }
            Err(e) => {
                eprintln!("  DMEM upload failed: {e}");
                harness.check_bool("SEC2 DMEM upload", false);
            }
        }
    } else {
        println!("  No data section");
        harness.check_bool("SEC2 DMEM upload", true);
    }

    // Phase 6: Start SEC2
    println!("\n━━━ Phase 6: SEC2 Start ━━━\n");
    match ember.falcon_start_cpu(&bdf, SEC2_BASE) {
        Ok(r) => {
            println!("  SEC2 start: ok={}, pc={:?}, cpuctl={:?}", r.ok, r.pc, r.cpuctl);
            harness.check_bool("SEC2 start", r.ok);
        }
        Err(e) => {
            eprintln!("  SEC2 start failed: {e}");
            harness.check_bool("SEC2 start", false);
        }
    }

    // Phase 7: Poll SEC2
    println!("\n━━━ Phase 7: SEC2 Poll ━━━\n");
    std::thread::sleep(Duration::from_millis(200));
    match ember.falcon_poll(&bdf, SEC2_BASE, 5000, 0xDEAD_A5A5) {
        Ok(r) => {
            let snap_count = r.snapshots.len();
            println!("  Snapshots: {snap_count}");
            println!("  PC trace: {:?}", r.pc_trace);
            if let Some(fin) = &r.final_state {
                println!("  Final: pc={:?}, cpuctl={:?}, mb0={:?}, sctl={:?}",
                    fin.pc, fin.cpuctl, fin.mailbox0, fin.sctl);
                let hs = fin.sctl.map_or(false, |s| s & 0x2 != 0);
                println!("  HS mode: {hs}");
                harness.check_bool("SEC2 HS mode reached", hs);
            } else {
                println!("  No final state available");
                harness.check_bool("SEC2 HS mode reached", false);
            }
            harness.check_bool("SEC2 poll completed", snap_count > 0);
        }
        Err(e) => {
            eprintln!("  Poll error: {e}");
            harness.check_bool("SEC2 poll completed", false);
            harness.check_bool("SEC2 HS mode reached", false);
        }
    }

    // Phase 8: Real FECS firmware
    println!("\n━━━ Phase 8: FECS Real Firmware ━━━\n");
    println!("  Uploading FECS inst ({} bytes)...", fw.fecs_inst.len());
    match ember.falcon_upload_imem(&bdf, FECS_BASE, 0, &fw.fecs_inst, 0, false) {
        Ok(r) => {
            println!("  FECS IMEM: ok={}, bytes={:?}", r.ok, r.bytes);
            harness.check_bool("FECS IMEM upload", r.ok);
        }
        Err(e) => {
            eprintln!("  FECS IMEM failed: {e}");
            harness.check_bool("FECS IMEM upload", false);
        }
    }

    if !fw.fecs_data.is_empty() {
        println!("  Uploading FECS data ({} bytes)...", fw.fecs_data.len());
        match ember.falcon_upload_dmem(&bdf, FECS_BASE, 0, &fw.fecs_data) {
            Ok(r) => {
                println!("  FECS DMEM: ok={}, bytes={:?}", r.ok, r.bytes);
                harness.check_bool("FECS DMEM upload", r.ok);
            }
            Err(e) => {
                eprintln!("  FECS DMEM failed: {e}");
                harness.check_bool("FECS DMEM upload", false);
            }
        }
    }

    println!("  Starting FECS...");
    match ember.falcon_start_cpu(&bdf, FECS_BASE) {
        Ok(r) => {
            println!("  FECS start: ok={}, pc={:?}, cpuctl={:?}", r.ok, r.pc, r.cpuctl);
            harness.check_bool("FECS start", r.ok);
        }
        Err(e) => {
            eprintln!("  FECS start failed: {e}");
            harness.check_bool("FECS start", false);
        }
    }

    std::thread::sleep(Duration::from_millis(200));
    match ember.falcon_poll(&bdf, FECS_BASE, 5000, 0xDEAD_A5A5) {
        Ok(r) => {
            let snap_count = r.snapshots.len();
            println!("  FECS poll: {snap_count} snapshots, pc_trace={:?}", r.pc_trace);
            if let Some(fin) = &r.final_state {
                println!("  FECS final: pc={:?}, cpuctl={:?}, mb0={:?}",
                    fin.pc, fin.cpuctl, fin.mailbox0);
            }
        }
        Err(e) => eprintln!("  FECS poll: {e}"),
    }

    // Cleanup: halt falcons
    println!("\n━━━ Cleanup ━━━\n");
    let _ = ember.mmio_write(&bdf, SEC2_CPUCTL, 0x10);
    let _ = ember.mmio_write(&bdf, PMU_CPUCTL, 0x10);
    println!("  Halted SEC2 + PMU.");

    harness.finish();
}

struct FirmwareSet {
    acr_bl_code: Vec<u8>,
    acr_bl_code_offset: usize,
    sec2_image: Vec<u8>,
    sec2_code: Vec<u8>,
    sec2_data: Vec<u8>,
    sec2_sig: Vec<u8>,
    fecs_inst: Vec<u8>,
    fecs_data: Vec<u8>,
}

fn load_firmware(dir: &str) -> Option<FirmwareSet> {
    let base = PathBuf::from(dir);
    let acr_bl = std::fs::read(base.join("acr/bl.bin")).ok()?;
    let sec2_image = std::fs::read(base.join("sec2/image.bin")).ok()?;
    let sec2_sig = std::fs::read(base.join("sec2/sig.bin")).ok()?;
    let fecs_inst = std::fs::read(base.join("gr/fecs_inst.bin")).ok()?;
    let fecs_data = std::fs::read(base.join("gr/fecs_data.bin")).ok()?;

    // Parse ACR BL header: magic(4) + ver(4) + total(4) + hdr(4) + code_off(4) + code_size(4)
    let code_off = u32::from_le_bytes(acr_bl.get(16..20)?.try_into().ok()?) as usize;
    let code_sz = u32::from_le_bytes(acr_bl.get(20..24)?.try_into().ok()?) as usize;
    let code_sz = code_sz.min(acr_bl.len().saturating_sub(code_off));
    let acr_bl_code = acr_bl[code_off..code_off + code_sz].to_vec();

    // SEC2 image: code at 0x200, size 0xFD00; data at 0x16200, size 0x500
    let sec2_code_off = 0x200_usize;
    let sec2_code_sz = 0xFD00_usize.min(sec2_image.len().saturating_sub(sec2_code_off));
    let sec2_code = sec2_image[sec2_code_off..sec2_code_off + sec2_code_sz].to_vec();

    let sec2_data_off = 0x16200_usize.min(sec2_image.len());
    let sec2_data_sz = 0x500_usize.min(sec2_image.len().saturating_sub(sec2_data_off));
    let sec2_data = if sec2_data_sz > 0 {
        sec2_image[sec2_data_off..sec2_data_off + sec2_data_sz].to_vec()
    } else {
        Vec::new()
    };

    Some(FirmwareSet {
        acr_bl_code_offset: code_off,
        acr_bl_code,
        sec2_image,
        sec2_code,
        sec2_data,
        sec2_sig,
        fecs_inst,
        fecs_data,
    })
}

fn mmio_rd(ember: &EmberClient, bdf: &str, offset: u32) -> u32 {
    ember.mmio_read(bdf, offset).map(|r| r.value).unwrap_or(0xDEAD_DEAD)
}

fn connect_ember(bdf: &str) -> Option<EmberClient> {
    let slug = bdf.replace(':', "-");
    for c in [
        format!("/run/coralreef/fleet/ember-{slug}.sock"),
        "/run/coralreef/ember.sock".to_string(),
    ] {
        if Path::new(&c).exists() {
            let client = EmberClient::connect(&c);
            if client.mmio_read(bdf, 0).is_ok() {
                return Some(client);
            }
        }
    }
    let fleet_path = FleetDiscovery::resolve_path();
    if let Ok(fleet) = FleetDiscovery::load(&fleet_path) {
        for dev in &fleet.file().devices {
            if dev.bdf == bdf {
                if let Some(sock) = &dev.socket {
                    return Some(EmberClient::connect(sock));
                }
            }
        }
    }
    None
}

fn extract_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
