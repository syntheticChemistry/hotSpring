// SPDX-License-Identifier: AGPL-3.0-or-later

use hotspring_barracuda::ember_types::MmioBatchOp;
use hotspring_barracuda::fleet_client::{
    EmberClient, FleetDiscovery, discover_diesel_ember_socket,
};

use super::registers::DMEMC_AINCW;

/// Read a BAR0 register via ember, returning 0xDEAD_DEAD on error.
pub fn r32(ember: &EmberClient, bdf: &str, offset: u32) -> u32 {
    match ember.mmio_read(bdf, offset) {
        Ok(r) => r.value,
        Err(e) => {
            eprintln!("  WARN: ember.mmio.read({offset:#010x}): {e}");
            0xDEAD_DEAD
        }
    }
}

/// Write a BAR0 register via ember, ignoring errors (GPU may be in D3cold).
pub fn w32(ember: &EmberClient, bdf: &str, offset: u32, value: u32) {
    if let Err(e) = ember.mmio_write(bdf, offset, value) {
        eprintln!("  WARN: ember.mmio.write({offset:#010x}, {value:#010x}): {e}");
    }
}

/// Build a CSDATA batch: one DMEMC setup write followed by N DMEMD word writes.
/// The AINCW flag auto-increments the DMEM address on each DMEMD write.
pub fn csdata_batch_ops(dmemc: u32, dmemd: u32, words: &[u32], byte_offset: u32) -> Vec<MmioBatchOp> {
    let mut ops = Vec::with_capacity(1 + words.len());
    ops.push(MmioBatchOp::write(dmemc, DMEMC_AINCW | byte_offset));
    for &w in words {
        ops.push(MmioBatchOp::write(dmemd, w));
    }
    ops
}

/// PIO-upload firmware code to Falcon IMEM via ember.falcon.upload_imem.
///
/// Note: ember handles the page-by-page IMEMC/IMEMD/IMETTAG protocol internally.
pub fn upload_imem(ember: &EmberClient, bdf: &str, base: u32, imem_words: &[u32]) {
    // Convert words to bytes for the RPC (ember does the page-by-page PIO)
    let bytes: Vec<u8> = imem_words.iter().flat_map(|w| w.to_le_bytes()).collect();
    match ember.falcon_upload_imem(bdf, base, 0, &bytes, 0, false) {
        Ok(r) if r.ok => {}
        Ok(r) => eprintln!("  WARN: falcon.upload_imem ok=false bytes={:?}", r.bytes),
        Err(e) => eprintln!("  WARN: falcon.upload_imem error: {e}"),
    }
}

/// PIO-upload firmware data to Falcon DMEM at byte_offset via ember.falcon.upload_dmem.
pub fn upload_dmem(ember: &EmberClient, bdf: &str, base: u32, byte_offset: u32, data_words: &[u32]) {
    let bytes: Vec<u8> = data_words.iter().flat_map(|w| w.to_le_bytes()).collect();
    match ember.falcon_upload_dmem(bdf, base, byte_offset, &bytes) {
        Ok(r) if r.ok => {}
        Ok(r) => eprintln!("  WARN: falcon.upload_dmem ok=false bytes={:?}", r.bytes),
        Err(e) => eprintln!("  WARN: falcon.upload_dmem error: {e}"),
    }
}

/// Discover and connect to the toadstool-ember instance for a given BDF.
///
/// Search order: diesel engine scan → fleet discovery → slug-based socket → legacy.
/// Accepts `--ember-socket` CLI override to skip discovery.
pub fn connect_ember(bdf: &str, override_socket: Option<&str>) -> EmberClient {
    if let Some(sock) = override_socket {
        return EmberClient::connect(sock);
    }
    if let Some(sock) = discover_diesel_ember_socket(bdf) {
        eprintln!("  diesel engine: found ember at {}", sock.display());
        return EmberClient::connect(sock.to_string_lossy().as_ref());
    }
    if let Ok(disc) = FleetDiscovery::load_default() {
        if let Some(sock) = disc.file().routes.get(bdf) {
            return EmberClient::connect(sock);
        }
    }
    for candidate in hotspring_barracuda::fleet_client::ember_socket_candidates(bdf) {
        if candidate.exists() {
            return EmberClient::connect(candidate.to_string_lossy().as_ref());
        }
    }
    eprintln!(
        "FATAL: no ember socket found for BDF {bdf}.\n\
         Start toadstool-ember (diesel engine) and verify: toadstool device list\n\
         Override with: --ember-socket /path/to/ember.sock"
    );
    std::process::exit(1);
}
