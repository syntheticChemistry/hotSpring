// SPDX-License-Identifier: AGPL-3.0-or-later

//! Native MILC gauge configuration format reader/writer.
//!
//! The MILC format (v5, 1997) is the de facto standard for quenched and
//! dynamical configurations produced by the MILC collaboration. This module
//! enables direct interop with MILC-generated configs for validation.
//!
//! # Binary Layout
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │ magic_number: u32 (20103, big-endian)       │
//! │ dims: [u32; 4] (nx, ny, nz, nt, big-end.)  │
//! │ time_stamp: [u8; 64] (null-terminated)      │
//! │ order: u32 (0=NATURAL, 1=EVEN_ODD)          │
//! │ checksum: [u32; 2] (sum29, sum31)           │
//! │ link_trace: f32 (header diagnostic)         │
//! │ plaquette: f32 (OUR VALIDATION TARGET)      │
//! ├─────────────────────────────────────────────┤
//! │ Gauge field data:                           │
//! │   for site in natural_order(nx,ny,nz,nt):   │
//! │     for mu in 0..4:                         │
//! │       U_mu: 18 × f32/f64 (big-endian)      │
//! │       (re00, im00, re01, im01, ..., im22)   │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! # Site ordering
//!
//! MILC NATURAL_ORDER: site index(x,y,z,t) = x + nx*(y + ny*(z + nz*t))
//! i.e., x varies fastest (innermost loop).
//!
//! Our internal convention (wilson.rs):
//! site index(x,y,z,t) = t*Nx*Ny*Nz + x*Ny*Nz + y*Nz + z
//!
//! The remapping handles this difference transparently on read/write.

use super::complex_f64::Complex64;
use super::su3::Su3Matrix;
use super::wilson::Lattice;

use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MILC_MAGIC_V5: u32 = 20103;
const NATURAL_ORDER: u32 = 0;
#[allow(dead_code)]
const EVEN_ODD_ORDER: u32 = 1;

/// Parsed header of a MILC gauge configuration file.
#[derive(Clone, Debug)]
pub struct MilcHeader {
    pub magic: u32,
    pub dims: [usize; 4],
    pub time_stamp: String,
    pub order: u32,
    pub checksum: [u32; 2],
    pub link_trace: f32,
    pub plaquette: f32,
}

/// Errors specific to MILC format I/O.
#[derive(Debug)]
pub enum MilcError {
    Io(io::Error),
    BadMagic(u32),
    ChecksumMismatch { expected: [u32; 2], computed: [u32; 2] },
    UnsupportedOrder(u32),
}

impl From<io::Error> for MilcError {
    fn from(e: io::Error) -> Self {
        MilcError::Io(e)
    }
}

impl std::fmt::Display for MilcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MilcError::Io(e) => write!(f, "MILC I/O error: {e}"),
            MilcError::BadMagic(m) => write!(f, "Bad MILC magic: {m} (expected {MILC_MAGIC_V5})"),
            MilcError::ChecksumMismatch { expected, computed } => {
                write!(f, "Checksum mismatch: header={expected:?}, computed={computed:?}")
            }
            MilcError::UnsupportedOrder(o) => write!(f, "Unsupported MILC order: {o}"),
        }
    }
}

impl std::error::Error for MilcError {}

/// Read a MILC v5 gauge configuration from a file path.
///
/// Returns a `Lattice` with the gauge field populated and the `MilcHeader`
/// for validation (contains the plaquette value reported by MILC).
pub fn read_milc_config(path: &Path) -> Result<(Lattice, MilcHeader), MilcError> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    read_milc(&mut reader)
}

/// Read MILC format from any reader.
pub fn read_milc<R: Read>(reader: &mut R) -> Result<(Lattice, MilcHeader), MilcError> {
    let header = read_header(reader)?;

    if header.magic != MILC_MAGIC_V5 {
        return Err(MilcError::BadMagic(header.magic));
    }
    if header.order != NATURAL_ORDER {
        return Err(MilcError::UnsupportedOrder(header.order));
    }

    let [nx, ny, nz, nt] = header.dims;
    let vol = nx * ny * nz * nt;
    let mut lattice = Lattice::cold_start([nx, ny, nz, nt], 6.0);

    let mut sum29: u32 = 0;
    let mut sum31: u32 = 0;

    for milc_idx in 0..vol {
        let (x, y, z, t) = milc_natural_to_coords(milc_idx, nx, ny, nz);

        for mu in 0..4 {
            let matrix = read_su3_f32_be(reader)?;
            let bytes = matrix_to_bytes(&matrix);
            update_checksum(&bytes, milc_idx as u32, &mut sum29, &mut sum31);

            let our_idx = t * nx * ny * nz + x * ny * nz + y * nz + z;
            lattice.links[our_idx * 4 + mu] = matrix;
        }
    }

    let computed = [sum29, sum31];
    if computed != header.checksum {
        return Err(MilcError::ChecksumMismatch {
            expected: header.checksum,
            computed,
        });
    }

    Ok((lattice, header))
}

/// Write a lattice to MILC v5 format.
pub fn write_milc_config(path: &Path, lattice: &Lattice) -> Result<MilcHeader, MilcError> {
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_milc(&mut writer, lattice)
}

/// Write MILC format to any writer.
pub fn write_milc<W: Write>(writer: &mut W, lattice: &Lattice) -> Result<MilcHeader, MilcError> {
    let [nx, ny, nz, nt] = lattice.dims;
    let vol = nx * ny * nz * nt;
    let plaquette = lattice.average_plaquette() as f32;
    let link_trace = compute_link_trace(lattice) as f32;

    // First pass: compute checksum
    let mut sum29: u32 = 0;
    let mut sum31: u32 = 0;
    for milc_idx in 0..vol {
        let (x, y, z, t) = milc_natural_to_coords(milc_idx, nx, ny, nz);
        let our_idx = t * nx * ny * nz + x * ny * nz + y * nz + z;
        for mu in 0..4 {
            let bytes = matrix_to_bytes(&lattice.links[our_idx * 4 + mu]);
            update_checksum(&bytes, milc_idx as u32, &mut sum29, &mut sum31);
        }
    }

    let header = MilcHeader {
        magic: MILC_MAGIC_V5,
        dims: [nx, ny, nz, nt],
        time_stamp: format!("hotSpring {} ecoPrimals/strandGate", chrono_stamp()),
        order: NATURAL_ORDER,
        checksum: [sum29, sum31],
        link_trace,
        plaquette,
    };

    write_header(writer, &header)?;

    // Write gauge field in MILC natural order
    for milc_idx in 0..vol {
        let (x, y, z, t) = milc_natural_to_coords(milc_idx, nx, ny, nz);
        let our_idx = t * nx * ny * nz + x * ny * nz + y * nz + z;
        for mu in 0..4 {
            write_su3_f32_be(writer, &lattice.links[our_idx * 4 + mu])?;
        }
    }

    writer.flush()?;
    Ok(header)
}

/// Validate round-trip: read a MILC config, compute plaquette, compare to header.
pub fn validate_milc_roundtrip(path: &Path) -> Result<MilcRoundtripResult, MilcError> {
    let (lattice, header) = read_milc_config(path)?;
    let our_plaquette = lattice.average_plaquette();
    let milc_plaquette = header.plaquette as f64;
    let delta = (our_plaquette - milc_plaquette).abs();
    let relative_delta = delta / milc_plaquette;

    Ok(MilcRoundtripResult {
        milc_plaquette,
        our_plaquette,
        absolute_delta: delta,
        relative_delta,
        checksum_valid: true, // already validated in read_milc
        dims: header.dims,
    })
}

/// Result of a MILC round-trip validation.
#[derive(Clone, Debug)]
pub struct MilcRoundtripResult {
    pub milc_plaquette: f64,
    pub our_plaquette: f64,
    pub absolute_delta: f64,
    pub relative_delta: f64,
    pub checksum_valid: bool,
    pub dims: [usize; 4],
}

impl std::fmt::Display for MilcRoundtripResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [nx, ny, nz, nt] = self.dims;
        write!(
            f,
            "MILC Round-trip ({}×{}×{}×{}):\n  MILC header ⟨P⟩ = {:.8}\n  Our computed ⟨P⟩ = {:.8}\n  |Δ| = {:.2e} (relative: {:.2e})\n  Checksum: {}",
            nx, ny, nz, nt,
            self.milc_plaquette,
            self.our_plaquette,
            self.absolute_delta,
            self.relative_delta,
            if self.checksum_valid { "PASS" } else { "FAIL" },
        )
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────

/// MILC natural order: index = x + nx*(y + ny*(z + nz*t))
fn milc_natural_to_coords(idx: usize, nx: usize, ny: usize, nz: usize) -> (usize, usize, usize, usize) {
    let x = idx % nx;
    let rem = idx / nx;
    let y = rem % ny;
    let rem = rem / ny;
    let z = rem % nz;
    let t = rem / nz;
    (x, y, z, t)
}

fn read_header<R: Read>(reader: &mut R) -> Result<MilcHeader, MilcError> {
    let magic = read_u32_be(reader)?;
    let dims = [
        read_u32_be(reader)? as usize,
        read_u32_be(reader)? as usize,
        read_u32_be(reader)? as usize,
        read_u32_be(reader)? as usize,
    ];

    let mut ts_buf = [0u8; 64];
    reader.read_exact(&mut ts_buf)?;
    let time_stamp = String::from_utf8_lossy(&ts_buf)
        .trim_end_matches('\0')
        .to_string();

    let order = read_u32_be(reader)?;
    let sum29 = read_u32_be(reader)?;
    let sum31 = read_u32_be(reader)?;
    let link_trace = read_f32_be(reader)?;
    let plaquette = read_f32_be(reader)?;

    Ok(MilcHeader {
        magic,
        dims,
        time_stamp,
        order,
        checksum: [sum29, sum31],
        link_trace,
        plaquette,
    })
}

fn write_header<W: Write>(writer: &mut W, h: &MilcHeader) -> Result<(), MilcError> {
    write_u32_be(writer, h.magic)?;
    for &d in &h.dims {
        write_u32_be(writer, d as u32)?;
    }

    let mut ts_buf = [0u8; 64];
    let ts_bytes = h.time_stamp.as_bytes();
    let len = ts_bytes.len().min(63);
    ts_buf[..len].copy_from_slice(&ts_bytes[..len]);
    writer.write_all(&ts_buf)?;

    write_u32_be(writer, h.order)?;
    write_u32_be(writer, h.checksum[0])?;
    write_u32_be(writer, h.checksum[1])?;
    write_f32_be(writer, h.link_trace)?;
    write_f32_be(writer, h.plaquette)?;
    Ok(())
}

fn read_su3_f32_be<R: Read>(reader: &mut R) -> Result<Su3Matrix, MilcError> {
    let mut m = Su3Matrix::IDENTITY;
    for row in 0..3 {
        for col in 0..3 {
            let re = read_f32_be(reader)? as f64;
            let im = read_f32_be(reader)? as f64;
            m.m[row][col] = Complex64 { re, im };
        }
    }
    Ok(m)
}

fn write_su3_f32_be<W: Write>(writer: &mut W, m: &Su3Matrix) -> Result<(), MilcError> {
    for row in 0..3 {
        for col in 0..3 {
            let c = m.m[row][col];
            write_f32_be(writer, c.re as f32)?;
            write_f32_be(writer, c.im as f32)?;
        }
    }
    Ok(())
}

fn matrix_to_bytes(m: &Su3Matrix) -> [u8; 72] {
    let mut buf = [0u8; 72];
    let mut offset = 0;
    for row in 0..3 {
        for col in 0..3 {
            let c = m.m[row][col];
            buf[offset..offset + 4].copy_from_slice(&(c.re as f32).to_be_bytes());
            offset += 4;
            buf[offset..offset + 4].copy_from_slice(&(c.im as f32).to_be_bytes());
            offset += 4;
        }
    }
    buf
}

/// MILC checksum: two running sums with different bit rotations.
/// rank = site_index for serial (no MPI) case.
fn update_checksum(data: &[u8], rank: u32, sum29: &mut u32, sum31: &mut u32) {
    let mut val: u32 = 0;
    for chunk in data.chunks(4) {
        if chunk.len() == 4 {
            val ^= u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
    }
    *sum29 ^= val.rotate_left(rank % 29);
    *sum31 ^= val.rotate_left(rank % 31);
}

fn compute_link_trace(lattice: &Lattice) -> f64 {
    let [nx, ny, nz, nt] = lattice.dims;
    let vol = nx * ny * nz * nt;
    let mut trace_sum = 0.0;
    for site in 0..vol {
        for mu in 0..4 {
            let m = &lattice.links[site * 4 + mu];
            for i in 0..3 {
                trace_sum += m.m[i][i].re;
            }
        }
    }
    trace_sum / (vol as f64 * 4.0 * 3.0)
}

fn chrono_stamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{now}")
}

// ─── Big-endian I/O primitives ────────────────────────────────────────

fn read_u32_be<R: Read>(r: &mut R) -> Result<u32, io::Error> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_f32_be<R: Read>(r: &mut R) -> Result<f32, io::Error> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_be_bytes(buf))
}

fn write_u32_be<W: Write>(w: &mut W, v: u32) -> Result<(), io::Error> {
    w.write_all(&v.to_be_bytes())
}

fn write_f32_be<W: Write>(w: &mut W, v: f32) -> Result<(), io::Error> {
    w.write_all(&v.to_be_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_order_coords() {
        let (nx, ny, nz) = (4, 4, 4);
        // First site should be (0,0,0,0)
        assert_eq!(milc_natural_to_coords(0, nx, ny, nz), (0, 0, 0, 0));
        // x varies fastest
        assert_eq!(milc_natural_to_coords(1, nx, ny, nz), (1, 0, 0, 0));
        assert_eq!(milc_natural_to_coords(4, nx, ny, nz), (0, 1, 0, 0));
        assert_eq!(milc_natural_to_coords(16, nx, ny, nz), (0, 0, 1, 0));
        assert_eq!(milc_natural_to_coords(64, nx, ny, nz), (0, 0, 0, 1));
    }

    #[test]
    fn test_cold_start_roundtrip() {
        let lattice = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let plaq_before = lattice.average_plaquette();

        let mut buffer: Vec<u8> = Vec::new();
        let header = write_milc(&mut buffer, &lattice).unwrap();

        // Verify header plaquette matches
        assert!((header.plaquette as f64 - plaq_before).abs() < 1e-5);

        let mut cursor = std::io::Cursor::new(&buffer);
        let (lattice_back, header_back) = read_milc(&mut cursor).unwrap();

        let plaq_after = lattice_back.average_plaquette();
        assert!((plaq_before - plaq_after).abs() < 1e-6,
            "Round-trip plaquette mismatch: {plaq_before} vs {plaq_after}");
        assert_eq!(header_back.dims, [4, 4, 4, 4]);
    }

    #[test]
    fn test_checksum_deterministic() {
        let data = [0u8; 72];
        let mut sum29_a: u32 = 0;
        let mut sum31_a: u32 = 0;
        let mut sum29_b: u32 = 0;
        let mut sum31_b: u32 = 0;

        update_checksum(&data, 0, &mut sum29_a, &mut sum31_a);
        update_checksum(&data, 0, &mut sum29_b, &mut sum31_b);

        assert_eq!(sum29_a, sum29_b);
        assert_eq!(sum31_a, sum31_b);
    }
}
