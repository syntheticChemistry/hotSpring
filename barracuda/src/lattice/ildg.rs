// SPDX-License-Identifier: AGPL-3.0-or-later

//! ILDG gauge configuration I/O for MILC/Bazavov ecosystem interop.
//!
//! Reads and writes SU(3) gauge configurations in the ILDG binary format
//! (LIME container with `ildg-format`, `ildg-binary-data`, and
//! `ildg-data-lfn` records). This enables bidirectional data flow with
//! the MILC code and any ILDG-compliant tool.
//!
//! # ILDG binary format (v1.2)
//!
//! Gauge data is stored as big-endian IEEE f64 (or f32) with index ordering:
//!
//! ```text
//! double U[Nt][Nz][Ny][Nx][4][3][3][2];
//!           ↑ slowest              ↑ fastest (re/im)
//! ```
//!
//! Our internal convention (`wilson.rs`) uses:
//!
//! ```text
//! links[t*NxNyNz + x*NyNz + y*Nz + z][mu]  →  Su3Matrix.m[row][col].(re,im)
//! ```
//!
//! The spatial index order differs (we use x>y>z fastest, ILDG uses z>y>x
//! fastest), so a remapping is needed on read/write.
//!
//! # References
//!
//! - ILDG file format 1.2: <https://www-zeuthen.desy.de/apewww/ILDG/specifications/ildg-file-format-1.2.pdf>
//! - LIME 1.2: <https://usqcd-software.github.io/c-lime/lime_1p2.pdf>
//! - SciDAC/USQCD c-lime: <https://usqcd-software.github.io/c-lime/>

pub use super::qcdml::ildg_crc;

use super::lime::{LimeReader, LimeWriter};
use super::su3::Su3Matrix;
use super::wilson::Lattice;

use std::io::{self, Cursor, Read, Write};

/// Metadata for an ILDG gauge configuration file.
#[derive(Clone, Debug)]
pub struct IldgMetadata {
    /// Lattice dimensions `[Nx, Ny, Nz, Nt]`.
    pub dims: [usize; 4],
    /// Inverse bare coupling β = 6/g².
    pub beta: f64,
    /// Quark mass (0.0 for quenched).
    pub mass: f64,
    /// Number of dynamical flavors (0 = quenched).
    pub nf: usize,
    /// Gauge action type (e.g. "Wilson", "Symanzik").
    pub action: String,
    /// Fermion action (e.g. "staggered", "HISQ", "none").
    pub fermion_action: String,
    /// Creator software identifier.
    pub creator: String,
    /// Ensemble identifier (arbitrary string, e.g. "hotspring_b6.0_m0.1_L16").
    pub ensemble_id: String,
    /// HMC trajectory index for this configuration.
    pub trajectory: usize,
    /// Average plaquette measured on this configuration.
    pub plaquette: f64,
    /// Floating-point precision in bits (32 or 64).
    pub precision_bits: u32,
    /// Logical file name for ILDG catalogs.
    pub lfn: String,
}

impl IldgMetadata {
    /// Construct metadata for a given lattice with sensible defaults.
    pub fn for_lattice(lattice: &Lattice, trajectory: usize) -> Self {
        let plaq = lattice.average_plaquette();
        let [nx, ny, nz, nt] = lattice.dims;
        Self {
            dims: lattice.dims,
            beta: lattice.beta,
            mass: 0.0,
            nf: 0,
            action: "Wilson".to_string(),
            fermion_action: "none".to_string(),
            creator: format!("hotSpring-barracuda {}", env!("CARGO_PKG_VERSION")),
            ensemble_id: format!("hotspring_b{:.2}_L{}", lattice.beta, nx),
            trajectory,
            plaquette: plaq,
            precision_bits: 64,
            lfn: format!(
                "/hotspring/b{:.2}/{}x{}x{}x{}/conf.{:06}",
                lattice.beta, nx, ny, nz, nt, trajectory
            ),
        }
    }
}

/// Write a gauge configuration to an ILDG/LIME file.
///
/// Produces a valid ILDG file with three LIME records:
/// 1. `ildg-format` — XML metadata (dimensions, precision, field type)
/// 2. `ildg-binary-data` — gauge field in ILDG index order, big-endian
/// 3. `ildg-data-lfn` — logical file name string
pub fn write_gauge_config<W: Write>(
    writer: W,
    lattice: &Lattice,
    metadata: &IldgMetadata,
) -> io::Result<()> {
    let mut lime = LimeWriter::new(writer);

    // Record 1: ildg-format XML
    let format_xml = format_xml(metadata);
    lime.begin_message("ildg-format", format_xml.as_bytes())?;

    // Record 2: ildg-binary-data (gauge field)
    let binary_data = lattice_to_ildg_binary(lattice, metadata.precision_bits);
    lime.end_message("ildg-binary-data", &binary_data)?;

    // Record 3: ildg-data-lfn (separate single-record message)
    lime.write_record("ildg-data-lfn", metadata.lfn.as_bytes())?;

    lime.flush()?;
    Ok(())
}

/// Write a gauge configuration to a file path.
pub fn write_gauge_config_file(
    path: &str,
    lattice: &Lattice,
    metadata: &IldgMetadata,
) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let buffered = io::BufWriter::new(file);
    write_gauge_config(buffered, lattice, metadata)
}

/// Read a gauge configuration from an ILDG/LIME file.
///
/// Returns the loaded `Lattice` and parsed metadata.
pub fn read_gauge_config<R: Read>(reader: R) -> io::Result<(Lattice, IldgMetadata)> {
    let lime = LimeReader::new(reader);
    let records = lime.read_all()?;

    let mut format_xml: Option<String> = None;
    let mut binary_data: Option<Vec<u8>> = None;
    let mut lfn: Option<String> = None;

    for rec in &records {
        match rec.header.record_type.as_str() {
            "ildg-format" => {
                format_xml = Some(String::from_utf8_lossy(&rec.data).to_string());
            }
            "ildg-binary-data" => {
                binary_data = Some(rec.data.clone());
            }
            "ildg-data-lfn" => {
                lfn = Some(String::from_utf8_lossy(&rec.data).to_string());
            }
            _ => {} // skip unknown record types
        }
    }

    let xml = format_xml
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing ildg-format record"))?;
    let data = binary_data.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "missing ildg-binary-data record",
        )
    })?;

    let meta = parse_format_xml(&xml, lfn.as_deref())?;

    let lattice = ildg_binary_to_lattice(&data, &meta)?;

    Ok((lattice, meta))
}

/// Read a gauge configuration from a file path.
pub fn read_gauge_config_file(path: &str) -> io::Result<(Lattice, IldgMetadata)> {
    let file = std::fs::File::open(path)?;
    let buffered = io::BufReader::new(file);
    read_gauge_config(buffered)
}

/// Convert a `Lattice` to ILDG binary format (big-endian, ILDG index order).
fn lattice_to_ildg_binary(lattice: &Lattice, precision_bits: u32) -> Vec<u8> {
    let [nx, ny, nz, nt] = lattice.dims;
    let n_links = nx * ny * nz * nt * 4;
    let floats_per_link = 18; // 3×3 complex = 9 × 2

    let bytes_per_float = if precision_bits == 32 { 4 } else { 8 };
    let mut buf = Vec::with_capacity(n_links * floats_per_link * bytes_per_float);

    // ILDG ordering: t(slow) > z > y > x > mu > row > col > re/im(fast)
    for t in 0..nt {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let coords = [x, y, z, t];
                    for mu in 0..4 {
                        let u = lattice.link(coords, mu);
                        for row in 0..3 {
                            for col in 0..3 {
                                if precision_bits == 32 {
                                    buf.extend_from_slice(&(u.m[row][col].re as f32).to_be_bytes());
                                    buf.extend_from_slice(&(u.m[row][col].im as f32).to_be_bytes());
                                } else {
                                    buf.extend_from_slice(&u.m[row][col].re.to_be_bytes());
                                    buf.extend_from_slice(&u.m[row][col].im.to_be_bytes());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    buf
}

/// Convert ILDG binary data back to a `Lattice`.
fn ildg_binary_to_lattice(data: &[u8], meta: &IldgMetadata) -> io::Result<Lattice> {
    let [nx, ny, nz, nt] = meta.dims;
    let vol = nx * ny * nz * nt;
    let bytes_per_float = if meta.precision_bits == 32 { 4 } else { 8 };
    let expected_size = vol * 4 * 18 * bytes_per_float;

    if data.len() != expected_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "ILDG binary data size mismatch: got {} bytes, expected {} ({}x{}x{}x{}, {}bit)",
                data.len(),
                expected_size,
                nx,
                ny,
                nz,
                nt,
                meta.precision_bits
            ),
        ));
    }

    let mut links = vec![Su3Matrix::IDENTITY; vol * 4];
    let mut cursor = Cursor::new(data);

    // ILDG ordering: t(slow) > z > y > x > mu > row > col > re/im(fast)
    for t in 0..nt {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let coords = [x, y, z, t];
                    let site_idx = t * (nx * ny * nz) + x * (ny * nz) + y * nz + z;

                    for mu in 0..4 {
                        let mut m = Su3Matrix::ZERO;
                        for row in 0..3 {
                            for col in 0..3 {
                                let (re, im) = if meta.precision_bits == 32 {
                                    let mut buf4 = [0u8; 4];
                                    cursor.read_exact(&mut buf4)?;
                                    let re = f32::from_be_bytes(buf4) as f64;
                                    cursor.read_exact(&mut buf4)?;
                                    let im = f32::from_be_bytes(buf4) as f64;
                                    (re, im)
                                } else {
                                    let mut buf8 = [0u8; 8];
                                    cursor.read_exact(&mut buf8)?;
                                    let re = f64::from_be_bytes(buf8);
                                    cursor.read_exact(&mut buf8)?;
                                    let im = f64::from_be_bytes(buf8);
                                    (re, im)
                                };
                                m.m[row][col] = super::complex_f64::Complex64::new(re, im);
                            }
                        }
                        links[site_idx * 4 + mu] = m;
                    }
                    let _ = coords; // used for documentation clarity
                }
            }
        }
    }

    Ok(Lattice {
        dims: meta.dims,
        links,
        beta: meta.beta,
    })
}

/// Generate the `ildg-format` XML record content.
fn format_xml(meta: &IldgMetadata) -> String {
    let [nx, ny, nz, nt] = meta.dims;
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<ildgFormat xmlns="http://www.lqcd.org/ildg"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://www.lqcd.org/ildg http://www.lqcd.org/ildg/filefmt.xsd">
  <version>1.0</version>
  <field>su3gauge</field>
  <precision>{precision}</precision>
  <lx>{nx}</lx>
  <ly>{ny}</ly>
  <lz>{nz}</lz>
  <lt>{nt}</lt>
  <creator>{creator}</creator>
  <ensemble>{ensemble}</ensemble>
  <beta>{beta}</beta>
  <mass>{mass}</mass>
  <nf>{nf}</nf>
  <action>{action}</action>
  <fermion_action>{fermion_action}</fermion_action>
  <trajectory>{trajectory}</trajectory>
  <plaquette>{plaquette}</plaquette>
</ildgFormat>"#,
        precision = meta.precision_bits,
        creator = meta.creator,
        ensemble = meta.ensemble_id,
        beta = meta.beta,
        mass = meta.mass,
        nf = meta.nf,
        action = meta.action,
        fermion_action = meta.fermion_action,
        trajectory = meta.trajectory,
        plaquette = meta.plaquette,
    )
}

/// Parse the `ildg-format` XML to extract metadata.
///
/// Uses simple string matching rather than an XML parser dependency.
fn parse_format_xml(xml: &str, lfn: Option<&str>) -> io::Result<IldgMetadata> {
    let extract = |tag: &str| -> Option<String> {
        let open = format!("<{tag}>");
        let close = format!("</{tag}>");
        let start = xml.find(&open)? + open.len();
        let end = xml[start..].find(&close)? + start;
        Some(xml[start..end].trim().to_string())
    };

    let extract_f64 =
        |tag: &str| -> f64 { extract(tag).and_then(|s| s.parse().ok()).unwrap_or(0.0) };
    let extract_usize =
        |tag: &str| -> usize { extract(tag).and_then(|s| s.parse().ok()).unwrap_or(0) };

    let nx = extract_usize("lx");
    let ny = extract_usize("ly");
    let nz = extract_usize("lz");
    let nt = extract_usize("lt");

    if nx == 0 || ny == 0 || nz == 0 || nt == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid ILDG dimensions: {nx}x{ny}x{nz}x{nt}"),
        ));
    }

    let precision_bits = extract_usize("precision") as u32;
    let precision_bits = if precision_bits == 32 || precision_bits == 64 {
        precision_bits
    } else {
        64
    };

    Ok(IldgMetadata {
        dims: [nx, ny, nz, nt],
        beta: extract_f64("beta"),
        mass: extract_f64("mass"),
        nf: extract_usize("nf"),
        action: extract("action").unwrap_or_else(|| "Wilson".to_string()),
        fermion_action: extract("fermion_action").unwrap_or_else(|| "none".to_string()),
        creator: extract("creator").unwrap_or_default(),
        ensemble_id: extract("ensemble").unwrap_or_default(),
        trajectory: extract_usize("trajectory"),
        plaquette: extract_f64("plaquette"),
        precision_bits,
        lfn: lfn.unwrap_or_default().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_cold_lattice() {
        let lattice = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let meta = IldgMetadata::for_lattice(&lattice, 100);

        let mut buf = Vec::new();
        write_gauge_config(&mut buf, &lattice, &meta).unwrap();

        let (loaded, loaded_meta) = read_gauge_config(Cursor::new(&buf)).unwrap();

        assert_eq!(loaded.dims, lattice.dims);
        assert!((loaded.beta - lattice.beta).abs() < 1e-10);
        assert_eq!(loaded_meta.trajectory, 100);
        assert_eq!(loaded_meta.dims, [4, 4, 4, 4]);

        // Every link should be identity
        for idx in 0..loaded.volume() {
            let x = loaded.site_coords(idx);
            for mu in 0..4 {
                let u = loaded.link(x, mu);
                let diff = (u - Su3Matrix::IDENTITY).norm_sq();
                assert!(
                    diff < 1e-20,
                    "link at site {x:?} mu={mu} not identity: diff={diff}"
                );
            }
        }
    }

    #[test]
    fn roundtrip_hot_lattice() {
        let lattice = Lattice::hot_start([4, 6, 4, 8], 5.8, 42);
        let meta = IldgMetadata::for_lattice(&lattice, 500);

        let mut buf = Vec::new();
        write_gauge_config(&mut buf, &lattice, &meta).unwrap();

        let (loaded, loaded_meta) = read_gauge_config(Cursor::new(&buf)).unwrap();

        assert_eq!(loaded.dims, lattice.dims);
        assert!((loaded_meta.beta - 5.8).abs() < 1e-10);

        let plaq_orig = lattice.average_plaquette();
        let plaq_loaded = loaded.average_plaquette();
        assert!(
            (plaq_orig - plaq_loaded).abs() < 1e-12,
            "plaquette mismatch: orig={plaq_orig}, loaded={plaq_loaded}"
        );

        // Check every link matches
        for idx in 0..lattice.volume() {
            let x = lattice.site_coords(idx);
            for mu in 0..4 {
                let u_orig = lattice.link(x, mu);
                let u_loaded = loaded.link(x, mu);
                let diff = (u_orig - u_loaded).norm_sq();
                assert!(
                    diff < 1e-20,
                    "link mismatch at site {x:?} mu={mu}: diff={diff}"
                );
            }
        }
    }

    #[test]
    fn roundtrip_32bit_precision() {
        let lattice = Lattice::hot_start([4, 4, 4, 4], 6.0, 99);
        let mut meta = IldgMetadata::for_lattice(&lattice, 10);
        meta.precision_bits = 32;

        let mut buf = Vec::new();
        write_gauge_config(&mut buf, &lattice, &meta).unwrap();

        let (loaded, _) = read_gauge_config(Cursor::new(&buf)).unwrap();

        let plaq_orig = lattice.average_plaquette();
        let plaq_loaded = loaded.average_plaquette();
        assert!(
            (plaq_orig - plaq_loaded).abs() < 1e-5,
            "32-bit roundtrip plaquette: orig={plaq_orig}, loaded={plaq_loaded}"
        );
    }

    #[test]
    fn metadata_xml_roundtrip() {
        let meta = IldgMetadata {
            dims: [16, 16, 16, 32],
            beta: 6.0,
            mass: 0.1,
            nf: 2,
            action: "Wilson".to_string(),
            fermion_action: "staggered".to_string(),
            creator: "hotSpring-test".to_string(),
            ensemble_id: "test_ensemble".to_string(),
            trajectory: 42,
            plaquette: 0.598,
            precision_bits: 64,
            lfn: "/test/lfn".to_string(),
        };

        let xml = format_xml(&meta);
        let parsed = parse_format_xml(&xml, Some("/test/lfn")).unwrap();

        assert_eq!(parsed.dims, meta.dims);
        assert!((parsed.beta - meta.beta).abs() < 1e-10);
        assert!((parsed.mass - meta.mass).abs() < 1e-10);
        assert_eq!(parsed.nf, meta.nf);
        assert_eq!(parsed.action, "Wilson");
        assert_eq!(parsed.fermion_action, "staggered");
        assert_eq!(parsed.trajectory, 42);
        assert_eq!(parsed.precision_bits, 64);
    }

    #[test]
    fn file_size_correct() {
        let lattice = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let meta = IldgMetadata::for_lattice(&lattice, 0);

        let mut buf = Vec::new();
        write_gauge_config(&mut buf, &lattice, &meta).unwrap();

        let n_links = 4 * 4 * 4 * 4 * 4;
        let data_bytes = n_links * 18 * 8; // 18 f64 per link
        // Should have 3 LIME headers (144 each) + xml + binary data + lfn + padding
        assert!(
            buf.len() > data_bytes,
            "file too small: {} < {data_bytes}",
            buf.len()
        );
    }
}
