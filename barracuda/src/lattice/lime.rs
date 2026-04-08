// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure-Rust LIME (Lattice Interchange Message Encapsulation) I/O.
//!
//! LIME is the binary container format used by the ILDG (International
//! Lattice Data Grid) for gauge configuration interchange. Each LIME file
//! contains a sequence of records, each with a 144-byte header followed by
//! a payload padded to an 8-byte boundary.
//!
//! This implementation follows the LIME 1.2 specification:
//!   <https://usqcd-software.github.io/c-lime/lime_1p2.pdf>
//!
//! No C FFI or external dependencies — pure Rust byte-level I/O.
//!
//! # Wire format
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │ Record Header (144 bytes)                            │
//! │  magic:   u32 = 0x456789ab (big-endian)              │
//! │  version: u16 (big-endian)                           │
//! │  flags:   u16 (MB=bit15, ME=bit14)                   │
//! │  length:  u64 (big-endian, payload byte count)       │
//! │  type:    [u8; 128] (NUL-padded ASCII string)        │
//! ├──────────────────────────────────────────────────────┤
//! │ Payload (length bytes)                               │
//! ├──────────────────────────────────────────────────────┤
//! │ Padding (0-7 NUL bytes, total = ceil8(length))       │
//! └──────────────────────────────────────────────────────┘
//! ```

use std::io::{self, Read, Seek, Write};

/// LIME magic number: `0x456789ab`.
pub const LIME_MAGIC: u32 = 0x456789ab;

/// Current LIME version we write.
pub const LIME_VERSION: u16 = 1;

/// Header size in bytes.
pub const LIME_HEADER_SIZE: usize = 144;

/// Maximum record type string length (NUL-padded to 128 bytes).
pub const LIME_TYPE_MAX: usize = 128;

/// Flag: Message Begin — this record starts a new LIME message.
const FLAG_MB: u16 = 1 << 15;

/// Flag: Message End — this record ends the current LIME message.
const FLAG_ME: u16 = 1 << 14;

/// A parsed LIME record header.
#[derive(Clone, Debug)]
pub struct LimeHeader {
    /// LIME version from the header.
    pub version: u16,
    /// Whether this record starts a new message (MB flag).
    pub message_begin: bool,
    /// Whether this record ends the current message (ME flag).
    pub message_end: bool,
    /// Payload byte count (before padding).
    pub data_length: u64,
    /// Record type string (ASCII, up to 128 bytes, NUL-terminated).
    pub record_type: String,
}

/// A complete LIME record (header + payload bytes).
#[derive(Clone, Debug)]
pub struct LimeRecord {
    /// Parsed header.
    pub header: LimeHeader,
    /// Raw payload bytes.
    pub data: Vec<u8>,
}

/// Reads LIME records from a byte stream.
pub struct LimeReader<R: Read> {
    reader: R,
    finished: bool,
}

impl<R: Read> LimeReader<R> {
    /// Wrap a reader (file, cursor, etc.) as a LIME record stream.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            finished: false,
        }
    }

    /// Read the next LIME record, or `None` at EOF.
    pub fn next_record(&mut self) -> io::Result<Option<LimeRecord>> {
        if self.finished {
            return Ok(None);
        }

        let mut hdr_buf = [0u8; LIME_HEADER_SIZE];
        match self.reader.read_exact(&mut hdr_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                self.finished = true;
                return Ok(None);
            }
            Err(e) => return Err(e),
        }

        let header = parse_header(&hdr_buf)?;

        let mut data = vec![0u8; header.data_length as usize];
        self.reader.read_exact(&mut data)?;

        let padding = pad_to_8(header.data_length) as usize;
        if padding > 0 {
            let mut pad = vec![0u8; padding];
            self.reader.read_exact(&mut pad)?;
        }

        Ok(Some(LimeRecord { header, data }))
    }

    /// Consume the reader, returning all records.
    pub fn read_all(mut self) -> io::Result<Vec<LimeRecord>> {
        let mut records = Vec::new();
        while let Some(rec) = self.next_record()? {
            records.push(rec);
        }
        Ok(records)
    }
}

/// Writes LIME records to a byte stream.
pub struct LimeWriter<W: Write> {
    writer: W,
    in_message: bool,
}

impl<W: Write> LimeWriter<W> {
    /// Wrap a writer (file, Vec<u8>, etc.) as a LIME record sink.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            in_message: false,
        }
    }

    /// Write a single-record message (MB=1, ME=1).
    ///
    /// Most ILDG files use single-record messages for each logical record.
    pub fn write_record(&mut self, record_type: &str, data: &[u8]) -> io::Result<()> {
        self.write_raw(record_type, data, true, true)
    }

    /// Start a multi-record message (MB=1, ME=0).
    pub fn begin_message(&mut self, record_type: &str, data: &[u8]) -> io::Result<()> {
        self.write_raw(record_type, data, true, false)?;
        self.in_message = true;
        Ok(())
    }

    /// Continue a multi-record message (MB=0, ME=0).
    pub fn continue_message(&mut self, record_type: &str, data: &[u8]) -> io::Result<()> {
        self.write_raw(record_type, data, false, false)
    }

    /// End a multi-record message (MB=0, ME=1).
    pub fn end_message(&mut self, record_type: &str, data: &[u8]) -> io::Result<()> {
        self.write_raw(record_type, data, false, true)?;
        self.in_message = false;
        Ok(())
    }

    /// Flush the underlying writer.
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    /// Consume the writer, returning the inner writer.
    pub fn into_inner(self) -> W {
        self.writer
    }

    fn write_raw(
        &mut self,
        record_type: &str,
        data: &[u8],
        mb: bool,
        me: bool,
    ) -> io::Result<()> {
        let mut hdr = [0u8; LIME_HEADER_SIZE];
        let data_len = data.len() as u64;

        // Magic
        hdr[0..4].copy_from_slice(&LIME_MAGIC.to_be_bytes());
        // Version
        hdr[4..6].copy_from_slice(&LIME_VERSION.to_be_bytes());
        // Flags
        let flags = if mb { FLAG_MB } else { 0 } | if me { FLAG_ME } else { 0 };
        hdr[6..8].copy_from_slice(&flags.to_be_bytes());
        // Data length
        hdr[8..16].copy_from_slice(&data_len.to_be_bytes());
        // Record type (NUL-padded)
        let type_bytes = record_type.as_bytes();
        let copy_len = type_bytes.len().min(LIME_TYPE_MAX - 1);
        hdr[16..16 + copy_len].copy_from_slice(&type_bytes[..copy_len]);

        self.writer.write_all(&hdr)?;
        self.writer.write_all(data)?;

        let padding = pad_to_8(data_len) as usize;
        if padding > 0 {
            self.writer.write_all(&vec![0u8; padding])?;
        }

        Ok(())
    }
}

impl<W: Write + Seek> LimeWriter<W> {
    /// Returns the current byte position in the output stream.
    pub fn position(&mut self) -> io::Result<u64> {
        self.writer.seek(io::SeekFrom::Current(0))
    }
}

fn parse_header(buf: &[u8; LIME_HEADER_SIZE]) -> io::Result<LimeHeader> {
    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != LIME_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad LIME magic: 0x{magic:08x} (expected 0x{LIME_MAGIC:08x})"),
        ));
    }

    let version = u16::from_be_bytes([buf[4], buf[5]]);
    let flags = u16::from_be_bytes([buf[6], buf[7]]);
    let data_length = u64::from_be_bytes([buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15]]);

    let type_slice = &buf[16..LIME_HEADER_SIZE];
    let nul_pos = type_slice.iter().position(|&b| b == 0).unwrap_or(LIME_TYPE_MAX);
    let record_type = String::from_utf8_lossy(&type_slice[..nul_pos]).to_string();

    Ok(LimeHeader {
        version,
        message_begin: flags & FLAG_MB != 0,
        message_end: flags & FLAG_ME != 0,
        data_length,
        record_type,
    })
}

/// Compute padding bytes needed to reach the next 8-byte boundary.
const fn pad_to_8(len: u64) -> u64 {
    let rem = len % 8;
    if rem == 0 { 0 } else { 8 - rem }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_single_record() {
        let payload = b"Hello, LIME!";
        let mut buf = Vec::new();
        {
            let mut w = LimeWriter::new(&mut buf);
            w.write_record("test-type", payload).unwrap();
            w.flush().unwrap();
        }

        let reader = LimeReader::new(buf.as_slice());
        let records = reader.read_all().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].header.record_type, "test-type");
        assert!(records[0].header.message_begin);
        assert!(records[0].header.message_end);
        assert_eq!(records[0].data, payload);
    }

    #[test]
    fn roundtrip_multiple_records() {
        let mut buf = Vec::new();
        {
            let mut w = LimeWriter::new(&mut buf);
            w.write_record("type-a", b"first").unwrap();
            w.write_record("type-b", b"second-longer-payload").unwrap();
            w.write_record("type-c", &[0u8; 0]).unwrap();
            w.flush().unwrap();
        }

        let records = LimeReader::new(buf.as_slice()).read_all().unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].header.record_type, "type-a");
        assert_eq!(records[0].data, b"first");
        assert_eq!(records[1].header.record_type, "type-b");
        assert_eq!(records[1].data, b"second-longer-payload");
        assert_eq!(records[2].header.record_type, "type-c");
        assert!(records[2].data.is_empty());
    }

    #[test]
    fn header_size_is_144() {
        assert_eq!(LIME_HEADER_SIZE, 144);
        assert_eq!(4 + 2 + 2 + 8 + 128, 144);
    }

    #[test]
    fn padding_calculation() {
        assert_eq!(pad_to_8(0), 0);
        assert_eq!(pad_to_8(1), 7);
        assert_eq!(pad_to_8(7), 1);
        assert_eq!(pad_to_8(8), 0);
        assert_eq!(pad_to_8(9), 7);
        assert_eq!(pad_to_8(16), 0);
    }

    #[test]
    fn roundtrip_binary_payload() {
        let payload: Vec<u8> = (0..255).collect();
        let mut buf = Vec::new();
        {
            let mut w = LimeWriter::new(&mut buf);
            w.write_record("binary-data", &payload).unwrap();
        }
        let records = LimeReader::new(buf.as_slice()).read_all().unwrap();
        assert_eq!(records[0].data, payload);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(&0xDEADBEEFu32.to_be_bytes());
        let mut reader = LimeReader::new(buf.as_slice());
        assert!(reader.next_record().is_err());
    }

    #[test]
    fn multi_record_message() {
        let mut buf = Vec::new();
        {
            let mut w = LimeWriter::new(&mut buf);
            w.begin_message("ildg-format", b"<xml/>").unwrap();
            w.end_message("ildg-binary-data", b"\x00\x01\x02").unwrap();
        }
        let records = LimeReader::new(buf.as_slice()).read_all().unwrap();
        assert_eq!(records.len(), 2);
        assert!(records[0].header.message_begin);
        assert!(!records[0].header.message_end);
        assert!(!records[1].header.message_begin);
        assert!(records[1].header.message_end);
    }
}
