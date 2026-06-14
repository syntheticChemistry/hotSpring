// SPDX-License-Identifier: AGPL-3.0-or-later

//! Minimal zero-dependency base64 codec (RFC 4648 standard alphabet).
//!
//! Replaces the external `base64` crate with ~50 lines of inline encode/decode.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode bytes to standard base64 with `=` padding.
pub fn encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Decode standard base64 (with or without `=` padding) to bytes.
pub fn decode(input: &[u8]) -> Result<Vec<u8>, Base64Error> {
    const INV: [u8; 256] = {
        let mut t = [0xFFu8; 256];
        let alpha = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut i = 0;
        while i < 64 {
            t[alpha[i] as usize] = i as u8;
            i += 1;
        }
        t
    };

    let trimmed: Vec<u8> = input
        .iter()
        .copied()
        .filter(|&b| b != b'=' && !b.is_ascii_whitespace())
        .collect();
    let mut out = Vec::with_capacity(trimmed.len() * 3 / 4);

    for quad in trimmed.chunks(4) {
        let vals: Vec<u8> = quad
            .iter()
            .map(|&b| {
                let v = INV[b as usize];
                if v == 0xFF {
                    Err(Base64Error(b))
                } else {
                    Ok(v)
                }
            })
            .collect::<Result<_, _>>()?;

        let n = vals.len();
        let triple = ((vals[0] as u32) << 18)
            | (vals.get(1).copied().unwrap_or(0) as u32 * (1 << 12))
            | (vals.get(2).copied().unwrap_or(0) as u32 * (1 << 6))
            | vals.get(3).copied().unwrap_or(0) as u32;

        out.push((triple >> 16) as u8);
        if n > 2 {
            out.push((triple >> 8) as u8);
        }
        if n > 3 {
            out.push(triple as u8);
        }
    }
    Ok(out)
}

/// Error returned when base64 input contains invalid characters.
#[derive(Debug)]
pub struct Base64Error(pub u8);

impl std::fmt::Display for Base64Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid base64 byte: 0x{:02X}", self.0)
    }
}

impl std::error::Error for Base64Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(encode(b"hello"), "aGVsbG8=");
        assert_eq!(encode(b""), "");
        assert_eq!(encode(b"a"), "YQ==");
        assert_eq!(encode(b"ab"), "YWI=");
        assert_eq!(encode(b"abc"), "YWJj");
    }

    #[test]
    fn roundtrip() {
        for input in &[
            b"" as &[u8],
            b"a",
            b"ab",
            b"abc",
            b"hello world",
            b"\x00\xFF\x80",
        ] {
            let encoded = encode(input);
            let decoded = decode(encoded.as_bytes()).unwrap();
            assert_eq!(&decoded, input, "roundtrip failed for {:?}", input);
        }
    }

    #[test]
    fn invalid_byte() {
        assert!(decode(b"!!!").is_err());
    }
}
