// SPDX-License-Identifier: AGPL-3.0-or-later

// Suppress dead-code warnings when included via `#[path]` by experiment binaries
// that only use a subset of Bar0View / Bar0Map / SafeBar0.
#![allow(dead_code)]

//! Safe RAII wrappers for PCI BAR0 MMIO via Linux `sysfs` resource files.
//!
//! The entire unsafe surface is confined to this module:
//! - `mmap`/`munmap` via `rustix`
//! - `read_volatile`/`write_volatile` for MMIO register access
//!
//! All public methods are safe Rust with bounds-checked, alignment-checked offsets.
//!
//! # Usage (requires `low-level` feature)
//!
//! ```ignore
//! let bar = Bar0View::open("0000:02:00.0")?;
//! let pmc = bar.read_u32(0x0000_0200)?;
//! ```

use std::fmt;
use std::io;

use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};

/// Default BAR0 mapping size (16 MiB). Overridden by actual resource file
/// size when available, so this serves as a fallback for platforms where
/// file metadata isn't reliable for mmap'd resources.
const BAR0_MAP_SIZE_DEFAULT: usize = 16 * 1024 * 1024;

/// Determine the actual BAR0 resource size from file metadata, falling back
/// to `BAR0_MAP_SIZE_DEFAULT` when metadata is unavailable.
pub(crate) fn bar0_map_size(file: &std::fs::File) -> usize {
    file.metadata()
        .map(|m| m.len() as usize)
        .ok()
        .filter(|&sz| sz > 0)
        .unwrap_or(BAR0_MAP_SIZE_DEFAULT)
}

/// Resolve a BDF string to its `resource0` sysfs path.
///
/// Supports `HOTSPRING_SYSFS_PCI` override (for mock testing).
fn bdf_to_resource0(bdf: &str) -> String {
    let sysfs_base =
        std::env::var("HOTSPRING_SYSFS_PCI").unwrap_or_else(|_| "/sys/bus/pci/devices".into());
    format!("{sysfs_base}/{bdf}/resource0")
}

// ── Bar0Error ────────────────────────────────────────────────────────────────

/// Errors from checked BAR0 access.
#[derive(Debug, Clone)]
pub enum Bar0Error {
    /// PCIe link is down — all reads return `0xFFFF_FFFF`.
    DeadLink { offset: u32 },
    /// Register offset is not 4-byte aligned.
    Unaligned { offset: u32 },
    /// Register offset falls outside all allowed domains.
    OutOfDomain { offset: u32, end: u32, domains: Vec<&'static str> },
    /// Write to a deny-listed register offset was rejected.
    DenyListed { offset: u32, reason: &'static str },
    /// Offset arithmetic overflow.
    Overflow { offset: u32, width: u32 },
    /// Offset exceeds the mapped region.
    OutOfBounds { offset: u32, map_len: usize },
}

impl fmt::Display for Bar0Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeadLink { offset } =>
                write!(f, "BAR0 dead link at {offset:#x} (0xFFFFFFFF)"),
            Self::Unaligned { offset } =>
                write!(f, "BAR0 offset {offset:#x} is not 4-byte aligned"),
            Self::OutOfDomain { offset, end, domains } =>
                write!(f, "offset {offset:#x}..{end:#x} outside allowed domains: {domains:?}"),
            Self::DenyListed { offset, reason } =>
                write!(f, "write to {offset:#x} denied: {reason}"),
            Self::Overflow { offset, width } =>
                write!(f, "offset {offset:#x} + {width} overflows"),
            Self::OutOfBounds { offset, map_len } =>
                write!(f, "offset {offset:#x} out of BAR0 range (len={map_len:#x})"),
        }
    }
}

impl std::error::Error for Bar0Error {}

// ── Read-only MMIO ────────────────────────────────────────────────────────────

/// Read-only RAII mapping of a PCI BAR0 `resource0` file.
///
/// Provides bounds-checked volatile register reads. Released via `munmap` on drop.
pub struct Bar0View {
    base: *const u8,
    len: usize,
}

impl Bar0View {
    /// Map `resource0` for the given PCI device (e.g. `"0000:02:00.0"`).
    ///
    /// Respects `HOTSPRING_SYSFS_PCI` environment variable for the sysfs base.
    ///
    /// # Errors
    ///
    /// Returns an error string if `resource0` cannot be opened or mapped.
    pub fn open(bdf: &str) -> Result<Self, String> {
        let resource_path = bdf_to_resource0(bdf);
        let file = std::fs::File::options()
            .read(true)
            .open(&resource_path)
            .map_err(|e| format!("cannot open {resource_path}: {e}"))?;

        let map_size = bar0_map_size(&file);

        // SAFETY: `resource0` is a kernel-exported PCI BAR mmap. READ+SHARED
        // mirrors hardware registers without writeback. The kernel guarantees
        // coverage of at least `map_size` bytes.
        let mm = unsafe {
            mmap(
                std::ptr::null_mut(),
                map_size,
                ProtFlags::READ,
                MapFlags::SHARED,
                &file,
                0,
            )
        }
        .map_err(|_| format!("mmap of {resource_path} failed"))?;

        Ok(Self {
            base: mm.cast::<u8>(),
            len: map_size,
        })
    }

    /// Read a little-endian `u32` register at `offset` bytes from BAR0 base.
    ///
    /// # Errors
    ///
    /// Returns an error if `offset + 4 > mapping length`.
    pub fn read_u32(&self, offset: u32) -> Result<u32, String> {
        let end = (offset as usize).saturating_add(4);
        if end > self.len {
            return Err(format!(
                "offset {offset:#x} out of BAR0 range (len={:#x})",
                self.len
            ));
        }
        // SAFETY: `end <= self.len`; `self.base` valid for `self.len` bytes;
        // `read_volatile` prevents MMIO read elision or reordering.
        let bytes = unsafe {
            let ptr = self.base.add(offset as usize);
            [
                std::ptr::read_volatile(ptr),
                std::ptr::read_volatile(ptr.add(1)),
                std::ptr::read_volatile(ptr.add(2)),
                std::ptr::read_volatile(ptr.add(3)),
            ]
        };
        Ok(u32::from_le_bytes(bytes))
    }
}

impl Drop for Bar0View {
    fn drop(&mut self) {
        // SAFETY: `self.base` was returned by `mmap` with `self.len`.
        if unsafe { munmap(self.base.cast_mut().cast(), self.len) }.is_err() {
            log::warn!("Bar0View munmap failed");
        }
    }
}

// SAFETY: BAR0View holds an immutable mmap of a PCI BAR0 resource file.
// The kernel guarantees physical memory backing is stable for the file's
// lifetime. No interior mutability: concurrent reads of MMIO registers
// are naturally idempotent (hardware provides the ordering guarantees via
// volatile reads). Ownership transfer between threads is safe because:
// 1. The mapping is read-only — no data races on write.
// 2. Drop calls munmap exactly once (RAII, not reference-counted).
// 3. No pointers alias outside this struct.
unsafe impl Send for Bar0View {}

// ── Read-write MMIO ──────────────────────────────────────────────────────────

/// Read-write RAII mapping of a PCI BAR0 `resource0` file.
///
/// Enables register writes (e.g. engine init, PRAMIN windows) in addition to reads.
/// Use [`Bar0View`] when writes are not needed — it is more restrictive and safer.
pub struct Bar0Map {
    ptr: *mut u8,
    len: usize,
}

impl Bar0Map {
    /// Map `resource0` from a raw sysfs path.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the file cannot be opened or the mmap fails.
    pub fn open(path: &str) -> io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;
        let map_size = bar0_map_size(&file);
        // SAFETY: `resource0` is a kernel PCI BAR file. READ|WRITE|SHARED
        // gives direct MMIO access. Caller must ensure no concurrent driver access.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                map_size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                &file,
                0,
            )
            .map_err(io::Error::from)?
        };
        Ok(Self {
            ptr: ptr.cast(),
            len: map_size,
        })
    }

    /// Map `resource0` for a PCI BDF (e.g. `"0000:02:00.0"`).
    ///
    /// Respects `HOTSPRING_SYSFS_PCI` environment variable for the sysfs base.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the file cannot be opened or the mmap fails.
    pub fn open_bdf(bdf: &str) -> io::Result<Self> {
        Self::open(&bdf_to_resource0(bdf))
    }

    /// Read a `u32` MMIO register at `offset` bytes from BAR0 base.
    ///
    /// # Panics
    ///
    /// Panics if `offset` is not 4-byte aligned or `offset + 4 > mapping length`.
    #[must_use]
    pub fn r32(&self, offset: u32) -> u32 {
        assert!(offset % 4 == 0, "r32: offset {offset:#x} is not 4-byte aligned");
        assert!(
            (offset as usize + 4) <= self.len,
            "r32 out-of-bounds: {offset:#x}"
        );
        // SAFETY: bounds checked above; volatile prevents MMIO elision.
        unsafe { std::ptr::read_volatile(self.ptr.add(offset as usize).cast::<u32>()) }
    }

    /// Read a `u32` MMIO register with dead-link detection.
    ///
    /// Returns `Err(Bar0Error::DeadLink)` if the read value is `0xFFFF_FFFF`
    /// (PCIe link down sentinel).
    ///
    /// # Errors
    ///
    /// - `Bar0Error::Unaligned` if offset is not 4-byte aligned.
    /// - `Bar0Error::OutOfBounds` if offset exceeds the mapping.
    /// - `Bar0Error::DeadLink` if the read returns the dead-link sentinel.
    pub fn r32_checked(&self, offset: u32) -> Result<u32, Bar0Error> {
        if offset % 4 != 0 {
            return Err(Bar0Error::Unaligned { offset });
        }
        if (offset as usize + 4) > self.len {
            return Err(Bar0Error::OutOfBounds { offset, map_len: self.len });
        }
        // SAFETY: bounds and alignment checked above.
        let val = unsafe {
            std::ptr::read_volatile(self.ptr.add(offset as usize).cast::<u32>())
        };
        if val == 0xFFFF_FFFF {
            return Err(Bar0Error::DeadLink { offset });
        }
        Ok(val)
    }

    /// Return the total size of the mapped region in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Write a `u32` MMIO register at `offset` bytes from BAR0 base.
    ///
    /// # Panics
    ///
    /// Panics if `offset` is not 4-byte aligned or `offset + 4 > mapping length`.
    pub fn w32(&self, offset: u32, val: u32) {
        assert!(offset % 4 == 0, "w32: offset {offset:#x} is not 4-byte aligned");
        assert!(
            (offset as usize + 4) <= self.len,
            "w32 out-of-bounds: {offset:#x}"
        );
        // SAFETY: bounds checked above; volatile prevents MMIO store coalescing.
        unsafe { std::ptr::write_volatile(self.ptr.add(offset as usize).cast::<u32>(), val) }
    }
}

impl Drop for Bar0Map {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was returned by `mmap` with `self.len`.
        if unsafe { munmap(self.ptr.cast(), self.len) }.is_err() {
            log::warn!("Bar0Map munmap failed");
        }
    }
}

// SAFETY: Bar0Map holds a read-write mmap of a PCI BAR0 resource file.
// Ownership transfer between threads is safe because:
// 1. Writes go through volatile stores — no Rust-level data races.
// 2. MMIO ordering is guaranteed by the hardware (PCIe posted writes
//    are serialized by the device's BAR decoder).
// 3. Drop calls munmap exactly once (RAII).
// 4. Callers must ensure no concurrent kernel driver writes to the
//    same BAR0 region (enforced by VFIO exclusion at the OS level).
unsafe impl Send for Bar0Map {}

// ── Domain-validated wrapper ─────────────────────────────────────────────────

/// A named BAR0 offset range (e.g. PMC, GR, FECS).
#[derive(Debug, Clone, Copy)]
pub struct Bar0Domain {
    pub name: &'static str,
    pub start: u32,
    pub end: u32,
}

/// A deny-listed absolute offset with a human-readable reason.
#[derive(Debug, Clone, Copy)]
pub struct DenyEntry {
    pub offset: u32,
    pub reason: &'static str,
}

/// Validates MMIO offsets against a set of allowed register domains before
/// forwarding to the underlying [`Bar0Map`].
///
/// Prevents accidental out-of-range MMIO from experiments that probe
/// GPU engines. Callers declare which domains they intend to touch at
/// construction time; any access outside those domains is rejected.
///
/// Optionally enforces a **deny-list** of absolute offsets where writes
/// are always rejected (e.g. `ENGCTL` registers that irreversibly destroy
/// falcon security state).
pub struct SafeBar0 {
    inner: Bar0Map,
    domains: Vec<Bar0Domain>,
    deny_list: Vec<DenyEntry>,
}

impl SafeBar0 {
    /// Wrap an existing [`Bar0Map`] with domain validation.
    pub fn new(inner: Bar0Map, domains: Vec<Bar0Domain>) -> Self {
        Self { inner, domains, deny_list: Vec::new() }
    }

    /// Wrap an existing [`Bar0Map`] with domain validation and a write deny-list.
    pub fn with_deny_list(
        inner: Bar0Map,
        domains: Vec<Bar0Domain>,
        deny_list: Vec<DenyEntry>,
    ) -> Self {
        Self { inner, domains, deny_list }
    }

    /// Open a BAR0 mapping for a BDF with domain validation.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying mmap fails.
    pub fn open(bdf: &str, domains: Vec<Bar0Domain>) -> io::Result<Self> {
        let inner = Bar0Map::open_bdf(bdf)?;
        Ok(Self { inner, domains, deny_list: Vec::new() })
    }

    /// Open a BAR0 mapping for a BDF with domain validation and a deny-list.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying mmap fails.
    pub fn open_with_deny_list(
        bdf: &str,
        domains: Vec<Bar0Domain>,
        deny_list: Vec<DenyEntry>,
    ) -> io::Result<Self> {
        let inner = Bar0Map::open_bdf(bdf)?;
        Ok(Self { inner, domains, deny_list })
    }

    /// Check that `offset..offset+width` falls within at least one domain.
    pub fn check_offset(&self, offset: u32, width: u32) -> Result<(), Bar0Error> {
        if offset % 4 != 0 {
            return Err(Bar0Error::Unaligned { offset });
        }
        let end = offset.checked_add(width).ok_or(Bar0Error::Overflow { offset, width })?;
        for d in &self.domains {
            if offset >= d.start && end <= d.end {
                return Ok(());
            }
        }
        let domain_list: Vec<&str> = self.domains.iter().map(|d| d.name).collect();
        Err(Bar0Error::OutOfDomain { offset, end, domains: domain_list })
    }

    /// Check whether `offset` is on the write deny-list.
    fn check_deny_list(&self, offset: u32) -> Result<(), Bar0Error> {
        for entry in &self.deny_list {
            if offset == entry.offset {
                return Err(Bar0Error::DenyListed { offset, reason: entry.reason });
            }
        }
        Ok(())
    }

    /// Read a `u32` register, validating the offset is within an allowed domain.
    ///
    /// # Errors
    ///
    /// Returns a `Bar0Error` if the offset is outside all allowed domains or misaligned.
    pub fn r32(&self, offset: u32) -> Result<u32, Bar0Error> {
        self.check_offset(offset, 4)?;
        Ok(self.inner.r32(offset))
    }

    /// Read a `u32` register with domain validation and dead-link detection.
    ///
    /// # Errors
    ///
    /// Returns `Bar0Error::DeadLink` if the read returns `0xFFFF_FFFF`.
    pub fn r32_checked(&self, offset: u32) -> Result<u32, Bar0Error> {
        self.check_offset(offset, 4)?;
        self.inner.r32_checked(offset)
    }

    /// Write a `u32` register, validating domain and deny-list.
    ///
    /// # Errors
    ///
    /// Returns a `Bar0Error` if the offset is outside all allowed domains,
    /// misaligned, or on the write deny-list.
    pub fn w32(&self, offset: u32, val: u32) -> Result<(), Bar0Error> {
        self.check_offset(offset, 4)?;
        self.check_deny_list(offset)?;
        self.inner.w32(offset, val);
        Ok(())
    }

    /// Total mapping size in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Access the underlying [`Bar0Map`] directly (bypassing domain checks).
    #[must_use]
    pub fn inner(&self) -> &Bar0Map {
        &self.inner
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_domains() -> Vec<Bar0Domain> {
        vec![
            Bar0Domain { name: "PMC", start: 0x0000, end: 0x1000 },
            Bar0Domain { name: "PMU", start: 0x10_A000, end: 0x10_A400 },
        ]
    }

    fn make_safe(domains: Vec<Bar0Domain>, deny_list: Vec<DenyEntry>) -> SafeBar0 {
        // SafeBar0 with a dummy Bar0Map — only tests check_offset / check_deny_list
        // which don't touch the mmap at all. We construct directly to avoid needing
        // actual hardware.
        SafeBar0 {
            inner: unsafe { std::mem::zeroed() },
            domains,
            deny_list,
        }
    }

    #[test]
    fn check_offset_in_domain_passes() {
        let s = make_safe(test_domains(), vec![]);
        assert!(s.check_offset(0x0200, 4).is_ok());
        assert!(s.check_offset(0x0000, 4).is_ok());
        assert!(s.check_offset(0x0FFC, 4).is_ok());
        assert!(s.check_offset(0x10_A000, 4).is_ok());
        assert!(s.check_offset(0x10_A3C0, 4).is_ok());
    }

    #[test]
    fn check_offset_spanning_boundary_fails() {
        let s = make_safe(test_domains(), vec![]);
        // PMC domain ends at 0x1000; offset 0x0FFC + 8 = 0x1004 spans past end
        let err = s.check_offset(0x0FFC, 8);
        assert!(matches!(err, Err(Bar0Error::OutOfDomain { .. })));
    }

    #[test]
    fn check_offset_overflow_fails() {
        let s = make_safe(test_domains(), vec![]);
        let err = s.check_offset(0xFFFF_FFFC, 8);
        assert!(matches!(err, Err(Bar0Error::Overflow { .. })));
    }

    #[test]
    fn check_offset_empty_domains_reject() {
        let s = make_safe(vec![], vec![]);
        let err = s.check_offset(0x0200, 4);
        assert!(matches!(err, Err(Bar0Error::OutOfDomain { .. })));
    }

    #[test]
    fn alignment_rejection() {
        let s = make_safe(test_domains(), vec![]);
        let err = s.check_offset(0x0201, 4);
        assert!(matches!(err, Err(Bar0Error::Unaligned { offset: 0x0201 })));

        let err = s.check_offset(0x0202, 4);
        assert!(matches!(err, Err(Bar0Error::Unaligned { offset: 0x0202 })));
    }

    #[test]
    fn deny_list_rejects_write() {
        let deny = vec![DenyEntry {
            offset: 0x10_A3C0,
            reason: "ENGCTL destroys falcon security state",
        }];
        let s = make_safe(test_domains(), deny);
        let err = s.check_deny_list(0x10_A3C0);
        assert!(matches!(err, Err(Bar0Error::DenyListed { offset: 0x10_A3C0, .. })));

        // Other offsets pass
        assert!(s.check_deny_list(0x10_A100).is_ok());
    }

    #[test]
    fn bar0_map_size_zero_returns_default() {
        let tmp = std::env::temp_dir().join("bar0_test_empty");
        {
            let _ = std::fs::File::create(&tmp).unwrap();
        }
        let file = std::fs::File::open(&tmp).unwrap();
        assert_eq!(bar0_map_size(&file), BAR0_MAP_SIZE_DEFAULT);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn bar0_map_size_nonzero() {
        use std::io::Write;
        let tmp = std::env::temp_dir().join("bar0_test_nonzero");
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            f.write_all(&[0u8; 4096]).unwrap();
        }
        let file = std::fs::File::open(&tmp).unwrap();
        assert_eq!(bar0_map_size(&file), 4096);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn bar0_error_display() {
        let e = Bar0Error::DeadLink { offset: 0x200 };
        assert!(format!("{e}").contains("dead link"));
        assert!(format!("{e}").contains("0x200"));

        let e = Bar0Error::Unaligned { offset: 0x201 };
        assert!(format!("{e}").contains("not 4-byte aligned"));

        let e = Bar0Error::DenyListed { offset: 0x3C0, reason: "ENGCTL" };
        assert!(format!("{e}").contains("denied"));
    }
}
