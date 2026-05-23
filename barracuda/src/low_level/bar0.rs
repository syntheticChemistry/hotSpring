// SPDX-License-Identifier: AGPL-3.0-or-later

// This file is shared across multiple experiment binaries via `#[path]` inclusion.
// Each binary uses a different subset of Bar0View / Bar0Map; suppress dead-code
// warnings that arise when the file is included by a binary that only needs one type.
#![allow(dead_code)]

//! Safe RAII wrappers for PCI BAR0 MMIO via Linux `sysfs` resource files.
//!
//! The entire unsafe surface is confined to this module:
//! - `mmap`/`munmap` via `rustix`
//! - `read_volatile`/`write_volatile` for MMIO register access
//!
//! All public methods are safe Rust with bounds-checked offsets.
//!
//! # Usage (requires `low-level` feature)
//!
//! ```ignore
//! let bar = Bar0View::open("0000:02:00.0")?;
//! let pmc = bar.read_u32(0x0000_0200)?;
//! ```

use std::io;

use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};

/// Default BAR0 mapping size (16 MiB). Overridden by actual resource file
/// size when available, so this serves as a fallback for platforms where
/// file metadata isn't reliable for mmap'd resources.
const BAR0_MAP_SIZE_DEFAULT: usize = 16 * 1024 * 1024;

/// Determine the actual BAR0 resource size from file metadata, falling back
/// to `BAR0_MAP_SIZE_DEFAULT` when metadata is unavailable.
fn bar0_map_size(file: &std::fs::File) -> usize {
    file.metadata()
        .map(|m| m.len() as usize)
        .filter(|&sz| sz > 0)
        .unwrap_or(BAR0_MAP_SIZE_DEFAULT)
}

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
    /// # Errors
    ///
    /// Returns an error string if `resource0` cannot be opened or mapped.
    pub fn open(bdf: &str) -> Result<Self, String> {
        let sysfs_base =
            std::env::var("HOTSPRING_SYSFS_PCI").unwrap_or_else(|_| "/sys/bus/pci/devices".into());
        let resource_path = format!("{sysfs_base}/{bdf}/resource0");
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
    /// Map `resource0` for the given PCI device path (e.g. from `/sys/bus/pci/devices/…`).
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

    /// Read a `u32` MMIO register at `offset` bytes from BAR0 base.
    ///
    /// # Panics
    ///
    /// Panics if `offset + 4 > mapping length`.
    #[must_use]
    pub fn r32(&self, offset: u32) -> u32 {
        assert!(
            (offset as usize + 4) <= self.len,
            "r32 out-of-bounds: {offset:#x}"
        );
        // SAFETY: bounds checked above; volatile prevents MMIO elision.
        unsafe { std::ptr::read_volatile(self.ptr.add(offset as usize).cast::<u32>()) }
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
    /// Panics if `offset + 4 > mapping length`.
    pub fn w32(&self, offset: u32, val: u32) {
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

/// Validates MMIO offsets against a set of allowed register domains before
/// forwarding to the underlying [`Bar0Map`].
///
/// Prevents accidental out-of-range MMIO from experiments that probe
/// GPU engines. Callers declare which domains they intend to touch at
/// construction time; any access outside those domains is rejected.
///
/// # memmap2 evaluation
///
/// `memmap2::MmapRaw` was evaluated as an alternative to raw `rustix::mm::mmap`.
/// It provides RAII but does not support `MAP_SHARED` on `/sys` resource files
/// reliably (it targets regular files). The current `rustix` approach is more
/// direct and already has RAII via `Drop`. No migration warranted.
pub struct SafeBar0 {
    inner: Bar0Map,
    domains: Vec<Bar0Domain>,
}

impl SafeBar0 {
    /// Wrap an existing [`Bar0Map`] with domain validation.
    pub fn new(inner: Bar0Map, domains: Vec<Bar0Domain>) -> Self {
        Self { inner, domains }
    }

    /// Open a BAR0 mapping with domain validation.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying mmap fails.
    pub fn open(path: &str, domains: Vec<Bar0Domain>) -> io::Result<Self> {
        let inner = Bar0Map::open(path)?;
        Ok(Self { inner, domains })
    }

    fn check_offset(&self, offset: u32, width: u32) -> Result<(), String> {
        let end = offset.checked_add(width).ok_or_else(|| {
            format!("offset {offset:#x} + {width} overflows")
        })?;
        for d in &self.domains {
            if offset >= d.start && end <= d.end {
                return Ok(());
            }
        }
        let domain_list: Vec<&str> = self.domains.iter().map(|d| d.name).collect();
        Err(format!(
            "offset {offset:#x}..{end:#x} outside allowed domains: {domain_list:?}"
        ))
    }

    /// Read a `u32` register, validating the offset is within an allowed domain.
    ///
    /// # Errors
    ///
    /// Returns an error string if the offset is outside all allowed domains.
    pub fn r32(&self, offset: u32) -> Result<u32, String> {
        self.check_offset(offset, 4)?;
        Ok(self.inner.r32(offset))
    }

    /// Write a `u32` register, validating the offset is within an allowed domain.
    ///
    /// # Errors
    ///
    /// Returns an error string if the offset is outside all allowed domains.
    pub fn w32(&self, offset: u32, val: u32) -> Result<(), String> {
        self.check_offset(offset, 4)?;
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
