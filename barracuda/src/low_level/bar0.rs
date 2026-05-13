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

// SAFETY: BAR0 is a hardware-memory-mapped region; Send is appropriate for
// experiment binaries that transfer ownership between threads.
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

// SAFETY: same rationale as Bar0View.
unsafe impl Send for Bar0Map {}
