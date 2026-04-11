// SPDX-License-Identifier: AGPL-3.0-or-later

//! CLI parsing and human-readable formatting for lattice geometry flags.

/// Parse lattice dimensions from CLI arguments.
///
/// Supports three formats (checked in priority order):
///   - `--dims=Nx,Ny,Nz,Nt` — fully specified
///   - `--ns=Ns` + optional `--nt=Nt` — cubic spatial with optional temporal override
///   - `--lattice=N` — isotropic N^4
///
/// Returns `None` if no geometry flag is found (caller should use its default).
pub fn parse_dims_from_args(args: &[String]) -> Option<[usize; 4]> {
    let mut dims_val: Option<String> = None;
    let mut ns_val: Option<usize> = None;
    let mut nt_val: Option<usize> = None;
    let mut lattice_val: Option<usize> = None;

    for arg in args {
        if let Some(v) = arg.strip_prefix("--dims=") {
            dims_val = Some(v.to_string());
        } else if let Some(v) = arg.strip_prefix("--ns=") {
            ns_val = v.parse().ok();
        } else if let Some(v) = arg.strip_prefix("--nt=") {
            nt_val = v.parse().ok();
        } else if let Some(v) = arg.strip_prefix("--lattice=") {
            lattice_val = v.parse().ok();
        }
    }

    if let Some(d) = dims_val {
        let parts: Vec<usize> = d.split(',').filter_map(|s| s.parse().ok()).collect();
        if parts.len() == 4 {
            return Some([parts[0], parts[1], parts[2], parts[3]]);
        }
        eprintln!("error: --dims requires exactly 4 comma-separated values (Nx,Ny,Nz,Nt)");
        std::process::exit(1);
    }

    if let Some(ns) = ns_val {
        let nt = nt_val.unwrap_or(ns);
        return Some([ns, ns, ns, nt]);
    }

    if let Some(nt) = nt_val
        && let Some(l) = lattice_val
    {
        return Some([l, l, l, nt]);
    }

    lattice_val.map(|l| [l, l, l, l])
}

/// Format lattice dimensions as a human-readable string.
///
/// Returns "N^4" for isotropic, "Ns^3 x Nt" for cubic spatial with
/// different temporal, or "Nx x Ny x Nz x Nt" for fully anisotropic.
pub fn format_dims(dims: [usize; 4]) -> String {
    let [nx, ny, nz, nt] = dims;
    if nx == ny && ny == nz && nz == nt {
        format!("{nx}⁴")
    } else if nx == ny && ny == nz {
        format!("{nx}³×{nt}")
    } else {
        format!("{nx}×{ny}×{nz}×{nt}")
    }
}

/// Format lattice dimensions for use in ensemble/file identifiers.
///
/// Returns "L8" for isotropic 8^4, "L8_Nt16" for 8^3 x 16,
/// or "8x8x8x16" for fully anisotropic.
pub fn format_dims_id(dims: [usize; 4]) -> String {
    let [nx, ny, nz, nt] = dims;
    if nx == ny && ny == nz && nz == nt {
        format!("L{nx}")
    } else if nx == ny && ny == nz {
        format!("L{nx}_Nt{nt}")
    } else {
        format!("{nx}x{ny}x{nz}x{nt}")
    }
}

/// Minimum spatial dimension (for bounding Wilson loop extents).
pub fn min_spatial_dim(dims: [usize; 4]) -> usize {
    dims[0].min(dims[1]).min(dims[2])
}
