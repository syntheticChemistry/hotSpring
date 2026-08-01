// SPDX-License-Identifier: AGPL-3.0-or-later

/// Linearly interpolate `computed` FES onto a target grid point.
pub(crate) fn interp(x: f64, grid: &[f64], values: &[f64]) -> f64 {
    if x <= grid[0] {
        return values[0];
    }
    if x >= *grid.last().unwrap() {
        return *values.last().unwrap();
    }
    // Binary search for bracketing interval
    let mut lo = 0;
    let mut hi = grid.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if grid[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (x - grid[lo]) / (grid[hi] - grid[lo]);
    values[lo] + t * (values[hi] - values[lo])
}

/// Bilinear interpolation on a 2D grid.
pub(crate) fn interp_2d(x: f64, y: f64, grid_x: &[f64], grid_y: &[f64], values: &[Vec<f64>]) -> f64 {
    let nx = grid_x.len();
    let ny = grid_y.len();

    // Clamp x
    let x = x.max(grid_x[0]).min(*grid_x.last().unwrap());
    let y = y.max(grid_y[0]).min(*grid_y.last().unwrap());

    // Find bracketing x
    let mut ix = 0;
    for i in 0..nx - 1 {
        if grid_x[i + 1] >= x {
            ix = i;
            break;
        }
    }
    if x >= *grid_x.last().unwrap() {
        ix = nx - 2;
    }

    // Find bracketing y
    let mut iy = 0;
    for j in 0..ny - 1 {
        if grid_y[j + 1] >= y {
            iy = j;
            break;
        }
    }
    if y >= *grid_y.last().unwrap() {
        iy = ny - 2;
    }

    let tx = if (grid_x[ix + 1] - grid_x[ix]).abs() > 1e-15 {
        (x - grid_x[ix]) / (grid_x[ix + 1] - grid_x[ix])
    } else {
        0.0
    };
    let ty = if (grid_y[iy + 1] - grid_y[iy]).abs() > 1e-15 {
        (y - grid_y[iy]) / (grid_y[iy + 1] - grid_y[iy])
    } else {
        0.0
    };

    let v00 = values[ix][iy];
    let v10 = values[ix + 1][iy];
    let v01 = values[ix][iy + 1];
    let v11 = values[ix + 1][iy + 1];

    v00 * (1.0 - tx) * (1.0 - ty) + v10 * tx * (1.0 - ty) + v01 * (1.0 - tx) * ty + v11 * tx * ty
}
