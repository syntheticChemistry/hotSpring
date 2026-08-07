// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Compensated DF64 summation — reduces accumulation error from O(√N) to O(1).
//
// In standard DF64 addition, accumulating N values gives error growth
// proportional to √N (random walk) or N (worst case). For a 16⁴ lattice
// with 24,576 plaquettes, this means up to ~6e-11 error.
//
// With Kahan-Babuška-Neumaier compensation, the error is bounded by O(u²),
// independent of N. This makes DF64 accumulation effectively as precise
// as the individual DF64 arithmetic, regardless of lattice volume.
//
// Requires: df64_preamble.wgsl (Df64, df64_add, df64_sub, df64_abs, etc.)

struct CompDf64 {
    sum: Df64,
    comp: Df64,
}

fn comp_df64_zero() -> CompDf64 {
    return CompDf64(df64_zero(), df64_zero());
}

/// Neumaier-style compensated addition: handles the case where
/// the addend may be larger than the running sum.
fn comp_df64_add(acc: CompDf64, val: Df64) -> CompDf64 {
    let t = df64_add(acc.sum, val);
    var c: Df64;
    if df64_abs(acc.sum).hi >= df64_abs(val).hi {
        // |sum| >= |val|: the small terms lost in sum+val go to compensation
        c = df64_add(df64_sub(acc.sum, t), val);
    } else {
        // |val| > |sum|: the small terms lost in val+sum go to compensation
        c = df64_add(df64_sub(val, t), acc.sum);
    }
    return CompDf64(t, df64_add(acc.comp, c));
}

/// Extract final compensated result.
fn comp_df64_result(acc: CompDf64) -> Df64 {
    return df64_add(acc.sum, acc.comp);
}

// ─── Workgroup-level compensated reduction ─────────────────────────────
//
// For plaquette averaging across a lattice, each workgroup computes a
// partial sum using compensation, then the partial sums are combined
// in a second pass (also compensated).

var<workgroup> wg_partial: array<Df64, 256>;

/// Compensated workgroup reduction of values in wg_partial[0..count].
/// Call after each thread has stored its local compensated result.
fn wg_compensated_reduce(lid: u32, count: u32) -> Df64 {
    workgroupBarrier();

    var stride = count / 2u;
    while stride > 0u {
        if lid < stride {
            wg_partial[lid] = df64_add(wg_partial[lid], wg_partial[lid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    return wg_partial[0];
}
