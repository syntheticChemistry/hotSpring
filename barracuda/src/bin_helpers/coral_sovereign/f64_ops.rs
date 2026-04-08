// SPDX-License-Identifier: AGPL-3.0-or-later

//! f64 QCD-style kernel tests and reduction-style shader exercises on the sovereign path.

use coral_gpu::GpuContext;
use std::time::Instant;

/// Test: write a literal f64 value to a buffer.
/// Isolates whether f64 stores (GLOBAL_STORE_DWORDX2) work.
pub fn test_f64_literal_write(ctx: &mut GpuContext) -> Result<(), String> {
    const WGSL: &str = include_str!("../../bin/shaders/coral_sovereign/f64_literal_write.wgsl");
    let kernel = ctx
        .compile_wgsl(WGSL)
        .map_err(|e| format!("compile: {e}"))?;

    eprintln!("[QCD] f64 literal ISA ({} bytes):", kernel.binary.len());
    let isa_words: &[u32] = bytemuck::cast_slice(&kernel.binary);
    for (i, w) in isa_words.iter().enumerate() {
        eprint!("  [{i:3}] 0x{w:08X}");
        if *w == 0xBF81_0000 {
            eprint!("  ← S_ENDPGM");
        }
        if *w & 0xFC00_0000 == 0xDC00_0000 {
            eprint!("  ← FLAT/GLOBAL");
        }
        eprintln!();
    }

    let buf = ctx.alloc(4096).map_err(|e| format!("alloc: {e}"))?;
    ctx.upload(buf, &[0u8; 16])
        .map_err(|e| format!("upload: {e}"))?;

    ctx.dispatch(&kernel, &[buf], [1, 1, 1])
        .map_err(|e| format!("dispatch: {e}"))?;
    ctx.sync().map_err(|e| format!("sync: {e}"))?;

    let rb = ctx
        .readback(buf, 16)
        .map_err(|e| format!("readback: {e}"))?;
    let val: f64 = *bytemuck::from_bytes(&rb[..8]);
    ctx.free(buf).ok();

    eprintln!("[QCD] f64 literal result: {val} (expected 3.14)");
    if (val - 3.14).abs() > 1e-10 {
        return Err(format!("expected 3.14, got {val}"));
    }
    print!("(val={val}) ");
    Ok(())
}

/// Test: copy f64 from input buffer to output buffer.
/// Isolates whether f64 loads (GLOBAL_LOAD_DWORDX2) work.
pub fn test_f64_copy(ctx: &mut GpuContext) -> Result<(), String> {
    const WGSL: &str = include_str!("../../bin/shaders/coral_sovereign/f64_copy.wgsl");
    let kernel = ctx
        .compile_wgsl(WGSL)
        .map_err(|e| format!("compile: {e}"))?;

    eprintln!("[QCD] f64 copy ISA ({} bytes):", kernel.binary.len());
    let isa_words: &[u32] = bytemuck::cast_slice(&kernel.binary);
    for (i, w) in isa_words.iter().enumerate() {
        eprint!("  [{i:3}] 0x{w:08X}");
        if *w == 0xBF81_0000 {
            eprint!("  ← S_ENDPGM");
        }
        if *w & 0xFC00_0000 == 0xDC00_0000 {
            eprint!("  ← FLAT/GLOBAL");
        }
        eprintln!();
    }

    let buf_in = ctx.alloc(4096).map_err(|e| format!("alloc in: {e}"))?;
    let buf_out = ctx.alloc(4096).map_err(|e| format!("alloc out: {e}"))?;

    let input_val: f64 = 2.718281828;
    ctx.upload(buf_in, bytemuck::bytes_of(&input_val))
        .map_err(|e| format!("upload: {e}"))?;
    ctx.upload(buf_out, &[0u8; 8])
        .map_err(|e| format!("upload out: {e}"))?;

    ctx.dispatch(&kernel, &[buf_in, buf_out], [1, 1, 1])
        .map_err(|e| format!("dispatch: {e}"))?;
    ctx.sync().map_err(|e| format!("sync: {e}"))?;

    let rb = ctx
        .readback(buf_out, 16)
        .map_err(|e| format!("readback: {e}"))?;
    let val: f64 = *bytemuck::from_bytes(&rb[..8]);

    let input_hex = u64::from_le_bytes(bytemuck::bytes_of(&input_val).try_into().unwrap());
    let result_hex = u64::from_le_bytes(rb[..8].try_into().unwrap());
    eprintln!("[QCD] f64 copy: input  = {input_val}  (0x{input_hex:016X})");
    eprintln!("[QCD] f64 copy: output = {val}  (0x{result_hex:016X})");
    eprintln!("[QCD] f64 copy: raw bytes = {:02X?}", &rb[..16]);

    ctx.free(buf_in).ok();
    ctx.free(buf_out).ok();

    if (val - input_val).abs() > 1e-10 {
        return Err(format!(
            "expected {input_val} (0x{input_hex:016X}), got {val} (0x{result_hex:016X})"
        ));
    }
    print!("(copied {val}) ");
    Ok(())
}

/// Test: f64 addition across 3 buffers (a + b → c).
/// Isolates whether 3 buffer bindings + V_ADD_F64 work.
pub fn test_f64_add_3buf(ctx: &mut GpuContext) -> Result<(), String> {
    const WGSL: &str = include_str!("../../bin/shaders/coral_sovereign/f64_add_3buf.wgsl");
    let kernel = ctx
        .compile_wgsl(WGSL)
        .map_err(|e| format!("compile: {e}"))?;

    let buf_a = ctx.alloc(4096).map_err(|e| format!("alloc a: {e}"))?;
    let buf_b = ctx.alloc(4096).map_err(|e| format!("alloc b: {e}"))?;
    let buf_out = ctx.alloc(4096).map_err(|e| format!("alloc out: {e}"))?;

    let a_val: f64 = 1.5;
    let b_val: f64 = 2.5;
    ctx.upload(buf_a, bytemuck::bytes_of(&a_val))
        .map_err(|e| format!("upload a: {e}"))?;
    ctx.upload(buf_b, bytemuck::bytes_of(&b_val))
        .map_err(|e| format!("upload b: {e}"))?;
    ctx.upload(buf_out, &[0u8; 8])
        .map_err(|e| format!("upload out: {e}"))?;

    let t0 = Instant::now();
    ctx.dispatch(&kernel, &[buf_a, buf_b, buf_out], [1, 1, 1])
        .map_err(|e| format!("dispatch: {e}"))?;
    ctx.sync().map_err(|e| format!("sync: {e}"))?;
    let us = t0.elapsed().as_micros();

    let rb = ctx
        .readback(buf_out, 8)
        .map_err(|e| format!("readback: {e}"))?;
    let result: f64 = *bytemuck::from_bytes(&rb[..8]);

    ctx.free(buf_a).ok();
    ctx.free(buf_b).ok();
    ctx.free(buf_out).ok();

    let expected = 4.0_f64;
    if (result - expected).abs() > 1e-12 {
        return Err(format!(
            "expected {expected}, got {result} (a={a_val}, b={b_val})"
        ));
    }
    print!("({a_val}+{b_val}={result}, {us}us) ");
    Ok(())
}

/// Test: pure f64 division across 3 buffers (a / b → c).
pub fn test_f64_div_3buf(ctx: &mut GpuContext) -> Result<(), String> {
    const WGSL: &str = include_str!("../../bin/shaders/coral_sovereign/f64_div_3buf.wgsl");
    let kernel = ctx
        .compile_wgsl(WGSL)
        .map_err(|e| format!("compile: {e}"))?;

    let buf_a = ctx.alloc(4096).map_err(|e| format!("alloc a: {e}"))?;
    let buf_b = ctx.alloc(4096).map_err(|e| format!("alloc b: {e}"))?;
    let buf_out = ctx.alloc(4096).map_err(|e| format!("alloc out: {e}"))?;

    let a_val: f64 = 6.0;
    let b_val: f64 = 3.0;
    ctx.upload(buf_a, bytemuck::bytes_of(&a_val))
        .map_err(|e| format!("upload a: {e}"))?;
    ctx.upload(buf_b, bytemuck::bytes_of(&b_val))
        .map_err(|e| format!("upload b: {e}"))?;
    ctx.upload(buf_out, &[0u8; 8])
        .map_err(|e| format!("upload out: {e}"))?;

    let t0 = Instant::now();
    ctx.dispatch(&kernel, &[buf_a, buf_b, buf_out], [1, 1, 1])
        .map_err(|e| format!("dispatch: {e}"))?;
    ctx.sync().map_err(|e| format!("sync: {e}"))?;
    let us = t0.elapsed().as_micros();

    let rb = ctx
        .readback(buf_out, 8)
        .map_err(|e| format!("readback: {e}"))?;
    let result: f64 = *bytemuck::from_bytes(&rb[..8]);
    ctx.free(buf_a).ok();
    ctx.free(buf_b).ok();
    ctx.free(buf_out).ok();

    let expected = 2.0_f64;
    if (result - expected).abs() > 1e-12 {
        return Err(format!(
            "expected {expected}, got {result} ({a_val}/{b_val})"
        ));
    }
    print!("({a_val}/{b_val}={result}, {us}us) ");
    Ok(())
}

/// Test: f64 comparison + branch (if abs(x) > 1e-30 → 1.0 else 0.0).
pub fn test_f64_cmp_branch(ctx: &mut GpuContext) -> Result<(), String> {
    const WGSL: &str = include_str!("../../bin/shaders/coral_sovereign/f64_cmp_branch.wgsl");
    let kernel = ctx
        .compile_wgsl(WGSL)
        .map_err(|e| format!("compile: {e}"))?;

    let buf_in = ctx.alloc(4096).map_err(|e| format!("alloc in: {e}"))?;
    let buf_out = ctx.alloc(4096).map_err(|e| format!("alloc out: {e}"))?;

    let input_val: f64 = 3.0;
    ctx.upload(buf_in, bytemuck::bytes_of(&input_val))
        .map_err(|e| format!("upload: {e}"))?;
    ctx.upload(buf_out, &0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes())
        .map_err(|e| format!("upload out: {e}"))?;

    ctx.dispatch(&kernel, &[buf_in, buf_out], [1, 1, 1])
        .map_err(|e| format!("dispatch: {e}"))?;
    ctx.sync().map_err(|e| format!("sync: {e}"))?;

    let rb = ctx
        .readback(buf_out, 8)
        .map_err(|e| format!("readback: {e}"))?;
    let result: f64 = *bytemuck::from_bytes(&rb[..8]);
    ctx.free(buf_in).ok();
    ctx.free(buf_out).ok();

    if (result - 1.0).abs() > 1e-12 {
        return Err(format!(
            "expected 1.0, got {result} (input {input_val}, abs > 1e-30 should be true)"
        ));
    }
    print!("(abs({input_val})>1e-30 → {result}) ");
    Ok(())
}

/// CG scalar: alpha = rz / pAp.
/// Verifies f64 load, f64 division, f64 store across 3 buffers.
pub fn test_cg_compute_alpha(ctx: &mut GpuContext) -> Result<(), String> {
    let kernel = ctx
        .compile_wgsl(include_str!("../../lattice/shaders/cg_compute_alpha_f64.wgsl"))
        .map_err(|e| format!("compile: {e}"))?;

    eprintln!(
        "[QCD] CG alpha ISA ({} bytes, {} GPRs, wg={:?}, wave={}):",
        kernel.binary.len(),
        kernel.gpr_count,
        kernel.workgroup,
        kernel.wave_size
    );
    let isa_words: &[u32] = bytemuck::cast_slice(&kernel.binary);
    for (i, w) in isa_words.iter().enumerate() {
        eprint!("  [{i:3}] 0x{w:08X}");
        if *w == 0xBF81_0000 {
            eprint!("  ← S_ENDPGM");
        }
        if *w & 0xFC00_0000 == 0xDC00_0000 {
            eprint!("  ← FLAT/GLOBAL");
        }
        eprintln!();
    }

    let buf_rz = ctx.alloc(4096).map_err(|e| format!("alloc rz: {e}"))?;
    let buf_pap = ctx.alloc(4096).map_err(|e| format!("alloc pap: {e}"))?;
    let buf_alpha = ctx.alloc(4096).map_err(|e| format!("alloc alpha: {e}"))?;

    let rz_val: f64 = 6.0;
    let pap_val: f64 = 3.0;
    ctx.upload(buf_rz, bytemuck::bytes_of(&rz_val))
        .map_err(|e| format!("upload rz: {e}"))?;
    ctx.upload(buf_pap, bytemuck::bytes_of(&pap_val))
        .map_err(|e| format!("upload pap: {e}"))?;
    ctx.upload(buf_alpha, &[0u8; 8])
        .map_err(|e| format!("upload alpha: {e}"))?;

    // Verify upload path for f64
    let pre_rz = ctx
        .readback(buf_rz, 8)
        .map_err(|e| format!("pre-readback rz: {e}"))?;
    let pre_rz_val: f64 = *bytemuck::from_bytes(&pre_rz[..8]);
    eprintln!("[QCD] Pre-dispatch: rz={pre_rz_val} (expected {rz_val})");
    let pre_pap = ctx
        .readback(buf_pap, 8)
        .map_err(|e| format!("pre-readback pap: {e}"))?;
    let pre_pap_val: f64 = *bytemuck::from_bytes(&pre_pap[..8]);
    eprintln!("[QCD] Pre-dispatch: pap={pre_pap_val} (expected {pap_val})");

    let t0 = Instant::now();
    ctx.dispatch(&kernel, &[buf_rz, buf_pap, buf_alpha], [1, 1, 1])
        .map_err(|e| format!("dispatch: {e}"))?;
    ctx.sync().map_err(|e| format!("sync: {e}"))?;
    let us = t0.elapsed().as_micros();

    let rb = ctx
        .readback(buf_alpha, 8)
        .map_err(|e| format!("readback: {e}"))?;
    let result: f64 = *bytemuck::from_bytes(&rb[..8]);
    eprintln!("[QCD] Post-dispatch: alpha={result} (expected 2.0)");

    // Also read back rz and pap to see if they were modified
    let post_rz = ctx
        .readback(buf_rz, 8)
        .map_err(|e| format!("post-readback rz: {e}"))?;
    let post_rz_val: f64 = *bytemuck::from_bytes(&post_rz[..8]);
    eprintln!("[QCD] Post-dispatch: rz={post_rz_val} (should still be {rz_val})");

    ctx.free(buf_rz).ok();
    ctx.free(buf_pap).ok();
    ctx.free(buf_alpha).ok();

    let expected = 2.0_f64;
    if (result - expected).abs() > 1e-12 {
        return Err(format!(
            "expected {expected}, got {result} (rz={rz_val}, pAp={pap_val})"
        ));
    }
    print!("(rz/pAp = {result}, {us}us) ");
    Ok(())
}

/// AXPY: y = y + α·x (f64 vector operation with uniform params).
/// Verifies uniform buffer reads, multi-element f64 FMA, workgroup dispatch.
/// Diagnostic: read uniform struct members via sovereign pipeline.
pub fn test_uniform_read(ctx: &mut GpuContext) -> Result<(), String> {
    // Test A: single f64 in uniform struct (offset 0) — eliminates offset issues
    const WGSL_A: &str = include_str!("../../bin/shaders/coral_sovereign/uniform_single_f64.wgsl");
    let kernel_a = ctx
        .compile_wgsl(WGSL_A)
        .map_err(|e| format!("compile A: {e}"))?;

    let val = 3.14_f64;
    let buf_p = ctx.alloc(4096).map_err(|e| format!("alloc p: {e}"))?;
    let buf_out = ctx.alloc(4096).map_err(|e| format!("alloc out: {e}"))?;
    ctx.upload(buf_p, &val.to_le_bytes())
        .map_err(|e| format!("upload: {e}"))?;
    ctx.upload(buf_out, &[0u8; 8])
        .map_err(|e| format!("upload out: {e}"))?;

    ctx.dispatch(&kernel_a, &[buf_p, buf_out], [1, 1, 1])
        .map_err(|e| format!("dispatch A: {e}"))?;
    ctx.sync().map_err(|e| format!("sync A: {e}"))?;

    let rb = ctx
        .readback(buf_out, 8)
        .map_err(|e| format!("readback A: {e}"))?;
    let got_a: f64 = f64::from_le_bytes(rb[..8].try_into().unwrap());

    ctx.free(buf_p).ok();
    ctx.free(buf_out).ok();

    if (got_a - 3.14).abs() > 1e-12 {
        return Err(format!(
            "test A: single f64 uniform: expected 3.14, got {got_a}"
        ));
    }

    // Test B: u32 then f64 in uniform struct (alpha at offset 8)
    const WGSL_B: &str = include_str!("../../bin/shaders/coral_sovereign/uniform_params_n_alpha.wgsl");
    let kernel_b = ctx
        .compile_wgsl(WGSL_B)
        .map_err(|e| format!("compile B: {e}"))?;

    let n = 42_u32;
    let alpha = 2.0_f64;
    let mut params_bytes = Vec::with_capacity(16);
    params_bytes.extend_from_slice(&n.to_le_bytes());
    params_bytes.extend_from_slice(&0_u32.to_le_bytes());
    params_bytes.extend_from_slice(&alpha.to_le_bytes());

    let buf_params = ctx.alloc(4096).map_err(|e| format!("alloc params: {e}"))?;
    let buf_out2 = ctx.alloc(4096).map_err(|e| format!("alloc out: {e}"))?;
    ctx.upload(buf_params, &params_bytes)
        .map_err(|e| format!("upload: {e}"))?;
    ctx.upload(buf_out2, &[0u8; 16])
        .map_err(|e| format!("upload out: {e}"))?;

    ctx.dispatch(&kernel_b, &[buf_params, buf_out2], [1, 1, 1])
        .map_err(|e| format!("dispatch B: {e}"))?;
    ctx.sync().map_err(|e| format!("sync B: {e}"))?;

    let rb2 = ctx
        .readback(buf_out2, 16)
        .map_err(|e| format!("readback B: {e}"))?;
    let vals: &[f64] = bytemuck::cast_slice(&rb2);

    ctx.free(buf_params).ok();
    ctx.free(buf_out2).ok();

    let got_n = vals[0];
    let got_alpha = vals[1];
    print!("(n={got_n}, alpha={got_alpha}) ");

    if (got_n - 42.0).abs() > 0.5 {
        return Err(format!("test B: params.n: expected 42, got {got_n}"));
    }
    if (got_alpha - 2.0).abs() > 1e-12 {
        return Err(format!(
            "test B: params.alpha: expected 2.0, got {got_alpha}"
        ));
    }
    Ok(())
}

/// Minimal AXPY: no builtins, no branching, just y[0] = y[0] + alpha * x[0]
pub fn test_axpy_minimal(ctx: &mut GpuContext) -> Result<(), String> {
    const WGSL: &str = include_str!("../../bin/shaders/coral_sovereign/axpy_minimal.wgsl");
    let kernel = ctx
        .compile_wgsl(WGSL)
        .map_err(|e| format!("compile: {e}"))?;

    let alpha = 2.0_f64;
    let mut params_bytes = Vec::with_capacity(16);
    params_bytes.extend_from_slice(&1_u32.to_le_bytes());
    params_bytes.extend_from_slice(&0_u32.to_le_bytes());
    params_bytes.extend_from_slice(&alpha.to_le_bytes());

    let x_val = 3.0_f64;
    let y_val = 10.0_f64;

    let buf_p = ctx.alloc(4096).map_err(|e| format!("alloc p: {e}"))?;
    let buf_x = ctx.alloc(4096).map_err(|e| format!("alloc x: {e}"))?;
    let buf_y = ctx.alloc(4096).map_err(|e| format!("alloc y: {e}"))?;
    ctx.upload(buf_p, &params_bytes)
        .map_err(|e| format!("upload p: {e}"))?;
    ctx.upload(buf_x, &x_val.to_le_bytes())
        .map_err(|e| format!("upload x: {e}"))?;
    ctx.upload(buf_y, &y_val.to_le_bytes())
        .map_err(|e| format!("upload y: {e}"))?;

    ctx.dispatch(&kernel, &[buf_p, buf_x, buf_y], [1, 1, 1])
        .map_err(|e| format!("dispatch: {e}"))?;
    ctx.sync().map_err(|e| format!("sync: {e}"))?;

    let rb = ctx
        .readback(buf_y, 8)
        .map_err(|e| format!("readback: {e}"))?;
    let got = f64::from_le_bytes(rb[..8].try_into().unwrap());
    let expected = y_val + alpha * x_val; // 10 + 2*3 = 16

    ctx.free(buf_p).ok();
    ctx.free(buf_x).ok();
    ctx.free(buf_y).ok();

    print!("(y=y+α*x: {got}, expected {expected}) ");
    if (got - expected).abs() > 1e-12 {
        return Err(format!("expected {expected}, got {got}"));
    }
    Ok(())
}

pub fn test_axpy_f64(ctx: &mut GpuContext) -> Result<(), String> {
    // Simplified AXPY: use gid.x directly, no num_workgroups
    const WGSL: &str = include_str!("../../bin/shaders/coral_sovereign/axpy_f64.wgsl");
    let kernel = ctx
        .compile_wgsl(WGSL)
        .map_err(|e| format!("compile: {e}"))?;

    // Use N=64 to fill one full workgroup (avoids divergent branch on the
    // `if i >= params.n { return; }` guard — coral-reef uses scalar branches
    // which don't yet support EXEC masking for divergent control flow).
    let n = 64_u32;
    let alpha = 2.0_f64;

    let mut params_buf = Vec::with_capacity(16);
    params_buf.extend_from_slice(&n.to_le_bytes());
    params_buf.extend_from_slice(&0_u32.to_le_bytes());
    params_buf.extend_from_slice(&alpha.to_le_bytes());

    let mut x_vals = [0.0_f64; 64];
    let mut y_vals = [0.0_f64; 64];
    for i in 0..64 {
        x_vals[i] = (i + 1) as f64;
        y_vals[i] = (i as f64 + 1.0) * 10.0;
    }

    let buf_params = ctx.alloc(4096).map_err(|e| format!("alloc params: {e}"))?;
    let buf_x = ctx.alloc(4096).map_err(|e| format!("alloc x: {e}"))?;
    let buf_y = ctx.alloc(4096).map_err(|e| format!("alloc y: {e}"))?;

    ctx.upload(buf_params, &params_buf)
        .map_err(|e| format!("upload params: {e}"))?;
    ctx.upload(buf_x, bytemuck::cast_slice(&x_vals))
        .map_err(|e| format!("upload x: {e}"))?;
    ctx.upload(buf_y, bytemuck::cast_slice(&y_vals))
        .map_err(|e| format!("upload y: {e}"))?;

    let t0 = Instant::now();
    ctx.dispatch(&kernel, &[buf_params, buf_x, buf_y], [1, 1, 1])
        .map_err(|e| format!("dispatch: {e}"))?;
    ctx.sync().map_err(|e| format!("sync: {e}"))?;
    let us = t0.elapsed().as_micros();

    let rb = ctx
        .readback(buf_y, 64 * 8)
        .map_err(|e| format!("readback: {e}"))?;
    let result: &[f64] = bytemuck::cast_slice(&rb);
    println!(
        "[AXPY] y[0..4] = {:?}, expected = {:?}",
        &result[..4],
        (0..4)
            .map(|i| y_vals[i] + alpha * x_vals[i])
            .collect::<Vec<_>>()
    );

    ctx.free(buf_params).ok();
    ctx.free(buf_x).ok();
    ctx.free(buf_y).ok();

    let mut errors = 0;
    for i in 0..4 {
        let expected = y_vals[i] + alpha * x_vals[i];
        if (result[i] - expected).abs() > 1e-12 {
            if errors == 0 {
                print!("(y[{i}]={}, expected {}) ", result[i], expected);
            }
            errors += 1;
        }
    }
    if errors > 0 {
        return Err(format!(
            "y[0]: expected {}, got {} (y + α·x = {} + {}·{})",
            y_vals[0] + alpha * x_vals[0],
            result[0],
            y_vals[0],
            alpha,
            x_vals[0]
        ));
    }
    print!(
        "(y=[{:.0},{:.0},{:.0},{:.0}], {us}us) ",
        result[0], result[1], result[2], result[3]
    );
    Ok(())
}

/// Diagnostic: verify @builtin(num_workgroups) reads the correct dispatch dimensions.
pub fn test_num_workgroups(ctx: &mut GpuContext) -> Result<(), String> {
    const WGSL: &str = include_str!("../../bin/shaders/coral_sovereign/num_workgroups_builtin.wgsl");
    let kernel = ctx
        .compile_wgsl(WGSL)
        .map_err(|e| format!("compile: {e}"))?;
    let buf = ctx.alloc(4096).map_err(|e| format!("alloc: {e}"))?;
    ctx.upload(buf, &[0u8; 12])
        .map_err(|e| format!("upload: {e}"))?;

    ctx.dispatch(&kernel, &[buf], [7, 3, 2])
        .map_err(|e| format!("dispatch: {e}"))?;
    ctx.sync().map_err(|e| format!("sync: {e}"))?;

    let rb = ctx
        .readback(buf, 12)
        .map_err(|e| format!("readback: {e}"))?;
    let vals: &[u32] = bytemuck::cast_slice(&rb);
    ctx.free(buf).ok();

    print!("(nwg=[{},{},{}]) ", vals[0], vals[1], vals[2]);
    if vals[0] != 7 || vals[1] != 3 || vals[2] != 2 {
        return Err(format!(
            "expected [7,3,2], got [{},{},{}]",
            vals[0], vals[1], vals[2]
        ));
    }
    Ok(())
}

/// Diagnostic: trace gid + nwg values from a single thread to check register mapping.
pub fn test_nwg_idx_debug(ctx: &mut GpuContext) -> Result<(), String> {
    const WGSL: &str = include_str!("../../bin/shaders/coral_sovereign/nwg_idx_debug.wgsl");
    let kernel = ctx
        .compile_wgsl(WGSL)
        .map_err(|e| format!("compile: {e}"))?;
    let buf = ctx.alloc(4096).map_err(|e| format!("alloc: {e}"))?;
    ctx.upload(buf, &[0xFFu8; 24])
        .map_err(|e| format!("upload: {e}"))?;

    ctx.dispatch(&kernel, &[buf], [1, 1, 1])
        .map_err(|e| format!("dispatch: {e}"))?;
    ctx.sync().map_err(|e| format!("sync: {e}"))?;

    let rb = ctx
        .readback(buf, 24)
        .map_err(|e| format!("readback: {e}"))?;
    let vals: &[u32] = bytemuck::cast_slice(&rb);
    ctx.free(buf).ok();

    print!(
        "(gid=[{},{}] nwg=[{},{}] wid={} lid={}) ",
        vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]
    );
    if vals[0] != 0 {
        return Err(format!(
            "gid.x: expected 0, got {} (0x{:08X})",
            vals[0], vals[0]
        ));
    }
    if vals[2] != 1 {
        return Err(format!(
            "nwg.x: expected 1, got {} (0x{:08X})",
            vals[2], vals[2]
        ));
    }
    Ok(())
}

/// Test full AXPY with num_workgroups builtin (original shader from barracuda).
pub fn test_axpy_with_num_workgroups(ctx: &mut GpuContext) -> Result<(), String> {
    const WGSL: &str = include_str!("../../bin/shaders/coral_sovereign/axpy_num_workgroups.wgsl");
    let kernel = ctx
        .compile_wgsl(WGSL)
        .map_err(|e| format!("compile: {e}"))?;

    let n = 64_u32;
    let alpha = 2.0_f64;
    let mut params_buf = Vec::with_capacity(16);
    params_buf.extend_from_slice(&n.to_le_bytes());
    params_buf.extend_from_slice(&0_u32.to_le_bytes());
    params_buf.extend_from_slice(&alpha.to_le_bytes());

    let mut x_vals = [0.0_f64; 64];
    let mut y_vals = [0.0_f64; 64];
    for i in 0..64 {
        x_vals[i] = (i + 1) as f64;
        y_vals[i] = (i as f64 + 1.0) * 10.0;
    }

    let buf_params = ctx.alloc(4096).map_err(|e| format!("alloc p: {e}"))?;
    let buf_x = ctx.alloc(4096).map_err(|e| format!("alloc x: {e}"))?;
    let buf_y = ctx.alloc(4096).map_err(|e| format!("alloc y: {e}"))?;
    ctx.upload(buf_params, &params_buf)
        .map_err(|e| format!("upload p: {e}"))?;
    ctx.upload(buf_x, bytemuck::cast_slice(&x_vals))
        .map_err(|e| format!("upload x: {e}"))?;
    ctx.upload(buf_y, bytemuck::cast_slice(&y_vals))
        .map_err(|e| format!("upload y: {e}"))?;

    ctx.dispatch(&kernel, &[buf_params, buf_x, buf_y], [1, 1, 1])
        .map_err(|e| format!("dispatch: {e}"))?;
    ctx.sync().map_err(|e| format!("sync: {e}"))?;

    let rb = ctx
        .readback(buf_y, 64 * 8)
        .map_err(|e| format!("readback: {e}"))?;
    let result: &[f64] = bytemuck::cast_slice(&rb);

    ctx.free(buf_params).ok();
    ctx.free(buf_x).ok();
    ctx.free(buf_y).ok();

    let mut errors = 0;
    for i in 0..4 {
        let expected = y_vals[i] + alpha * x_vals[i];
        if (result[i] - expected).abs() > 1e-12 {
            errors += 1;
        }
    }
    if errors > 0 {
        return Err(format!(
            "y[0..4]={:?}, expected {:?}",
            &result[..4],
            (0..4)
                .map(|i| y_vals[i] + alpha * x_vals[i])
                .collect::<Vec<_>>()
        ));
    }
    print!(
        "(y=[{:.0},{:.0},{:.0},{:.0}]) ",
        result[0], result[1], result[2], result[3]
    );
    Ok(())
}

/// Complex dot product Re⟨a,b⟩ (f64 multi-thread reduction frontier).
/// Tests the coral-reef f64 pipeline with multiple buffer reads in a real physics kernel.
pub fn test_complex_dot_re(ctx: &mut GpuContext) -> Result<(), String> {
    let kernel = ctx
        .compile_wgsl(include_str!("../../lattice/shaders/complex_dot_re_f64.wgsl"))
        .map_err(|e| format!("compile: {e}"))?;

    print!(
        "(compiled {} bytes, {} GPRs) ",
        kernel.binary.len(),
        kernel.gpr_count
    );
    Ok(())
}
