// SPDX-License-Identifier: AGPL-3.0-or-later

//! Basic sovereign dispatch tests (write constant, thread IDs, buffer read probe).

use coral_gpu::GpuContext;
use std::time::Instant;

/// Test 1: Write a constant value to a buffer via compiled WGSL shader.
/// This is the proven E2E path: compile → dispatch → readback → verify.
pub fn test_write_constant(ctx: &mut GpuContext) -> Result<(), String> {
    test_write_constant_inner(ctx, false)
}

pub fn test_write_constant_inner(ctx: &mut GpuContext, verbose: bool) -> Result<(), String> {
    const WGSL: &str = include_str!("../../bin/shaders/coral_sovereign/write_constant.wgsl");

    let t0 = Instant::now();
    let kernel = ctx
        .compile_wgsl(WGSL)
        .map_err(|e| format!("compile: {e}"))?;
    let compile_us = t0.elapsed().as_micros();

    if verbose {
        println!(
            "\n    ISA binary ({} bytes, {} GPRs, wg={:?}, wave={}):",
            kernel.binary.len(),
            kernel.gpr_count,
            kernel.workgroup,
            kernel.wave_size
        );
        let words: &[u32] = bytemuck::cast_slice(&kernel.binary);
        for (i, w) in words.iter().enumerate() {
            print!("      [{i:3}] 0x{w:08X}");
            if *w == 0xBF81_0000 {
                print!("  ← S_ENDPGM");
            }
            println!();
        }
    }

    let buf = ctx.alloc(4096).map_err(|e| format!("alloc: {e}"))?;

    // Pre-fill with sentinel to distinguish "dispatch didn't write" from "readback broken"
    let sentinel: u32 = 0xDEAD_BEEF;
    let mut init_data = vec![0u8; 4096];
    for chunk in init_data[..16].chunks_exact_mut(4) {
        chunk.copy_from_slice(&sentinel.to_le_bytes());
    }
    ctx.upload(buf, &init_data)
        .map_err(|e| format!("upload: {e}"))?;

    if verbose {
        // Verify upload: read back before dispatch
        let pre = ctx
            .readback(buf, 16)
            .map_err(|e| format!("pre-readback: {e}"))?;
        let pre_vals: &[u32] = bytemuck::cast_slice(&pre);
        println!("    Pre-dispatch readback (sentinel 0x{sentinel:08X}): {pre_vals:?}");
    }

    let t1 = Instant::now();
    ctx.dispatch(&kernel, &[buf], [1, 1, 1])
        .map_err(|e| format!("dispatch: {e}"))?;
    ctx.sync().map_err(|e| format!("sync: {e}"))?;
    let dispatch_us = t1.elapsed().as_micros();

    let readback = ctx
        .readback(buf, 16)
        .map_err(|e| format!("readback: {e}"))?;
    let values: &[u32] = bytemuck::cast_slice(&readback);
    let value = values[0];

    if verbose {
        println!("    Post-dispatch readback: {:?}", values);
        if values[1] == sentinel {
            println!(
                "    → Sentinel intact at [1]: upload+readback works, dispatch didn't modify [0]"
            );
        } else if values.iter().all(|&v| v == 0) {
            println!("    → All zeros: upload may not persist (mmap/domain issue)");
        }
    }

    ctx.free(buf).ok();

    if value != 42 {
        return Err(format!("expected 42, got {value}"));
    }

    print!("(compile {compile_us}us, dispatch {dispatch_us}us, value={value}) ",);
    Ok(())
}

/// Test 2: Write thread IDs to buffer (multi-thread dispatch, no buffer reads).
pub fn test_write_thread_id(ctx: &mut GpuContext) -> Result<(), String> {
    const WGSL: &str = include_str!("../../bin/shaders/coral_sovereign/write_thread_id.wgsl");

    let kernel = ctx
        .compile_wgsl(WGSL)
        .map_err(|e| format!("compile: {e}"))?;

    let buf_size = 64 * 4;
    let buf = ctx
        .alloc(buf_size as u64)
        .map_err(|e| format!("alloc: {e}"))?;
    ctx.upload(buf, &vec![0u8; buf_size])
        .map_err(|e| format!("upload: {e}"))?;

    let t0 = Instant::now();
    ctx.dispatch(&kernel, &[buf], [1, 1, 1])
        .map_err(|e| format!("dispatch: {e}"))?;
    ctx.sync().map_err(|e| format!("sync: {e}"))?;
    let dispatch_us = t0.elapsed().as_micros();

    let readback = ctx
        .readback(buf, buf_size)
        .map_err(|e| format!("readback: {e}"))?;
    let values: &[u32] = bytemuck::cast_slice(&readback);

    ctx.free(buf).ok();

    let mut errors = 0;
    for (i, &val) in values.iter().enumerate() {
        let expected = (i as u32) * 3 + 7;
        if val != expected {
            if errors < 3 {
                print!("(out[{i}]={val}, expected {expected}) ");
            }
            errors += 1;
        }
    }

    if errors > 0 {
        return Err(format!("{errors}/64 values wrong"));
    }

    print!("(dispatch {dispatch_us}us, 64 values correct) ");
    Ok(())
}

/// Test 3: Probe buffer-read capability (known frontier on RDNA2).
/// Reads from an input buffer, adds 1, writes to output.
pub fn test_buffer_read_probe(ctx: &mut GpuContext) -> Result<(), String> {
    const WGSL: &str = include_str!("../../bin/shaders/coral_sovereign/buffer_read_probe.wgsl");

    let kernel = ctx
        .compile_wgsl(WGSL)
        .map_err(|e| format!("compile: {e}"))?;

    let buf_in = ctx.alloc(4096).map_err(|e| format!("alloc in: {e}"))?;
    let buf_out = ctx.alloc(4096).map_err(|e| format!("alloc out: {e}"))?;

    let input_val: u32 = 100;
    ctx.upload(buf_in, &input_val.to_le_bytes())
        .map_err(|e| format!("upload in: {e}"))?;
    ctx.upload(buf_out, &[0u8; 4])
        .map_err(|e| format!("upload out: {e}"))?;

    ctx.dispatch(&kernel, &[buf_in, buf_out], [1, 1, 1])
        .map_err(|e| format!("dispatch: {e}"))?;
    ctx.sync().map_err(|e| format!("sync: {e}"))?;

    let readback = ctx
        .readback(buf_out, 4)
        .map_err(|e| format!("readback: {e}"))?;
    let value = u32::from_le_bytes(readback[..4].try_into().map_err(|_| "readback too short")?);

    ctx.free(buf_in).ok();
    ctx.free(buf_out).ok();

    if value != 101 {
        return Err(format!("expected 101, got {value} (input was 100)"));
    }

    print!("(value={value}, expected=101) ");
    Ok(())
}
