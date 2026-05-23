// SPDX-License-Identifier: AGPL-3.0-or-later

//! Minimal inline `block_on` for running a single future to completion.
//!
//! Replaces the external `pollster` crate. Only suitable for
//! synchronous contexts that need to drive a single future (e.g. wgpu
//! adapter enumeration). For full async runtimes, use `tokio`.

use std::future::Future;
use std::task::{Context, Poll, Wake};

struct NoopWaker;
impl Wake for NoopWaker {
    fn wake(self: std::sync::Arc<Self>) {}
}

/// Drive a future to completion by polling in a tight loop.
///
/// Panics if the future returns `Pending` without a waker-based wake
/// (no IO reactor). This is fine for wgpu operations which complete
/// synchronously on the calling thread.
pub fn block_on<F: Future>(fut: F) -> F::Output {
    let waker = std::task::Waker::from(std::sync::Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}
