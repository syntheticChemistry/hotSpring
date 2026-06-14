# Sarkas Version Note — hotSpring Control Study

**Date**: 2026-02-07
**Author**: Eastgate (AI-assisted)

## Finding

Sarkas HEAD (`7b60e210`, version 1.1.0) has a **dump file corruption bug** introduced
by commit `4b561baa` ("Added multithreading for dumping."). All `.npz` checkpoint
files contain NaN positions/velocities from ~step 10 onward, while the simulation
engine itself runs correctly (progress bar completes, timing is realistic).

This cascades to all post-processed observables (RDF, DSF, SSF, VACF) which read
from dump files.

## Resolution

Pinned to **Sarkas v1.0.0** (`fd908c41`, tagged release). This version:

- Produces valid dump files (positions, velocities, accelerations all non-NaN)
- Tracks energy correctly (zero NaN rows in production CSV)
- Temperature control works (mean 5004.6 K for target 5000 K in quickstart)
- All post-processed observables compute correctly

## Performance Note

v1.0.0 is ~150x slower than v1.1.0 for PP force computation (22 it/s vs 3400 it/s
for 1000 particles). The speedup in v1.1.0 came from Numba JIT optimizations in
the same commit range that introduced the dump bug.

This performance gap is **relevant to the BarraCUDA gap analysis**: even the Python
scientific stack has optimization/correctness tradeoffs that are non-trivial to resolve.

## Impact on Control Study

- **Quickstart (1000 particles, PP)**: ~8 min total — fine on any gate
- **DSF Study Lite (2000 particles, PP)**: ~27 min — fine on Eastgate
- **DSF Study Full (10000 particles, PP)**: hours — needs Strandgate
- **DSF Study PPPM (kappa=0 cases)**: segfaults — needs investigation separate from version

## Upstream

Consider filing a bug report against `murillo-group/sarkas` for commit `4b561baa`.
The fix likely involves proper memory copying before threaded dump writing.

