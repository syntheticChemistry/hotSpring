+++
title = "hotSpring sporePrint Directory"
description = "Content pipeline for primals.eco — validation summaries, notebooks, and frozen data"
date = 2026-05-10
template = "page.html"
render = false

[taxonomies]
springs = ["hotspring"]
+++

# sporeprint/ — Content for primals.eco

Files in this directory are published to [primals.eco](https://primals.eco) via
the sporePrint auto-refresh CI pipeline.

## How it works

1. When you push to `main`, your `notify-sporeprint.yml` workflow fires
2. If your dispatch payload includes `"content": "true"`, sporePrint CI
   clones this repo and copies `sporeprint/*.md` into `content/lab/`
3. A PR is created for human review before merging to the live site

## What goes here

- `validation-summary.md` — hotSpring's headline validation results (596/1,045 tests, 213 experiments, guideStone Level 6)
- Additional `.md` pages with Zola-compatible front matter
- Results, benchmarks, or experiment summaries you want visible on primals.eco

## Notebooks

**sporePrint notebooks (5)** in `notebooks/` — see `notebooks/NOTEBOOK_PATTERN.md` for the pattern.
Frozen data in `experiments/results/*.json` (6 JSON files). Render via `jupyter nbconvert --execute`.

**Paper baseline notebooks (13)** in `notebooks/papers/` — publishable Python baselines for 25 reproduced papers.
See `notebooks/papers/PAPER_NOTEBOOK_GUIDE.md` for the collaborator pattern. Live compute for small
problems (SEMF, Yukawa, spectral, small-lattice QCD/Higgs), frozen JSON for production runs.

## Front matter requirements

Every `.md` file needs Zola TOML front matter with `[taxonomies]` for cross-referencing:

```toml
+++
title = "Your Page Title"
description = "One-line summary"
date = 2026-05-06

[taxonomies]
primals = ["barracuda", "toadstool"]
springs = ["yourspring"]
+++
```

See [CONTENT_GUIDE.md](https://github.com/ecoPrimals/wateringHole/blob/main/sporePrint/CONTENT_GUIDE.md)
for full documentation.
