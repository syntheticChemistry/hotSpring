# sporeprint/ — Content for primals.eco

Files in this directory are published to [primals.eco](https://primals.eco) via
the sporePrint auto-refresh CI pipeline.

## How it works

1. When you push to `main`, your `notify-sporeprint.yml` workflow fires
2. If your dispatch payload includes `"content": "true"`, sporePrint CI
   clones this repo and copies `sporeprint/*.md` into `content/lab/`
3. A PR is created for human review before merging to the live site

## What goes here

- `validation-summary.md` — your spring's headline validation results
- Additional `.md` pages with Zola-compatible front matter
- Results, benchmarks, or experiment summaries you want visible on primals.eco

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
