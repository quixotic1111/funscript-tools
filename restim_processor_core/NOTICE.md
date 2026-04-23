# NOTICE

The `restim_processor_core` package is an extraction of the signal
processing engine from the **Restim Funscript Processor** project,
originally created by **edger477** and contributors.

- Upstream repository: https://github.com/edger477/funscript-tools
- This fork (extraction host): https://github.com/quixotic1111/funscript-tools

## Original authors and contributors

The core signal-processing algorithms (electrode projection, speed
processing, alpha/beta generation, curve quantization, traveling wave,
1D→2D conversion, etc.) were authored by:

- **edger477** — original project creator, primary author of the
  processing engine
- **Senorgif2** — contributor
- **lqr** — contributor

Subsequent modifications (Spatial 3D Linear pipeline extensions,
noise gate, one-euro smoothing, input sharpening, envelope compression,
and related tuning) were contributed by **quixotic1111** in the
downstream fork.

Individual authorship of every file is preserved in the git history
via `git log` and `git blame`.

## License status

As of this extraction, **no LICENSE file has been declared in the
upstream repository**. This package is therefore maintained as an
internal fork restructure and is intended to be installed locally by
tools that consume it (e.g. FunGen).

If this package is ever to be distributed, published, or released as
a standalone project, explicit licensing permission from the original
authors (primarily edger477) should be obtained, or a clean-room
reimplementation of the processing code should be performed.

## How this package is used

`restim_processor_core` is consumed by:

1. The **Restim Funscript Processor** GUI (same repo, `ui/` + top-level
   app scripts) — imports the library locally via its repo checkout.
2. **FunGen** (https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator)
   — installs this package editable into its Python environment to
   drive device-ready output from video-tracked motion files.
