# Cross-compiled wasm wheels

Wheels we build ourselves because PyPI carries only native (x86/arm) builds
for them: the Nixtla forecasting stack, statsforecast and its compiled kernel
library coreforecast. Pyodide does not build them either, and statsforecast
was this app's canonical "cannot install in the browser" example until we
started shipping these (v6.3.0, 2026-08-20).

- Built by `scripts/build-wasm-wheels.sh` (Docker; pyodide-build + the
  matching emscripten toolchain). Versions are pinned in that script.
- Served by `scripts/setup-runtimes.mjs` (the `pypi-wheels` phase), which
  copies them into `public/runtimes/pyodide-wheels/` beside the pure-Python
  PyPI wheels and refuses a wheel whose ABI tag does not match the Pyodide in
  `package.json`.
- Installed at runtime by `public/workers/pyodide-worker.mjs` (HOSTED_WHEELS)
  when a snippet imports statsforecast or coreforecast, with deps off; the
  worker lists the lock packages each wheel needs so nothing reaches PyPI.

These wheels are ABI-PINNED to one exact Pyodide. After any Pyodide upgrade,
re-run `bash scripts/build-wasm-wheels.sh` and then
`npm run setup:runtimes -- pypi-wheels`; the setup fails loudly on a stale
wheel rather than serving one that cannot load.

Note on statsforecast's `fugue` dependency: statsforecast.core imports fugue
unconditionally even though only its cluster backends (spark, dask, ray;
none installable in a browser) would ever use it, so fugue plus its pure
dependencies triad and adagio are hosted as ordinary PyPI wheels alongside
utilsforecast. Their `pandas<3` pins are metadata only; the stack was
verified live against Pyodide's pandas 3.x on 2026-08-20.
