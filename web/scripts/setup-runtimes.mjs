/**
 * Copies the self-hosted WebAssembly runtimes into public/runtimes/ so the
 * in-browser code runner loads them from our own origin rather than a CDN.
 *
 * Run with `npm run setup:runtimes`. The output lives under public/runtimes/,
 * which is gitignored: these are large binary assets fetched at deploy time,
 * not committed. Re-run after changing a runtime version.
 *
 * Phase 1 handles SQLite (SQL), phase 2 Pyodide (Python), phase 3 WebR (R).
 */
import { cp, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const outRoot = join(root, "public", "runtimes");

async function copyFile(from, to) {
  await mkdir(dirname(to), { recursive: true });
  await cp(from, to);
  const size = (await stat(to)).size;
  console.log(`  ${to.replace(root, ".")}  (${(size / 1024).toFixed(0)} KB)`);
}

/**
 * SQLite: the loader ES module plus its wasm. The loader locates the wasm
 * through a `locateFile` callback, so the two only need to be reachable at the
 * URL we serve them from.
 */
async function setupSqlite() {
  const dist = join(root, "node_modules", "@sqlite.org", "sqlite-wasm", "dist");
  if (!existsSync(dist)) {
    throw new Error(
      "@sqlite.org/sqlite-wasm is not installed. Run npm install first.",
    );
  }
  const version = JSON.parse(
    await readFile(
      join(root, "node_modules", "@sqlite.org", "sqlite-wasm", "package.json"),
      "utf8",
    ),
  ).version;

  console.log(`SQLite ${version}:`);
  const out = join(outRoot, "sqlite");
  // index.mjs is the bundler-friendly loader (default export sqlite3InitModule).
  await copyFile(join(dist, "index.mjs"), join(out, "sqlite3.mjs"));
  await copyFile(join(dist, "sqlite3.wasm"), join(out, "sqlite3.wasm"));
}

/**
 * The Python packages we host, so a Coding Companion snippet can import them and
 * run offline against our own origin. Everything they depend on is added
 * automatically from the lock file, so this stays a short, intentional list.
 */
const PYTHON_PACKAGES = [
  "numpy",
  "pandas",
  "matplotlib",
  "scikit-learn",
  "statsmodels", // regression and statistical models
  "pyarrow", // Parquet (and faster pandas), for the data upload feature
  "polars", // modern DataFrames
  "micropip", // installs our hosted pure-Python wheels (seaborn, openpyxl) at runtime
  // Web parsing for Ask Anything (design 2026-07-24): the model fetches pages
  // via read_url and parses them in the Python runtime. These are lock-listed
  // but were not mirrored, so `import bs4` used to 404 against our origin.
  "beautifulsoup4",
  "lxml",
  "html5lib",
  "requests",
  // Runtime dependencies of statsforecast (v6.3.0), which we ship as our own
  // wasm wheel. They are Pyodide-built, but a hosted wheel installs with deps
  // off, so anything it imports must be mirrored here or it 404s offline.
  "cloudpickle",
  "tqdm",
  "threadpoolctl",
  "narwhals", // utilsforecast's dataframe abstraction
  "packaging",
  "six", // triad (fugue's core) imports these two
  "fsspec",
  // scipy is pulled in automatically as a dependency of several of the above.
];

/**
 * Pure-Python packages that Pyodide does not build into its own distribution,
 * so we fetch their wheels from PyPI and host them ourselves. This keeps common
 * tools available offline and same-origin: seaborn (statistical plots) and
 * openpyxl (reading .xlsx, for the upload feature) plus its small dependency,
 * and the two chart-style helpers (design 2026-07-25) that mirror what ggrepel
 * and ggtext do in R.
 *
 * Each name is spelled as it is IMPORTED, not as PyPI distributes it
 * ("highlight_text", which PyPI resolves to highlight-text), because the wheel
 * manifest this writes is keyed by the name the worker matches against a
 * snippet's import statements.
 */
const PYPI_WHEELS = [
  "seaborn",
  "openpyxl",
  "et_xmlfile",
  "adjustText", // non-overlapping annotations, the adjust_text function
  "highlight_text", // coloured words inside a title or subtitle
  // The statsforecast support cast (v6.3.0), all pure Python. utilsforecast
  // is Nixtla's helper library; fugue (with triad and adagio) is imported
  // unconditionally at the top of statsforecast.core, even though only its
  // cluster backends would ever use it. Their pandas<3 pins are metadata
  // only: verified working against Pyodide's pandas 3.0.2 on 2026-08-20, and
  // deps=False installs never consult the pin anyway.
  "utilsforecast",
  "fugue",
  "triad",
  "adagio",
];

/**
 * Wheels we CROSS-COMPILED for this exact Pyodide ABI because PyPI only ships
 * native builds: the Nixtla forecasting stack (statsforecast and its compiled
 * kernel library coreforecast) has no public wasm build, so
 * scripts/build-wasm-wheels.sh produces one in a Docker container and the
 * artifacts are checked into vendor/pyodide-wasm-wheels/. This copies them
 * into the served wheel directory beside the PyPI ones and adds them to the
 * same manifest, keyed by import name like everything else.
 *
 * ABI-PINNED: these wheels are valid only for the Pyodide in package.json.
 * After a Pyodide upgrade, re-run scripts/build-wasm-wheels.sh; the tag check
 * below fails the setup loudly rather than serving an incompatible wheel.
 */
async function copyWasmWheels(out, manifest) {
  const { readdir } = await import("node:fs/promises");
  const vendorDir = join(root, "vendor", "pyodide-wasm-wheels");
  if (!existsSync(vendorDir)) {
    console.log("  vendor/pyodide-wasm-wheels: absent, skipping (forecast stack unavailable)");
    return;
  }
  const lock = JSON.parse(
    await readFile(join(root, "public", "runtimes", "pyodide", "pyodide-lock.json"), "utf8"),
  );
  const abi = lock.info.abi_version; // e.g. "2026_0"
  for (const file of (await readdir(vendorDir)).filter((f) => f.endsWith(".whl"))) {
    if (!file.includes("pyodide") && !file.includes("emscripten")) {
      throw new Error(`vendor wheel ${file} is not a wasm wheel`);
    }
    if (!file.includes(abi) && !file.includes(lock.info.platform)) {
      throw new Error(
        `vendor wheel ${file} does not match the Pyodide ABI ${abi} / ${lock.info.platform}; re-run scripts/build-wasm-wheels.sh`,
      );
    }
    const importName = file.split("-")[0];
    await copyFile(join(vendorDir, file), join(out, file));
    manifest[importName] = file;
  }
}

/** The transitive closure of `roots` over the lock file's `depends` edges. */
function packageClosure(lockPackages, roots) {
  const closure = new Set();
  const visit = (name) => {
    const key = name.toLowerCase();
    if (closure.has(key)) return;
    const pkg = lockPackages[key] ?? lockPackages[name];
    if (!pkg) {
      throw new Error(`package "${name}" is not in the Pyodide lock file`);
    }
    closure.add(key);
    for (const dep of pkg.depends ?? []) visit(dep);
  };
  for (const root of roots) visit(root);
  return [...closure].map((key) => lockPackages[key]);
}

/**
 * Pyodide: the core interpreter files plus only the wheels our curated package
 * set needs. The npm package ships the interpreter but not the wheels, so the
 * wheels are fetched once from the pinned CDN release into our own runtimes
 * directory. The browser then downloads, per run, only the wheels a snippet
 * actually imports.
 */
async function setupPyodide() {
  const pkgDir = join(root, "node_modules", "pyodide");
  if (!existsSync(pkgDir)) {
    throw new Error("pyodide is not installed. Run npm install first.");
  }
  const version = JSON.parse(
    await readFile(join(pkgDir, "package.json"), "utf8"),
  ).version;
  const cdn = `https://cdn.jsdelivr.net/pyodide/v${version}/full`;

  console.log(`Pyodide ${version}:`);
  const out = join(outRoot, "pyodide");

  // Core interpreter: the loader, its asm module and wasm, the standard library
  // and the lock file. loadPyodide reads the rest from indexURL at runtime.
  const core = [
    "pyodide.mjs",
    "pyodide.asm.mjs",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
    "pyodide-lock.json",
  ];
  for (const file of core) {
    await copyFile(join(pkgDir, file), join(out, file));
  }

  const lock = JSON.parse(await readFile(join(pkgDir, "pyodide-lock.json"), "utf8"));
  const wanted = packageClosure(lock.packages, PYTHON_PACKAGES);
  console.log(
    `  ${wanted.length} wheels (${PYTHON_PACKAGES.join(", ")} and dependencies):`,
  );
  for (const pkg of wanted) {
    await downloadFile(`${cdn}/${pkg.file_name}`, join(out, pkg.file_name));
  }
}

/**
 * Downloads the pure-Python wheels Pyodide does not build (see PYPI_WHEELS) from
 * PyPI into our own origin, so they can be installed offline and same-origin at
 * runtime (via micropip pointed at these files). Only "py3-none-any" wheels are
 * taken, since those are platform-independent and safe to run under Pyodide.
 */
async function setupPypiWheels() {
  console.log(`PyPI wheels (not built by Pyodide): ${PYPI_WHEELS.join(", ")}`);
  const out = join(outRoot, "pyodide-wheels");
  await mkdir(out, { recursive: true });
  // A manifest maps each import name to its (versioned) wheel file, so the
  // worker can install it with micropip without hard-coding a version.
  const manifest = {};
  for (const name of PYPI_WHEELS) {
    const meta = await fetch(`https://pypi.org/pypi/${name}/json`);
    if (!meta.ok) {
      throw new Error(`PyPI lookup for "${name}" failed: ${meta.status}`);
    }
    const data = await meta.json();
    const wheel = (data.urls ?? []).find(
      (u) => u.packagetype === "bdist_wheel" && u.filename.endsWith("-none-any.whl"),
    );
    if (!wheel) {
      throw new Error(`no pure-Python (none-any) wheel found for "${name}"`);
    }
    await downloadFile(wheel.url, join(out, wheel.filename));
    manifest[name] = wheel.filename;
  }
  await copyWasmWheels(out, manifest);
  await writeFile(join(out, "wheels.json"), JSON.stringify(manifest, null, 2) + "\n");
}

/** Fetches a URL to disk, reporting its size like copyFile does. Skips the
 * download when the file already exists (versioned names make this safe), so
 * re-running the setup only fetches what is new. */
async function downloadFile(url, to) {
  await mkdir(dirname(to), { recursive: true });
  if (existsSync(to)) {
    console.log(`  ${to.replace(root, ".")}  (cached)`);
    return;
  }
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`fetch ${url} failed: ${res.status} ${res.statusText}`);
  }
  const bytes = Buffer.from(await res.arrayBuffer());
  await writeFile(to, bytes);
  console.log(`  ${to.replace(root, ".")}  (${(bytes.length / 1024).toFixed(0)} KB)`);
}

/**
 * WebR: the whole distribution (interpreter wasm, base R virtual filesystem,
 * loader and channel worker) copied to our own origin, so the runtime loads
 * from us. Only user-installed R packages are fetched at runtime, from the WebR
 * package repository, which is why R packages are not bundled here.
 */
async function setupWebR() {
  const dist = join(root, "node_modules", "webr", "dist");
  if (!existsSync(dist)) {
    throw new Error("webr is not installed. Run npm install first.");
  }
  const version = JSON.parse(
    await readFile(join(root, "node_modules", "webr", "package.json"), "utf8"),
  ).version;

  console.log(`WebR ${version}:`);
  const out = join(outRoot, "webr");
  await mkdir(out, { recursive: true });
  // The distribution's files reference each other by relative path, so the
  // whole directory is copied rather than a hand-picked subset.
  await cp(dist, out, { recursive: true });
  const bytes = await directorySize(out);
  console.log(
    `  ${out.replace(root, ".")}  (${(bytes / 1048576).toFixed(0)} MB, self-hosted core)`,
  );
}

/**
 * The R packages we pre-bundle so the Coding Studio's R examples run instantly
 * and offline, without reaching the WebR package repository. tidyverse is the
 * data-analysis core (dplyr, ggplot2, tibble, tidyr, readr, and the rest);
 * readxl reads Excel files (for the data upload) and janitor cleans column
 * names; httr2 does web requests. Heavier sets (tidymodels, fpp2, fpp3) were
 * bundled briefly on 2026-07-24 and REVERTED the same day by the professor's
 * direction: they tripled the mirror for packages a session installs
 * on-demand in one line anyway. install.packages("tidymodels") still works,
 * served by the public webR repo.
 */
// ggtext and ggrepel serve the house chart style (design 2026-07-25): coloured
// words in a subtitle that replace a legend, and labels that do not sit on the
// geoms or each other. They add 9 packages and 4.2 MB (Rcpp, commonmark,
// gridtext, jpeg, litedown, markdown, png ride along), all prebuilt for
// WebAssembly, so nothing compiles.
const WEBR_PACKAGES = [
  "tidyverse",
  "readxl",
  "janitor",
  "httr2",
  "ggtext",
  "ggrepel",
];

// WebR 0.6.0 ships R 4.6, whose WebAssembly binaries live under this repo path.
const WEBR_REPO = "https://repo.r-wasm.org";
const WEBR_CONTRIB = "bin/emscripten/contrib/4.6";

/** Parses a CRAN PACKAGES index (DCF) into records keyed by package name. */
function parsePackagesIndex(text) {
  const records = new Map();
  for (const block of text.split(/\n\s*\n/)) {
    if (!block.trim()) continue;
    // Unfold DCF continuation lines (a field wraps onto lines that are indented).
    const unfolded = block.replace(/\n[ \t]+/g, " ");
    const fields = {};
    for (const line of unfolded.split("\n")) {
      const m = /^([A-Za-z0-9]+):\s?(.*)$/.exec(line);
      if (m) fields[m[1]] = m[2];
    }
    if (fields.Package) records.set(fields.Package, { block, fields });
  }
  return records;
}

/**
 * The runtime dependencies (Depends, Imports) of a record, names only.
 * LinkingTo is deliberately excluded: it supplies C/C++ headers needed only to
 * compile a package, and we ship pre-built binaries, so those header-only
 * packages (BH, cpp11, and the like) never need to be loaded at runtime.
 */
function hardDeps(fields) {
  const names = [];
  for (const key of ["Depends", "Imports"]) {
    const raw = fields[key];
    if (!raw) continue;
    for (const part of raw.split(",")) {
      const name = part.trim().replace(/\s*\(.*\)$/, ""); // drop version constraints
      if (name && name !== "R") names.push(name);
    }
  }
  return names;
}

/**
 * The transitive closure of `roots` over Depends/Imports/LinkingTo, keeping only
 * packages present in the repo index. Base and recommended packages that ship
 * with WebR are not in the index, so they fall away naturally.
 */
function webrClosure(index, roots) {
  const closure = new Set();
  const visit = (name) => {
    if (closure.has(name) || !index.has(name)) return;
    closure.add(name);
    for (const dep of hardDeps(index.get(name).fields)) visit(dep);
  };
  for (const root of roots) {
    if (!index.has(root)) {
      throw new Error(`R package "${root}" is not in the WebR repo index`);
    }
    visit(root);
  }
  return closure;
}

/**
 * Mirrors the pre-bundled R packages (and their dependency closure) from the
 * WebR repository into our own origin, with a filtered PACKAGES index listing
 * exactly what we host. At runtime the worker points R's repo at this mirror, so
 * library() loads from us. A package outside this set (tidymodels and friends)
 * is not listed here, so its on-demand install falls through to the real repo.
 */
/**
 * Writes PACKAGES.rds beside PACKAGES and PACKAGES.gz, when R is available here.
 *
 * R asks a repository for PACKAGES.rds FIRST and falls back to the gzip and plain
 * forms, so this file is purely an optimisation: without it every first R run in
 * the browser logs a console error containing a full HTML 404 page, in exactly the
 * place a student looks while debugging their own code.
 *
 * Generated with the local R rather than hand-serialised, because the format is
 * R's own and `saveRDS(read.dcf(...))` is precisely what a CRAN-style repository
 * contains: a character matrix of the DCF fields.
 *
 * OPTIONAL BY DESIGN, and that matters for where this runs. `npm run
 * setup:runtimes` populates public/runtimes on a DEVELOPMENT machine, and the
 * result is shipped inside the deploy bundle, so the production server never needs
 * R. If R is missing wherever this does run, the step is skipped with a note and
 * the mirror still works exactly as before.
 */
async function writePackagesRds(out) {
  const { spawnSync } = await import("node:child_process");
  const probe = spawnSync("Rscript", ["--version"], { encoding: "utf8" });
  if (probe.error || probe.status !== 0) {
    console.log(
      "  PACKAGES.rds: skipped (no Rscript here). R falls back to PACKAGES.gz,",
    );
    console.log(
      "    so the mirror works; the only cost is a 404 in the browser console",
    );
    console.log("    on a student's first R run.");
    return;
  }
  // Written by R, in R's own format. The directory travels in an ENVIRONMENT
  // VARIABLE rather than argv or string interpolation: a Windows path is full of
  // backslashes that would need escaping inside R source, and `Rscript -e ...
  // --args <path>` fails outright on the R build here ("The system cannot find
  // the path specified"). Sys.getenv has neither problem.
  // ONE -e per statement. A single -e carrying newline-separated statements runs
  // only the FIRST of them on this R build, and does so with exit status 0: the
  // assignment succeeded, nothing else ran, no file appeared, and the script
  // cheerfully reported success. Separate flags are the documented form.
  const statements = [
    'dir <- Sys.getenv("CHATISA_PACKAGES_DIR")',
    'db <- read.dcf(file.path(dir, "PACKAGES"))',
    'saveRDS(db, file.path(dir, "PACKAGES.rds"), compress = "xz")',
    "cat(nrow(db))",
  ];
  const run = spawnSync("Rscript", statements.flatMap((s) => ["-e", s]), {
    encoding: "utf8",
    env: { ...process.env, CHATISA_PACKAGES_DIR: out },
  });
  // Verify the FILE, not the exit status. See the note above: R exited 0 having
  // written nothing, so status alone reported a success that had not happened.
  const rds = join(out, "PACKAGES.rds");
  if (run.status !== 0 || !existsSync(rds)) {
    console.log(
      `  PACKAGES.rds: skipped (Rscript did not produce it${
        run.stderr?.trim() ? `: ${run.stderr.trim().slice(0, 120)}` : ""
      })`,
    );
    return;
  }
  const version = (probe.stdout || probe.stderr || "").trim().split("\n")[0];
  const size = (await stat(rds)).size;
  console.log(
    `  PACKAGES.rds  (${(size / 1024).toFixed(0)} KB, ${run.stdout.trim()} packages, via ${version})`,
  );
}

async function setupWebRPackages() {
  console.log(`R packages (mirror of ${WEBR_PACKAGES.join(", ")} + dependencies):`);
  const indexUrl = `${WEBR_REPO}/${WEBR_CONTRIB}/PACKAGES`;
  const res = await fetch(indexUrl);
  if (!res.ok) {
    throw new Error(`fetch ${indexUrl} failed: ${res.status} ${res.statusText}`);
  }
  const index = parsePackagesIndex(await res.text());
  const closure = webrClosure(index, WEBR_PACKAGES);
  const names = [...closure].sort();
  console.log(`  ${names.length} packages in the closure`);

  const out = join(outRoot, "webr-packages", WEBR_CONTRIB);
  await mkdir(out, { recursive: true });

  let total = 0;
  const blocks = [];
  for (const name of names) {
    const { block, fields } = index.get(name);
    const file = fields.File ?? `${name}_${fields.Version}.tgz`;
    const dest = join(out, file);
    // Re-downloads are skipped when the file already exists, so re-running the
    // setup after a version bump only fetches what changed.
    if (existsSync(dest)) {
      total += (await stat(dest)).size;
    } else {
      const url = `${WEBR_REPO}/${WEBR_CONTRIB}/${file}`;
      const dl = await fetch(url);
      if (!dl.ok) throw new Error(`fetch ${url} failed: ${dl.status}`);
      const bytes = Buffer.from(await dl.arrayBuffer());
      await writeFile(dest, bytes);
      total += bytes.length;
    }
    blocks.push(block.trim());
  }

  // A PACKAGES index containing only the mirrored subset, so available.packages()
  // against our mirror reports exactly what we host and nothing 404s. Both the
  // plain and gzip forms are written: R fetches PACKAGES.gz first, so providing
  // it avoids a noisy failed request before the plain fallback.
  const index_text = blocks.join("\n\n") + "\n";
  const { gzipSync } = await import("node:zlib");
  await writeFile(join(out, "PACKAGES"), index_text);
  await writeFile(join(out, "PACKAGES.gz"), gzipSync(Buffer.from(index_text)));
  await writePackagesRds(out);

  // An availability manifest for the browser (added 2026-07-26): which packages
  // we host, and every package WebR's repository serves. The UI reads this to
  // decide whether to offer a Run button at all, since a snippet needing a
  // package that cannot exist here can only produce an error.
  //
  // `repo` is names only, from the index already downloaded above, so this costs
  // one extra file and no extra request. Its ABSENCE is meaningful to the
  // client: with no manifest, an unmirrored package is "unknown" and Run is
  // still offered. So a stale or missing file degrades to today's behaviour
  // rather than hiding buttons.
  const manifest = {
    generated: "npm run setup:runtimes",
    contrib: WEBR_CONTRIB,
    mirrored: names.slice().sort(),
    repo: [...index.keys()].sort(),
  };
  await writeFile(
    join(outRoot, "webr-packages", "available.json"),
    JSON.stringify(manifest),
  );

  console.log(
    `  ${out.replace(root, ".")}  (${(total / 1048576).toFixed(0)} MB, ${names.length} packages + PACKAGES index)`,
  );
  console.log(
    `  available.json  (${names.length} mirrored, ${manifest.repo.length} in the WebR repository)`,
  );
}

/**
 * ggsql (experimental, alpha): the WebAssembly build that turns SQL with a
 * VISUALISE clause into a Vega-Lite chart spec, self-hosted so the SQL plot
 * runs from our own origin. Pinned to the version we tested; see the note in the
 * Coding Studio's SQL example. The Vega renderer itself is an npm dependency
 * (vega-embed) bundled by Next, not copied here.
 */
async function setupGgsql() {
  const pkgDir = join(root, "node_modules", "ggsql-wasm");
  if (!existsSync(pkgDir)) {
    throw new Error("ggsql-wasm is not installed. Run npm install first.");
  }
  const version = JSON.parse(
    await readFile(join(pkgDir, "package.json"), "utf8"),
  ).version;

  console.log(`ggsql-wasm ${version} (experimental):`);
  const out = join(outRoot, "ggsql");
  // The JS glue, the wasm, and the snippets it imports by relative path.
  await copyFile(join(pkgDir, "ggsql_wasm.js"), join(out, "ggsql_wasm.js"));
  await copyFile(
    join(pkgDir, "ggsql_wasm_bg.wasm"),
    join(out, "ggsql_wasm_bg.wasm"),
  );
  await cp(join(pkgDir, "snippets"), join(out, "snippets"), { recursive: true });
  console.log(`  ${join(out, "snippets").replace(root, ".")}  (wasm-bindgen snippets)`);
}

/** Total size of a directory tree, for the one-line report. */
async function directorySize(dir) {
  const { readdir } = await import("node:fs/promises");
  let total = 0;
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    total += entry.isDirectory()
      ? await directorySize(full)
      : (await stat(full)).size;
  }
  return total;
}

// Optional phase argument runs a single step (handy when only one changed), for
// example `node scripts/setup-runtimes.mjs webr-packages`. No argument runs all.
const PHASES = {
  sqlite: setupSqlite,
  pyodide: setupPyodide,
  "pypi-wheels": setupPypiWheels,
  webr: setupWebR,
  "webr-packages": setupWebRPackages,
  ggsql: setupGgsql,
};

async function main() {
  console.log(`Populating ${outRoot.replace(root, ".")}\n`);
  const only = process.argv[2];
  if (only) {
    const phase = PHASES[only];
    if (!phase) {
      throw new Error(
        `unknown phase "${only}". Options: ${Object.keys(PHASES).join(", ")}`,
      );
    }
    await phase();
    console.log("\nDone (single phase). These assets are gitignored.");
    return;
  }
  await setupSqlite();
  console.log("");
  await setupPyodide();
  console.log("");
  await setupPypiWheels();
  console.log("");
  await setupWebR();
  console.log("");
  await setupWebRPackages();
  console.log("");
  await setupGgsql();
  console.log("\nDone. These assets are gitignored; re-run after a version bump.");
}

main().catch((err) => {
  console.error("\nsetup:runtimes failed:", err.message);
  process.exit(1);
});
