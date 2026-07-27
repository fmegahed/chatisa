/**
 * "Which packages can I install?" logic for the Coding Studio help. Python is
 * authoritative: we read Pyodide's own package lock (public/runtimes/pyodide/
 * pyodide-lock.json), so we can tell a student whether a name is ready now,
 * installable with micropip, or not built for the browser. R stays guidance-only
 * (the full webR repo is not bundled), handled in the UI copy.
 */

/** PyPI-style normalization: lowercase and collapse runs of - _ . to a single -.
 * Applied to both the query and every package/import name so aliases line up. */
export function normalizePkg(name: string): string {
  return name.trim().toLowerCase().replace(/[-_.]+/g, "-");
}

/** What the Coding Studio preloads or hosts, so it is usable with a bare import.
 * A superset of the bundled loadPackage set plus our hosted wheels. */
export const BUNDLED_PYTHON = new Set(
  [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "statsmodels",
    "pyarrow",
    "polars",
    "scipy",
    "seaborn",
    "openpyxl",
    "micropip",
  ].map(normalizePkg),
);

/** Names that categorically cannot be installed in the browser (need compiling or
 * a native toolchain Pyodide does not provide), called out by name so the answer is
 * concrete rather than a vague "maybe". */
export const KNOWN_UNAVAILABLE_PYTHON = new Set(
  ["statsforecast", "pyreadr"].map(normalizePkg),
);

/**
 * R packages installed into every browser session from our own mirror, so
 * library() and pkg::fn() work with no download from the WebR repository.
 *
 * Mirrored, NOT retyped by hand: this must stay identical to BUNDLED_PACKAGES in
 * public/workers/webr-worker.mjs, which cannot be imported here because it is a
 * static ES-module worker loaded by URL rather than through the bundler.
 * tests/unit/packages.test.ts reads that file and fails if the two disagree.
 *
 * tidyverse pulls its whole dependency closure with it, which is why rvest,
 * dplyr, ggplot2, stringr, readr, purrr and tibble are all usable without being
 * named here.
 */
export const BUNDLED_R = ["tidyverse", "readxl", "janitor"] as const;

/**
 * Also hosted on our mirror and installable instantly, but not preinstalled: a
 * session pays for them only on first use. httr2 for web requests, ggtext and
 * ggrepel for the house chart style's subtitles and labels.
 */
export const MIRRORED_R = ["httr2", "ggtext", "ggrepel"] as const;

/**
 * Packages that arrive as part of the tidyverse install and are therefore
 * present in every session, without being named in BUNDLED_R.
 *
 * This exists because BUNDLED_R lists what we ask to be INSTALLED, not what ends
 * up installed, and the difference is most of what students actually type.
 * `dplyr::mutate()` and `ggplot2::ggplot()` are the two most common calls in this
 * app's R code and neither package is in BUNDLED_R.
 *
 * Not the full closure, which is 110 packages: the widely used ones, so a snippet
 * using them is classified "ready" rather than "unknown" even before the
 * generated manifest is available. tests/unit/runnable.test.ts checks each name
 * against the mirror on disk when the mirror is present, so a wrong entry here
 * cannot survive quietly.
 */
export const BUNDLED_R_CLOSURE = [
  "broom", "cli", "conflicted", "dbplyr", "dplyr", "dtplyr", "forcats",
  "ggplot2", "glue", "googledrive", "googlesheets4", "haven", "hms",
  "htmltools", "httr", "jsonlite", "knitr", "lubridate", "magrittr", "modelr",
  "pillar", "purrr", "readr", "reprex", "rlang", "rmarkdown", "rvest", "scales",
  "stringr", "tibble", "tidyr", "tidyselect", "vctrs", "withr", "xml2",
] as const;

export interface PyodideIndex {
  /** Canonical package names Pyodide builds (excludes its -tests helper entries). */
  packages: Set<string>;
  /** Normalized package name OR import name -> canonical package name. */
  byAlias: Map<string, string>;
}

/** Builds the lookup index from a parsed pyodide-lock.json. Each entry's `imports`
 * are the real import names, so `sklearn` resolves to `scikit-learn` and `bs4` to
 * `beautifulsoup4` without a hand-kept alias table. */
export function buildPyodideIndex(lock: {
  packages: Record<string, { name?: string; imports?: string[] }>;
}): PyodideIndex {
  const packages = new Set<string>();
  const byAlias = new Map<string, string>();
  for (const [key, entry] of Object.entries(lock.packages ?? {})) {
    if (key.endsWith("-tests")) continue; // Pyodide's per-package test bundles
    const canonical = entry.name ?? key;
    packages.add(canonical);
    byAlias.set(normalizePkg(canonical), canonical);
    for (const imp of entry.imports ?? []) byAlias.set(normalizePkg(imp), canonical);
  }
  return { packages, byAlias };
}

export type PackageStatus = "ready" | "installable" | "unavailable";

export interface PackageResult {
  status: PackageStatus;
  /** The canonical package name, when the query resolved to one. */
  canonical?: string;
  message: string;
}

/** Classifies a Python package query against the Pyodide index. Returns null for an
 * empty query so the UI can show nothing until the student types. */
export function classifyPythonPackage(
  query: string,
  index: PyodideIndex,
): PackageResult | null {
  const q = normalizePkg(query);
  if (!q) return null;
  // Resolve to a canonical name via the lock's aliases; fall back to the query
  // itself for a bundled package that Pyodide does not build (seaborn, openpyxl are
  // our hosted wheels, so they are not in the lock but are still ready to use).
  const canonical = index.byAlias.get(q) ?? (BUNDLED_PYTHON.has(q) ? q : undefined);
  if (canonical) {
    if (BUNDLED_PYTHON.has(normalizePkg(canonical))) {
      return {
        status: "ready",
        canonical,
        message: `${canonical} is ready to use. Just import it.`,
      };
    }
    return {
      status: "installable",
      canonical,
      message: `${canonical} is available. Just import it and the runtime fetches it the first time, or add it up front with: import micropip; await micropip.install("${canonical}")`,
    };
  }
  if (KNOWN_UNAVAILABLE_PYTHON.has(q)) {
    return {
      status: "unavailable",
      message: `${query} cannot be installed here: it needs compiling, which the browser runtime does not do.`,
    };
  }
  return {
    status: "unavailable",
    message: `${query} is not in the browser's built package set. micropip can still add it if it is a pure-Python package; packages with C or Fortran code (for example statsforecast, pyreadr) cannot be installed.`,
  };
}
