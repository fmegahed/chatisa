/**
 * What packages does this snippet need?
 *
 * Pure text analysis, deliberately: it has to answer before any runtime is
 * booted, because the answer decides whether a Run button is offered at all. If
 * this needed a runtime it would defeat its own purpose, since booting Pyodide
 * or WebR for every code block on a page is exactly the cost the lazy-load
 * design avoids.
 *
 * Over-reporting is safe (the classifier will find the package is fine);
 * under-reporting is safe too (the run proceeds and the runtime installs on
 * demand, as it always has). The one thing that must not happen is reporting a
 * package the snippet does not need, because that could hide a Run button on a
 * snippet that works. So every pattern here is anchored and conservative.
 */

/**
 * Python's standard library, which is present and never installable. Not
 * exhaustive over every stdlib module, only the ones that actually turn up in
 * teaching code; an unlisted stdlib module is reported as a requirement and the
 * classifier then declines to block on it.
 */
const PYTHON_STDLIB = new Set([
  "abc", "argparse", "array", "ast", "asyncio", "base64", "bisect", "builtins",
  "calendar", "cmath", "collections", "concurrent", "contextlib", "copy", "csv",
  "ctypes", "dataclasses", "datetime", "decimal", "difflib", "enum", "errno",
  "fnmatch", "fractions", "functools", "gc", "getpass", "glob", "gzip", "hashlib",
  "heapq", "hmac", "html", "http", "importlib", "inspect", "io", "ipaddress",
  "itertools", "json", "keyword", "locale", "logging", "math", "mimetypes",
  "numbers", "operator", "os", "pathlib", "pickle", "platform", "pprint",
  "queue", "random", "re", "secrets", "shutil", "signal", "site", "socket",
  "sqlite3", "statistics", "string", "struct", "subprocess", "sys", "tempfile",
  "textwrap", "threading", "time", "timeit", "tokenize", "traceback", "types",
  "typing", "unicodedata", "unittest", "urllib", "uuid", "warnings", "weakref",
  "zipfile", "zoneinfo",
  // Pyodide's own bridge modules, always present in the browser runtime.
  "js", "pyodide", "pyscript", "micropip",
]);

/**
 * R's base and recommended packages, shipped inside WebR itself. `stats` and
 * `utils` in particular appear constantly in teaching code, and treating them as
 * requirements would block almost every R snippet.
 */
const R_BASE = new Set([
  "base", "compiler", "datasets", "graphics", "grDevices", "grid", "methods",
  "parallel", "splines", "stats", "stats4", "tcltk", "tools", "utils",
  // WebR's own namespace, which provides install/canvas helpers.
  "webr",
]);

/** Strips comments and string literals so a package name inside them is not
 * mistaken for a real requirement. */
function stripPythonNoise(code: string): string {
  return code
    // Triple-quoted blocks first, or their contents leak into the line scan.
    .replace(/"""[\s\S]*?"""|'''[\s\S]*?'''/g, " ")
    .replace(/#[^\n]*/g, " ");
}

function stripRNoise(code: string): string {
  // R has no block comments. Quoted strings are kept, because library("pkg")
  // and install.packages("pkg") both name the package inside quotes.
  return code.replace(/#[^\n]*/g, " ");
}

/**
 * Top-level module names a Python snippet imports.
 *
 * Only the first component matters: `import matplotlib.pyplot as plt` needs
 * matplotlib. Relative imports are skipped: `from . import x` names nothing
 * installable.
 */
export function pythonRequirements(code: string): string[] {
  const text = stripPythonNoise(code);
  const found = new Set<string>();

  // `import a, b.c as d`
  for (const match of text.matchAll(/^[ \t]*import[ \t]+([^\n#]+)/gm)) {
    for (const part of match[1].split(",")) {
      const name = part.trim().split(/[ \t]+as[ \t]+/)[0].trim().split(".")[0];
      if (name && /^[A-Za-z_]\w*$/.test(name)) found.add(name);
    }
  }
  // `from a.b import c`. A leading dot is a relative import: not a package.
  for (const match of text.matchAll(/^[ \t]*from[ \t]+([A-Za-z_][\w.]*)[ \t]+import/gm)) {
    const name = match[1].split(".")[0];
    if (name) found.add(name);
  }
  // `micropip.install("x")` and `await micropip.install("x")`: an explicit
  // request, so it counts even though nothing is imported yet.
  for (const match of text.matchAll(/micropip\.install\(\s*["']([^"']+)["']/g)) {
    found.add(match[1].split(/[<>=!\[]/)[0].trim());
  }

  return [...found].filter((name) => !PYTHON_STDLIB.has(name)).sort();
}

/**
 * Package names an R snippet needs.
 *
 * Covers the four ways R code names a package, including the pkg::fn() form this
 * app's own style rules mandate, which a library()-only scan would miss entirely.
 */
export function rRequirements(code: string): string[] {
  const text = stripRNoise(code);
  const found = new Set<string>();

  const patterns = [
    // library(dplyr) / library("dplyr") / require(dplyr)
    /\b(?:library|require|loadNamespace|requireNamespace)\s*\(\s*["']?([A-Za-z][\w.]*)["']?/g,
    // install.packages("zoo") and install.packages(c("a","b")) handled below
    /\binstall\.packages\s*\(\s*["']([A-Za-z][\w.]*)["']/g,
    // dplyr::mutate() and dplyr:::internal()
    /\b([A-Za-z][\w.]*)\s*:::?\s*[A-Za-z._]/g,
  ];
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) found.add(match[1]);
  }
  // install.packages(c("a", "b"))
  for (const match of text.matchAll(/\binstall\.packages\s*\(\s*c\(([^)]*)\)/g)) {
    for (const quoted of match[1].matchAll(/["']([A-Za-z][\w.]*)["']/g)) {
      found.add(quoted[1]);
    }
  }

  return [...found].filter((name) => !R_BASE.has(name)).sort();
}

export type RunLanguageId = "python" | "r" | "sql";

/** Requirements for whichever language the block is in. SQL has no packages. */
export function requirementsFor(language: RunLanguageId, code: string): string[] {
  if (language === "python") return pythonRequirements(code);
  if (language === "r") return rRequirements(code);
  return [];
}
