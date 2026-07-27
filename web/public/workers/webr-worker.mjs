/**
 * In-browser R worker (WebR).
 *
 * A static ES-module worker, loaded by URL rather than through the bundler.
 * WebR's interpreter and base R are self-hosted under /runtimes/webr/, so the
 * runtime loads from our own origin. A runaway snippet is stopped by the manager
 * terminating this worker, which also terminates WebR's nested worker.
 *
 * Cross-origin isolation IS required, contrary to what this comment said until
 * 2026-07-26 ("WebR runs on its PostMessage channel, no cross-origin isolation
 * needed"). Without a SharedArrayBuffer, WebR falls back to a channel with no
 * networking at all, so rvest, httr2 and curl fail with "cannot open the
 * connection" and nothing says why. The route list that guarantees the headers is
 * lib/run/isolation.ts.
 *
 * Packages: tidyverse, readxl and janitor install from our own mirror at
 * /runtimes/webr-packages (see installBundledPackages). Anything else a student
 * asks for comes from the WebR repository on demand, which is the one thing here
 * that reaches beyond our origin.
 *
 * Protocol, shared with the other language workers:
 *   in:  { id, code }
 *   out: { id, ok: true, result } | { id, ok: false, error }
 * result: { text?, table?: { columns, rows }, imageDataUrl? }
 */

let readyPromise = null;

// Pre-bundled R packages, mirrored on our own origin (see scripts/setup-runtimes
// setupWebRPackages). They install from us rather than the WebR repository, so
// they work offline and on networks that block third-party CDNs.
const WEBR_MIRROR = "/runtimes/webr-packages";
const BUNDLED_PACKAGES = ["tidyverse", "readxl", "janitor"];
// An IndexedDB-backed R library, so the bundled packages install once per
// browser and later sessions load them instantly, without re-downloading.
const PERSIST_LIB = "/home/web_user/r-library";

async function getWebR() {
  if (!readyPromise) {
    readyPromise = (async () => {
      // The browser build (exports.browser); webr.mjs targets bundlers and
      // statically imports Node built-ins, which native ESM cannot resolve.
      const { WebR } = await import("/runtimes/webr/webr.js");
      const webR = new WebR({ baseUrl: "/runtimes/webr/" });
      await webR.init();
      // Make library()/require() download a missing package too, matching the
      // shim already in place for install.packages(). This is what lets a
      // student pull in their own packages (for example tidymodels), which are
      // not bundled and so come from the WebR repository on demand.
      await webR.evalRVoid("webr::shim_install()");
      return webR;
    })();
  }
  return readyPromise;
}

// The bundled-package preload is memoised separately from getWebR so it runs
// only for code runs (and the explicit prewarm), never for a keystroke-driven
// autocomplete or a data-viewer fetch, which must stay fast.
let packagesPromise = null;
function ensurePackages(webR) {
  if (!packagesPromise) packagesPromise = installBundledPackages(webR);
  return packagesPromise;
}

// Point R's libcurl at the ws-proxy exactly once, so rvest/httr2/curl reach the
// internet. Harmless when the page is not cross-origin isolated: the setenv
// succeeds, but R's synchronous networking still cannot run, and the request
// fails with libcurl's own error rather than silently.
let networkingProxy = null;
async function ensureNetworking(webR, wsProxy) {
  if (!wsProxy || networkingProxy === wsProxy) return;
  networkingProxy = wsProxy;
  await webR.evalRVoid(`Sys.setenv(ALL_PROXY=${rStr(wsProxy)})`);
}

/**
 * Makes the bundled tidyverse (plus readxl and janitor) available, installed
 * once from our own origin's mirror and cached in the browser's IndexedDB so
 * later sessions load it instantly and offline. Best-effort throughout: if
 * IndexedDB is unavailable (private browsing), it falls back to installing into
 * the in-memory library for this session; if the mirror or install fails, R
 * still works, the packages just are not preloaded.
 */
async function installBundledPackages(webR) {
  const mirror = self.location.origin + WEBR_MIRROR;
  let persistent = false;
  try {
    await webR.FS.mkdir(PERSIST_LIB).catch(() => {});
    await webR.FS.mount("IDBFS", {}, PERSIST_LIB);
    await webR.FS.syncfs(true); // load anything installed in a previous session
    await webR.evalRVoid(`.libPaths(c(${rStr(PERSIST_LIB)}, .libPaths()))`);
    persistent = true;
  } catch {
    // No usable IndexedDB: install fresh into the default library this session.
  }

  let haveAll = false;
  try {
    haveAll = await webR.evalRBoolean(
      `all(c(${BUNDLED_PACKAGES.map(rStr).join(",")}) %in% rownames(installed.packages()))`,
    );
  } catch {
    haveAll = false;
  }
  if (haveAll) return;

  try {
    // mount:false forces a plain .tgz install from our mirror, which is what we
    // host (the repository's filesystem-image variant is not mirrored).
    await webR.installPackages(BUNDLED_PACKAGES, {
      repos: mirror,
      quiet: true,
      mount: false,
    });
    if (persistent) await webR.FS.syncfs(false); // persist for next time
  } catch {
    // Leave R usable without the preloaded packages rather than failing the run.
  }
}

/** Quotes a string for embedding in R code. */
function rStr(s) {
  return '"' + String(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"') + '"';
}

/** Draws a captured plot (an ImageBitmap) onto white and returns a PNG URL. */
async function bitmapToPngDataUrl(bitmap) {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d");
  // R's canvas device is transparent, so paint white first or the plot reads
  // as light-on-black.
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0);
  const blob = await canvas.convertToBlob({ type: "image/png" });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return `data:image/png;base64,${btoa(binary)}`;
}

/** One page of a data.frame's rows (as strings), for the data viewer. */
async function fetchFrameR(webR, name, offset, limit) {
  const shelter = await new webR.Shelter();
  try {
    const rcode = `
local({
  v <- tryCatch(get(${JSON.stringify(name)}, envir = globalenv()), error = function(e) NULL)
  if (is.null(v) || !is.data.frame(v)) stop("__notframe__")
  total <- nrow(v)
  from <- ${Number(offset)} + 1L
  to <- min(${Number(offset)} + ${Number(limit)}, total)
  page <- if (from <= total) v[from:to, , drop = FALSE] else v[0, , drop = FALSE]
  rows <- lapply(seq_len(nrow(page)), function(i) {
    as.character(unlist(lapply(page[i, , drop = FALSE], format)))
  })
  list(total = total, columns = names(v), rows = rows)
})
`;
    const obj = await shelter.captureR(rcode, {
      withAutoprint: false,
      captureStreams: false,
      captureConditions: false,
      captureGraphics: false,
    });
    const js = await obj.result.toJs();
    const named = {};
    (js.names || []).forEach((n, i) => {
      named[n] = js.values[i];
    });
    const columns = named.columns?.values ?? [];
    const rowsList = named.rows?.values ?? [];
    const rows = rowsList.map((rowVec) => {
      const cells = rowVec.values ?? [];
      const row = {};
      columns.forEach((c, i) => {
        row[c] = cells[i];
      });
      return row;
    });
    return { columns, rows, totalRows: named.total?.values?.[0] ?? 0 };
  } finally {
    await shelter.purge();
  }
}

/** Runtime autocomplete: pkg:: exports, obj$ names, or names in scope. */
async function completeR(webR, prefix) {
  // The editor sends the whole document before the cursor, but the token being
  // completed (a name, obj$field, or pkg::export) is always on the current
  // line. R's regexes below anchor with ^.*? and R's "." does not cross
  // newlines, so a multi-line prefix would never match; reduce it to the last
  // line first. (Splitting on both \n and \r keeps it correct on any newline.)
  const lastLine = String(prefix).split(/\r?\n/).pop() ?? "";
  const escaped =
    '"' +
    lastLine.replace(/\\/g, "\\\\").replace(/"/g, '\\"') +
    '"';
  const rcode = `
local({
  p <- ${escaped}
  out <- list(partial = "", labels = character(0), types = character(0))
  if (grepl("::[A-Za-z0-9._]*$", p, perl = TRUE)) {
    pkg <- sub("^.*?([A-Za-z][A-Za-z0-9.]*)::[A-Za-z0-9._]*$", "\\\\1", p, perl = TRUE)
    partial <- sub("^.*::([A-Za-z0-9._]*)$", "\\\\1", p, perl = TRUE)
    ex <- tryCatch(getNamespaceExports(pkg), error = function(e)
      tryCatch(ls(getNamespace(pkg)), error = function(e) character(0)))
    ex <- sort(ex[startsWith(ex, partial)])
    out$partial <- partial; out$labels <- head(ex, 200)
    out$types <- rep("function", length(head(ex, 200)))
  } else if (grepl("[$][A-Za-z0-9._]*$", p, perl = TRUE)) {
    obj <- sub("^.*?([A-Za-z.][A-Za-z0-9._]*)[$][A-Za-z0-9._]*$", "\\\\1", p, perl = TRUE)
    partial <- sub("^.*[$]([A-Za-z0-9._]*)$", "\\\\1", p, perl = TRUE)
    nm <- tryCatch(names(get(obj, envir = globalenv())), error = function(e) character(0))
    nm <- sort(nm[startsWith(nm, partial)])
    out$partial <- partial; out$labels <- head(nm, 200)
    out$types <- rep("property", length(head(nm, 200)))
  } else {
    partial <- sub("^.*?([A-Za-z.][A-Za-z0-9._]*)$", "\\\\1", p, perl = TRUE)
    if (!grepl("^[A-Za-z.]", partial)) partial <- ""
    cand <- ls(envir = globalenv())
    if (nzchar(partial)) {
      ap <- tryCatch(apropos(paste0("^", partial), ignore.case = FALSE),
                     error = function(e) character(0))
      cand <- unique(c(cand, ap))
    }
    cand <- sort(cand[startsWith(cand, partial)])
    out$partial <- partial; out$labels <- head(cand, 200)
    out$types <- rep("variable", length(head(cand, 200)))
  }
  out
})
`;
  const shelter = await new webR.Shelter();
  try {
    const obj = await shelter.captureR(rcode, {
      withAutoprint: false,
      captureStreams: false,
      captureConditions: false,
      captureGraphics: false,
    });
    const js = await obj.result.toJs();
    const named = {};
    (js.names || []).forEach((n, i) => {
      named[n] = js.values[i];
    });
    const labels = named.labels?.values ?? [];
    const types = named.types?.values ?? [];
    return {
      partial: named.partial?.values?.[0] ?? "",
      options: labels.map((label, i) => ({
        label,
        type: types[i] || "variable",
        detail: "",
      })),
    };
  } finally {
    await shelter.purge();
  }
}

/** R code that reads an uploaded file (at `path`) with the student's options
 * and either previews a sample or imports it under `name`. Returns a JSON string
 * (via jsonlite) that the worker parses. */
function buildRFileCode(mode, name, format, options, path) {
  const skip = Number(options.skipRows) || 0;
  const header = options.header !== false ? "TRUE" : "FALSE";
  const sheet = options.sheet ? rStr(options.sheet) : "NULL";
  const object = options.object ? rStr(options.object) : "NULL";
  return `
local({
  path <- ${rStr(path)}; skip <- ${skip}L; header <- ${header}
  sheet <- ${sheet}; objname <- ${object}; fmt <- ${rStr(format)}
  sheets <- NULL; objects <- NULL
  if (fmt == "csv") {
    df <- readr::read_csv(path, skip = skip, col_names = header, show_col_types = FALSE)
  } else if (fmt == "json") {
    df <- as.data.frame(jsonlite::fromJSON(path))
  } else if (fmt == "xlsx") {
    sheets <- readxl::excel_sheets(path)
    s <- if (is.null(sheet)) sheets[1] else sheet
    df <- readxl::read_excel(path, sheet = s, skip = skip, col_names = header)
  } else {
    e <- new.env()
    if (grepl("[.]rds$", path, ignore.case = TRUE)) {
      assign("data", readRDS(path), envir = e)
    } else {
      load(path, envir = e)
    }
    objects <- ls(e)
    which <- if (is.null(objname)) objects[1] else objname
    df <- as.data.frame(get(which, envir = e))
  }
  df <- as.data.frame(df)
  names(df) <- make.names(names(df), unique = TRUE)
  if (${mode === "import" ? "TRUE" : "FALSE"}) {
    assign(${rStr(name)}, df, envir = globalenv())
    jsonlite::toJSON(list(ok = TRUE, nrows = nrow(df), ncols = ncol(df)), auto_unbox = TRUE)
  } else {
    h <- utils::head(df, 20)
    for (nm in names(h)) h[[nm]] <- as.character(h[[nm]])
    jsonlite::toJSON(list(
      columns = as.list(names(df)), rows = h, totalRows = nrow(df),
      sheets = if (is.null(sheets)) NULL else as.list(sheets),
      objects = if (is.null(objects)) NULL else as.list(objects)
    ), dataframe = "rows", auto_unbox = TRUE, na = "null")
  }
})
`;
}

/** R that serializes a named data.frame to CSV/TSV text via readr, or signals a
 * non-frame with the __notframe__ sentinel. */
function buildRExportCode(name, format) {
  const fn = format === "tsv" ? "format_tsv" : "format_csv";
  return `
local({
  v <- tryCatch(get(${JSON.stringify(name)}, envir = globalenv()), error = function(e) NULL)
  if (is.null(v) || !is.data.frame(v)) stop("__notframe__")
  readr::${fn}(v)
})
`;
}

/** R that restores an uploaded .RData image (5d): loads every object into a temp env,
 * then either reports the members and which collide with globalenv (preview), or assigns
 * each into globalenv under the conflict rule (import). Read-only against the file; base R
 * only. Returns a JSON string. */
function buildRRestoreCode(mode, options, path) {
  const rule = rStr(options.conflict || "rename");
  return `
local({
  e <- new.env(); load(${rStr(path)}, envir = e)
  members <- ls(e); existing <- ls(envir = globalenv())
  if (${mode === "import" ? "TRUE" : "FALSE"}) {
    rule <- ${rule}; restored <- character(); skipped <- character(); renamed <- character()
    for (nm in members) {
      if (nm %in% existing) {
        if (rule == "skip") { skipped <- c(skipped, nm); next }
        if (rule == "overwrite") { assign(nm, get(nm, envir = e), envir = globalenv()); restored <- c(restored, nm); existing <- c(existing, nm); next }
        t <- nm; k <- 2L; while (t %in% existing) { t <- paste0(nm, "_", k); k <- k + 1L }
        assign(t, get(nm, envir = e), envir = globalenv()); renamed <- c(renamed, paste0(nm, " to ", t)); existing <- c(existing, t); next
      }
      assign(nm, get(nm, envir = e), envir = globalenv()); restored <- c(restored, nm); existing <- c(existing, nm)
    }
    jsonlite::toJSON(list(ok = TRUE, restored = as.list(restored),
      skipped = as.list(skipped), renamed = as.list(renamed)), auto_unbox = TRUE)
  } else {
    jsonlite::toJSON(list(restore = TRUE, members = lapply(members, function(nm)
      list(name = nm, collides = nm %in% existing))), auto_unbox = TRUE)
  }
})
`;
}

/** A plain, em-dash-free sentence describing what a restore did (shared shape across
 * languages: {restored, skipped, renamed} arrays). */
function restoreNoteFrom(parsed) {
  const arr = (x) => (Array.isArray(x) ? x : x == null ? [] : [x]);
  const restored = arr(parsed.restored), skipped = arr(parsed.skipped), renamed = arr(parsed.renamed);
  const parts = [];
  if (restored.length) parts.push(`Restored ${restored.join(", ")}.`);
  if (renamed.length) parts.push(`Renamed to avoid a clash: ${renamed.join("; ")}.`);
  if (skipped.length) parts.push(`Skipped (already in your session): ${skipped.join(", ")}.`);
  return parts.join(" ") || "Nothing to restore.";
}

/** R that saves the whole global environment to `path` as an .RData image, or returns the
 * __empty__ sentinel when the environment holds nothing. Read-only: it reads globalenv() and
 * never removes or restarts anything. */
function buildRSaveImageCode(path, names) {
  const filter = Array.isArray(names)
    ? `ns <- intersect(ns, c(${names.map(rStr).join(", ")}))`
    : "";
  return `
local({
  ns <- ls(envir = globalenv())
  ${filter}
  if (length(ns) == 0) return("__empty__")
  save(list = ns, envir = globalenv(), file = ${rStr(path)})
  "__ok__"
})
`;
}

/** Maps the resolver's source label to an R package name for help lookup, or "" when
 * none is needed (base R). */
function sourceToPackage(source) {
  const s = String(source || "").toLowerCase();
  if (s === "dplyr") return "dplyr";
  if (s === "ggplot2") return "ggplot2";
  return ""; // base R and unknown: let help() search without a package
}

/** R that renders the Rd help page for one topic to plain text, or signals no help
 * with {found:false}. Uses do.call so a character topic passes through help()'s NSE,
 * .getHelpFile to read the parsed Rd, and Rd2txt to render. Returns a JSON string. */
function buildRDocCode(name, qualifier, source) {
  const pkg = qualifier ? String(qualifier) : sourceToPackage(source);
  const pkgArg = pkg ? `, package = ${rStr(pkg)}` : "";
  // With no explicit package, search the whole installed library, not just the
  // attached packages, so a bundled function (tidyr::pivot_longer, readr::read_csv,
  // stringr::str_detect, ...) resolves even before the student runs library().
  const tryAll = pkg ? "" : ", try.all.packages = TRUE";
  return `
local({
  topic <- ${rStr(name)}
  paths <- tryCatch(
    as.character(do.call(utils::help, list(topic${pkgArg}${tryAll}))),
    error = function(e) character(0)
  )
  if (length(paths) == 0) return(jsonlite::toJSON(list(found = FALSE), auto_unbox = TRUE))
  rd <- tryCatch(utils:::.getHelpFile(paths[[1]]), error = function(e) NULL)
  if (is.null(rd)) return(jsonlite::toJSON(list(found = FALSE), auto_unbox = TRUE))
  tmp <- tempfile()
  ok <- tryCatch({ tools::Rd2txt(rd, out = tmp, package = ${pkg ? rStr(pkg) : "\"\""}); TRUE },
                 error = function(e) FALSE)
  if (!ok) return(jsonlite::toJSON(list(found = FALSE), auto_unbox = TRUE))
  txt <- paste(readLines(tmp, warn = FALSE), collapse = "\\n")
  if (nchar(txt) > 12000) txt <- substr(txt, 1, 12000)
  jsonlite::toJSON(list(found = TRUE, text = txt), auto_unbox = TRUE)
})
`;
}

self.onmessage = async (event) => {
  const { id, code, keepState, withVariables, dataRequest, completeAt, prewarm, preparePackages, exportRequest, exportWorkspace, names, docRequest, fileOp, wsProxy } =
    event.data ?? {};
  let webR;
  try {
    webR = await getWebR();
  } catch {
    self.postMessage({
      id,
      ok: false,
      error:
        "The R runtime could not be loaded. Check your connection and try again.",
    });
    return;
  }

  // Upload: preview a sample of a data file, or import it as a named object.
  if (fileOp) {
    const rawText =
      fileOp.format === "csv" || fileOp.format === "json"
        ? new TextDecoder().decode(fileOp.bytes.slice(0, 50000))
        : undefined;
    const isRestore = !!fileOp.options?.restore && fileOp.format === "rdata";
    const ext = { csv: "csv", json: "json", xlsx: "xlsx", rdata: "RData" }[fileOp.format] ?? "dat";
    const path = `/tmp/__upload.${ext}`;
    const shelter = await new webR.Shelter();
    try {
      await ensurePackages(webR);
      await webR.FS.writeFile(path, fileOp.bytes);
      const obj = await shelter.captureR(
        isRestore
          ? buildRRestoreCode(fileOp.mode, fileOp.options ?? {}, path)
          : buildRFileCode(fileOp.mode, fileOp.name, fileOp.format, fileOp.options ?? {}, path),
        { withAutoprint: false, captureStreams: false, captureConditions: false, captureGraphics: false },
      );
      const parsed = JSON.parse((await obj.result.toArray())[0]);
      if (fileOp.mode === "import") {
        const variables = await inspectR(webR, shelter);
        self.postMessage({
          id,
          ok: true,
          result: {
            text: isRestore
              ? restoreNoteFrom(parsed)
              : `Loaded ${fileOp.name} (${parsed.nrows} rows x ${parsed.ncols} columns).`,
            variables,
          },
        });
      } else if (isRestore) {
        self.postMessage({
          id,
          ok: true,
          preview: { restore: true, columns: [], rows: [], members: parsed.members ?? [] },
        });
      } else {
        self.postMessage({
          id,
          ok: true,
          preview: {
            rawText,
            columns: parsed.columns ?? [],
            rows: parsed.rows ?? [],
            totalRows: parsed.totalRows,
            sheets: parsed.sheets ?? undefined,
            objects: parsed.objects ?? undefined,
          },
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (fileOp.mode === "preview") {
        self.postMessage({ id, ok: true, preview: { rawText, columns: [], rows: [], parseError: message } });
      } else {
        self.postMessage({ id, ok: false, error: friendlyError(message) });
      }
    } finally {
      await shelter.purge();
    }
    return;
  }

  // Prewarm request (sent when the R tab opens): load the runtime and the
  // bundled packages in the background, so they are ready by the time the
  // student runs their first line. Reports completion so the UI can drop any
  // "preparing" indicator.
  if (prewarm) {
    try {
      await ensurePackages(webR);
      await ensureNetworking(webR, wsProxy);
      // `preparePackages` (2026-07-26) names the packages ONE SPECIFIC snippet
      // needs, so they are fetched before the student presses Run rather than
      // during their run. The names are computed on the client from the snippet
      // (lib/sandbox/requirements) and are already known to be obtainable, since
      // a snippet needing an impossible package is not offered a Run button.
      //
      // Quoted through rStr and filtered to plain package names, because these
      // strings become R code. Anything else is dropped rather than escaped.
      if (Array.isArray(preparePackages) && preparePackages.length) {
        const safe = preparePackages
          .filter((n) => typeof n === "string" && /^[A-Za-z][\w.]*$/.test(n))
          .slice(0, 12);
        if (safe.length) {
          const missing = await webR.evalRString(
            `paste(setdiff(c(${safe.map(rStr).join(",")}), rownames(installed.packages())), collapse=",")`,
          );
          const todo = missing.split(",").filter(Boolean);
          // One call, so a shared dependency is resolved once.
          if (todo.length) await webR.installPackages(todo, { quiet: true });
        }
      }
      self.postMessage({ id, ok: true });
    } catch {
      self.postMessage({ id, ok: true }); // best-effort; R still works
    }
    return;
  }

  // Autocomplete request.
  if (completeAt) {
    try {
      self.postMessage({
        id,
        ok: true,
        completions: await completeR(webR, completeAt.prefix ?? ""),
      });
    } catch (error) {
      self.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }

  // Documentation request: render the Rd help page for one topic to plain text.
  // Read-only; never runs the student's code. Falls back to {found:false} on any
  // failure (including a WebR build that ships without the package help database).
  if (docRequest) {
    const shelter = await new webR.Shelter();
    try {
      const src = String(docRequest.source || "").toLowerCase();
      if (src === "dplyr" || src === "ggplot2" || docRequest.qualifier) {
        await ensurePackages(webR); // dplyr/ggplot2 (via tidyverse) carry their help
      }
      const obj = await shelter.captureR(
        buildRDocCode(docRequest.name, docRequest.qualifier, docRequest.source),
        { withAutoprint: false, captureStreams: false, captureConditions: false, captureGraphics: false },
      );
      const parsed = JSON.parse((await obj.result.toArray())[0]);
      self.postMessage({ id, ok: true, doc: parsed });
    } catch {
      self.postMessage({ id, ok: true, doc: { found: false } });
    } finally {
      await shelter.purge();
    }
    return;
  }

  // Export one data.frame to CSV/TSV rather than running code.
  if (exportRequest) {
    const shelter = await new webR.Shelter();
    try {
      await ensurePackages(webR); // readr is bundled; format_csv/format_tsv live there
      const obj = await shelter.captureR(
        buildRExportCode(exportRequest.name, exportRequest.format),
        { withAutoprint: false, captureStreams: false, captureConditions: false, captureGraphics: false },
      );
      const text = (await obj.result.toArray())[0];
      self.postMessage({ id, ok: true, exported: { text } });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      self.postMessage({
        id,
        ok: false,
        error: /__notframe__/.test(message)
          ? "That variable is not a data frame."
          : message,
      });
    } finally {
      await shelter.purge();
    }
    return;
  }

  // Export the whole global environment as one .RData byte image (read-only).
  if (exportWorkspace) {
    const path = "/tmp/__ws.RData";
    const shelter = await new webR.Shelter();
    try {
      const obj = await shelter.captureR(buildRSaveImageCode(path, names), {
        withAutoprint: false,
        captureStreams: false,
        captureConditions: false,
        captureGraphics: false,
      });
      const status = (await obj.result.toArray())[0];
      if (status === "__empty__") {
        self.postMessage({ id, ok: true, exported: { empty: true } });
      } else {
        const bytes = await webR.FS.readFile(path);
        await webR.FS.unlink(path).catch(() => {});
        self.postMessage({ id, ok: true, exported: { bytes, skipped: [], empty: false } });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      self.postMessage({ id, ok: false, error: friendlyError(message) });
    } finally {
      await shelter.purge();
    }
    return;
  }

  // Data-viewer request: return a page of a data.frame rather than running code.
  if (dataRequest) {
    try {
      const data = await fetchFrameR(
        webR,
        dataRequest.name,
        dataRequest.offset,
        dataRequest.limit,
      );
      self.postMessage({ id, ok: true, data });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      self.postMessage({
        id,
        ok: false,
        error: /__notframe__/.test(message)
          ? "That variable is not a data frame."
          : message,
      });
    }
    return;
  }

  // Preload the bundled packages before running student code, so library()
  // finds tidyverse (and friends) already installed from our mirror rather than
  // reaching out to the WebR repository. Only code runs pay this (once); it is
  // usually already done from the tab-open prewarm.
  await ensurePackages(webR);
  await ensureNetworking(webR, wsProxy);

  // A shelter scopes every R object this run allocates; purging it frees them
  // so one run does not leak into the next.
  const shelter = await new webR.Shelter();
  try {
    const { output, images } = await shelter.captureR(code, {
      // Print results as an R console would, so typing `1:10` shows its value.
      withAutoprint: true,
      captureStreams: true,
      // TRUE, which is webR's default, and load-bearing.
      //
      // This used to be false, with a comment claiming errors would still
      // re-throw "(the default)". They do not: false is exactly what turns that
      // off. Measured 2026-07-26 against webR 0.6: with false, stop("boom")
      // RESOLVES successfully and "Error: boom" arrives as an ordinary stderr
      // line. So every failed R run was reported to the student as a success,
      // in the neutral "Output" panel rather than the red "Error" one, and
      // announced through aria-live="polite" instead of role="alert". Nothing
      // downstream could tell a failure had happened either.
      //
      // The professor's production screenshot of the Coding Tutor shows the
      // symptom exactly: an "Output" panel reading "Error: cannot open the
      // connection". Python never had this: its worker throws, so it always
      // classified correctly, which is why the two languages behaved
      // differently for the same mistake.
      captureConditions: true,
    });

    // Capturing conditions takes message() and warning() OFF the stderr stream
    // and hands them over as R condition objects instead, so they have to be
    // rendered back or the student silently loses them: "Loading required
    // package: rvest", dplyr's masking notes, and every warning R raises are all
    // messages, not stdout.
    const lines = [];
    for (const entry of output) {
      if (
        (entry.type === "stdout" || entry.type === "stderr") &&
        typeof entry.data === "string"
      ) {
        lines.push(entry.data);
        continue;
      }
      if (entry.type !== "message" && entry.type !== "warning") continue;
      const condition = await conditionMessage(entry.data);
      if (!condition) continue;
      // R itself prefixes warnings when it prints them, and a warning that
      // reads like ordinary output is how a student misses it.
      lines.push(entry.type === "warning" ? `Warning: ${condition}` : condition);
    }
    const text = lines.join("\n");

    let imageDataUrl;
    if (images && images.length > 0) {
      // The last plot is the final state a student drew.
      imageDataUrl = await bitmapToPngDataUrl(images[images.length - 1]);
    }

    let variables;
    if (withVariables) variables = await inspectR(webR, shelter);

    self.postMessage({
      id,
      ok: true,
      result: { text: text || undefined, imageDataUrl, variables },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    self.postMessage({ id, ok: false, error: friendlyError(message) });
  } finally {
    await shelter.purge();
    // R's global environment persists across runs on its own, which a Sandbox
    // session wants but the inline Run button does not. So for a one-off run,
    // clear the global environment to keep each run independent.
    if (!keepState) {
      try {
        await webR.evalRVoid(
          "rm(list = ls(envir = globalenv()), envir = globalenv())",
        );
      } catch {
        // A failed cleanup must not mask the real result.
      }
    }
  }
};

/**
 * The user's variables in the global environment, each as "name\tclass\tsize",
 * for the Variables pane. Best-effort: a failure never fails the run.
 */
const INSPECT_R = `
local({
  ns <- ls(envir = globalenv())
  if (length(ns) == 0) return(character(0))
  vapply(ns, function(n) {
    v <- tryCatch(get(n, envir = globalenv()), error = function(e) NULL)
    cls <- paste(class(v), collapse = ",")
    sz <- if (is.data.frame(v))
            paste0(nrow(v), " obs. of ", ncol(v), " variable",
                   if (ncol(v) == 1) "" else "s")
          else if (!is.null(dim(v))) paste(dim(v), collapse = " x ")
          else paste0("len ", length(v))
    cols <- if (is.data.frame(v))
      paste(sprintf("%s:%s", names(v),
                    vapply(v, function(c) class(c)[1], character(1))),
            collapse = ",")
    else ""
    paste(n, cls, sz, cols, sep = "\\t")
  }, character(1))
})
`;

async function inspectR(webR, shelter) {
  try {
    const obj = await shelter.captureR(INSPECT_R, {
      withAutoprint: false,
      captureStreams: false,
      captureConditions: false,
      captureGraphics: false,
    });
    const rows = await obj.result.toArray();
    return rows
      .filter((r) => typeof r === "string")
      .map((r) => {
        const [name, type, info, colspec] = r.split("\t");
        const entry = { name, type: type || "", info: info || "" };
        if (colspec) {
          entry.columns = colspec
            .split(",")
            .filter(Boolean)
            .map((c) => {
              const i = c.lastIndexOf(":");
              return i > 0
                ? { name: c.slice(0, i), type: c.slice(i + 1) }
                : { name: c, type: "" };
            });
        }
        return entry;
      });
  } catch {
    return [];
  }
}

/** Adds a hint when a failure looks like a package that could not be installed. */
/**
 * The message text out of an R condition object.
 *
 * With captureConditions on, message() and warning() arrive as R conditions
 * rather than strings: a list whose "message" element is a character vector and
 * whose "call" element is an R language object.
 *
 * That `call` is why the whole condition cannot simply be converted: toJs() on
 * it fails with "This R object cannot be converted to JS" (measured against webR
 * 0.6). So only the one element we want is pulled across, in R space, and the
 * language object is never touched.
 *
 * Returns null rather than throwing: losing one line of output must never turn a
 * working run into a failed one.
 */
async function conditionMessage(obj) {
  try {
    const message = await obj.get("message");
    const js = await message.toJs();
    const parts = Array.isArray(js?.values) ? js.values : [];
    // R's message() appends a newline of its own; warning() does not. The panel
    // supplies its own line breaks, so a trailing one would show as a gap.
    return parts.join("").replace(/\n+$/, "") || null;
  } catch {
    return null;
  }
}

function friendlyError(message) {
  // webR tags a thrown R error with a one-letter type prefix and, for a
  // top-level error, names its own internal call: the raw string is
  // "T: Error in `eval(ei, envir)`: boom". Neither part means anything to a
  // student, so both are removed before the message is shown.
  //
  // The "eval(" below is text INSIDE R's error message being matched by a
  // regex. Nothing here evaluates anything; this function only rewrites a
  // string for display.
  let cleaned = message.replace(/^[A-Z]:\s+/, "");
  cleaned = cleaned.replace(/^Error in `eval\((?:ei, envir|.*?)\)`:\s*/, "Error: ");
  if (
    /could not.*(download|install)|unable to.*install|package.*not.*available|no such/i.test(
      cleaned,
    )
  ) {
    return `${cleaned}\n\nIf you were installing a package, check the name and your connection. Packages download from the WebR repository the first time you use them.`;
  }
  return cleaned;
}
