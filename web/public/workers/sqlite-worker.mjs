/**
 * In-browser SQLite worker.
 *
 * A static ES-module worker, loaded by URL rather than through the bundler,
 * because Turbopack cannot trace the wasm loader and its .wasm. It runs one SQL
 * snippet per message against a fresh in-memory database, so nothing persists
 * between runs and one run cannot see another's tables.
 *
 * Protocol, shared with the other language workers:
 *   in:  { id, code }
 *   out: { id, ok: true, result } | { id, ok: false, error }
 * result: { text?, table?: { columns, rows }, imageDataUrl? }
 */

let initPromise = null;
// Reused across runs only in a Sandbox session (keepState), so a table created
// in one run is still there in the next. The inline Run button never sets it.
let sessionDb = null;

async function getSqlite() {
  if (!initPromise) {
    const { default: sqlite3InitModule } = await import(
      "/runtimes/sqlite/sqlite3.mjs"
    );
    initPromise = sqlite3InitModule({
      // The loader fetches its wasm through this callback.
      locateFile: (file) => `/runtimes/sqlite/${file}`,
    });
  }
  return initPromise;
}

/** The tables in the session database, with row counts and columns, for the
 * Variables pane. Best-effort: a failure here never fails the run. */
function inspectTables(db) {
  try {
    const tables = [];
    db.exec({
      sql: "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
      rowMode: "object",
      resultRows: tables,
    });
    return tables.map((t) => {
      const name = t.name;
      const cols = [];
      db.exec({ sql: `PRAGMA table_info("${name}")`, rowMode: "object", resultRows: cols });
      const count = [];
      db.exec({ sql: `SELECT count(*) AS n FROM "${name}"`, rowMode: "object", resultRows: count });
      const n = count[0]?.n ?? 0;
      return {
        name,
        type: "table",
        info: `${n} row${n === 1 ? "" : "s"}`,
        columns: cols.map((c) => ({
          name: c.name,
          type: c.type || "?",
        })),
      };
    });
  } catch {
    return [];
  }
}

/** One page of a table's rows, for the data viewer. */
function fetchTablePage(db, name, offset, limit) {
  const safe = String(name).replace(/"/g, '""');
  const count = [];
  db.exec({
    sql: `SELECT count(*) AS n FROM "${safe}"`,
    rowMode: "object",
    resultRows: count,
  });
  const rows = [];
  const columns = [];
  db.exec({
    sql: `SELECT * FROM "${safe}" LIMIT ${Number(limit)} OFFSET ${Number(offset)}`,
    rowMode: "object",
    resultRows: rows,
    columnNames: columns,
  });
  return { columns, rows, totalRows: count[0]?.n ?? 0 };
}

/** A single CSV field, quoted only when it must be. */
function csvField(value) {
  if (value === null || value === undefined) return "";
  const s = String(value);
  return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

/** A single delimited field, quoted only when it must be (contains the delimiter,
 * a quote, or a newline). Works for both comma and tab. */
function delimitedField(value, delimiter) {
  if (value === null || value === undefined) return "";
  const s = String(value);
  const mustQuote =
    s.includes(delimiter) || s.includes('"') || s.includes("\n") || s.includes("\r");
  return mustQuote ? '"' + s.replace(/"/g, '""') + '"' : s;
}

/** One named table serialized to CSV/TSV via a read-only SELECT. Never re-runs
 * the student's CREATE/INSERT/UPDATE/DELETE: it reads the already-built table. */
function exportTableDelimited(db, name, format) {
  const delimiter = format === "tsv" ? "\t" : ",";
  const exists = [];
  db.exec({
    sql: "SELECT 1 AS ok FROM sqlite_schema WHERE type='table' AND name = ?",
    bind: [String(name)],
    rowMode: "object",
    resultRows: exists,
  });
  if (exists.length === 0) {
    throw new Error("That object is not a table you can export.");
  }
  const rows = [];
  const columns = [];
  db.exec({
    sql: `SELECT * FROM "${String(name).replace(/"/g, '""')}"`,
    rowMode: "array",
    resultRows: rows,
    columnNames: columns,
  });
  const lines = [columns.map((c) => delimitedField(c, delimiter)).join(delimiter)];
  for (const row of rows) {
    lines.push(row.map((v) => delimitedField(v, delimiter)).join(delimiter));
  }
  return lines.join("\n") + "\n";
}

/** The whole in-memory database as a byte image: a complete, re-openable .sqlite file with
 * every user table. Read-only: it serializes the existing DB and re-runs no statement. */
function exportDatabaseBytes(sqlite3, db) {
  // sqlite3_js_db_export returns a Uint8Array copy of the database's serialized image.
  return sqlite3.capi.sqlite3_js_db_export(db.pointer);
}

// --- Restore an uploaded .sqlite workspace (5d) -------------------------------

const qid = (name) => `"${String(name).replace(/"/g, '""')}"`;

/** Opens a byte image into a throwaway in-memory DB (a full, separate connection).
 * Confirmed API: allocFromTypedArray + sqlite3_deserialize with FREEONCLOSE|RESIZEABLE. */
function deserializeInto(sqlite3, bytes) {
  const db = new sqlite3.oo1.DB();
  const p = sqlite3.wasm.allocFromTypedArray(bytes);
  const rc = sqlite3.capi.sqlite3_deserialize(
    db.pointer,
    "main",
    p,
    bytes.length,
    bytes.length,
    sqlite3.capi.SQLITE_DESERIALIZE_FREEONCLOSE |
      sqlite3.capi.SQLITE_DESERIALIZE_RESIZEABLE,
  );
  if (rc !== 0) {
    db.close();
    throw new Error("That file is not a valid SQLite database.");
  }
  return db;
}

/** User table names (excludes sqlite_* internal tables). */
function userTableNames(db) {
  const rows = [];
  db.exec({
    sql: "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
    rowMode: "array",
    resultRows: rows,
  });
  return rows.map((r) => r[0]);
}

/** The first free table name: `name`, else `name_2`, ... not in `taken`. */
function uniqueTableName(name, taken) {
  if (!taken.has(name)) return name;
  let n = 2;
  while (taken.has(`${name}_${n}`)) n++;
  return `${name}_${n}`;
}

/** Copies one table from `src` into `main` under `target`, preserving the schema
 * by reusing src's own CREATE (with the first name token swapped to `target`). */
function copyTable(src, main, srcName, target) {
  const schema = [];
  src.exec({
    sql: "SELECT sql FROM sqlite_schema WHERE type='table' AND name=?",
    bind: [srcName],
    rowMode: "array",
    resultRows: schema,
  });
  const createSql = schema[0]?.[0];
  if (!createSql) return;
  const create = createSql.replace(
    /^(\s*CREATE\s+TABLE\s+)(?:IF\s+NOT\s+EXISTS\s+)?("(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]*\]|[A-Za-z_][\w$]*)/i,
    (_m, head) => `${head}${qid(target)}`,
  );
  main.exec(create);
  const cols = [];
  src.exec({ sql: `SELECT * FROM ${qid(srcName)} LIMIT 0`, columnNames: cols });
  const rows = [];
  src.exec({ sql: `SELECT * FROM ${qid(srcName)}`, rowMode: "array", resultRows: rows });
  if (rows.length === 0) return;
  const stmt = main.prepare(
    `INSERT INTO ${qid(target)} (${cols.map(qid).join(",")}) VALUES (${cols
      .map(() => "?")
      .join(",")})`,
  );
  try {
    for (const r of rows) {
      stmt.bind(r).step();
      stmt.reset();
    }
  } finally {
    stmt.finalize();
  }
}

/** A plain, em-dash-free sentence describing what a restore did. */
function restoreNote(restored, skipped, renamed) {
  const parts = [];
  if (restored.length) parts.push(`Restored ${restored.join(", ")}.`);
  if (renamed.length) parts.push(`Renamed to avoid a clash: ${renamed.join("; ")}.`);
  if (skipped.length) parts.push(`Skipped (already in your session): ${skipped.join(", ")}.`);
  return parts.join(" ") || "Nothing to restore.";
}

/** Every table in the session as CSV, for handing to the ggsql plot engine. */
function dumpTables(db) {
  const tables = [];
  const names = [];
  db.exec({
    sql: "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
    rowMode: "array",
    resultRows: names,
  });
  for (const [name] of names) {
    const rows = [];
    const columns = [];
    db.exec({
      sql: `SELECT * FROM "${String(name).replace(/"/g, '""')}"`,
      rowMode: "array",
      resultRows: rows,
      columnNames: columns,
    });
    const lines = [columns.map(csvField).join(",")];
    for (const row of rows) lines.push(row.map(csvField).join(","));
    tables.push({ name, csv: lines.join("\n") });
  }
  return tables;
}

const SQL_KEYWORDS = [
  "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "JOIN",
  "LEFT JOIN", "INNER JOIN", "ON", "INSERT INTO", "VALUES", "UPDATE", "SET",
  "DELETE", "CREATE TABLE", "DROP TABLE", "LIMIT", "OFFSET", "DISTINCT", "AS",
  "AND", "OR", "NOT", "NULL", "IS NULL", "IN", "LIKE", "BETWEEN", "COUNT",
  "SUM", "AVG", "MIN", "MAX", "ROUND", "CAST",
];

/** Table names, column names, and keywords matching the word before the cursor. */
function completeSql(db, prefix) {
  const m = /([A-Za-z_][A-Za-z0-9_]*)$/.exec(prefix);
  const partial = m ? m[1] : "";
  const options = [];
  try {
    const tables = [];
    db.exec({
      sql: "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
      rowMode: "object",
      resultRows: tables,
    });
    for (const t of tables) {
      options.push({ label: t.name, type: "table", detail: "table" });
      const cols = [];
      db.exec({
        sql: `PRAGMA table_info("${String(t.name).replace(/"/g, '""')}")`,
        rowMode: "object",
        resultRows: cols,
      });
      for (const c of cols) {
        options.push({
          label: c.name,
          type: "property",
          detail: `${t.name} (${c.type || "?"})`,
        });
      }
    }
  } catch {
    // best-effort
  }
  for (const k of SQL_KEYWORDS) options.push({ label: k, type: "keyword", detail: "" });
  const lower = partial.toLowerCase();
  const seen = new Set();
  const filtered = options.filter((o) => {
    if (!o.label.toLowerCase().startsWith(lower)) return false;
    const key = o.label + o.type;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return { partial, options: filtered.slice(0, 200) };
}

/** Splits delimited text into rows of fields, honouring quoted fields ("" is an
 * escaped quote). Newlines inside quotes are kept. */
function parseDelimited(text, delimiter) {
  const rows = [];
  let field = "";
  let row = [];
  let i = 0;
  let inQuotes = false;
  while (i < text.length) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
        inQuotes = false; i++; continue;
      }
      field += c; i++; continue;
    }
    if (c === '"') { inQuotes = true; i++; continue; }
    if (c === delimiter) { row.push(field); field = ""; i++; continue; }
    if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); rows.push(row); field = ""; row = []; i++; continue;
    }
    field += c; i++;
  }
  if (field.length > 0 || row.length > 0) { row.push(field); rows.push(row); }
  return rows;
}

/** Makes column names unique and non-empty. */
function uniqueColumns(names) {
  const seen = new Set();
  return names.map((name, i) => {
    let out = name && name.trim() ? name.trim() : `col${i + 1}`;
    while (seen.has(out)) out = `${out}_`;
    seen.add(out);
    return out;
  });
}

/** Turns CSV text into {columns, rows} using the student's options. */
function tableFromCsv(text, options) {
  const delimiter = options.delimiter || ",";
  const skip = Number(options.skipRows) || 0;
  const header = options.header !== false;
  let cells = parseDelimited(text, delimiter).filter(
    (r) => !(r.length === 1 && r[0] === ""),
  );
  cells = cells.slice(skip);
  if (cells.length === 0) throw new Error("No rows to read with these settings.");
  let columns;
  if (header) {
    columns = uniqueColumns(cells[0]);
    cells = cells.slice(1);
  } else {
    columns = cells[0].map((_, i) => `col${i + 1}`);
  }
  const rows = cells.map((r) => {
    const o = {};
    columns.forEach((c, i) => (o[c] = r[i] ?? null));
    return o;
  });
  return { columns, rows };
}

/** Turns JSON text into {columns, rows}: an array of records, a columnar object,
 * or a single record. */
function tableFromJson(text) {
  const data = JSON.parse(text);
  let records;
  if (Array.isArray(data)) {
    records = data;
  } else if (data && typeof data === "object") {
    const values = Object.values(data);
    const columnar =
      values.length > 0 && values.every((v) => Array.isArray(v));
    if (columnar) {
      const keys = Object.keys(data);
      const len = Math.max(...values.map((v) => v.length));
      records = [];
      for (let i = 0; i < len; i++) {
        const rec = {};
        keys.forEach((k) => (rec[k] = data[k][i] ?? null));
        records.push(rec);
      }
    } else {
      records = [data];
    }
  } else {
    throw new Error("This JSON is not a table of records.");
  }
  const columns = [];
  for (const rec of records)
    for (const k of Object.keys(rec ?? {})) if (!columns.includes(k)) columns.push(k);
  const rows = records.map((rec) => {
    const o = {};
    columns.forEach((c) => (o[c] = rec?.[c] ?? null));
    return o;
  });
  return { columns, rows };
}

/** Creates a table `name` from parsed rows, inferring numeric columns. */
function importTable(db, name, columns, rows) {
  const q = (s) => `"${String(s).replace(/"/g, '""')}"`;
  const numeric = columns.map(
    (c) =>
      rows.length > 0 &&
      rows.every((r) => {
        const v = r[c];
        return v === null || v === "" || typeof v === "number" || !isNaN(Number(v));
      }),
  );
  db.exec(`DROP TABLE IF EXISTS ${q(name)}`);
  db.exec(
    `CREATE TABLE ${q(name)} (${columns
      .map((c, i) => `${q(c)} ${numeric[i] ? "REAL" : "TEXT"}`)
      .join(", ")})`,
  );
  const stmt = db.prepare(
    `INSERT INTO ${q(name)} VALUES (${columns.map(() => "?").join(",")})`,
  );
  try {
    for (const r of rows) {
      const vals = columns.map((c, i) => {
        const v = r[c];
        if (v === null || v === undefined || v === "") return null;
        if (numeric[i]) return Number(v);
        return typeof v === "object" ? JSON.stringify(v) : String(v);
      });
      stmt.bind(vals);
      stmt.step();
      stmt.reset();
    }
  } finally {
    stmt.finalize();
  }
}

self.onmessage = async (event) => {
  const { id, code, keepState, withVariables, dataRequest, completeAt, dumpTablesRequest, exportRequest, exportWorkspace, names, docRequest, fileOp } =
    event.data ?? {};

  // SQLite has no runtime help database, so there is no local doc text to render.
  // Answer found:false so the pane falls back to the curated blurb and the link.
  if (docRequest) {
    self.postMessage({ id, ok: true, doc: { found: false } });
    return;
  }

  // Export one table to CSV/TSV via a read-only re-select (never re-runs the
  // student's data-changing statements).
  if (exportRequest) {
    try {
      const sqlite3 = await getSqlite();
      const db = keepState
        ? (sessionDb ??= new sqlite3.oo1.DB())
        : new sqlite3.oo1.DB();
      const text = exportTableDelimited(db, exportRequest.name, exportRequest.format);
      self.postMessage({ id, ok: true, exported: { text } });
    } catch (error) {
      self.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }

  // Export the database as one .sqlite byte image: the whole thing, or only the
  // selected tables (5e). Read-only; re-runs nothing.
  if (exportWorkspace) {
    try {
      const sqlite3 = await getSqlite();
      const db = keepState
        ? (sessionDb ??= new sqlite3.oo1.DB())
        : new sqlite3.oo1.DB();
      const all = userTableNames(db);
      const wanted = Array.isArray(names) ? all.filter((n) => names.includes(n)) : all;
      if (wanted.length === 0) {
        self.postMessage({ id, ok: true, exported: { empty: true } });
      } else if (!Array.isArray(names)) {
        const bytes = exportDatabaseBytes(sqlite3, db);
        self.postMessage({ id, ok: true, exported: { bytes, skipped: [], empty: false } });
      } else {
        // A subset: copy just the wanted tables into a throwaway DB and export that.
        const sub = new sqlite3.oo1.DB();
        try {
          for (const n of wanted) copyTable(db, sub, n, n);
          const bytes = exportDatabaseBytes(sqlite3, sub);
          self.postMessage({ id, ok: true, exported: { bytes, skipped: [], empty: false } });
        } finally {
          sub.close();
        }
      }
    } catch (error) {
      self.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }

  // Restore an uploaded .sqlite workspace: copy every table into the session,
  // handling name clashes per the chosen rule. Read-only against the file; never
  // runs a student statement. Guarded before the csv/json path (binary bytes).
  if (fileOp && (fileOp.format === "sqlite" || fileOp.options?.restore)) {
    let src = null;
    try {
      const sqlite3 = await getSqlite();
      const main = (sessionDb ??= new sqlite3.oo1.DB());
      src = deserializeInto(sqlite3, fileOp.bytes);
      const names = userTableNames(src);
      const existing = new Set(userTableNames(main));
      if (fileOp.mode === "preview") {
        self.postMessage({
          id,
          ok: true,
          preview: {
            restore: true,
            columns: [],
            rows: [],
            members: names.map((n) => ({ name: n, collides: existing.has(n) })),
          },
        });
      } else {
        const rule = fileOp.options?.conflict ?? "rename";
        const restored = [], skipped = [], renamed = [];
        for (const n of names) {
          if (existing.has(n)) {
            if (rule === "skip") {
              skipped.push(n);
              continue;
            }
            if (rule === "overwrite") {
              main.exec(`DROP TABLE IF EXISTS ${qid(n)}`);
              copyTable(src, main, n, n);
              restored.push(n);
              existing.add(n);
              continue;
            }
            const t = uniqueTableName(n, existing);
            copyTable(src, main, n, t);
            renamed.push(`${n} to ${t}`);
            existing.add(t);
            continue;
          }
          copyTable(src, main, n, n);
          restored.push(n);
          existing.add(n);
        }
        self.postMessage({
          id,
          ok: true,
          result: { text: restoreNote(restored, skipped, renamed), variables: inspectTables(main) },
        });
      }
    } catch (error) {
      self.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    } finally {
      if (src) src.close();
    }
    return;
  }

  // Upload: preview a sample of a data file, or import it as a table.
  if (fileOp) {
    const rawText = new TextDecoder().decode(fileOp.bytes.slice(0, 50000));
    let parsed;
    try {
      const text = new TextDecoder().decode(fileOp.bytes);
      parsed =
        fileOp.format === "json"
          ? tableFromJson(text)
          : tableFromCsv(text, fileOp.options ?? {});
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (fileOp.mode === "preview") {
        self.postMessage({ id, ok: true, preview: { rawText, columns: [], rows: [], parseError: message } });
      } else {
        self.postMessage({ id, ok: false, error: message });
      }
      return;
    }
    try {
      if (fileOp.mode === "import") {
        const sqlite3 = await getSqlite();
        const db = (sessionDb ??= new sqlite3.oo1.DB());
        importTable(db, fileOp.name, parsed.columns, parsed.rows);
        self.postMessage({
          id,
          ok: true,
          result: {
            text: `Loaded ${fileOp.name} (${parsed.rows.length} rows x ${parsed.columns.length} columns).`,
            variables: inspectTables(db),
          },
        });
      } else {
        self.postMessage({
          id,
          ok: true,
          preview: {
            rawText,
            columns: parsed.columns,
            rows: parsed.rows.slice(0, 20),
            totalRows: parsed.rows.length,
          },
        });
      }
    } catch (error) {
      self.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }

  // Dump every session table as CSV, so the ggsql plot engine can register them.
  if (dumpTablesRequest) {
    try {
      const sqlite3 = await getSqlite();
      const db = keepState
        ? (sessionDb ??= new sqlite3.oo1.DB())
        : new sqlite3.oo1.DB();
      self.postMessage({ id, ok: true, tables: dumpTables(db) });
    } catch (error) {
      self.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }

  // Autocomplete request.
  if (completeAt) {
    try {
      const sqlite3 = await getSqlite();
      const db = keepState
        ? (sessionDb ??= new sqlite3.oo1.DB())
        : new sqlite3.oo1.DB();
      self.postMessage({
        id,
        ok: true,
        completions: completeSql(db, completeAt.prefix ?? ""),
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

  // Data-viewer request: return a page of a table rather than running code.
  if (dataRequest) {
    try {
      const sqlite3 = await getSqlite();
      const db = keepState
        ? (sessionDb ??= new sqlite3.oo1.DB())
        : new sqlite3.oo1.DB();
      const data = fetchTablePage(
        db,
        dataRequest.name,
        dataRequest.offset,
        dataRequest.limit,
      );
      self.postMessage({ id, ok: true, data });
    } catch (error) {
      self.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }

  try {
    const sqlite3 = await getSqlite();
    // A session keeps one database across runs; the inline Run button uses a
    // fresh one every time so no state leaks between snippets.
    const db = keepState
      ? (sessionDb ??= new sqlite3.oo1.DB())
      : new sqlite3.oo1.DB();
    try {
      const rows = [];
      let columns = [];
      db.exec({
        sql: code,
        rowMode: "object",
        resultRows: rows,
        columnNames: columns,
      });

      const result =
        rows.length > 0
          ? {
              table: { columns, rows },
              text:
                rows.length === 1
                  ? "1 row"
                  : `${rows.length} rows`,
            }
          : { text: "Statement ran. No rows returned." };

      if (withVariables) result.variables = inspectTables(db);
      self.postMessage({ id, ok: true, result });
    } finally {
      // Keep a session's database open; close a one-off run's.
      if (!keepState) db.close();
    }
  } catch (error) {
    self.postMessage({
      id,
      ok: false,
      // SQLite's messages are already student-readable ("no such table: x").
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
