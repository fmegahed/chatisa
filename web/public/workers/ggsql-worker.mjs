/**
 * In-browser ggsql worker (experimental, alpha).
 *
 * ggsql turns a SQL query with a VISUALISE clause into a Vega-Lite chart spec,
 * which the Plots pane renders. It is a plot ADD-ON: the SQLite worker stays the
 * SQL workhorse, and the data a plot needs is handed here as CSV (dumped from
 * the SQL session) and registered into ggsql's own embedded SQLite.
 *
 * Pinned to ggsql-wasm 0.4.1, self-hosted under /runtimes/ggsql/. The notation
 * may change in later versions; see the Coding Studio's SQL example comment.
 *
 * Protocol:
 *   in:  { id, prewarm }                      -> warm the wasm
 *   in:  { id, query, tables: [{name, csv}] } -> a ggsql VISUALISE query
 *   out: { id, ok: true, spec } | { id, ok: false, error }
 * spec is a Vega-Lite JSON object (already parsed).
 */

let contextPromise = null;

async function getContext() {
  if (!contextPromise) {
    contextPromise = (async () => {
      const mod = await import("/runtimes/ggsql/ggsql_wasm.js");
      // wasm-bindgen default export initialises the module; point it at the
      // self-hosted wasm rather than letting it guess a bundler-relative URL.
      await mod.default({
        module_or_path: "/runtimes/ggsql/ggsql_wasm_bg.wasm",
      });
      return new mod.GgsqlContext();
    })();
  }
  return contextPromise;
}

self.onmessage = async (event) => {
  const { id, query, tables, prewarm } = event.data ?? {};

  let ctx;
  try {
    ctx = await getContext();
  } catch {
    self.postMessage({
      id,
      ok: false,
      error: "The plotting engine could not be loaded.",
    });
    return;
  }

  if (prewarm) {
    self.postMessage({ id, ok: true });
    return;
  }

  try {
    // Register the data fresh each time, so a plot always reflects the current
    // state of the session's tables.
    for (const table of tables ?? []) {
      try {
        ctx.unregister(table.name);
      } catch {
        // Not previously registered; fine.
      }
      ctx.register_csv(table.name, new TextEncoder().encode(table.csv));
    }
    const spec = JSON.parse(ctx.execute(query));
    self.postMessage({ id, ok: true, spec });
  } catch (error) {
    self.postMessage({
      id,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
