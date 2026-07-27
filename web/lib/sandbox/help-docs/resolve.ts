import type { DocEntry, HelpRequest, HelpLanguage, SqlDialect } from "./types";

interface Ref {
  source: string;
  url: string;
  blurb: string;
}

// --- R: dplyr, ggplot2, and base R ---------------------------------------
const DPLYR = "https://dplyr.tidyverse.org/reference/";
const GGPLOT = "https://ggplot2.tidyverse.org/reference/";
const BASE_R = "https://stat.ethz.ch/R-manual/R-devel/library/base/html/";

const R_REFS: Record<string, Ref> = {
  // dplyr
  summarise: { source: "dplyr", url: `${DPLYR}summarise.html`, blurb: "Summarise each group down to one row (dplyr)." },
  summarize: { source: "dplyr", url: `${DPLYR}summarise.html`, blurb: "Summarise each group down to one row (dplyr)." },
  mutate: { source: "dplyr", url: `${DPLYR}mutate.html`, blurb: "Create or change columns (dplyr)." },
  filter: { source: "dplyr", url: `${DPLYR}filter.html`, blurb: "Keep rows that match a condition (dplyr)." },
  select: { source: "dplyr", url: `${DPLYR}select.html`, blurb: "Keep or drop columns by name (dplyr)." },
  arrange: { source: "dplyr", url: `${DPLYR}arrange.html`, blurb: "Order rows by column values (dplyr)." },
  group_by: { source: "dplyr", url: `${DPLYR}group_by.html`, blurb: "Group rows for grouped operations (dplyr)." },
  count: { source: "dplyr", url: `${DPLYR}count.html`, blurb: "Count rows per group (dplyr)." },
  // ggplot2
  ggplot: { source: "ggplot2", url: `${GGPLOT}ggplot.html`, blurb: "Start a ggplot2 plot (ggplot2)." },
  aes: { source: "ggplot2", url: `${GGPLOT}aes.html`, blurb: "Map data to visual aesthetics (ggplot2)." },
  geom_point: { source: "ggplot2", url: `${GGPLOT}geom_point.html`, blurb: "Scatterplot layer (ggplot2)." },
  geom_line: { source: "ggplot2", url: `${GGPLOT}geom_path.html`, blurb: "Line layer, documented with geom_path (ggplot2)." },
  labs: { source: "ggplot2", url: `${GGPLOT}labs.html`, blurb: "Set titles and axis labels (ggplot2)." },
  theme_bw: { source: "ggplot2", url: `${GGPLOT}ggtheme.html`, blurb: "A complete black and white theme (ggplot2)." },
  // base R
  mean: { source: "base R", url: `${BASE_R}mean.html`, blurb: "Arithmetic mean of a vector (base R)." },
  sum: { source: "base R", url: `${BASE_R}sum.html`, blurb: "Sum of the values (base R)." },
  paste: { source: "base R", url: `${BASE_R}paste.html`, blurb: "Concatenate strings (base R)." },
  c: { source: "base R", url: `${BASE_R}c.html`, blurb: "Combine values into a vector (base R)." },
  seq: { source: "base R", url: `${BASE_R}seq.html`, blurb: "Generate regular sequences (base R)." },
};

// --- Python: pandas, NumPy, builtins -------------------------------------
const PANDAS_API = "https://pandas.pydata.org/docs/reference/api/";
const NUMPY_API = "https://numpy.org/doc/stable/reference/generated/";
const PY_BUILTINS = "https://docs.python.org/3/library/functions.html#";

const PANDAS_DATAFRAME_METHODS = new Set([
  "groupby", "merge", "join", "pivot_table", "head", "tail", "describe",
  "apply", "assign", "sort_values", "reset_index", "set_index", "drop",
  "fillna", "dropna", "agg", "rename",
]);
const PANDAS_TOPLEVEL: Record<string, string> = {
  read_csv: "pandas.read_csv.html",
  read_excel: "pandas.read_excel.html",
  concat: "pandas.concat.html",
  merge: "pandas.merge.html",
  DataFrame: "pandas.DataFrame.html",
  Series: "pandas.Series.html",
  to_datetime: "pandas.to_datetime.html",
};
const NUMPY_FUNCS: Record<string, string> = {
  array: "numpy.array.html",
  arange: "numpy.arange.html",
  linspace: "numpy.linspace.html",
  zeros: "numpy.zeros.html",
  ones: "numpy.ones.html",
  mean: "numpy.mean.html",
  where: "numpy.where.html",
};
const PY_BUILTIN_NAMES = new Set([
  "len", "print", "range", "sum", "min", "max", "enumerate", "zip", "map",
  "filter", "sorted", "open", "list", "dict", "set", "tuple", "int", "float",
  "str", "bool", "abs", "round", "type", "isinstance",
]);

// --- SQL (SQLite only) ----------------------------------------------------
const SQLITE_AGG = "https://www.sqlite.org/lang_aggfunc.html";
const SQLITE_DATE = "https://www.sqlite.org/lang_datefunc.html";
const SQLITE_SCALAR = "https://www.sqlite.org/lang_corefunc.html";
const SQLITE_SELECT = "https://www.sqlite.org/lang_select.html";
const SQLITE_WITH = "https://www.sqlite.org/lang_with.html";

const SQLITE_FUNCS: Record<string, Ref> = {
  COUNT: { source: "SQLite", url: SQLITE_AGG, blurb: "Count rows or non-null values (SQLite aggregate)." },
  AVG: { source: "SQLite", url: SQLITE_AGG, blurb: "Average of the values (SQLite aggregate)." },
  SUM: { source: "SQLite", url: SQLITE_AGG, blurb: "Sum of the values (SQLite aggregate)." },
  MIN: { source: "SQLite", url: SQLITE_AGG, blurb: "Minimum value (SQLite aggregate)." },
  MAX: { source: "SQLite", url: SQLITE_AGG, blurb: "Maximum value (SQLite aggregate)." },
  TOTAL: { source: "SQLite", url: SQLITE_AGG, blurb: "Sum that returns 0.0 over no rows (SQLite aggregate)." },
  GROUP_CONCAT: { source: "SQLite", url: SQLITE_AGG, blurb: "Concatenate values across a group (SQLite aggregate)." },
  DATE: { source: "SQLite", url: SQLITE_DATE, blurb: "Date value from a time string (SQLite date function)." },
  DATETIME: { source: "SQLite", url: SQLITE_DATE, blurb: "Date and time value (SQLite date function)." },
  STRFTIME: { source: "SQLite", url: SQLITE_DATE, blurb: "Format a date or time (SQLite date function)." },
  COALESCE: { source: "SQLite", url: SQLITE_SCALAR, blurb: "First non-null argument (SQLite core function)." },
  ROUND: { source: "SQLite", url: SQLITE_SCALAR, blurb: "Round a number (SQLite core function)." },
  LENGTH: { source: "SQLite", url: SQLITE_SCALAR, blurb: "Length of a string or blob (SQLite core function)." },
};
const SQLITE_KEYWORDS: Record<string, Ref> = {
  SELECT: { source: "SQLite", url: SQLITE_SELECT, blurb: "Query rows from tables (SQLite SELECT)." },
  FROM: { source: "SQLite", url: SQLITE_SELECT, blurb: "The tables a query reads from (SQLite SELECT)." },
  WHERE: { source: "SQLite", url: SQLITE_SELECT, blurb: "Filter rows in a query (SQLite SELECT)." },
  JOIN: { source: "SQLite", url: SQLITE_SELECT, blurb: "Combine rows from two tables (SQLite SELECT)." },
  "GROUP BY": { source: "SQLite", url: SQLITE_SELECT, blurb: "Group rows for aggregates (SQLite SELECT)." },
  HAVING: { source: "SQLite", url: SQLITE_SELECT, blurb: "Filter groups after aggregation (SQLite SELECT)." },
  "ORDER BY": { source: "SQLite", url: SQLITE_SELECT, blurb: "Sort the result rows (SQLite SELECT)." },
  LIMIT: { source: "SQLite", url: SQLITE_SELECT, blurb: "Cap the number of result rows (SQLite SELECT)." },
  WITH: { source: "SQLite", url: SQLITE_WITH, blurb: "Common table expressions (SQLite WITH)." },
};

function rEntry(name: string, qualifier?: string): DocEntry | null {
  const ref = R_REFS[name];
  if (!ref) return null;
  // Honor an explicit namespace: stats::filter must not borrow dplyr::filter's
  // curated link and blurb. When the qualifier names a different package than the
  // curated entry, drop it and let the reference-home fallback plus the live
  // help() text (which is namespace-correct) stand on their own.
  if (qualifier) {
    const q = qualifier.toLowerCase();
    const pkg = ref.source.toLowerCase() === "base r" ? "base" : ref.source.toLowerCase();
    if (q !== pkg) return null;
  }
  return { symbol: name, source: ref.source, url: ref.url, blurb: ref.blurb };
}

function pyEntry(req: HelpRequest): DocEntry | null {
  const { name } = req;
  if (PANDAS_DATAFRAME_METHODS.has(name)) {
    return {
      symbol: req.qualifier ? `${req.qualifier}.${name}` : name,
      source: "pandas",
      url: `${PANDAS_API}pandas.DataFrame.${name}.html`,
      blurb: `pandas.DataFrame.${name}: a DataFrame method (pandas).`,
    };
  }
  if (PANDAS_TOPLEVEL[name]) {
    return { symbol: name, source: "pandas", url: `${PANDAS_API}${PANDAS_TOPLEVEL[name]}`, blurb: `pandas.${name} (pandas).` };
  }
  if (NUMPY_FUNCS[name]) {
    return { symbol: name, source: "NumPy", url: `${NUMPY_API}${NUMPY_FUNCS[name]}`, blurb: `numpy.${name} (NumPy).` };
  }
  if (PY_BUILTIN_NAMES.has(name)) {
    return { symbol: name, source: "Python", url: `${PY_BUILTINS}${name}`, blurb: `${name}: a Python built-in function.` };
  }
  return null;
}

function sqlEntry(req: HelpRequest, dialect: SqlDialect): DocEntry | null {
  const key = req.name.toUpperCase();
  // DATE_TRUNC is a Postgres/BigQuery function with no SQLite equivalent.
  if (key === "DATE_TRUNC") {
    return {
      symbol: "DATE_TRUNC",
      source: "SQLite",
      url: SQLITE_DATE,
      blurb: "Truncate a timestamp to a unit.",
      note: "SQLite has no DATE_TRUNC. Use strftime() to truncate dates. DATE_TRUNC is a PostgreSQL and BigQuery function, which do not run here.",
    };
  }
  const ref = SQLITE_FUNCS[key] ?? SQLITE_KEYWORDS[key];
  if (!ref) return null;
  const entry: DocEntry = { symbol: key, source: ref.source, url: ref.url, blurb: ref.blurb };
  if (dialect !== "sqlite") {
    entry.note = `Coding Studio runs SQLite, so only SQLite runs here. The ${dialect} form of ${key} may differ; the link is the SQLite reference.`;
  }
  return entry;
}

/**
 * Resolves a clicked symbol to a documentation entry, or null when the symbol is
 * not in the curated tables (the HELP tab then shows a graceful fallback). The
 * curated map is intentionally small: it covers the requirement's examples and
 * the common tidyverse/pandas/SQLite names a student meets first. A full offline
 * doc corpus is out of scope.
 */
export function resolveDoc(
  req: HelpRequest,
  opts: { dialect?: SqlDialect } = {},
): DocEntry | null {
  if (req.language === "r") return rEntry(req.name, req.qualifier);
  if (req.language === "python") return pyEntry(req);
  return sqlEntry(req, opts.dialect ?? "sqlite");
}

const HOME: Record<HelpLanguage, DocEntry> = {
  r: {
    symbol: "R",
    source: "R",
    url: "https://www.rdocumentation.org/",
    blurb: "No bundled link for this symbol. Search the R documentation.",
  },
  python: {
    symbol: "Python",
    source: "Python",
    url: "https://docs.python.org/3/",
    blurb: "No bundled link for this symbol. Search the Python documentation.",
  },
  sql: {
    symbol: "SQLite",
    source: "SQLite",
    url: "https://www.sqlite.org/docs.html",
    blurb: "No bundled link for this symbol. Browse the SQLite documentation.",
  },
};

/** A per-language reference home, shown when a clicked symbol is not in the
 *  curated map so the HELP tab still offers a useful link. */
export function referenceHome(language: HelpLanguage): DocEntry {
  return HOME[language];
}
