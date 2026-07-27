/** The three runnable Coding Studio languages, for documentation lookup. */
export type HelpLanguage = "r" | "python" | "sql";

/** Whether the clicked token reads as a callable or a language keyword. A hint,
 *  not authoritative: `resolveDoc` tries both tables regardless. */
export type SymbolKind = "function" | "keyword";

/** SQL dialects the resolver's signature admits. Only "sqlite" is wired here,
 *  because sqlite-wasm is the only engine that runs in the browser. */
export type SqlDialect =
  | "sqlite"
  | "postgres"
  | "mysql"
  | "bigquery"
  | "snowflake";

/** A resolved click: the token and, when known, the receiver it hangs off
 *  (for Python `df.groupby`, `qualifier` is `df`). */
export interface HelpRequest {
  name: string;
  qualifier?: string;
  kind: SymbolKind;
  language: HelpLanguage;
}

/** What the HELP tab shows: the symbol, its source, the canonical doc URL, an
 *  optional bundled blurb, and an optional honesty note (SQL dialects). */
export interface DocEntry {
  symbol: string;
  source: string;
  url: string;
  blurb?: string;
  note?: string;
}
