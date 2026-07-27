/**
 * Helpers for the Upload Dataset flow: mapping a file to a format, a safe
 * variable name, and sensible parsing defaults, plus which formats each
 * language can read (so the file picker only offers what will work).
 */

import type { ImportOptions } from "@/lib/run/manager";

export type UploadFormat =
  | "csv"
  | "json"
  | "xlsx"
  | "parquet"
  | "rdata"
  | "pkl"
  | "sqlite";

/** Formats each language can read, given what we bundle. `pkl` (Python), `sqlite`
 * (SQL), and a multi-object `.RData` (R) are whole-workspace restores; the rest
 * import a single object. */
const SUPPORTED: Record<string, UploadFormat[]> = {
  python: ["csv", "json", "xlsx", "parquet", "pkl"],
  r: ["csv", "json", "xlsx", "rdata"],
  sql: ["csv", "json", "sqlite"],
};

/**
 * True when a file is a whole-workspace artifact to restore (all its objects),
 * rather than a single dataset to import as one variable. Our own exports: `.pkl`
 * (Python pickle), `.sqlite` (SQL database), and `.RData`/`.rda` (R image). A single
 * R object saved as `.rds` is NOT a workspace, so it stays a one-object import.
 */
export function isWorkspaceFile(filename: string): boolean {
  const lower = filename.toLowerCase();
  return (
    lower.endsWith(".pkl") ||
    lower.endsWith(".sqlite") ||
    lower.endsWith(".sqlite3") ||
    lower.endsWith(".db") ||
    lower.endsWith(".rdata") ||
    lower.endsWith(".rda")
  );
}

/** The formats the current language supports. */
export function supportedFormats(languageId: string): UploadFormat[] {
  return SUPPORTED[languageId] ?? ["csv", "json"];
}

/** The `accept` attribute for the file picker, scoped to the language. */
export function acceptFor(languageId: string): string {
  const ext: Record<UploadFormat, string[]> = {
    csv: [".csv", ".tsv", ".txt"],
    json: [".json"],
    xlsx: [".xlsx", ".xls"],
    parquet: [".parquet"],
    rdata: [".rdata", ".rda", ".rds"],
    pkl: [".pkl"],
    sqlite: [".sqlite", ".sqlite3", ".db"],
  };
  return supportedFormats(languageId)
    .flatMap((f) => ext[f])
    .join(",");
}

/** The upload format implied by a filename, or null if unrecognised. */
export function formatFromName(filename: string): UploadFormat | null {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".csv") || lower.endsWith(".tsv") || lower.endsWith(".txt"))
    return "csv";
  if (lower.endsWith(".json")) return "json";
  if (lower.endsWith(".xlsx") || lower.endsWith(".xls")) return "xlsx";
  if (lower.endsWith(".parquet")) return "parquet";
  if (lower.endsWith(".rdata") || lower.endsWith(".rda") || lower.endsWith(".rds"))
    return "rdata";
  if (lower.endsWith(".pkl")) return "pkl";
  if (
    lower.endsWith(".sqlite") ||
    lower.endsWith(".sqlite3") ||
    lower.endsWith(".db")
  )
    return "sqlite";
  return null;
}

/**
 * A valid variable/table name from a filename: the base name, non-word
 * characters to underscores, and a letter prefix if it would start with a
 * digit. "2024 sales.xlsx" becomes "x2024_sales".
 */
export function nameFromFile(filename: string): string {
  const base = filename.replace(/\.[^.]+$/, "");
  let cleaned = base
    .trim()
    .replace(/[^A-Za-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .toLowerCase();
  if (!cleaned) cleaned = "dataset";
  if (/^[0-9]/.test(cleaned)) cleaned = "x" + cleaned;
  return cleaned;
}

/** Guesses the CSV delimiter from the first non-empty line: whichever of comma,
 * tab or semicolon appears most. Defaults to a comma. */
export function detectDelimiter(rawText: string): string {
  const line =
    rawText.split(/\r?\n/).find((l) => l.trim().length > 0) ?? "";
  const counts: Record<string, number> = {
    ",": (line.match(/,/g) ?? []).length,
    "\t": (line.match(/\t/g) ?? []).length,
    ";": (line.match(/;/g) ?? []).length,
  };
  let best = ",";
  for (const d of Object.keys(counts)) if (counts[d] > counts[best]) best = d;
  return best;
}

/** The default import options for a format. */
export function defaultOptions(format: UploadFormat): ImportOptions {
  if (format === "csv") return { skipRows: 0, header: true, delimiter: "," };
  if (format === "xlsx") return { skipRows: 0, header: true };
  return {};
}

/** The first free name: `name`, else `name_2`, `name_3`, ... not in `taken`. The
 * suffix uses `_` so a valid identifier stays valid in R, Python, and SQL. */
export function uniqueName(name: string, taken: Set<string>): string {
  if (!taken.has(name)) return name;
  let n = 2;
  while (taken.has(`${name}_${n}`)) n++;
  return `${name}_${n}`;
}

/** Default options for a whole-workspace restore: load everything, and rename on a
 * name collision so nothing already in the session is lost without a choice. */
export function defaultRestoreOptions(): ImportOptions {
  return { restore: true, conflict: "rename" };
}
