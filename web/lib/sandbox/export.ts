/**
 * Helpers for exporting one tabular object (a data frame, a DataFrame, a table)
 * from the Coding Studio session to a downloaded CSV or TSV. The serialization
 * itself runs in the language worker; these are the pure main-thread pieces plus
 * the browser download, factored from the existing Download Script pattern.
 */

export type ExportFormat = "csv" | "tsv";

/** The field separator for a format. */
export function delimiterFor(format: ExportFormat): string {
  return format === "tsv" ? "\t" : ",";
}

/** The file extension for a format (also its short label). */
export function extensionFor(format: ExportFormat): string {
  return format;
}

/** The MIME type for a format. */
export function mimeFor(format: ExportFormat): string {
  return format === "tsv" ? "text/tab-separated-values" : "text/csv";
}

/**
 * A safe download name: the student (email local part), the object name, and a
 * timestamp, e.g. megahefm-grades-20260723-1430.csv. Mirrors the Download Script
 * naming so the two feel like one feature.
 */
export function exportFilename(opts: {
  userEmail: string;
  name: string;
  format: ExportFormat;
  date?: Date;
}): string {
  const who = (opts.userEmail.split("@")[0] || "sandbox").replace(
    /[^a-zA-Z0-9._-]/g,
    "",
  );
  const safeName =
    opts.name.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "") || "data";
  const d = opts.date ?? new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
  return `${who}-${safeName}-${stamp}.${extensionFor(opts.format)}`;
}

/**
 * Downloads text as a file. Same Blob-and-anchor mechanism as the Download Script
 * button (Sandbox.tsx download); the object URL is revoked after the click.
 */
export function downloadText(text: string, filename: string, mime: string): void {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** The three languages whose whole session can be exported as one file. */
export type WorkspaceLanguage = "python" | "r" | "sql";

/** The whole-environment file extension per language: R saves an .RData image, SQL a
 * .sqlite database file, Python a .pkl (pickle) of its serializable values. */
export function workspaceExtensionFor(lang: WorkspaceLanguage): string {
  return lang === "r" ? "RData" : lang === "sql" ? "sqlite" : "pkl";
}

/** The binary MIME type for a workspace artifact. All three are opaque blobs to the
 * browser; a specific type only helps the OS label the download. */
export function workspaceMimeFor(lang: WorkspaceLanguage): string {
  if (lang === "sql") return "application/vnd.sqlite3";
  // .RData and .pkl have no registered MIME type.
  return "application/octet-stream";
}

/**
 * A safe download name for a whole-environment export, e.g.
 * chatisa-workspace-r-20260723-1430.RData. A workspace is a session-wide artifact, not a
 * per-object one, so it is branded chatisa-workspace-<language> plus a timestamp rather than
 * being named after one object.
 */
export function exportWorkspaceFilename(opts: {
  lang: WorkspaceLanguage;
  date?: Date;
}): string {
  const d = opts.date ?? new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
  return `chatisa-workspace-${opts.lang}-${stamp}.${workspaceExtensionFor(opts.lang)}`;
}

/**
 * Downloads binary bytes as a file. The binary sibling of downloadText: same Blob-and-anchor
 * mechanism (see the Download Script button and the plot PNG export), with a binary MIME type
 * and no charset. The object URL is revoked after the click.
 */
export function downloadBytes(
  bytes: Uint8Array,
  filename: string,
  mime: string,
): void {
  // Copy into a fresh ArrayBuffer so a subarray view cannot leak neighbouring bytes.
  const blob = new Blob([bytes.slice()], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
