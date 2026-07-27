"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FilePreview, ImportOptions, RunOutcome } from "@/lib/run/manager";
import {
  defaultOptions,
  defaultRestoreOptions,
  detectDelimiter,
  isWorkspaceFile,
  type UploadFormat,
} from "@/lib/sandbox/upload";

type ConflictRule = "rename" | "overwrite" | "skip";

export interface UploadFile {
  filename: string;
  name: string;
  format: UploadFormat;
  bytes: Uint8Array;
}

/**
 * The Import Dataset dialog: shows the actual file (its raw text for csv/json,
 * a rendered grid for the binary formats), a live preview of how it parses with
 * the current options, and the options a beginner needs (skip rows, header,
 * delimiter, Excel sheet, RData object). Import loads it into the session.
 */
export function ImportDialog(props: {
  file: UploadFile;
  dark: boolean;
  previewFile: (req: {
    name: string;
    format: UploadFormat;
    bytes: Uint8Array;
    options: ImportOptions;
  }) => Promise<FilePreview | null>;
  importFile: (req: {
    name: string;
    format: UploadFormat;
    bytes: Uint8Array;
    options: ImportOptions;
  }) => Promise<RunOutcome>;
  onImported: (outcome: RunOutcome) => void;
  onClose: () => void;
}) {
  const { file } = props;
  // A whole-workspace file (5c export) restores every object, rather than importing
  // one dataset. Detected by name so a .RData image restores while a single-object
  // .rds imports.
  const restore = isWorkspaceFile(file.filename);
  const isPickle = file.format === "pkl";
  const isText = file.format === "csv" || file.format === "json";
  // Decoded once, for the raw view and the delimiter guess.
  const rawText = useMemo(
    () => (isText ? new TextDecoder().decode(file.bytes.slice(0, 50000)) : ""),
    [file.bytes, isText],
  );

  const [name, setName] = useState(file.name);
  const [options, setOptions] = useState<ImportOptions>(() => {
    if (isWorkspaceFile(file.filename)) return defaultRestoreOptions();
    const base = defaultOptions(file.format);
    if (file.format === "csv") base.delimiter = detectDelimiter(rawText);
    return base;
  });
  // Restore-only controls, kept separate so changing them does not re-run the
  // (read-only) preview: they only matter when the student clicks Restore.
  const [conflict, setConflict] = useState<ConflictRule>("rename");
  const [trusted, setTrusted] = useState(false);
  const [preview, setPreview] = useState<FilePreview | null>(null);
  const [previewing, setPreviewing] = useState(true);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);

  // Re-preview whenever the parse options change, debounced so a worker call does
  // not fire on every keystroke. For a restore, the preview only lists the file's
  // members and collisions (conflict/trust are applied at Restore time), so it runs
  // once per file.
  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(() => {
      setPreviewing(true);
      const req = restore
        ? { name, format: file.format, bytes: file.bytes, options: { restore: true } }
        : { name, format: file.format, bytes: file.bytes, options };
      props
        .previewFile(req)
        .then((p) => {
          if (cancelled) return;
          setPreview(p);
          if (restore) return;
          // Adopt sheet/object defaults the first time they are known.
          if (p?.sheets && p.sheets.length > 0 && !options.sheet) {
            setOptions((o) => ({ ...o, sheet: p.sheets![0] }));
          }
          if (p?.objects && p.objects.length > 0 && !options.object) {
            setOptions((o) => ({ ...o, object: p.objects![0] }));
          }
        })
        .finally(() => !cancelled && setPreviewing(false));
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [restore ? "restore" : options, name, file]);

  useEffect(() => {
    panelRef.current?.focus();
  }, []);

  const set = useCallback(
    (patch: Partial<ImportOptions>) => setOptions((o) => ({ ...o, ...patch })),
    [],
  );

  const doImport = useCallback(async () => {
    setImporting(true);
    setError(null);
    const outcome = await props.importFile({
      name,
      format: file.format,
      bytes: file.bytes,
      options: restore ? { restore: true, conflict, trusted } : options,
    });
    setImporting(false);
    if (outcome.ok) props.onImported(outcome);
    else setError(outcome.error ?? "The file could not be imported.");
  }, [name, options, file, props, restore, conflict, trusted]);

  const nameValid = /^[A-Za-z][A-Za-z0-9_.]*$/.test(name);
  const members = preview?.members ?? [];
  // The restore action is blocked until a pickle is trusted (it can run code), and
  // when there is nothing to restore.
  const restoreDisabled =
    importing || previewing || members.length === 0 || (isPickle && !trusted);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`${restore ? "Restore" : "Import"} ${file.filename}`}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onKeyDown={(e) => e.key === "Escape" && props.onClose()}
      onClick={(e) => e.target === e.currentTarget && props.onClose()}
    >
      <div
        ref={panelRef}
        tabIndex={-1}
        data-sb-theme={props.dark ? "dark" : "light"}
        className="sb-root flex max-h-[85vh] w-full max-w-4xl flex-col rounded-card border border-[var(--sb-border)] bg-[var(--sb-bg)] text-[var(--sb-text)] shadow-xl outline-none"
      >
        <div className="flex items-center justify-between border-b border-[var(--sb-border)] px-4 py-3">
          <h2 className="text-lg font-bold">
            {restore ? "Restore" : "Import"} {file.filename}
          </h2>
          <button
            type="button"
            onClick={props.onClose}
            className="rounded-card border border-[var(--sb-border)] px-2 py-1 text-sm font-bold hover:border-[var(--sb-accent)]"
          >
            Close
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
          {restore ? (
            <>
              <p className="text-sm text-[var(--sb-muted)]">
                This adds the items below from {file.filename} to your session.
                Nothing you already have changes unless you choose it to.
              </p>
              <div>
                <h3 className="mb-1 text-sm font-bold">
                  What it will restore{" "}
                  {previewing ? (
                    <span className="font-normal text-[var(--sb-muted)]">
                      (reading...)
                    </span>
                  ) : (
                    <span className="font-normal text-[var(--sb-muted)]">
                      ({members.length} item{members.length === 1 ? "" : "s"})
                    </span>
                  )}
                </h3>
                {members.length > 0 ? (
                  <ul className="max-h-48 overflow-auto rounded-card border border-[var(--sb-border)] p-2 text-sm">
                    {members.map((m) => (
                      <li
                        key={m.name}
                        className="flex items-center justify-between px-1 py-0.5"
                      >
                        <span className="font-mono">{m.name}</span>
                        {m.collides ? (
                          <span className="rounded border border-[var(--sb-border)] px-1.5 text-xs font-bold text-[var(--sb-accent)]">
                            already exists
                          </span>
                        ) : null}
                      </li>
                    ))}
                  </ul>
                ) : !previewing ? (
                  <p className="text-sm text-[var(--sb-muted)]">
                    This file has nothing to restore.
                  </p>
                ) : null}
              </div>
              {members.some((m) => m.collides) ? (
                <fieldset className="rounded-card border border-[var(--sb-border)] p-3">
                  <legend className="px-1 text-sm font-bold">
                    If a name already exists
                  </legend>
                  {(
                    [
                      ["rename", "Keep both (rename the incoming one)"],
                      ["overwrite", "Replace the existing one"],
                      ["skip", "Keep the existing one (skip the incoming)"],
                    ] as [ConflictRule, string][]
                  ).map(([val, label]) => (
                    <label
                      key={val}
                      className="flex items-center gap-2 py-0.5 text-sm"
                    >
                      <input
                        type="radio"
                        name="conflict"
                        checked={conflict === val}
                        onChange={() => setConflict(val)}
                      />
                      {label}
                    </label>
                  ))}
                </fieldset>
              ) : null}
              {isPickle ? (
                <label className="flex items-start gap-2 rounded-card border border-[var(--sb-border)] p-3 text-sm">
                  <input
                    type="checkbox"
                    checked={trusted}
                    onChange={(e) => setTrusted(e.target.checked)}
                    className="mt-0.5"
                  />
                  <span>
                    <span className="font-bold">I trust this file.</span> A pickle
                    can run code when it is opened, so only restore one you created
                    or got from someone you trust.
                  </span>
                </label>
              ) : null}
            </>
          ) : (
            <>
          {/* Options */}
          <div className="flex flex-wrap items-end gap-4">
            <label className="text-sm">
              <span className="mb-1 block font-bold">Load as</span>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                aria-invalid={!nameValid}
                className="w-40 rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] px-2 py-1"
              />
            </label>
            {file.format === "csv" ? (
              <>
                <label className="text-sm">
                  <span className="mb-1 block font-bold">Skip rows</span>
                  <input
                    type="number"
                    min={0}
                    value={options.skipRows ?? 0}
                    onChange={(e) => set({ skipRows: Math.max(0, Number(e.target.value) || 0) })}
                    className="w-20 rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] px-2 py-1"
                  />
                </label>
                <label className="text-sm">
                  <span className="mb-1 block font-bold">Separator</span>
                  <select
                    value={options.delimiter ?? ","}
                    onChange={(e) => set({ delimiter: e.target.value })}
                    className="rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] px-2 py-1"
                  >
                    <option value=",">Comma</option>
                    <option value={"\t"}>Tab</option>
                    <option value=";">Semicolon</option>
                  </select>
                </label>
              </>
            ) : null}
            {file.format === "xlsx" && preview?.sheets ? (
              <label className="text-sm">
                <span className="mb-1 block font-bold">Sheet</span>
                <select
                  value={options.sheet ?? preview.sheets[0]}
                  onChange={(e) => set({ sheet: e.target.value })}
                  className="rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] px-2 py-1"
                >
                  {preview.sheets.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            {file.format === "xlsx" ? (
              <label className="text-sm">
                <span className="mb-1 block font-bold">Skip rows</span>
                <input
                  type="number"
                  min={0}
                  value={options.skipRows ?? 0}
                  onChange={(e) => set({ skipRows: Math.max(0, Number(e.target.value) || 0) })}
                  className="w-20 rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] px-2 py-1"
                />
              </label>
            ) : null}
            {(file.format === "csv" || file.format === "xlsx") ? (
              <label className="flex items-center gap-2 text-sm font-bold">
                <input
                  type="checkbox"
                  checked={options.header ?? true}
                  onChange={(e) => set({ header: e.target.checked })}
                />
                First row is the header
              </label>
            ) : null}
            {file.format === "rdata" && preview?.objects ? (
              <label className="text-sm">
                <span className="mb-1 block font-bold">Object</span>
                <select
                  value={options.object ?? preview.objects[0]}
                  onChange={(e) => set({ object: e.target.value })}
                  className="rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] px-2 py-1"
                >
                  {preview.objects.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
          </div>

          {/* The actual file */}
          {isText ? (
            <div>
              <h3 className="mb-1 text-sm font-bold">The file</h3>
              <pre className="max-h-40 overflow-auto rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] p-2 text-xs">
                {rawText}
              </pre>
            </div>
          ) : null}

          {/* How it parses */}
          <div>
            <h3 className="mb-1 text-sm font-bold">
              How it will import{" "}
              {previewing ? (
                <span className="font-normal text-[var(--sb-muted)]">(reading...)</span>
              ) : preview?.totalRows != null ? (
                <span className="font-normal text-[var(--sb-muted)]">
                  ({preview.totalRows} rows)
                </span>
              ) : null}
            </h3>
            {preview?.parseError ? (
              <p className="rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] p-2 text-sm text-[var(--sb-accent)]">
                Could not read the data with these settings. Look at the file
                above and adjust the options (for example the number of rows to
                skip, or the separator).
              </p>
            ) : preview && preview.columns.length > 0 ? (
              <div className="max-h-56 overflow-auto rounded-card border border-[var(--sb-border)]">
                <table className="min-w-full text-left text-xs">
                  <thead className="bg-[var(--sb-panel)]">
                    <tr>
                      {preview.columns.map((c) => (
                        <th key={c} className="px-2 py-1 font-bold">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row, i) => (
                      <tr key={i} className="border-t border-[var(--sb-border)]">
                        {preview.columns.map((c) => (
                          <td key={c} className="px-2 py-1">
                            {String(row[c] ?? "")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </div>
            </>
          )}

          {error ? (
            <p role="alert" className="text-sm text-[var(--sb-accent)]">
              {error}
            </p>
          ) : null}
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-[var(--sb-border)] px-4 py-3">
          <button
            type="button"
            onClick={props.onClose}
            className="rounded-card border border-[var(--sb-border)] px-3 py-1 text-sm font-bold hover:border-[var(--sb-accent)]"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={doImport}
            disabled={
              restore
                ? restoreDisabled
                : importing || !nameValid || preview == null || !!preview?.parseError
            }
            className="rounded-card bg-[var(--sb-accent)] px-3 py-1 text-sm font-bold text-white disabled:cursor-not-allowed disabled:opacity-60"
          >
            {restore
              ? importing
                ? "Restoring..."
                : "Restore"
              : importing
                ? "Importing..."
                : "Import"}
          </button>
        </div>
      </div>
    </div>
  );
}
