"use client";

import { useEffect, useState } from "react";
import type { DataPage } from "@/lib/run/manager";
import { ExportMenu } from "@/components/sandbox/ExportMenu";
import type { ExportFormat } from "@/lib/sandbox/export";

export type GetData = (
  name: string,
  offset: number,
  limit: number,
) => Promise<{ ok: boolean; data?: DataPage; error?: string }>;

const PAGE_SIZE = 100;

/**
 * A spreadsheet-style view of a data frame or table, paged through the session
 * (100 rows at a time). The data is fetched from the student's own in-browser
 * session and never leaves the tab. Sorting and filtering are on the roadmap;
 * this first version is scroll plus row and column counts.
 */
export function DataView({
  getData,
  name,
  onExport,
}: {
  getData: GetData;
  name: string;
  onExport?: (name: string, format: ExportFormat) => void;
}) {
  const [page, setPage] = useState(0);
  const [refreshKey, setRefreshKey] = useState(0);
  const [state, setState] = useState<{
    loading: boolean;
    data?: DataPage;
    error?: string;
  }>({ loading: true });

  useEffect(() => {
    // A genuine data fetch: state is set from the resolved request (guarded so a
    // superseded request cannot overwrite a newer one), not a cascading render.
    let alive = true;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setState((s) => ({ ...s, loading: true }));
    getData(name, page * PAGE_SIZE, PAGE_SIZE).then((res) => {
      if (!alive) return;
      setState(
        res.ok && res.data
          ? { loading: false, data: res.data }
          : { loading: false, error: res.error ?? "Could not load the data." },
      );
    });
    return () => {
      alive = false;
    };
  }, [getData, name, page, refreshKey]);

  const data = state.data;
  const total = data?.totalRows ?? 0;
  const from = total === 0 ? 0 : page * PAGE_SIZE + 1;
  const to = Math.min((page + 1) * PAGE_SIZE, total);
  const maxPage = Math.max(0, Math.ceil(total / PAGE_SIZE) - 1);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 border-b border-[var(--sb-border)] px-3 py-1.5 text-xs text-[var(--sb-muted)]">
        <span className="font-mono font-bold text-[var(--sb-text)]">{name}</span>
        {data ? (
          <span>
            {data.columns.length} columns, {total} rows
          </span>
        ) : null}
        <span className="ml-auto flex items-center gap-2">
          <button
            type="button"
            onClick={() => setRefreshKey((k) => k + 1)}
            className="rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] px-2 py-0.5 font-bold hover:border-[var(--sb-accent)]"
          >
            Refresh
          </button>
          {onExport ? (
            <ExportMenu
              name={name}
              onExport={(format) => onExport(name, format)}
            />
          ) : null}
        </span>
      </div>

      <div
        tabIndex={0}
        aria-label={`Data for ${name}`}
        className="min-h-0 flex-1 overflow-auto"
      >
        {state.loading ? (
          <p className="p-3 text-sm text-[var(--sb-muted)]">Loading data...</p>
        ) : state.error ? (
          <p role="alert" className="p-3 text-sm text-[var(--sb-accent)]">
            {state.error}
          </p>
        ) : data ? (
          <Grid data={data} startIndex={from} />
        ) : null}
      </div>

      {data ? (
        <div className="flex items-center gap-2 border-t border-[var(--sb-border)] px-3 py-1.5 text-xs text-[var(--sb-muted)]">
          <span>
            Rows {from} to {to} of {total}
          </span>
          {total > PAGE_SIZE ? (
            <span className="ml-auto flex items-center gap-2">
              <button
                type="button"
                disabled={page <= 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                className="rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] px-2 py-0.5 font-bold hover:border-[var(--sb-accent)] disabled:opacity-40"
              >
                Previous
              </button>
              <span>
                Page {page + 1} of {maxPage + 1}
              </span>
              <button
                type="button"
                disabled={page >= maxPage}
                onClick={() => setPage((p) => Math.min(maxPage, p + 1))}
                className="rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] px-2 py-0.5 font-bold hover:border-[var(--sb-accent)] disabled:opacity-40"
              >
                Next
              </button>
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function Grid({ data, startIndex }: { data: DataPage; startIndex: number }) {
  return (
    <table className="border-collapse text-xs">
      <thead>
        <tr>
          <th className="sticky top-0 border border-[var(--sb-border)] bg-[var(--sb-header)] px-2 py-1" />
          {data.columns.map((col) => (
            <th
              key={col}
              className="sticky top-0 border border-[var(--sb-border)] bg-[var(--sb-header)] px-2 py-1 text-left font-mono font-bold text-[var(--sb-text)]"
            >
              {col}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.rows.map((row, i) => (
          <tr key={i}>
            <td className="border border-[var(--sb-border)] bg-[var(--sb-header)] px-2 py-0.5 text-right text-[var(--sb-muted)]">
              {startIndex + i}
            </td>
            {data.columns.map((col) => (
              <td
                key={col}
                className="whitespace-nowrap border border-[var(--sb-border)] px-2 py-0.5 font-mono"
              >
                {formatCell(row[col])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "NA";
  return String(value);
}
