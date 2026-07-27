// components/project/GenericDeliverable.tsx
"use client";

import {
  applyGenericOp,
  type CoachSpec,
  type GenericContent,
} from "@/lib/project/coach-framework";

export function GenericDeliverable({
  spec,
  content,
  onChange,
  lastUpdatedBy,
}: {
  spec: CoachSpec;
  content: GenericContent;
  onChange: (next: GenericContent) => void;
  lastUpdatedBy: string | null;
}) {
  return (
    <section aria-label={`${spec.title} worksheet`} className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl">{spec.title} worksheet</h2>
        {lastUpdatedBy ? (
          <p className="text-sm text-dark-tan">Last updated by {lastUpdatedBy}</p>
        ) : null}
      </div>

      {spec.fields.length > 0 ? (
        <fieldset className="flex flex-col gap-3">
          <legend className="text-lg font-bold">Details</legend>
          {spec.fields.map((f) => {
            const id = `gf-${f.key}`;
            const value = content.fields[f.key] ?? "";
            const set = (v: string) =>
              onChange(applyGenericOp(spec, content, { kind: "setField", path: f.key, value: v }));
            return (
              <div key={f.key} className="flex flex-col gap-1">
                <label htmlFor={id} className="text-sm font-bold">
                  {f.label}
                </label>
                {f.multiline ? (
                  <textarea
                    id={id}
                    rows={2}
                    value={value}
                    onChange={(e) => set(e.target.value)}
                    className="rounded border border-medium-tan bg-paper p-2"
                  />
                ) : (
                  <input
                    id={id}
                    value={value}
                    onChange={(e) => set(e.target.value)}
                    className="rounded border border-medium-tan bg-paper p-2"
                  />
                )}
              </div>
            );
          })}
        </fieldset>
      ) : null}

      {spec.tables.map((table) => {
        const rows = content.tables[table.key] ?? [];
        return (
          <fieldset key={table.key} className="flex flex-col gap-3">
            <legend className="text-lg font-bold">{table.label}</legend>
            {rows.map((row, index) => (
              <div
                key={index}
                className="grid gap-2 rounded-card border border-medium-tan bg-light-tan p-3 sm:grid-cols-2"
              >
                {table.columns.map((col) => {
                  const id = `gt-${table.key}-${index}-${col.key}`;
                  const set = (v: string) =>
                    onChange(
                      applyGenericOp(spec, content, {
                        kind: "setRow",
                        table: table.key,
                        index,
                        row: { [col.key]: v },
                      }),
                    );
                  return (
                    <div key={col.key} className="flex flex-col gap-1">
                      <label htmlFor={id} className="text-xs font-bold">
                        {col.label}
                      </label>
                      <input
                        id={id}
                        value={row[col.key] ?? ""}
                        onChange={(e) => set(e.target.value)}
                        className="rounded border border-medium-tan bg-paper p-1.5 text-sm"
                      />
                    </div>
                  );
                })}
              </div>
            ))}
            <div>
              <button
                type="button"
                onClick={() => onChange(applyGenericOp(spec, content, { kind: "addRow", table: table.key }))}
                className="rounded-card border border-medium-tan bg-paper px-3 py-1.5 text-sm font-bold hover:border-miami-red hover:text-miami-red"
              >
                Add row
              </button>
            </div>
          </fieldset>
        );
      })}
    </section>
  );
}
