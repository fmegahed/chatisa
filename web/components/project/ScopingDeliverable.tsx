// components/project/ScopingDeliverable.tsx
"use client";

import { applyScopingOp, type ScopingContent, type ScopingTable } from "@/lib/project/scoping";
import {
  FIELD_SECTIONS,
  TABLE_SECTIONS,
  readField,
} from "@/components/project/scoping-fields";

function rowsFor(content: ScopingContent, table: ScopingTable): Record<string, string>[] {
  switch (table) {
    case "goals":
      return content.goals;
    case "data.internalSources":
      return content.data.internalSources;
    case "data.externalSources":
      return content.data.externalSources;
    case "analysis":
      return content.analysis;
    case "stakeholders":
      return content.stakeholders;
  }
}

export function ScopingDeliverable({
  content,
  onChange,
  lastUpdatedBy,
}: {
  content: ScopingContent;
  onChange: (next: ScopingContent) => void;
  lastUpdatedBy: string | null;
}) {
  return (
    <section aria-label="Project scoping worksheet" className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl">Scoping worksheet</h2>
        {lastUpdatedBy ? (
          <p className="text-sm text-dark-tan">Last updated by {lastUpdatedBy}</p>
        ) : null}
      </div>

      {FIELD_SECTIONS.map((section) => (
        <fieldset key={section.heading} className="flex flex-col gap-3">
          <legend className="text-lg font-bold">{section.heading}</legend>
          {section.fields.map((f) => {
            const id = `sf-${f.path}`;
            const value = readField(content, f.path);
            const set = (v: string) =>
              onChange(applyScopingOp(content, { kind: "setField", path: f.path, value: v }));
            return (
              <div key={f.path} className="flex flex-col gap-1">
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
      ))}

      {TABLE_SECTIONS.map((section) => {
        const rows = rowsFor(content, section.table);
        const atCap = section.capped && rows.length >= 3;
        return (
          <fieldset key={section.heading} className="flex flex-col gap-3">
            <legend className="text-lg font-bold">{section.heading}</legend>
            {rows.map((row, index) => (
              <div
                key={index}
                className="grid gap-2 rounded-card border border-medium-tan bg-light-tan p-3 sm:grid-cols-2"
              >
                {section.columns.map((col) => {
                  const id = `tf-${section.table}-${index}-${col.key}`;
                  const set = (v: string) =>
                    onChange(
                      applyScopingOp(content, {
                        kind: "setRow",
                        table: section.table,
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
                disabled={atCap}
                onClick={() => onChange(applyScopingOp(content, { kind: "addRow", table: section.table }))}
                className="rounded-card border border-medium-tan bg-paper px-3 py-1.5 text-sm font-bold hover:border-miami-red hover:text-miami-red disabled:cursor-not-allowed disabled:opacity-60"
              >
                Add row
              </button>
              {atCap ? (
                <span className="ml-2 text-sm text-dark-tan">Up to three rows.</span>
              ) : null}
            </div>
          </fieldset>
        );
      })}
    </section>
  );
}
