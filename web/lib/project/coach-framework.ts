// lib/project/coach-framework.ts
import { z } from "zod";
import type { GenericOp } from "@/lib/project/coach-engine";

export interface CoachFieldDef {
  key: string;
  label: string;
  multiline?: boolean;
}
export interface CoachTableDef {
  key: string;
  label: string;
  columns: { key: string; label: string }[];
}
export interface CoachSpec {
  type: string;
  title: string;
  fields: CoachFieldDef[];
  tables: CoachTableDef[];
  systemPrompt: string;
}

export interface GenericContent {
  fields: Record<string, string>;
  tables: Record<string, Record<string, string>[]>;
}

export function buildEmptyContent(spec: CoachSpec): GenericContent {
  return {
    fields: Object.fromEntries(spec.fields.map((f) => [f.key, ""])),
    tables: Object.fromEntries(spec.tables.map((t) => [t.key, []])),
  };
}

export function coachContentSchema(spec: CoachSpec): z.ZodType<GenericContent> {
  const fieldShape = Object.fromEntries(
    spec.fields.map((f) => [f.key, z.string().default("")]),
  );
  const tableShape = Object.fromEntries(
    spec.tables.map((t) => {
      const rowShape = Object.fromEntries(
        t.columns.map((c) => [c.key, z.string().default("")]),
      );
      return [t.key, z.array(z.object(rowShape)).default([])];
    }),
  );
  return z.object({
    fields: z.object(fieldShape),
    tables: z.object(tableShape),
  }) as unknown as z.ZodType<GenericContent>;
}

function emptyRow(table: CoachTableDef): Record<string, string> {
  return Object.fromEntries(table.columns.map((c) => [c.key, ""]));
}

/** Pure. Returns content unchanged on an unknown field, table, or bad index. */
export function applyGenericOp(
  spec: CoachSpec,
  content: GenericContent,
  op: GenericOp,
): GenericContent {
  const next: GenericContent = structuredClone(content);

  if (op.kind === "setField") {
    if (!spec.fields.some((f) => f.key === op.path)) return content;
    next.fields[op.path] = op.value;
    return next;
  }

  const table = spec.tables.find((t) => t.key === op.table);
  if (!table) return content;
  const rows = next.tables[table.key] ?? (next.tables[table.key] = []);

  if (op.kind === "addRow") {
    rows.push(emptyRow(table));
    return next;
  }

  // setRow
  if (op.index < 0 || op.index >= rows.length) return content;
  const merged = { ...emptyRow(table), ...rows[op.index] };
  for (const col of table.columns) {
    if (col.key in op.row) merged[col.key] = op.row[col.key];
  }
  rows[op.index] = merged;
  return next;
}
