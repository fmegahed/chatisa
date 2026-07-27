// lib/project/scoping.ts
import { z } from "zod";

/**
 * The project scoping worksheet as a typed deliverable (ADR-010). Ten sections,
 * ported from the legacy scoping worksheet. Every text field defaults to an
 * empty string and every table to an empty array, so a fresh deliverable is
 * fully shaped and directly editable before the coach has filled anything.
 *
 * Bounded tables (goals, data sources, analysis) cap at three rows to match the
 * worksheet. Stakeholders is unbounded.
 */

const str = () => z.string().default("");

const goalRow = z.object({ goal: str(), constraints: str() });
const dataSource = z.object({
  name: str(),
  contains: str(),
  granularity: str(),
  frequency: str(),
  identifiers: str(),
  owner: str(),
  storage: str(),
  comments: str(),
});
const analysisRow = z.object({ type: str(), purpose: str(), validation: str() });
const stakeholderRow = z.object({
  orgDept: str(),
  involvement: str(),
  counterpart: str(),
});

export const scopingContentSchema = z.object({
  projectName: str(),
  organizationName: str(),
  contacts: str(),
  problem: z.object({
    whatProblem: str(),
    whoAffected: str(),
    howMuch: str(),
    whyPriority: str(),
  }),
  goals: z.array(goalRow).max(3).default([]),
  data: z.object({
    internalSources: z.array(dataSource).max(3).default([]),
    externalSources: z.array(dataSource).max(3).default([]),
    idealData: str(),
  }),
  analysis: z.array(analysisRow).max(3).default([]),
  ethics: z.object({
    privacy: str(),
    transparency: str(),
    discriminationEquity: str(),
    socialLicense: str(),
    accountability: str(),
    other: str(),
  }),
  stakeholders: z.array(stakeholderRow).default([]),
  experiment: z.object({
    successMeasure: str(),
    howTested: str(),
    duration: str(),
  }),
});

export type ScopingContent = z.infer<typeof scopingContentSchema>;

export function emptyScopingContent(): ScopingContent {
  // Parsing an empty object applies every default, so the shape stays in one
  // place (the schema) rather than being duplicated here.
  return scopingContentSchema.parse({
    problem: {},
    data: {},
    ethics: {},
    experiment: {},
  });
}

// ---- deliverable mutation operations (called by the coach's tools in 2B) ----

export type ScopingTable =
  | "goals"
  | "data.internalSources"
  | "data.externalSources"
  | "analysis"
  | "stakeholders";

export type ScopingOp =
  | { kind: "setField"; path: string; value: string }
  | { kind: "addRow"; table: ScopingTable }
  | { kind: "setRow"; table: ScopingTable; index: number; row: Record<string, string> };

/** Every scalar field path a setField op may target. */
const FIELD_PATHS = new Set<string>([
  "projectName",
  "organizationName",
  "contacts",
  "problem.whatProblem",
  "problem.whoAffected",
  "problem.howMuch",
  "problem.whyPriority",
  "data.idealData",
  "ethics.privacy",
  "ethics.transparency",
  "ethics.discriminationEquity",
  "ethics.socialLicense",
  "ethics.accountability",
  "ethics.other",
  "experiment.successMeasure",
  "experiment.howTested",
  "experiment.duration",
]);

/** The empty row and its known keys, per table. */
const ROW_SHAPES: Record<ScopingTable, () => Record<string, string>> = {
  goals: () => ({ goal: "", constraints: "" }),
  "data.internalSources": () => ({
    name: "", contains: "", granularity: "", frequency: "",
    identifiers: "", owner: "", storage: "", comments: "",
  }),
  "data.externalSources": () => ({
    name: "", contains: "", granularity: "", frequency: "",
    identifiers: "", owner: "", storage: "", comments: "",
  }),
  analysis: () => ({ type: "", purpose: "", validation: "" }),
  stakeholders: () => ({ orgDept: "", involvement: "", counterpart: "" }),
};

const TABLE_CAPS: Record<ScopingTable, number> = {
  goals: 3,
  "data.internalSources": 3,
  "data.externalSources": 3,
  analysis: 3,
  stakeholders: Number.POSITIVE_INFINITY,
};

function getTable(content: ScopingContent, table: ScopingTable): Record<string, string>[] {
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

/**
 * Applies one operation and returns a NEW content object. On anything malformed
 * (unknown path, unknown table, out-of-range index) it returns the content
 * unchanged, so a bad tool call from the model is a no-op rather than a crash.
 */
export function applyScopingOp(content: ScopingContent, op: ScopingOp): ScopingContent {
  const next: ScopingContent = structuredClone(content);

  if (op.kind === "setField") {
    if (!FIELD_PATHS.has(op.path)) return content;
    const parts = op.path.split(".");
    // Two levels at most, matching FIELD_PATHS.
    if (parts.length === 1) {
      (next as unknown as Record<string, unknown>)[parts[0]] = op.value;
    } else {
      const parent = (next as unknown as Record<string, Record<string, unknown>>)[parts[0]];
      parent[parts[1]] = op.value;
    }
    return next;
  }

  if (op.kind === "addRow") {
    if (!(op.table in ROW_SHAPES)) return content;
    const rows = getTable(next, op.table);
    if (rows.length >= TABLE_CAPS[op.table]) return content;
    rows.push(ROW_SHAPES[op.table]());
    return next;
  }

  // setRow
  if (!(op.table in ROW_SHAPES)) return content;
  const rows = getTable(next, op.table);
  if (op.index < 0 || op.index >= rows.length) return content;
  const known = ROW_SHAPES[op.table]();
  const merged = { ...known, ...rows[op.index] };
  for (const key of Object.keys(known)) {
    if (key in op.row) merged[key] = op.row[key];
  }
  rows[op.index] = merged;
  return next;
}
