// components/project/scoping-fields.ts
import type { ScopingTable } from "@/lib/project/scoping";

export interface FieldDef {
  path: string;
  label: string;
  multiline?: boolean;
}
export interface FieldSection {
  heading: string;
  fields: FieldDef[];
}
export interface TableSection {
  heading: string;
  table: ScopingTable;
  columns: { key: string; label: string }[];
  capped: boolean;
}

export const FIELD_SECTIONS: FieldSection[] = [
  {
    heading: "Project",
    fields: [
      { path: "projectName", label: "Project name" },
      { path: "organizationName", label: "Organization" },
      { path: "contacts", label: "Contacts (names and titles)", multiline: true },
    ],
  },
  {
    heading: "Problem",
    fields: [
      { path: "problem.whatProblem", label: "What is the problem", multiline: true },
      { path: "problem.whoAffected", label: "Who is affected", multiline: true },
      { path: "problem.howMuch", label: "How much does it cost", multiline: true },
      { path: "problem.whyPriority", label: "Why is it a priority now", multiline: true },
    ],
  },
  {
    heading: "Ideal data",
    fields: [{ path: "data.idealData", label: "Ideal data", multiline: true }],
  },
  {
    heading: "Ethics",
    fields: [
      { path: "ethics.privacy", label: "Privacy", multiline: true },
      { path: "ethics.transparency", label: "Transparency", multiline: true },
      { path: "ethics.discriminationEquity", label: "Discrimination and equity", multiline: true },
      { path: "ethics.socialLicense", label: "Social license", multiline: true },
      { path: "ethics.accountability", label: "Accountability", multiline: true },
      { path: "ethics.other", label: "Other", multiline: true },
    ],
  },
  {
    heading: "Experiment",
    fields: [
      { path: "experiment.successMeasure", label: "How success is measured", multiline: true },
      { path: "experiment.howTested", label: "How it will be tested", multiline: true },
      { path: "experiment.duration", label: "Duration", multiline: false },
    ],
  },
];

const DATA_COLUMNS = [
  { key: "name", label: "Name" },
  { key: "contains", label: "Contains" },
  { key: "granularity", label: "Granularity" },
  { key: "frequency", label: "Frequency" },
  { key: "identifiers", label: "Identifiers" },
  { key: "owner", label: "Owner" },
  { key: "storage", label: "Storage" },
  { key: "comments", label: "Comments" },
];

export const TABLE_SECTIONS: TableSection[] = [
  {
    heading: "Goals",
    table: "goals",
    columns: [
      { key: "goal", label: "Goal" },
      { key: "constraints", label: "Constraints" },
    ],
    capped: true,
  },
  { heading: "Internal data sources", table: "data.internalSources", columns: DATA_COLUMNS, capped: true },
  { heading: "External data sources", table: "data.externalSources", columns: DATA_COLUMNS, capped: true },
  {
    heading: "Analysis",
    table: "analysis",
    columns: [
      { key: "type", label: "Type" },
      { key: "purpose", label: "Purpose" },
      { key: "validation", label: "Validation" },
    ],
    capped: true,
  },
  {
    heading: "Stakeholders",
    table: "stakeholders",
    columns: [
      { key: "orgDept", label: "Org or department" },
      { key: "involvement", label: "Involvement" },
      { key: "counterpart", label: "Counterpart" },
    ],
    capped: false,
  },
];

/** Reads a scalar value at a one or two level path. */
export function readField(content: unknown, path: string): string {
  const parts = path.split(".");
  let node: unknown = content;
  for (const p of parts) {
    if (node && typeof node === "object") node = (node as Record<string, unknown>)[p];
  }
  return typeof node === "string" ? node : "";
}
