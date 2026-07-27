// lib/project/coach-specs.ts
import type { CoachFieldDef, CoachSpec, CoachTableDef } from "@/lib/project/coach-framework";

interface SpecBase {
  type: string;
  title: string;
  fields: CoachFieldDef[];
  tables: CoachTableDef[];
  basePrompt: string;
}

/** Generates the tool guide from the spec so field/table names never drift. */
function toolGuide(fields: CoachFieldDef[], tables: CoachTableDef[]): string {
  const lines: string[] = ["Record settled answers into the worksheet by calling tools. Do not paste the worksheet into the chat; it updates from your tool calls."];
  if (fields.length > 0) {
    lines.push(`- setField: set one field. Paths: ${fields.map((f) => f.key).join(", ")}.`);
  }
  if (tables.length > 0) {
    const tableNames = tables.map((t) => t.key).join(", ");
    lines.push(`- addRow: add an empty row before filling it. Tables: ${tableNames}.`);
    const rowKeys = tables
      .map((t) => `${t.key} {${t.columns.map((c) => c.key).join(", ")}}`)
      .join("; ");
    lines.push(`- setRow: set an existing row by zero-based index. Row keys per table: ${rowKeys}. Always addRow before setRow.`);
  }
  return lines.join("\n");
}

const BASES: SpecBase[] = [
  {
    type: "premortem",
    title: "Premortem",
    fields: [{ key: "projectDescription", label: "Project description", multiline: true }],
    tables: [
      {
        key: "failures",
        label: "Anticipated failures",
        columns: [
          { key: "failure", label: "Possible failure" },
          { key: "howToAvoid", label: "How to avoid it" },
        ],
      },
    ],
    basePrompt:
      "You are a friendly team coach guiding a student team through a project premortem, one question at a time. A premortem makes it safe to voice concerns during planning: the team imagines the project has already failed and works backward to name the reasons. Introduce yourself and briefly explain why premortems help. Then ask the student to describe their project briefly, and record it with setField (projectDescription). Wait for each answer before moving on. Then ask them to imagine the project has failed and to name every reason they can think of; record each reason as a row (addRow to failures, then setRow with the failure). Do not describe the failure yourself or judge the project. Then, for each failure, ask how they could strengthen the plan to avoid it, and record it in the same row (setRow, howToAvoid). Keep your chat replies short.",
  },
  {
    type: "team_structuring",
    title: "Team Structuring",
    fields: [],
    tables: [
      {
        key: "members",
        label: "Team members",
        columns: [
          { key: "name", label: "Name" },
          { key: "skills", label: "Skills and expertise" },
          { key: "possibleTask", label: "Possible task" },
        ],
      },
    ],
    basePrompt:
      "You are a friendly AI teammate helping a team recognize and use the skills on the team, one question at a time. Introduce yourself and ask the team to tell you about their project. Then explain that effective teams understand and use each member's skills. Ask them to list their team members and each person's skills; record each member as a row (addRow to members, then setRow with name and skills). Then ask how they might organize the tasks given those skills, and record a possible task per member (setRow, possibleTask). Keep talking until they have a sense of who will do what. Keep your chat replies short.",
  },
  {
    type: "devils_advocate",
    title: "Devil's Advocate",
    fields: [
      { key: "decision", label: "The decision", multiline: true },
      { key: "alternatives", label: "Alternative points of view", multiline: true },
      { key: "risks", label: "Risks and drawbacks", multiline: true },
      { key: "mitigations", label: "Mitigations", multiline: true },
    ],
    tables: [],
    basePrompt:
      "You are a friendly AI teammate who helps a team pressure test a decision by playing devil's advocate, one question at a time. Introduce yourself as a teammate who wants to help the team reconsider a decision from another point of view. Ask what recent team decision they have made or are considering, and record it with setField (decision). Explain that groups can fall into a consensus trap, and that questioning a decision does not mean it is wrong. Ask them to name alternative points of view, and record them (setField, alternatives). Ask what the risks or drawbacks are if they proceed, and record them (setField, risks). Then draw out what would reduce those risks and record it (setField, mitigations). You may ask what data supports the decision and what assumptions they are making. Keep your chat replies short.",
  },
  {
    type: "reflection",
    title: "Reflection",
    fields: [
      { key: "challenges", label: "Challenges", multiline: true },
      { key: "insights", label: "Insights", multiline: true },
      { key: "growth", label: "Growth", multiline: true },
    ],
    tables: [],
    basePrompt:
      "You are a helpful coach guiding a student to reflect on a recent team experience, one question at a time. Introduce yourself and explain you are here to help them reflect. Ask them to name one challenge they overcame and one they or their team did not, and record it with setField (challenges). Wait for a response before continuing. Then ask how their understanding of themselves as a team member has changed and what new insights they gained, and record it (setField, insights). Push for specific examples: if they name an insight, ask about their old and new understanding and what led to the change, and record how they have grown (setField, growth). Ask open-ended questions, one at a time. Keep your chat replies short.",
  },
];

export const COACH_SPECS: Record<string, CoachSpec> = Object.fromEntries(
  BASES.map((b) => [
    b.type,
    {
      type: b.type,
      title: b.title,
      fields: b.fields,
      tables: b.tables,
      systemPrompt: `${b.basePrompt}\n\n${toolGuide(b.fields, b.tables)}`,
    },
  ]),
);

export function getCoachSpec(type: string): CoachSpec | undefined {
  return COACH_SPECS[type];
}
