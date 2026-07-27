// lib/documents/coach-docx.ts
import "server-only";
import {
  AlignmentType,
  Document,
  HeadingLevel,
  Packer,
  Paragraph,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} from "docx";
import type { ScopingContent, ScopingTable } from "@/lib/project/scoping";
import { FIELD_SECTIONS, TABLE_SECTIONS, readField } from "@/components/project/scoping-fields";
import type { CoachSpec, GenericContent } from "@/lib/project/coach-framework";

export interface ScopingDocHeader {
  projectName: string;
  courseLabel: string;
  organization: string;
  members: string[];
}

export const FONT = "Arial";
export const MIAMI_RED = "C41230"; // app --color-miami-red; docx wants hex, no #
const PAGE = { width: 12240, height: 15840, margin: 720 };

export function labelledValue(label: string, value: string): Paragraph {
  return new Paragraph({
    spacing: { after: 80 },
    children: [
      new TextRun({ text: `${label}: `, bold: true, font: FONT, size: 22 }),
      new TextRun({ text: value || "Not recorded", font: FONT, size: 22 }),
    ],
  });
}

export function heading(text: string): Paragraph {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, font: FONT, bold: true, size: 26, color: MIAMI_RED })],
  });
}

function cell(text: string, bold?: boolean): TableCell {
  return new TableCell({
    children: [new Paragraph({ children: [new TextRun({ text, bold, font: FONT, size: 20 })] })],
  });
}

export function tableFor(
  columns: { key: string; label: string }[],
  rows: Record<string, string>[],
): (Table | Paragraph)[] {
  if (rows.length === 0) {
    return [
      new Paragraph({
        children: [new TextRun({ text: "None recorded.", italics: true, font: FONT, size: 22 })],
      }),
    ];
  }
  return [
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      rows: [
        new TableRow({ tableHeader: true, children: columns.map((c) => cell(c.label, true)) }),
        ...rows.map((r) => new TableRow({ children: columns.map((c) => cell(r[c.key] ?? "")) })),
      ],
    }),
  ];
}

/** Centered H1 title (Miami Red) plus the course, organization, and team. */
export function coverBlocks(header: ScopingDocHeader, title: string, size = 36): Paragraph[] {
  const blocks: Paragraph[] = [
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
      children: [new TextRun({ text: title, font: FONT, bold: true, size, color: MIAMI_RED })],
    }),
    labelledValue("Course", header.courseLabel),
  ];
  if (header.organization) blocks.push(labelledValue("Organization", header.organization));
  if (header.members.length > 0) blocks.push(labelledValue("Team", header.members.join(", ")));
  return blocks;
}

function scopingRows(content: ScopingContent, table: ScopingTable): Record<string, string>[] {
  switch (table) {
    case "goals": return content.goals;
    case "data.internalSources": return content.data.internalSources;
    case "data.externalSources": return content.data.externalSources;
    case "analysis": return content.analysis;
    case "stakeholders": return content.stakeholders;
  }
}

/** The scoping worksheet body (no cover). */
export function scopingBlocks(content: ScopingContent): (Paragraph | Table)[] {
  const blocks: (Paragraph | Table)[] = [];
  for (const section of FIELD_SECTIONS) {
    blocks.push(heading(section.heading));
    for (const f of section.fields) blocks.push(labelledValue(f.label, readField(content, f.path)));
  }
  for (const section of TABLE_SECTIONS) {
    blocks.push(heading(section.heading));
    for (const node of tableFor(section.columns, scopingRows(content, section.table))) blocks.push(node);
  }
  return blocks;
}

/** A generic coach's worksheet body (no cover). */
export function genericBlocks(spec: CoachSpec, content: GenericContent): (Paragraph | Table)[] {
  const blocks: (Paragraph | Table)[] = [];
  if (spec.fields.length > 0) {
    blocks.push(heading("Details"));
    for (const f of spec.fields) blocks.push(labelledValue(f.label, content.fields[f.key] ?? ""));
  }
  for (const table of spec.tables) {
    blocks.push(heading(table.label));
    for (const node of tableFor(table.columns, content.tables[table.key] ?? [])) blocks.push(node);
  }
  return blocks;
}

export async function docFromChildren(children: (Paragraph | Table)[]): Promise<Buffer> {
  const doc = new Document({
    sections: [
      {
        properties: {
          page: {
            size: { width: PAGE.width, height: PAGE.height },
            margin: { top: PAGE.margin, bottom: PAGE.margin, left: PAGE.margin, right: PAGE.margin },
          },
        },
        children,
      },
    ],
  });
  return Buffer.from(await Packer.toBuffer(doc));
}

/** One document, every started deliverable as its own section on a new page. */
export async function renderProjectDeliverablesDocx(
  header: ScopingDocHeader,
  sections: { title: string; blocks: (Paragraph | Table)[] }[],
): Promise<Buffer> {
  const children: (Paragraph | Table)[] = coverBlocks(
    header,
    `${header.projectName || "Project"}: all deliverables`,
  );
  sections.forEach((s, i) => {
    children.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        pageBreakBefore: i > 0,
        spacing: { before: 240, after: 120 },
        children: [new TextRun({ text: s.title, font: FONT, bold: true, size: 32, color: MIAMI_RED })],
      }),
    );
    for (const b of s.blocks) children.push(b);
  });
  if (sections.length === 0) {
    children.push(
      new Paragraph({
        children: [new TextRun({ text: "No deliverables have been started yet.", font: FONT, size: 22 })],
      }),
    );
  }
  return docFromChildren(children);
}
