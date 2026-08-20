import "server-only";
import {
  AlignmentType,
  BorderStyle,
  Document,
  HeadingLevel,
  Packer,
  Paragraph,
  TabStopType,
  TextRun,
} from "docx";
import {
  LAYOUT,
  TEMPLATES,
  type TemplateId,
} from "@/lib/prompts/fsb-standards";
import type { CoverLetterContent, ResumeContent } from "@/lib/documents/schema";

/**
 * Word export.
 *
 * Metrics come from the Word originals the user supplied rather than from the
 * rendered PDFs: US Letter, 0.5 inch margins all round, Arial headings, and for
 * Standard 1 a Times New Roman body at 10pt.
 *
 * One deliberate departure. The FSB template aligns dates and locations by
 * repeating left tab stops, an artifact of the file having been round-tripped
 * through Google Docs. It looks right until a student edits a line, at which
 * point the columns drift. A single right-aligned tab stop at the right margin
 * renders identically and survives editing, which matters because the whole
 * point of the .docx is that the student can change it.
 */

const rightTab = { type: TabStopType.RIGHT, position: LAYOUT.contentWidthTwips };

/** docx takes half-points. */
const pt = (points: number) => points * 2;

function bodyFontFor(template: TemplateId): string {
  return TEMPLATES[template].bodyFont === "serif"
    ? LAYOUT.resumeBodyFont
    : LAYOUT.headingFont;
}

function nameBlock(
  name: string,
  contact: { email: string | null; phone: string | null; linkedin: string | null },
): Paragraph[] {
  // The letterhead is byte-identical between the resume and cover letter
  // templates, so the two documents look like a matched pair.
  const line = [contact.email, contact.phone, contact.linkedin]
    .filter((v): v is string => Boolean(v && v.trim()))
    .join(" | ");

  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [
        new TextRun({
          text: name,
          bold: true,
          font: LAYOUT.headingFont,
          size: pt(LAYOUT.nameSizePt),
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
      children: [
        new TextRun({
          text: line,
          font: LAYOUT.headingFont,
          size: pt(LAYOUT.contactSizePt),
        }),
      ],
    }),
  ];
}

function sectionHeading(text: string, template: TemplateId): Paragraph {
  return new Paragraph({
    spacing: { before: 160, after: 60 },
    // Standards 1 and 2 rule under each header; Standard 3 does not.
    border: TEMPLATES[template].sectionRules
      ? { bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000" } }
      : undefined,
    children: [
      new TextRun({
        text: text.toUpperCase(),
        bold: true,
        font: LAYOUT.headingFont,
        size: pt(LAYOUT.headingSizePt),
      }),
    ],
  });
}

/** A line with something on the left and something right-aligned opposite it. */
function twoColumn(
  left: { text: string; bold?: boolean; italics?: boolean },
  right: string | null,
  font: string,
): Paragraph {
  const children = [
    new TextRun({
      text: left.text,
      bold: left.bold,
      italics: left.italics,
      font,
      size: pt(LAYOUT.bodySizePt),
    }),
  ];
  if (right) {
    children.push(
      new TextRun({ text: "\t", font, size: pt(LAYOUT.bodySizePt) }),
      new TextRun({ text: right, font, size: pt(LAYOUT.bodySizePt) }),
    );
  }
  return new Paragraph({ tabStops: [rightTab], children });
}

function bullet(text: string, font: string): Paragraph {
  return new Paragraph({
    bullet: { level: 0 },
    spacing: { after: 20 },
    children: [new TextRun({ text, font, size: pt(LAYOUT.bodySizePt) })],
  });
}

export async function renderResumeDocx(
  content: ResumeContent,
  template: TemplateId,
): Promise<Buffer> {
  const font = bodyFontFor(template);
  const style = TEMPLATES[template];
  const children: Paragraph[] = [...nameBlock(content.name, content.contact)];

  // Education
  children.push(sectionHeading("Education", template));
  children.push(
    twoColumn(
      { text: style.schoolLine, bold: true },
      content.education.location || "Oxford, OH",
      font,
    ),
  );
  // Models sometimes stuff the school and city into the degree string too,
  // printing "Miami University... | Oxford, OH Bachelor of Science..." under
  // the school line that already says both (v6.2.0 video review). The school
  // line above is authoritative; strip it out of the degree.
  {
    const scrub = (s: string) =>
      s
        .replace(style.schoolLine, "")
        .replace(content.education.location || "Oxford, OH", "")
        .replace(/^[\s|,·-]+/, "")
        .trim();
    content = {
      ...content,
      education: {
        ...content.education,
        degree: content.education.degree ? scrub(content.education.degree) || null : null,
      },
    };
  }
  if (content.education.degree) {
    children.push(
      twoColumn(
        { text: content.education.degree, italics: true },
        content.education.graduation,
        font,
      ),
    );
  }
  if (content.education.majorMinor) {
    children.push(
      twoColumn({ text: content.education.majorMinor }, content.education.gpa, font),
    );
  }
  for (const honor of content.education.honors) {
    children.push(bullet(honor, font));
  }

  // Experience and the rest. The structured Education block above and the
  // Skills/Certifications block below are authoritative; models sometimes
  // emit their own Education or Skills sections as well, which printed the
  // same content twice on the exported resume (user-visible bug, found on
  // the v6.2.0 video review, 2026-07-29).
  const OWN_SECTIONS = /^\s*(education|skills?(\s*\/?\s*certifications?)?)\s*$/i;
  for (const section of content.sections.filter(
    (s) => !OWN_SECTIONS.test(s.heading),
  )) {
    children.push(sectionHeading(section.heading, template));
    for (const entry of section.entries) {
      const place = [entry.location].filter(Boolean).join(", ");
      if (style.entryOrder === "organization-first") {
        children.push(
          twoColumn({ text: entry.organization, bold: true }, place || null, font),
        );
        children.push(
          twoColumn({ text: entry.title, italics: true }, entry.dates, font),
        );
      } else {
        // Standard 3 leads with the title and folds the location onto the
        // organization line.
        children.push(twoColumn({ text: entry.title, bold: true }, entry.dates, font));
        children.push(
          twoColumn(
            { text: [entry.organization, place].filter(Boolean).join(", "), italics: true },
            null,
            font,
          ),
        );
      }
      for (const b of entry.bullets) children.push(bullet(b.text, font));
    }
  }

  if (content.skills.length > 0) {
    children.push(sectionHeading("Skills / Certifications", template));
    // The template lays these in columns because entries are typically one
    // word. A single wrapped line is the honest equivalent here and keeps the
    // file easy for a student to edit.
    children.push(
      new Paragraph({
        children: [
          new TextRun({
            text: content.skills.join("  ·  "),
            font,
            size: pt(LAYOUT.bodySizePt),
          }),
        ],
      }),
    );
  }

  const doc = new Document({
    sections: [
      {
        properties: {
          page: {
            size: { width: LAYOUT.pageWidthTwips, height: LAYOUT.pageHeightTwips },
            margin: {
              top: LAYOUT.marginTwips,
              bottom: LAYOUT.marginTwips,
              left: LAYOUT.marginTwips,
              right: LAYOUT.marginTwips,
            },
          },
        },
        children,
      },
    ],
  });
  return Buffer.from(await Packer.toBuffer(doc));
}

export async function renderCoverLetterDocx(
  content: CoverLetterContent,
): Promise<Buffer> {
  // The cover letter templates are Arial throughout, whichever resume standard
  // they pair with.
  const font = LAYOUT.coverLetterFont;
  const body = (text: string, after = 160) =>
    new Paragraph({
      spacing: { after },
      children: [new TextRun({ text, font, size: pt(LAYOUT.bodySizePt) })],
    });

  const children: Paragraph[] = [...nameBlock(content.name, content.contact)];

  if (content.date) children.push(body(content.date, 240));

  if (content.recipient.name) children.push(body(content.recipient.name, 0));
  children.push(body(content.recipient.company, 0));
  if (content.recipient.address) children.push(body(content.recipient.address, 240));

  children.push(body(content.salutation, 160));
  for (const paragraph of content.paragraphs) children.push(body(paragraph.text));

  // Models sometimes pack a signed name into the closing ("Sincerely, Jane
  // Doe"); the renderer prints the authoritative name below, which doubled
  // the signature on the exported letter (v6.2.1 video review, 2026-07-30).
  // Keep only the conventional phrase up to its comma.
  const closingMatch = /^\s*([A-Za-z ]+,)/.exec(content.closing);
  children.push(body(closingMatch ? closingMatch[1] : content.closing, 0));
  // Space to sign a printed copy, as the template specifies.
  children.push(body("", 240));
  children.push(body(content.name, 0));

  const doc = new Document({
    sections: [
      {
        properties: {
          page: {
            size: { width: LAYOUT.pageWidthTwips, height: LAYOUT.pageHeightTwips },
            margin: {
              top: LAYOUT.marginTwips,
              bottom: LAYOUT.marginTwips,
              left: LAYOUT.marginTwips,
              right: LAYOUT.marginTwips,
            },
          },
        },
        children,
      },
    ],
  });
  return Buffer.from(await Packer.toBuffer(doc));
}

export { HeadingLevel };
