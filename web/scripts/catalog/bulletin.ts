/**
 * Miami Bulletin parsers for the catalog pipeline (v6.7.0). Pure functions
 * over page HTML so they are tested against saved pages
 * (tests/fixtures/bulletin/) and a change in the bulletin's markup fails in
 * CI rather than silently dropping courses from students' checklists.
 *
 * Nothing is dropped quietly: a requirement row that is not a heading, a
 * comment, a course, an "or" alternative, or a total is returned in
 * `unparsed` so the refresh report can show it.
 */

export interface ProgramGroup {
  /** The area heading the courses sit under ("Required Courses"). */
  title: string;
  /** A plain comment heading inside the area ("Elective Courses"), if any. */
  subtitle: string | null;
  /** The bulletin's own instruction ("Select two of the following"), if any. */
  instruction: string | null;
  /** One entry per requirement row; several codes mean an "or" row. */
  items: { codes: string[] }[];
  /** Free text the bulletin attaches after the courses. */
  notes: string[];
}

export interface ParsedCourse {
  code: string;
  /** Cross-listed or graduate equivalents ("ISA 501" for ISA 401). */
  altCodes: string[];
  title: string;
  /** The lower bound of a variable range ("1-3" gives 1). */
  credits: number;
  description: string;
  prereqText: string;
  /** AND of OR-groups of course codes; empty when there are none. */
  prereq: string[][];
  /** Instructor permission, consent, or class standing can substitute. */
  prereqUncertain: boolean;
}

const CODE = /\b([A-Z]{3})\s(\d{3}[A-Z]?)\b/g;
const INSTRUCTION = /^(select|complete|choose|take)\b/i;

export function text(fragment: string): string {
  return decode(fragment.replace(/<br\s*\/?>/gi, " ").replace(/<[^>]+>/g, " "))
    .replace(/\s+/g, " ")
    .trim();
}

function decode(s: string): string {
  return s
    .replace(/&#160;|&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&rsquo;|&#8217;/g, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)));
}

function codesIn(s: string): string[] {
  return [...s.matchAll(CODE)].map((m) => `${m[1]} ${m[2]}`);
}

type Row =
  | { kind: "header"; text: string; area: boolean }
  | { kind: "course"; codes: string[] }
  | { kind: "or"; codes: string[] }
  | { kind: "skip" }
  | { kind: "unknown"; text: string };

function classifyRow(attrs: string, body: string): Row {
  if (/\bhidden\b|\blistsum\b/.test(attrs)) return { kind: "skip" };
  const comment = /<span class="courselistcomment([^"]*)"[^>]*>([\s\S]*?)<\/span>/.exec(body);
  const codeCell = /<td class="codecol[^"]*"[^>]*>([\s\S]*?)<\/td>/.exec(body);
  if (codeCell) {
    const codes = codesIn(text(codeCell[1]));
    if (codes.length === 0) return { kind: "unknown", text: text(body) };
    const isOr = /\borclass\b/.test(attrs) || /\borclass\b/.test(codeCell[0]) || /^or\b/i.test(text(codeCell[1]));
    return { kind: isOr ? "or" : "course", codes };
  }
  if (comment) {
    const t = text(comment[2]);
    return t ? { kind: "header", text: t, area: /areaheader/.test(comment[1]) || /areaheader/.test(attrs) } : { kind: "skip" };
  }
  const t = text(body);
  return t ? { kind: "unknown", text: t } : { kind: "skip" };
}

/** Every requirement table on a program page, in order. */
export function parseProgram(html: string): { groups: ProgramGroup[]; unparsed: string[] } {
  const tables = html.match(/<table class="sc_courselist"[\s\S]*?<\/table>/g) ?? [];
  const rows: Row[] = [];
  for (const t of tables) {
    for (const m of t.matchAll(/<tr([^>]*)>([\s\S]*?)<\/tr>/g)) rows.push(classifyRow(m[1], m[2]));
  }

  const groups: ProgramGroup[] = [];
  const unparsed: string[] = [];
  let title = "Required courses";
  let subtitle: string | null = null;
  let instruction: string | null = null;
  let group: ProgramGroup | null = null;

  rows.forEach((row, i) => {
    if (row.kind === "skip") return;
    if (row.kind === "unknown") {
      unparsed.push(row.text);
      return;
    }
    if (row.kind === "header") {
      const next = rows.slice(i + 1).find((r) => r.kind !== "skip");
      // A comment after a group's courses with nothing but the end of the
      // table after it is the bulletin's note on that group.
      if (!next && group && !INSTRUCTION.test(row.text)) {
        group.notes.push(row.text);
        return;
      }
      group = null;
      if (INSTRUCTION.test(row.text)) {
        instruction = row.text;
      } else if (row.area) {
        title = row.text;
        subtitle = null;
        instruction = null;
      } else {
        subtitle = row.text;
        instruction = null;
      }
      return;
    }
    if (row.kind === "or" && group && group.items.length > 0) {
      const last = group.items[group.items.length - 1];
      for (const c of row.codes) if (!last.codes.includes(c)) last.codes.push(c);
      return;
    }
    if (!group) {
      group = { title, subtitle, instruction, items: [], notes: [] };
      groups.push(group);
    }
    for (const c of row.codes) group.items.push({ codes: [c] });
  });

  return { groups, unparsed };
}

/**
 * Prerequisite text to AND-of-OR groups. Commas and "and" separate terms,
 * "or" separates options, "One of (a, b)" is one group, a cross-listed
 * pair "STA 463/STA 563" is one option. Instructor permission, consent, or
 * class standing makes the result uncertain: the course terms are still
 * returned, but callers must not treat them as certain.
 */
export function parsePrereq(raw: string): { groups: string[][]; uncertain: boolean } {
  const src = decode(raw).replace(/\s+/g, " ").trim();
  if (!src) return { groups: [], uncertain: false };
  const uncertain = /permission|consent|approval|standing|junior|senior|sophomore|freshman|instructor/i.test(src);
  // Only the course requirement before an "; or permission" alternative.
  let s = src.split(/;\s*or\b/i)[0];
  // "A/B" cross-listed pairs are one option: keep the first code.
  s = s.replace(/\b([A-Z]{3} \d{3}[A-Z]?)\/[A-Z]{3} \d{3}[A-Z]?\b/g, "$1");
  // Grade conditions ("with a grade of "C" or better") carry an "or" that is
  // not an alternative.
  s = s.replace(/with a (?:minimum )?grade (?:of )?(?:at least )?"?[A-F][+-]?"? or (?:better|higher)/gi, " ");
  // A parenthesised list is one explicit group of alternatives: "One of
  // (A, B)", "(A or B) and (C or D)". Words in parentheses ("(College
  // Algebra)") are left as text.
  s = s.replace(/\(([^()]*)\)/g, (_, inner: string) => {
    const codes = codesIn(inner);
    return codes.length ? ` ${codes.join(" | ")} ` : ` ${inner} `;
  });
  // So is an explicit "one of A, B, C or D" list (FIN 401, review fix). A
  // bare comma elsewhere means "and" in the Bulletin's style.
  const code = String.raw`[A-Z]{3} \d{3}[A-Z]?`;
  s = s.replace(
    new RegExp(String.raw`one of\s+(${code}(?:\s*(?:,|\bor\b)\s*(?:or\s+)?${code})*)`, "gi"),
    // Joined with "|", which the ambiguity check below does not read as "or".
    (_, list: string) => ` ${codesIn(list).join(" | ")} `,
  );
  const segments = s.split(/[,;]/).map((x) => x.trim());
  // "A or B and C" has no reliable grouping, and "A, B, or C" contradicts the
  // comma-means-and reading: both are left for nobody to infer from.
  const ambiguous =
    segments.some((x) => /\bor\b/i.test(x) && /\band\b/i.test(x.replace(/^and\s+/i, ""))) ||
    segments.some((x, i) => i > 0 && /^or\b/i.test(x));
  const terms = s.split(/\s*,\s*(?:and\s+)?|\s+and\s+|;/i);
  const groups: string[][] = [];
  for (const term of terms) {
    const options = [...new Set(term.split(/\s+or\s+|\s*\|\s*/i).flatMap(codesIn))];
    if (options.length) groups.push(options);
  }
  // Text that names no course ("determined by professor", "completion of
  // the certificate coursework") is a requirement nobody can infer, which
  // is not the same as having none.
  return { groups, uncertain: uncertain || ambiguous || groups.length === 0 };
}

/** Every course block on a /courses-instruction/<prefix>/ page. */
export function parseCoursePage(html: string): ParsedCourse[] {
  const out: ParsedCourse[] = [];
  for (const block of html.split(/<div class="courseblock"[^>]*>/).slice(1)) {
    const head = /<p class="courseblocktitle"[^>]*>([\s\S]*?)<\/p>/.exec(block);
    if (!head) continue;
    const m = /^([A-Z]{3} \d{3}[A-Z]?(?:\/[A-Z]{3} \d{3}[A-Z]?)*)\.\s+(.+?)\.?\s+\(([\d.]+)(?:\s*-\s*[\d.]+)?(?:;[^)]*)?\)\s*$/.exec(text(head[1]));
    if (!m) continue;
    const [code, ...altCodes] = m[1].split("/");
    const descHtml = /<p class="courseblockdesc"[^>]*>([\s\S]*?)<\/p>/.exec(block)?.[1] ?? "";
    const desc = text(descHtml);
    const pre = /Prerequisites?:\s*(.*?)(?:\s*(?:Co-requisites?|Cross-listed with|Credit\/No-credit)\b|$)/i.exec(desc);
    const prereqText = pre ? pre[1].trim() : "";
    const crossListed = /Cross-listed with\s+([^.]+)/i.exec(desc);
    const { groups, uncertain } = parsePrereq(prereqText);
    out.push({
      code,
      altCodes: [...new Set([...altCodes, ...(crossListed ? codesIn(crossListed[1]) : [])])].filter((c) => c !== code),
      title: m[2].trim(),
      credits: Number(m[3]),
      description: pre ? desc.slice(0, pre.index).trim() : desc,
      prereqText,
      prereq: groups,
      prereqUncertain: uncertain,
    });
  }
  return out;
}
