/**
 * Simple Syllabus evidence for the catalog pipeline (v6.7.0).
 *
 * Miami publishes syllabi at syllabus.miamioh.edu; the library and each
 * syllabus are public JSON. Only course-level evidence is kept: weekly
 * topics, stated outcomes, and named tools. The instructor section is never
 * read, policy boilerplate is skipped, and any email address that appears
 * in kept text is removed, because the data describes courses, not people.
 */

import { text } from "./bulletin";

export const LIBRARY_URL =
  "https://syllabus.miamioh.edu/api2/doc-library-search?term_statuses%5B%5D=future&term_statuses%5B%5D=current";
export const DOC_URL = "https://syllabus.miamioh.edu/api2/doc-full-page-get?code=";

export interface Evidence {
  topics: string[];
  tools: string[];
  outcomes: string[];
  /** Textbook titles only: never authors, ISBNs, or publishers. */
  readings: string[];
}

export interface CourseEvidence extends Evidence {
  terms: string[];
  sections: number;
}

type FetchJson = (url: string) => Promise<unknown>;

interface LibraryPage {
  pagination: { total: number; returned: number; page: number; page_size: number };
  items: { code: string; title: string; term_name: string }[];
}

/**
 * Every published section in the current and future terms, grouped by
 * course code ("ISA 225"). The library caps a page at 500 items and says so
 * only in `pagination`, so the whole list is paged through.
 */
export async function listSections(
  fetchJson: FetchJson,
  pageSize = 500,
): Promise<Map<string, { code: string; term: string }[]>> {
  const out = new Map<string, { code: string; term: string }[]>();
  for (let page = 0, seen = 0; ; page++) {
    const body = (await fetchJson(`${LIBRARY_URL}&page_size=${pageSize}&page=${page}`)) as LibraryPage;
    for (const item of body.items) {
      const m = /^([A-Z]{3}) (\d{3}[A-Z]?)\b/.exec(item.title);
      if (!m) continue;
      const key = `${m[1]} ${m[2]}`;
      const list = out.get(key) ?? [];
      list.push({ code: item.code, term: item.term_name });
      out.set(key, list);
    }
    seen += body.items.length;
    if (body.items.length === 0 || seen >= body.pagination.total) break;
  }
  return out;
}

/** Tools named anywhere in the kept sections. A closed list: no guessing. */
const TOOLS: [string, RegExp][] = [
  ["Python", /\bpython\b|\bjupyter\b|\bpandas\b/i],
  ["R", /\bRStudio\b|\bR\s+(?:programming|language|studio|markdown)\b|\busing R\b|\bin R\b|\btidyverse\b|\bggplot2?\b/],
  ["SQL", /\bSQL\b/],
  // Case-sensitive: "excel at" is a verb, "Excel" is the spreadsheet.
  ["Excel", /\bExcel\b|\b[Ss]preadsheets?\b/],
  ["Tableau", /\btableau\b/i],
  ["Power BI", /\bpower\s?bi\b/i],
  ["SAS", /\bSAS\b/],
  ["SPSS", /\bSPSS\b/],
  ["JMP", /\bJMP\b/],
  ["Minitab", /\bminitab\b/i],
  ["Snowflake", /\bsnowflake\b/i],
  ["Databricks", /\bdatabricks\b/i],
  ["Spark", /\bspark\b/i],
  ["AWS", /\bAWS\b|\bamazon web services\b/i],
  ["Azure", /\bazure\b/i],
  ["Git", /\bgit\b|\bgithub\b/i],
  ["Bloomberg", /\bbloomberg\b/i],
  ["QuickBooks", /\bquickbooks\b/i],
  ["SAP", /\bSAP\b/],
  ["Salesforce", /\bsalesforce\b/i],
];

const EMAIL = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g;
const NOT_A_TOPIC = /^(review|exam|midterm|final|quiz|holiday|no class|spring break|fall break|thanksgiving|reading day|tbd|week\b|date\b|topic\b)/i;

function clean(s: string): string {
  return s.replace(EMAIL, "").replace(/\(\s*(questions?:?)?\s*\)/gi, "").replace(/\s+/g, " ").trim();
}

function heading(html: string): string {
  return text(/<h\d[^>]*>([\s\S]*?)<\/h\d>/.exec(html)?.[1] ?? "");
}

/** The topic column of the calendar table, one entry per line of a cell. */
function calendarTopics(html: string): string[] {
  const rows = [...html.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/g)].map((m) =>
    [...m[1].matchAll(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/g)].map((c) => c[1]),
  );
  if (rows.length === 0) {
    return [...html.matchAll(/<li[^>]*>([\s\S]*?)<\/li>/g)].map((m) => text(m[1]));
  }
  const headerIndex = rows.findIndex((r) => r.some((c) => /topic/i.test(text(c))));
  const col = headerIndex >= 0 ? rows[headerIndex].findIndex((c) => /topic/i.test(text(c))) : 1;
  const body = rows.slice(headerIndex + 1);
  const out: string[] = [];
  for (const r of body) {
    const cell = r[col] ?? "";
    for (const line of cell.split(/<br\s*\/?>|<\/p>|<\/li>|\n/i)) {
      const t = clean(text(line));
      if (t && !NOT_A_TOPIC.test(t) && !/^Ch\.\s*\d/.test(t)) out.push(t);
    }
  }
  return out;
}

/** Course-level evidence from one syllabus (doc-full-page-get response). */
export function extractEvidence(doc: unknown): Evidence {
  const components: { html?: string; component_type?: string }[] =
    (doc as { items?: { doc_data?: { components?: unknown[] } }[] }).items?.[0]?.doc_data?.components as never ?? [];
  const topics: string[] = [];
  const outcomes: string[] = [];
  const readings: string[] = [];
  let toolText = "";
  for (const c of components) {
    const html = c.html ?? "";
    if (c.component_type === "instructor") continue;
    const h = heading(html);
    if (/calendar|schedule|topics|course outline/i.test(h)) {
      topics.push(...calendarTopics(html));
      toolText += " " + text(html);
    } else if (/outcome|objective|goals/i.test(h)) {
      const items = [...html.matchAll(/<li[^>]*>([\s\S]*?)<\/li>/g)].map((m) => clean(text(m[1])));
      outcomes.push(...items.filter(Boolean));
      toolText += " " + text(html);
    } else if (c.component_type === "material" || /software|material|technology|tools|readings/i.test(h)) {
      // Book titles only; tools are looked for in the titles too
      // ("Python for Data Analysis"), never in the author lines.
      const titles = bookTitles(html);
      readings.push(...titles);
      toolText += " " + titles.join(" ");
    }
  }
  const tools = TOOLS.filter(([, re]) => re.test(toolText)).map(([name]) => name);
  return { topics: dedupe(topics), tools, outcomes: dedupe(outcomes), readings: dedupe(readings) };
}

/**
 * The first text cell of each reading row is the book's title; the cells
 * after it (ISBN, authors, publisher) are never read.
 */
function bookTitles(html: string): string[] {
  const out: string[] = [];
  for (const row of html.split(/<li\b/).slice(1)) {
    const m = /class="cell-0 cell-basic[^"]*"[\s\S]*?<span[^>]*class="cell-content"[^>]*>([\s\S]*?)<\/span>/.exec(row);
    const title = m ? clean(text(m[1])) : "";
    if (title) out.push(title);
  }
  return out;
}

function dedupe(list: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const s of list) {
    const k = s.toLowerCase();
    if (s && !seen.has(k)) {
      seen.add(k);
      out.push(s);
    }
  }
  return out;
}

/** All sections of one course, merged and sorted for a stable diff. */
export function mergeEvidence(sections: { term: string; evidence: Evidence }[]): CourseEvidence {
  const sortCi = (a: string, b: string) => a.toLowerCase().localeCompare(b.toLowerCase());
  const cap = (list: string[], n: number) => list.slice(0, n);
  return {
    terms: [...new Set(sections.map((s) => s.term))].sort(),
    sections: sections.length,
    topics: cap(dedupe(sections.flatMap((s) => s.evidence.topics)).sort(sortCi), 80),
    tools: [...new Set(sections.flatMap((s) => s.evidence.tools))].sort(),
    outcomes: cap(dedupe(sections.flatMap((s) => s.evidence.outcomes)).sort(sortCi), 30),
    readings: cap(dedupe(sections.flatMap((s) => s.evidence.readings ?? [])).sort(sortCi), 20),
  };
}
