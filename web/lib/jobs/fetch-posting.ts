import "server-only";
import { lookup } from "node:dns/promises";
import { isIP } from "node:net";
import { logger } from "@/lib/log";

/**
 * Fetching a job posting from a URL the student supplies.
 *
 * This is a server-side fetch of an address chosen by a user, which is a
 * server-side request forgery surface: without controls it can be pointed at
 * cloud metadata endpoints, at localhost, or at anything else reachable from
 * the server but not from the internet. Every address is therefore resolved and
 * checked before a request is made, and again after each redirect, because a
 * public hostname can redirect to a private one.
 *
 * The other half of the design is honesty. The large job boards will usually
 * fail or return a login wall, and their terms discourage automated access. A
 * failure here must say so plainly, because a silent failure produces a generic
 * interview and the student never learns why.
 */

// Raised from 750KB after real employer pages (US Bank, Google) came in over
// 1.2MB: modern career pages are large JavaScript apps, and the description we
// actually want is a few KB of structured data inside them. The cap is a
// download guard, not a content guard, and the 12s timeout bounds it further.
export const MAX_POSTING_BYTES = 4_000_000;
export const FETCH_TIMEOUT_MS = 12_000;
const MAX_REDIRECTS = 3;

export type PostingOutcome =
  | "fetched"
  | "blocked_host"
  | "login_required"
  | "unreachable"
  | "not_html"
  | "too_large"
  | "not_a_posting"
  | "empty";

export interface PostingResult {
  outcome: PostingOutcome;
  text: string | null;
  /** Shown to the student. Always actionable, never a raw error. */
  message: string;
}

/** Boards that reliably refuse automated access. Named so the student is told
 * up front rather than after a pointless wait. */
const KNOWN_BLOCKERS = [
  "linkedin.com",
  "indeed.com",
  "glassdoor.com",
  "ziprecruiter.com",
];

export function isKnownBlocker(url: string): boolean {
  try {
    const host = new URL(url).hostname.toLowerCase();
    return KNOWN_BLOCKERS.some((b) => host === b || host.endsWith(`.${b}`));
  } catch {
    return false;
  }
}

/** True for addresses that must never be requested from the server. */
export function isBlockedAddress(address: string): boolean {
  const version = isIP(address);
  if (version === 4) {
    const p = address.split(".").map(Number);
    if (p[0] === 10) return true; // private
    if (p[0] === 127) return true; // loopback
    if (p[0] === 0) return true; // this network
    if (p[0] === 169 && p[1] === 254) return true; // link-local, cloud metadata
    if (p[0] === 172 && p[1] >= 16 && p[1] <= 31) return true; // private
    if (p[0] === 192 && p[1] === 168) return true; // private
    if (p[0] === 100 && p[1] >= 64 && p[1] <= 127) return true; // CGNAT
    if (p[0] >= 224) return true; // multicast and reserved
    return false;
  }
  if (version === 6) {
    const a = address.toLowerCase();
    if (a === "::1" || a === "::") return true;
    if (a.startsWith("fc") || a.startsWith("fd")) return true; // unique local
    if (a.startsWith("fe80")) return true; // link-local
    // IPv4-mapped addresses carry the v4 rules with them.
    const mapped = /^::ffff:(\d+\.\d+\.\d+\.\d+)$/.exec(a);
    if (mapped) return isBlockedAddress(mapped[1]);
    return false;
  }
  return true; // not an IP at all
}

/** Resolves a hostname and refuses it if any address is internal. */
/** Exported for Ask Anything's read_url tool, which shares this SSRF guard. */
export async function hostIsSafe(hostname: string): Promise<boolean> {
  if (isIP(hostname)) return !isBlockedAddress(hostname);
  try {
    const results = await lookup(hostname, { all: true });
    if (results.length === 0) return false;
    return results.every((r) => !isBlockedAddress(r.address));
  } catch {
    return false;
  }
}

/** The named HTML entities worth decoding for a job posting. */
const NAMED_ENTITIES: Record<string, string> = {
  mdash: "\u2014", ndash: "\u2013", hellip: "...", rsquo: "'", lsquo: "'",
  rdquo: '"', ldquo: '"', bull: "\u2022", middot: "\u00b7", times: "x",
  divide: "\u00f7", deg: "\u00b0", trade: "(TM)", copy: "(c)", reg: "(R)",
  plusmn: "+/-", frac12: "1/2", frac14: "1/4", frac34: "3/4",
  prime: "'", Prime: '"', sbquo: ",", bdquo: '"', dagger: "+", Dagger: "++",
};

/** Strips markup down to readable text. */
export function htmlToText(html: string): string {
  const stripped = html
    .replace(/<script\b[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[\s\S]*?<\/style>/gi, " ")
    .replace(/<noscript\b[\s\S]*?<\/noscript>/gi, " ")
    .replace(/<nav\b[\s\S]*?<\/nav>/gi, " ")
    .replace(/<footer\b[\s\S]*?<\/footer>/gi, " ")
    .replace(/<header\b[\s\S]*?<\/header>/gi, " ")
    // Decode entities BEFORE stripping tags. Structured JobPosting data (P&G,
    // Workday and similar) carries its description as entity-encoded HTML
    // inside JSON, so "&lt;p&gt;" only becomes "<p>" at this step; decoding it
    // after the tag strip, as an earlier version did, left the tags visible in
    // the extracted text.
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    // Now the real and formerly-encoded tags are both real, so strip them.
    .replace(/<\/(p|div|li|h[1-6]|tr|section|article)>/gi, "\n")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "")
    .join("\n")
    .trim();

  // Decode text entities after the tags are gone, and twice, because some
  // sources double-encode them: Greenhouse ships "&amp;nbsp;", so one pass
  // leaves "&nbsp;" behind. A second pass on already-clean text is a no-op.
  return decodeTextEntities(decodeTextEntities(stripped));
}

/**
 * Decodes text-level entities. Kept separate from tag handling and run after
 * it, so it never turns an encoded "&lt;" into a "<" that the tag stripper has
 * already passed. `&amp;` is decoded last so "&amp;nbsp;" resolves in order.
 */
function decodeTextEntities(text: string): string {
  return text
    .replace(/&nbsp;/gi, " ")
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/&quot;/gi, '"')
    .replace(
      /&(mdash|ndash|hellip|rsquo|lsquo|rdquo|ldquo|bull|middot|times|divide|deg|trade|copy|reg|plusmn|frac12|frac14|frac34|prime|Prime|sbquo|bdquo|dagger|Dagger);/g,
      (whole, name) => NAMED_ENTITIES[name] ?? whole,
    )
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)))
    .replace(/&#x([0-9a-f]+);/gi, (_, code) => String.fromCharCode(parseInt(code, 16)))
    .replace(/&amp;/gi, "&");
}

/**
 * Pulls a job description out of schema.org JobPosting structured data.
 *
 * Large employers render the visible page with JavaScript, so a plain fetch of
 * P&G, Workday, Greenhouse or Lever returns a shell with 200 characters of text
 * and none of the actual posting. But the same sites embed a JobPosting block
 * in <script type="application/ld+json"> for search engines, and that block
 * carries the full description. Reading it is how this fetcher works at all on
 * the big career platforms rather than always bouncing the student to paste.
 *
 * Found on 2026-07-21 while testing real P&G links: the visible HTML yielded
 * 818 characters of company boilerplate, while the JSON-LD held the real
 * 8,215-character description.
 */
/**
 * The content-bearing fields of a schema.org JobPosting, in reading order.
 * Most sites use only `description`; Meta and others split the real posting
 * across the rest, so all are gathered when present.
 */
const JOB_CONTENT_FIELDS = [
  "description",
  "responsibilities",
  "qualifications",
  "skills",
  "experienceRequirements",
  "educationRequirements",
] as const;

/** Reads @type whether it is a string or an array of strings. */
function typeString(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    const found = value.find((v) => typeof v === "string");
    return typeof found === "string" ? found : "";
  }
  return "";
}

/** Looks a property up on an object regardless of key case. */
function pickField(object: unknown, name: string): string {
  if (!object || typeof object !== "object") return "";
  const wanted = name.toLowerCase();
  for (const [key, value] of Object.entries(object as Record<string, unknown>)) {
    if (key.toLowerCase() === wanted && typeof value === "string") return value;
  }
  return "";
}

export function extractJsonLdPosting(html: string): string | null {
  const blocks = html.matchAll(
    // The "+" in the type attribute may be HTML-entity-encoded. USAJobs writes
    // it as "application/ld&#x2B;json" (its nonce-based CSP encodes attribute
    // values), which a literal "+" match misses, dropping a clean JobPosting
    // block and falling back to noisy page text. Accept the plus in raw and
    // entity forms.
    /<script[^>]*type=["']application\/ld(?:\+|&#x2b;|&#43;|&plus;)json["'][^>]*>([\s\S]*?)<\/script>/gi,
  );
  for (const block of blocks) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(block[1].trim());
    } catch {
      continue;
    }
    // A block may be a single object, an array, or an @graph container.
    const candidates: unknown[] = Array.isArray(parsed)
      ? parsed
      : parsed && typeof parsed === "object" && "@graph" in parsed
        ? [...(parsed as { "@graph": unknown[] })["@graph"], parsed]
        : [parsed];

    for (const candidate of candidates) {
      if (
        candidate &&
        typeof candidate === "object" &&
        typeString((candidate as Record<string, unknown>)["@type"]) === "JobPosting"
      ) {
        // Gather every content field, not just description. schema.org splits
        // a JobPosting across description, responsibilities, qualifications and
        // skills, and some sites (Meta) put the bulk of the posting in the
        // latter three: reading description alone returned a 1,660-char overview
        // and dropped the 587-char responsibilities and 1,172-char
        // qualifications a resume tailorer most needs. Read case-insensitively,
        // because World Bank's Cornerstone site emits PascalCase keys.
        const parts = JOB_CONTENT_FIELDS.map((field) =>
          htmlToText(pickField(candidate, field)),
        ).filter((part) => part.trim() !== "");
        if (parts.length === 0) continue;

        const title = pickField(candidate, "title");
        const body = parts.join("\n\n");
        return title ? `${title}\n\n${body}` : body;
      }
    }
  }
  return null;
}

/**
 * Does this read like an actual job description, or like the site's furniture?
 *
 * Found by testing against real employers on 2026-07-21: a fabricated job id on
 * a large careers site returns HTTP 200 and serves the generic search page, and
 * a length check alone happily accepted 5,120 characters of cookie banner as
 * "the posting". A student would then have had their resume tailored to
 * navigation text. Silently wrong is worse than failing, so a page now has to
 * look like a posting before it is offered as one.
 */
const POSTING_SIGNALS = [
  /\bresponsibilit(y|ies)\b/i,
  /\bqualification/i,
  /\brequirements?\b/i,
  /\byou will\b/i,
  /\byears? of experience\b/i,
  /\bwhat you.ll (do|bring)\b/i,
  /\bthe (role|opportunity|position)\b/i,
  /\bwe are looking for\b/i,
  /\bskills? (and|&) \w+/i,
  /\bpreferred\b/i,
];

/** Text that means we landed on the furniture rather than a posting. */
const BOILERPLATE_SIGNALS = [
  /cookie (information|policy|preferences|consent)/i,
  /\bsearch (jobs|results)\b/i,
  /\bjob search\b/i,
  /\bdiscover open roles\b/i,
  /welcome to .{0,40}(careers|job search)/i,
];

export function looksLikePosting(text: string): boolean {
  const signals = POSTING_SIGNALS.filter((p) => p.test(text)).length;
  const boilerplate = BOILERPLATE_SIGNALS.filter((p) => p.test(text)).length;

  // Signals carry the weight, not length. A first measurement used a 600
  // character floor and rejected a genuine 578 character internship posting:
  // real postings for students are often short, while the careers pages this
  // is meant to catch are long. Length only rules out a stub.
  if (signals < 2) return false;
  if (text.length < 300) return false;
  // Boilerplate is fine inside a real posting (most pages carry a cookie
  // banner); it only condemns a page that has little else.
  if (boilerplate > 0 && signals < 4) return false;
  return true;
}

/**
 * Did we land on a search-results grid rather than one job description?
 *
 * The dangerous case, found on 2026-07-21 with a real Google Careers URL: the
 * page served a grid of unrelated openings ("Machine Learning Accelerators,
 * Sunnyvale... Site Reliability Manager, San Francisco... +24 more"), which
 * passed the posting heuristic because a jobs grid naturally contains the words
 * "responsibilities" and "qualifications" somewhere. A resume tailored to a
 * list of other people's jobs is worse than no help.
 *
 * Only consulted when there is no structured JobPosting data, since that is the
 * trustworthy path. A real multi-location posting (Deloitte lists 35 cities for
 * one job) reaches this only if it lacks structured data, so the signal is the
 * results-grid marker, not the location count.
 */
export function looksLikeListing(text: string): boolean {
  // "+24 more ; +23 more": the overflow marker a results grid uses, and one a
  // single posting never has.
  const overflowMarkers = (text.match(/\+\d+\s+more\b/gi) ?? []).length;
  if (overflowMarkers >= 1) return true;

  // A grid repeats "Job Title  City, ST, USA" many times. A single posting has
  // at most a handful of locations, listed together rather than interleaved
  // with distinct titles.
  const locationRuns = (text.match(/,\s+[A-Z]{2},?\s+USA\b/g) ?? []).length;
  if (locationRuns >= 8) return true;

  return false;
}

/** Heuristic for a page that returned a wall instead of a posting. */
export function looksLikeLoginWall(text: string): boolean {
  if (text.length > 2_000) return false;
  return /sign in|log in|create an account|enable javascript|are you a robot|verify you are human/i.test(
    text,
  );
}

/**
 * Workday's public job JSON, derived from a human job URL.
 *
 * Workday powers a large share of enterprise hiring, Miami University's own
 * careers site among them, and renders every page with JavaScript, so a plain
 * fetch yields nothing. But each tenant exposes an undocumented-but-stable JSON
 * endpoint that returns the full posting, so a human URL of the form
 *   https://TENANT.wdN.myworkdayjobs.com/en-US/SITE/job/PATH
 * maps to
 *   https://TENANT.wdN.myworkdayjobs.com/wday/cxs/TENANT/SITE/job/PATH
 * Same host, so it inherits the SSRF host check.
 */
/**
 * Ashby's public job board, derived from a human job URL.
 *
 * Ashby powers many AI and tech companies. Its single-job endpoint needs auth,
 * but the board endpoint is public and returns every posting, so the specific
 * job is found within it by id:
 *   https://jobs.ashbyhq.com/{org}/{jid}
 *   https://jobs.ashbyhq.com/{org}?ashby_jid={jid}
 *   https://www.ashbyhq.com/careers?ashby_jid={jid}   (Ashby's own board, org "ashby")
 * all read
 *   https://api.ashbyhq.com/posting-api/job-board/{org}?includeCompensation=true
 *
 * Company-embedded boards on a firm's own domain (for example openai.com/careers)
 * do not reveal the org slug in the URL, so those are not mappable here and fall
 * through to the ordinary path.
 */
export function ashbyBoard(rawUrl: string): { org: string; jid: string } | null {
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    return null;
  }
  const host = url.hostname.toLowerCase();
  if (!host.endsWith("ashbyhq.com")) return null;

  // The job id is the ashby_jid query parameter, or the last path segment.
  const queryJid = url.searchParams.get("ashby_jid");
  const pathParts = url.pathname.split("/").filter(Boolean);

  let org: string | null = null;
  let jid: string | null = queryJid;

  if (host === "jobs.ashbyhq.com") {
    // jobs.ashbyhq.com/{org}[/{jid}]
    org = pathParts[0] ?? null;
    jid = jid ?? pathParts[1] ?? null;
  } else if (host === "www.ashbyhq.com" || host === "ashbyhq.com") {
    // The company's own board on ashbyhq.com/careers is the "ashby" org.
    org = "ashby";
  } else {
    // {org}.ashbyhq.com
    org = host.replace(/\.ashbyhq\.com$/, "") || null;
    jid = jid ?? pathParts[pathParts.length - 1] ?? null;
  }

  if (!org || !jid || !/^[0-9a-f-]{16,}$/i.test(jid)) return null;
  return { org, jid };
}

interface AshbyJob {
  jobId?: string;
  id?: string;
  title?: string;
  descriptionHtml?: string;
  descriptionPlain?: string;
}

async function fetchAshby(org: string, jid: string): Promise<PostingResult> {
  const api = `https://api.ashbyhq.com/posting-api/job-board/${org}?includeCompensation=true`;
  if (!(await hostIsSafe("api.ashbyhq.com"))) {
    return { outcome: "blocked_host", text: null, message: "That address cannot be read from here. Paste the posting text instead." };
  }
  let data: { jobs?: AshbyJob[] };
  try {
    const res = await fetch(api, {
      headers: { accept: "application/json", "user-agent": "ChatISA/1.0 (Miami University; educational use)" },
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    });
    if (!res.ok) {
      return { outcome: "unreachable", text: null, message: "That posting could not be read. Paste the description below." };
    }
    data = (await res.json()) as { jobs?: AshbyJob[] };
  } catch {
    return { outcome: "unreachable", text: null, message: "That posting could not be opened. Paste the description below." };
  }

  const job = (data.jobs ?? []).find((j) => j.jobId === jid || j.id === jid);
  if (!job) {
    // The board loaded but this job is not in it: filled, or the wrong org.
    return { outcome: "not_a_posting", text: null, message: "That job is not on the board any more. Paste the description below if you still have it." };
  }

  const body = htmlToText(job.descriptionHtml ?? job.descriptionPlain ?? "");
  if (body.trim() === "") {
    return { outcome: "empty", text: null, message: "That posting had no description we could read. Paste it below." };
  }
  const title = job.title ?? "";
  return {
    outcome: "fetched",
    text: title ? `${title}\n\n${body}` : body,
    message: "We read the posting below. Check it, and edit or replace anything that came out wrong.",
  };
}

/**
 * Lever's public postings API, derived from a human job URL.
 *   https://jobs.lever.co/{company}/{id}
 * maps to
 *   https://api.lever.co/v0/postings/{company}/{id}?mode=json
 * Lever splits a posting across many fields (opening, description,
 * descriptionBody, lists, additional), and some employers put the whole thing
 * in an unexpected one, so all are gathered.
 */
export function leverApiUrl(rawUrl: string): string | null {
  const match = rawUrl.match(
    /^https:\/\/jobs\.lever\.co\/([^/]+)\/([0-9a-f-]{16,})/i,
  );
  if (!match) return null;
  const [, company, id] = match;
  return `https://api.lever.co/v0/postings/${company}/${id}?mode=json`;
}

interface LeverJob {
  text?: string;
  opening?: string;
  description?: string;
  descriptionBody?: string;
  additional?: string;
  salaryDescription?: string;
  lists?: { text?: string; content?: string }[];
}

async function fetchLever(apiUrl: string): Promise<PostingResult> {
  if (!(await hostIsSafe(new URL(apiUrl).hostname))) {
    return { outcome: "blocked_host", text: null, message: "That address cannot be read from here. Paste the posting text instead." };
  }
  let data: LeverJob;
  try {
    const res = await fetch(apiUrl, {
      headers: { accept: "application/json", "user-agent": "ChatISA/1.0 (Miami University; educational use)" },
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    });
    if (!res.ok) {
      return { outcome: "unreachable", text: null, message: "That posting could not be read. Paste the description below." };
    }
    data = (await res.json()) as LeverJob;
  } catch {
    return { outcome: "unreachable", text: null, message: "That posting could not be opened. Paste the description below." };
  }

  const parts = [
    data.opening,
    data.description,
    data.descriptionBody,
    ...(data.lists ?? []).map((l) =>
      [l.text, l.content].filter(Boolean).join(" "),
    ),
    data.additional,
    data.salaryDescription,
  ]
    .map((field) => htmlToText(field ?? ""))
    .filter((part) => part.trim() !== "");

  if (parts.length === 0) {
    return { outcome: "empty", text: null, message: "That posting had no description we could read. Paste it below." };
  }
  const title = data.text ?? "";
  const body = parts.join("\n\n");
  return {
    outcome: "fetched",
    text: title ? `${title}\n\n${body}` : body,
    message: "We read the posting below. Check it, and edit or replace anything that came out wrong.",
  };
}

/**
 * SmartRecruiters' public postings API, derived from a human job URL.
 *   https://jobs.smartrecruiters.com/{company}/{postingId}-{slug}
 * maps to
 *   https://api.smartrecruiters.com/v1/companies/{company}/postings/{postingId}
 * The posting body lives in jobAd.sections, one section per part of the ad.
 */
export function smartRecruitersApiUrl(rawUrl: string): string | null {
  const match = rawUrl.match(
    /^https:\/\/jobs\.smartrecruiters\.com\/([^/]+)\/(\d+)/i,
  );
  if (!match) return null;
  const [, company, postingId] = match;
  return `https://api.smartrecruiters.com/v1/companies/${company}/postings/${postingId}`;
}

interface SmartRecruitersJob {
  name?: string;
  jobAd?: { sections?: Record<string, { text?: string } | undefined> };
}

async function fetchSmartRecruiters(apiUrl: string): Promise<PostingResult> {
  if (!(await hostIsSafe(new URL(apiUrl).hostname))) {
    return { outcome: "blocked_host", text: null, message: "That address cannot be read from here. Paste the posting text instead." };
  }
  let data: SmartRecruitersJob;
  try {
    const res = await fetch(apiUrl, {
      headers: { accept: "application/json", "user-agent": "ChatISA/1.0 (Miami University; educational use)" },
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    });
    if (!res.ok) {
      return { outcome: "unreachable", text: null, message: "That posting could not be read. Paste the description below." };
    }
    data = (await res.json()) as SmartRecruitersJob;
  } catch {
    return { outcome: "unreachable", text: null, message: "That posting could not be opened. Paste the description below." };
  }

  // Ordered so the role and its requirements lead, with company blurb after.
  const order = [
    "jobDescription",
    "qualifications",
    "additionalInformation",
    "companyDescription",
  ];
  const sections = data.jobAd?.sections ?? {};
  const parts = order
    .map((key) => htmlToText(sections[key]?.text ?? ""))
    .filter((part) => part.trim() !== "");

  if (parts.length === 0) {
    return { outcome: "empty", text: null, message: "That posting had no description we could read. Paste it below." };
  }
  const title = data.name ?? "";
  const body = parts.join("\n\n");
  return {
    outcome: "fetched",
    text: title ? `${title}\n\n${body}` : body,
    message: "We read the posting below. Check it, and edit or replace anything that came out wrong.",
  };
}

/**
 * Greenhouse's public board API, derived from a human job URL.
 *
 * Greenhouse is one of the most common ATS for startups and tech companies
 * (Anthropic among them), and its job pages render with JavaScript. But it has
 * a clean public API that returns the posting as JSON:
 *   https://boards.greenhouse.io/{board}/jobs/{id}
 *   https://job-boards.greenhouse.io/{board}/jobs/{id}
 * both map to
 *   https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{id}?content=true
 * The API host is a fixed public Greenhouse host, so it passes the SSRF check.
 */
export function greenhouseApiUrl(rawUrl: string): string | null {
  const match = rawUrl.match(
    /^https:\/\/(?:job-boards|boards)\.greenhouse\.io\/([^/]+)\/jobs\/(\d+)/i,
  );
  if (!match) return null;
  const [, board, id] = match;
  return `https://boards-api.greenhouse.io/v1/boards/${board}/jobs/${id}?content=true`;
}

async function fetchGreenhouse(apiUrl: string): Promise<PostingResult> {
  if (!(await hostIsSafe(new URL(apiUrl).hostname))) {
    return {
      outcome: "blocked_host",
      text: null,
      message: "That address cannot be read from here. Paste the posting text instead.",
    };
  }
  let response: Response;
  try {
    response = await fetch(apiUrl, {
      headers: { accept: "application/json", "user-agent": "ChatISA/1.0 (Miami University; educational use)" },
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    });
  } catch {
    return { outcome: "unreachable", text: null, message: "That posting could not be opened. Paste the description below." };
  }
  if (!response.ok) {
    return { outcome: "unreachable", text: null, message: "That posting could not be read. Paste the description below." };
  }
  let data: { title?: string; content?: string };
  try {
    data = (await response.json()) as { title?: string; content?: string };
  } catch {
    return { outcome: "not_a_posting", text: null, message: "That link did not return a single job. Paste the description below." };
  }
  const content = data.content ?? "";
  if (content.trim() === "") {
    return { outcome: "empty", text: null, message: "That posting had no description we could read. Paste it below." };
  }
  const body = htmlToText(content);
  const title = data.title ?? "";
  return {
    outcome: "fetched",
    text: title ? `${title}\n\n${body}` : body,
    message: "We read the posting below. Check it, and edit or replace anything that came out wrong.",
  };
}

export function workdayCxsUrl(rawUrl: string): string | null {
  const match = rawUrl.match(
    /^https:\/\/([^.]+)\.(wd\d+)\.myworkdayjobs\.com\/[a-z-]+\/([^/]+)\/job\/(.+?)(?:\?|$)/i,
  );
  if (!match) return null;
  const [, tenant, wd, site, path] = match;
  return `https://${tenant}.${wd}.myworkdayjobs.com/wday/cxs/${tenant}/${site}/job/${path}`;
}

interface WorkdayJob {
  jobPostingInfo?: {
    title?: string;
    jobDescription?: string;
    location?: string;
  };
}

async function fetchWorkday(cxsUrl: string): Promise<PostingResult> {
  const host = new URL(cxsUrl).hostname;
  if (!(await hostIsSafe(host))) {
    return {
      outcome: "blocked_host",
      text: null,
      message: "That address cannot be read from here. Paste the posting text instead.",
    };
  }

  let response: Response;
  try {
    response = await fetch(cxsUrl, {
      headers: {
        accept: "application/json",
        "user-agent": "ChatISA/1.0 (Miami University; educational use)",
      },
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    });
  } catch {
    return {
      outcome: "unreachable",
      text: null,
      message: "That Workday posting could not be opened. Copy the description and paste it below.",
    };
  }

  if (!response.ok) {
    return {
      outcome: "unreachable",
      text: null,
      message: "That Workday posting could not be read. Copy the description and paste it below.",
    };
  }

  let data: WorkdayJob;
  try {
    data = (await response.json()) as WorkdayJob;
  } catch {
    return {
      outcome: "not_a_posting",
      text: null,
      message: "That Workday link did not return a single job. Open the specific job and paste its description below.",
    };
  }

  const info = data.jobPostingInfo;
  const description = info?.jobDescription ?? "";
  if (description.trim() === "") {
    return {
      outcome: "empty",
      text: null,
      message: "That Workday posting had no description we could read. Copy it and paste it below.",
    };
  }

  const title = info?.title ?? "";
  const body = htmlToText(description);
  return {
    outcome: "fetched",
    text: title ? `${title}\n\n${body}` : body,
    message:
      "We read the posting below. Check it, and edit or replace anything that came out wrong.",
  };
}

export async function fetchJobPosting(rawUrl: string): Promise<PostingResult> {
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    return {
      outcome: "unreachable",
      text: null,
      message: "That does not look like a web address. Paste the posting text instead.",
    };
  }

  if (url.protocol !== "https:") {
    return {
      outcome: "blocked_host",
      text: null,
      message: "Only https links can be read. Paste the posting text instead.",
    };
  }

  if (isKnownBlocker(url.href)) {
    return {
      outcome: "login_required",
      text: null,
      message:
        "This job board does not allow us to read its pages. Open the posting, copy the description, and paste it below.",
    };
  }

  // Workday renders with JavaScript, so read its JSON API rather than the shell.
  const cxs = workdayCxsUrl(url.href);
  if (cxs) return fetchWorkday(cxs);

  // Greenhouse likewise has a clean public API behind a JavaScript page.
  const gh = greenhouseApiUrl(url.href);
  if (gh) return fetchGreenhouse(gh);

  const lever = leverApiUrl(url.href);
  if (lever) return fetchLever(lever);

  const smartRecruiters = smartRecruitersApiUrl(url.href);
  if (smartRecruiters) return fetchSmartRecruiters(smartRecruiters);

  const ashby = ashbyBoard(url.href);
  if (ashby) return fetchAshby(ashby.org, ashby.jid);

  let current = url;
  for (let hop = 0; hop <= MAX_REDIRECTS; hop += 1) {
    if (!(await hostIsSafe(current.hostname))) {
      logger.warn({ host: current.hostname }, "posting fetch refused: internal address");
      return {
        outcome: "blocked_host",
        text: null,
        message: "That address cannot be read from here. Paste the posting text instead.",
      };
    }

    let response: Response;
    try {
      response = await fetch(current.href, {
        redirect: "manual",
        signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
        headers: {
          // Identifies the app rather than pretending to be a browser.
          "user-agent": "ChatISA/1.0 (Miami University; educational use)",
          accept: "text/html,application/xhtml+xml",
        },
      });
    } catch {
      return {
        outcome: "unreachable",
        text: null,
        message:
          "That page could not be opened. Check the link, or paste the posting text instead.",
      };
    }

    // Redirects are followed by hand so each new host is re-checked. Following
    // them automatically would let a public URL bounce to an internal one.
    if (response.status >= 300 && response.status < 400) {
      const location = response.headers.get("location");
      if (!location) break;
      try {
        current = new URL(location, current);
      } catch {
        break;
      }
      continue;
    }

    if (!response.ok) {
      return {
        outcome: response.status === 401 || response.status === 403
          ? "login_required"
          : "unreachable",
        text: null,
        message:
          response.status === 401 || response.status === 403
            ? "That page needs a login, so we cannot read it. Copy the description and paste it below."
            : "That page could not be read. Paste the posting text instead.",
      };
    }

    const contentType = response.headers.get("content-type") ?? "";
    if (!contentType.includes("html") && !contentType.includes("text")) {
      return {
        outcome: "not_html",
        text: null,
        message: "That link is not a web page we can read. Paste the posting text instead.",
      };
    }

    const declared = Number(response.headers.get("content-length") ?? 0);
    if (declared > MAX_POSTING_BYTES) {
      return {
        outcome: "too_large",
        text: null,
        message: "That page is too large to read. Paste the posting text instead.",
      };
    }

    const body = await response.text();
    if (body.length > MAX_POSTING_BYTES) {
      return {
        outcome: "too_large",
        text: null,
        message: "That page is too large to read. Paste the posting text instead.",
      };
    }

    // Prefer the structured JobPosting data. On a JavaScript-rendered site it
    // is the only place the real description exists; on a static one it is
    // cleaner than the surrounding page anyway.
    const structured = extractJsonLdPosting(body);
    const text = structured ?? htmlToText(body);

    if (text.length < 200) {
      return {
        outcome: "empty",
        text: null,
        message:
          "We opened that page but found almost no text on it, which usually means the posting is loaded by scripts. Copy the description and paste it below.",
      };
    }
    if (looksLikeLoginWall(text)) {
      return {
        outcome: "login_required",
        text: null,
        message:
          "That page asked us to sign in rather than showing the posting. Copy the description and paste it below.",
      };
    }
    if (!structured && looksLikeListing(text)) {
      // A results grid of many jobs, not one description. Tailoring to this
      // would emphasise a list of unrelated roles.
      return {
        outcome: "not_a_posting",
        text: null,
        message:
          "That link opened a list of jobs rather than one description. Open the specific job you want, copy its description, and paste it below.",
      };
    }
    if (!structured && !looksLikePosting(text)) {
      // We got a page, but it reads like a careers home page or a search
      // result rather than a job description. Handing this over would tailor
      // the student's resume to navigation text without either of us noticing.
      return {
        outcome: "not_a_posting",
        text: null,
        message:
          "That link opened a careers or search page rather than a single job description. Open the specific job, copy its description, and paste it below.",
      };
    }

    return {
      outcome: "fetched",
      text,
      message:
        "We read the posting below. Check it, and edit or replace anything that came out wrong.",
    };
  }

  return {
    outcome: "unreachable",
    text: null,
    message: "That link redirected too many times. Paste the posting text instead.",
  };
}
