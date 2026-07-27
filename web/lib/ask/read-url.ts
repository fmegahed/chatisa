import "server-only";
import {
  FETCH_TIMEOUT_MS,
  MAX_POSTING_BYTES,
  hostIsSafe,
  htmlToText,
  isKnownBlocker,
} from "@/lib/jobs/fetch-posting";

/**
 * The read_url tool's fetcher (slice C): a general page reader with the same
 * server-side request forgery discipline as the JobApp posting fetcher it
 * borrows from (https only, every host resolved and checked against private
 * ranges, redirects followed by hand and re-checked), minus that module's
 * job-posting heuristics. The model uses this to open a page the student
 * pasted or a link found by search; output is cleaned text, capped.
 */

export const READ_URL_TEXT_MAX = 8_000;
const MAX_REDIRECTS = 3;

export interface ReadUrlResult {
  url: string;
  text?: string;
  truncated?: boolean;
  error?: string;
}

export async function readUrl(rawUrl: string): Promise<ReadUrlResult> {
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    return { url: rawUrl, error: "That is not a valid web address." };
  }
  if (url.protocol !== "https:") {
    return { url: rawUrl, error: "Only https pages can be read." };
  }
  if (isKnownBlocker(url.href)) {
    return {
      url: rawUrl,
      error:
        "That site does not allow automated reading. Ask the student to paste the content.",
    };
  }

  let current = url;
  for (let hop = 0; hop <= MAX_REDIRECTS; hop += 1) {
    if (!(await hostIsSafe(current.hostname))) {
      return { url: rawUrl, error: "That address cannot be read from here." };
    }
    let response: Response;
    try {
      response = await fetch(current.href, {
        redirect: "manual",
        signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
        headers: {
          "user-agent": "ChatISA/1.0 (Miami University; educational use)",
          accept: "text/html,application/xhtml+xml,text/plain",
        },
      });
    } catch {
      return { url: rawUrl, error: "The page could not be opened." };
    }

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
        url: rawUrl,
        error:
          response.status === 401 || response.status === 403
            ? "The page requires a login."
            : `The page returned an error (HTTP ${response.status}).`,
      };
    }
    const contentType = response.headers.get("content-type") ?? "";
    if (!contentType.includes("html") && !contentType.includes("text")) {
      return {
        url: rawUrl,
        error:
          "That link is not a readable page. If it is a PDF, ask the student to attach it.",
      };
    }
    const body = await response.text();
    if (body.length > MAX_POSTING_BYTES) {
      return { url: rawUrl, error: "The page is too large to read." };
    }
    const text = htmlToText(body);
    if (text.trim().length < 80) {
      return {
        url: rawUrl,
        error:
          "The page had almost no readable text (it is probably rendered by scripts).",
      };
    }
    if (text.length > READ_URL_TEXT_MAX) {
      return {
        url: current.href,
        text: `${text.slice(0, READ_URL_TEXT_MAX)}\n[page truncated]`,
        truncated: true,
      };
    }
    return { url: current.href, text };
  }
  return { url: rawUrl, error: "The page redirected too many times." };
}
