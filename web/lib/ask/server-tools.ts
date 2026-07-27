import "server-only";
import { tool } from "ai";
import { z } from "zod";
import { askToolDefs } from "@/lib/ask/tools";
import { getPaper, searchPapers } from "@/lib/ask/paper-search";
import { readUrl } from "@/lib/ask/read-url";
import { MIAMI_STYLE_KINDS, getMiamiStyle } from "@/lib/ask/miami-style";

/**
 * The full Ask Anything tool set (slice C): the browser-executed code tools
 * from askToolDefs (no execute; they run on the student's WASM runtimes) plus
 * the server-executed research and style tools below, which run inside the
 * streaming request. The route hands this combined set to BOTH streamText and
 * convertToModelMessages, so history containing any tool's parts converts.
 */
export function askServerToolDefs() {
  return {
    ...askToolDefs(),
    search_papers: tool({
      description:
        "Search the academic literature: arXiv, Semantic Scholar, and OpenAlex in parallel, with encyclopedia background. Results are deduplicated across databases (a source field like 'arxiv+openalex' means corroborated), ranked by citations and recency, and include abstracts and links. Use for papers, methods, authors, and research topics. Cite the url of every paper you draw on.",
      inputSchema: z.object({
        query: z
          .string()
          .min(2)
          .max(300)
          .describe("Topic, method, title, or author keywords."),
        limit: z.number().int().min(1).max(10).optional()
          .describe("Papers to return (default 8)."),
      }),
      execute: async ({ query, limit }) => searchPapers(query, limit ?? 8),
    }),
    get_paper: tool({
      description:
        "One paper in depth by DOI or arXiv id: full abstract, a one-line tldr when available, citation and reference counts. Use after search_papers when the student wants detail on a specific paper.",
      inputSchema: z.object({
        doi: z.string().max(200).optional(),
        arxivId: z.string().max(50).optional().describe("For example 2501.01234"),
      }),
      execute: async ({ doi, arxivId }) => getPaper({ doi, arxivId }),
    }),
    read_url: tool({
      description:
        "Read one public https web page and return its text (cleaned, capped). Use when the student pastes a link or a search result needs opening. Cannot read pages behind logins, script-only pages, or PDFs (ask the student to attach PDFs).",
      inputSchema: z.object({
        url: z.string().min(8).max(2000),
      }),
      execute: async ({ url }) => {
        if (process.env.CHATISA_MOCK_LLM === "1") {
          return { url, text: `MOCK_PAGE_TEXT for ${url}` };
        }
        return readUrl(url);
      },
    }),
    get_miami_style: tool({
      description:
        `Miami University's visual style assets. Call this BEFORE producing Miami-themed output. Kinds: ${MIAMI_STYLE_KINDS.join(", ")}. "tikz" returns the house TikZ preamble and vocabulary for figures, "gantt" a styled timeline exemplar, "colors" the palette and conventions, "latex-doc" report styling. Build on what it returns; keep the palette exact.`,
      inputSchema: z.object({
        kind: z.enum(MIAMI_STYLE_KINDS),
      }),
      execute: async ({ kind }) => getMiamiStyle(kind),
    }),
  };
}
