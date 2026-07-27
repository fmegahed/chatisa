import { afterEach, describe, expect, it } from "vitest";
import {
  PAPER_OUTPUT_MAX,
  abstractFromInvertedIndex,
  distill,
  getPaper,
  mergePapers,
  normalizeArxiv,
  normalizeArxivId,
  normalizeDoi,
  normalizeOpenAlex,
  normalizeSemanticScholar,
  normalizeWikipedia,
  paperKey,
  rankPapers,
  searchPapers,
  type PaperRecord,
} from "@/lib/ask/paper-search";
import { resetRateLimits } from "@/lib/ratelimit";

afterEach(() => {
  resetRateLimits();
  delete process.env.CHATISA_MOCK_LLM;
});

// Trimmed recordings of each source's real response shape.
const ARXIV_ATOM = `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2501.01234v1</id>
    <published>2025-01-03T18:00:00Z</published>
    <title>Conformal Prediction for Time Series</title>
    <summary>  We study distribution-free uncertainty
      quantification &amp; coverage.  </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2404.09999v2</id>
    <published>2024-04-20T10:00:00Z</published>
    <title>Another Approach</title>
    <summary>Short abstract.</summary>
    <author><name>Grace Hopper</name></author>
  </entry>
</feed>`;

const S2_JSON = {
  total: 2,
  data: [
    {
      title: "Conformal Prediction for Time Series",
      authors: [{ name: "Ada Lovelace" }, { name: "Alan Turing" }],
      year: 2025,
      venue: "NeurIPS",
      citationCount: 214,
      externalIds: { DOI: "10.1000/CPTS.2025", ArXiv: "2501.01234" },
      url: "https://www.semanticscholar.org/paper/abc",
      abstract: "We study distribution-free uncertainty quantification.",
    },
    {
      title: "A Third Paper",
      authors: [{ name: "Katherine Johnson" }],
      year: 2020,
      venue: "JASA",
      citationCount: 1200,
      externalIds: {},
      url: "https://www.semanticscholar.org/paper/def",
      abstract: null,
    },
  ],
};

const OPENALEX_JSON = {
  results: [
    {
      title: "Conformal Prediction for Time Series",
      authorships: [{ author: { display_name: "Ada Lovelace" } }],
      publication_year: 2025,
      primary_location: { source: { display_name: "NeurIPS" } },
      cited_by_count: 190,
      doi: "https://doi.org/10.1000/cpts.2025",
      id: "https://openalex.org/W1",
      abstract_inverted_index: { Distribution: [0], free: [1], coverage: [2] },
    },
  ],
};

const WIKI_JSON = {
  pages: [
    {
      title: "Conformal prediction",
      key: "Conformal_prediction",
      excerpt: "<span class=\"searchmatch\">Conformal</span> prediction is a framework...",
    },
  ],
};

describe("source normalizers", () => {
  it("parses arXiv Atom entries", () => {
    const papers = normalizeArxiv(ARXIV_ATOM);
    expect(papers).toHaveLength(2);
    expect(papers[0]).toMatchObject({
      title: "Conformal Prediction for Time Series",
      authors: ["Ada Lovelace", "Alan Turing"],
      year: 2025,
      venue: "arXiv",
      url: "http://arxiv.org/abs/2501.01234v1",
      source: "arxiv",
    });
    expect(papers[0].abstract).toContain("uncertainty quantification & coverage");
  });

  it("parses Semantic Scholar results with lowercased DOIs", () => {
    const papers = normalizeSemanticScholar(S2_JSON);
    expect(papers[0].doi).toBe("10.1000/cpts.2025");
    expect(papers[0].citations).toBe(214);
    expect(papers[1].abstract).toBeNull();
  });

  it("parses OpenAlex and rebuilds the inverted-index abstract", () => {
    const papers = normalizeOpenAlex(OPENALEX_JSON);
    expect(papers[0].doi).toBe("10.1000/cpts.2025");
    expect(papers[0].abstract).toBe("Distribution free coverage");
    expect(
      abstractFromInvertedIndex({ b: [1], a: [0], c: [2] }),
    ).toBe("a b c");
  });

  it("parses Wikipedia search pages and strips markup", () => {
    const background = normalizeWikipedia(WIKI_JSON);
    expect(background[0].title).toBe("Conformal prediction");
    expect(background[0].extract).toBe("Conformal prediction is a framework...");
    expect(background[0].url).toContain("/wiki/Conformal_prediction");
  });
});

describe("merge, rank, distill", () => {
  it("dedupes by DOI then title and marks corroboration", () => {
    const merged = mergePapers([
      normalizeArxiv(ARXIV_ATOM),
      normalizeSemanticScholar(S2_JSON),
      normalizeOpenAlex(OPENALEX_JSON),
    ]);
    // The arXiv entry has no DOI, so it dedupes with S2/OpenAlex by title.
    const main = merged.find((p) => p.title.startsWith("Conformal"));
    expect(main?.source).toBe("arxiv+semanticscholar+openalex");
    expect(main?.citations).toBe(214);
    expect(main?.doi).toBe("10.1000/cpts.2025");
    expect(merged).toHaveLength(3); // main + Another Approach + A Third Paper
  });

  it("keys on DOI when present, title otherwise", () => {
    const withDoi = { doi: "10.1/x", title: "T" } as PaperRecord;
    const noDoi = { doi: null, title: "The  Same: Title!" } as PaperRecord;
    expect(paperKey(withDoi)).toBe("doi:10.1/x");
    expect(paperKey(noDoi)).toBe("title:thesametitle");
  });

  it("keeps each source's own relevance order ahead of citation count", () => {
    // The 2026-07-25 report: searching "ChatISA" put the paper the student
    // named (returned FIRST by arXiv and OpenAlex, 0 citations) below unrelated
    // full-text matches with a handful of citations, because the ranking had no
    // relevance term at all.
    const papers: PaperRecord[] = [
      {
        title: "Unrelated but cited",
        authors: [],
        year: 2023,
        venue: null,
        citations: 7,
        doi: null,
        url: null,
        source: "openalex",
        positions: [2],
        abstract: null,
      },
      {
        title: "ChatISA: the paper the student named",
        authors: [],
        year: 2024,
        venue: "arXiv",
        citations: 0,
        doi: null,
        url: null,
        source: "arxiv+openalex",
        positions: [0, 0],
        abstract: null,
      },
    ];
    expect(rankPapers(papers, 2026)[0].title).toContain("ChatISA");
  });

  it("ranks recent corroborated work above old uncited work", () => {
    const papers: PaperRecord[] = [
      { title: "Old uncited", authors: [], year: 2005, venue: null, citations: 0, doi: null, url: null, source: "arxiv", abstract: null },
      { title: "Recent corroborated", authors: [], year: 2025, venue: null, citations: 50, doi: null, url: null, source: "arxiv+openalex", abstract: null },
    ];
    expect(rankPapers(papers, 2026)[0].title).toBe("Recent corroborated");
  });

  it("distills to the output budget", () => {
    const papers: PaperRecord[] = Array.from({ length: 10 }, (_, i) => ({
      title: `Paper ${i}`,
      authors: Array.from({ length: 12 }, (_, j) => `Author ${j}`),
      year: 2024,
      venue: "Venue",
      citations: i,
      doi: `10.0/${i}`,
      url: `https://doi.org/10.0/${i}`,
      source: "openalex",
      abstract: "long ".repeat(1000),
    }));
    const out = distill({ papers, background: [], unavailable: [] }, 10);
    expect(JSON.stringify(out).length).toBeLessThanOrEqual(PAPER_OUTPUT_MAX);
    expect(out.papers[0].authors).toContain("and 6 more");
  });

  it("keeps internal ranking positions out of what the model sees", () => {
    const out = distill(
      {
        papers: [
          {
            title: "P",
            authors: [],
            year: 2025,
            venue: null,
            citations: 1,
            doi: null,
            url: null,
            source: "arxiv",
            positions: [0],
            abstract: null,
          },
        ],
        background: [],
        unavailable: [],
      },
      5,
    );
    expect(out.papers[0]).not.toHaveProperty("positions");
    expect(JSON.stringify(out)).not.toContain("positions");
  });
});

describe("identifier normalization", () => {
  it("strips version suffixes, prefixes, and URL wrappers from arXiv ids", () => {
    expect(normalizeArxivId("2407.15010")).toBe("2407.15010");
    // The version suffix is what search results actually carry, and it is not
    // a valid Semantic Scholar arXiv id.
    expect(normalizeArxivId("2407.15010v2")).toBe("2407.15010");
    expect(normalizeArxivId("arXiv:2407.15010v2")).toBe("2407.15010");
    expect(normalizeArxivId("http://arxiv.org/abs/2407.15010v2")).toBe(
      "2407.15010",
    );
    expect(normalizeArxivId("https://arxiv.org/pdf/2407.15010v2.pdf")).toBe(
      "2407.15010",
    );
    expect(normalizeArxivId("math.GT/0309136")).toBe("math.GT/0309136");
    expect(normalizeArxivId("not an id")).toBeNull();
    expect(normalizeArxivId("../../etc/passwd")).toBeNull();
  });

  it("strips resolver prefixes from DOIs and rejects non-DOIs", () => {
    expect(normalizeDoi("10.1000/CPTS.2025")).toBe("10.1000/cpts.2025");
    expect(normalizeDoi("https://doi.org/10.1000/cpts.2025")).toBe(
      "10.1000/cpts.2025",
    );
    expect(normalizeDoi("doi:10.1000/cpts.2025")).toBe("10.1000/cpts.2025");
    expect(normalizeDoi("2407.15010")).toBeNull();
    expect(normalizeDoi("10.1000/has space")).toBeNull();
  });
});

describe("getPaper", () => {
  /** A fetcher that answers per host from a status/body table and records the
   * URLs it was asked for. */
  function stubFetcher(
    routes: { match: string; status: number; body: string }[],
  ) {
    const seen: string[] = [];
    const fetcher = (async (url: RequestInfo | URL) => {
      const href = String(url);
      seen.push(href);
      const route = routes.find((r) => href.includes(r.match));
      if (!route) throw new Error(`unexpected url ${href}`);
      return new Response(route.body, { status: route.status });
    }) as typeof fetch;
    return { fetcher, seen };
  }

  const OPENALEX_WORK = JSON.stringify({
    title: "Conformal Prediction for Time Series",
    authorships: [{ author: { display_name: "Ada Lovelace" } }],
    publication_year: 2025,
    primary_location: { source: { display_name: "NeurIPS" } },
    cited_by_count: 190,
    doi: "https://doi.org/10.1000/cpts.2025",
    id: "https://openalex.org/W1",
    abstract_inverted_index: { Distribution: [0], free: [1] },
  });

  it("falls back to arXiv when Semantic Scholar is rate-limited", async () => {
    // The keyless Semantic Scholar pool answers 429 to nearly every request,
    // which used to break get_paper outright (it had no second source).
    const { fetcher } = stubFetcher([
      { match: "semanticscholar", status: 429, body: "Too Many Requests" },
      { match: "export.arxiv.org", status: 200, body: ARXIV_ATOM },
    ]);
    const detail = await getPaper({ arxivId: "2501.01234" }, { fetcher });
    expect(detail).toMatchObject({
      title: "Conformal Prediction for Time Series",
      source: "arxiv",
    });
  });

  it("falls back to OpenAlex for a DOI when the others fail", async () => {
    const { fetcher } = stubFetcher([
      { match: "semanticscholar", status: 500, body: "boom" },
      { match: "openalex", status: 200, body: OPENALEX_WORK },
    ]);
    const detail = await getPaper({ doi: "10.1000/cpts.2025" }, { fetcher });
    expect(detail).toMatchObject({
      title: "Conformal Prediction for Time Series",
      source: "openalex",
    });
  });

  it("normalizes a versioned arXiv id before querying anything", async () => {
    const { fetcher, seen } = stubFetcher([
      { match: "semanticscholar", status: 429, body: "429" },
      { match: "export.arxiv.org", status: 200, body: ARXIV_ATOM },
      { match: "openalex", status: 404, body: "not found" },
    ]);
    await getPaper({ arxivId: "https://arxiv.org/abs/2501.01234v1" }, { fetcher });
    expect(seen.length).toBeGreaterThan(0);
    for (const url of seen) {
      expect(url).toContain("2501.01234");
      // The version suffix must be gone from the id (the S2 path legitimately
      // contains "graph/v1", so match the id itself).
      expect(url).not.toMatch(/2501\.01234v\d/);
    }
  });

  it("says the paper is not on record when every source 404s", async () => {
    const { fetcher } = stubFetcher([
      { match: "semanticscholar", status: 404, body: "not found" },
      { match: "openalex", status: 404, body: "not found" },
    ]);
    const detail = await getPaper({ doi: "10.9999/nope" }, { fetcher });
    expect(detail).toHaveProperty("error");
    expect((detail as { error: string }).error).toMatch(/no record/i);
  });

  it("says the databases are unreachable when every source errors", async () => {
    const { fetcher } = stubFetcher([
      { match: "semanticscholar", status: 503, body: "down" },
      { match: "openalex", status: 503, body: "down" },
    ]);
    const detail = await getPaper({ doi: "10.1000/cpts.2025" }, { fetcher });
    expect((detail as { error: string }).error).toMatch(/could not be reached/i);
  });

  it("rejects an id that is neither a DOI nor an arXiv id", async () => {
    const detail = await getPaper(
      { arxivId: "the conformal prediction one" },
      {
        fetcher: (() => {
          throw new Error("must not fetch on an unusable id");
        }) as unknown as typeof fetch,
      },
    );
    expect((detail as { error: string }).error).toMatch(/DOI|arXiv id/);
  });
});

describe("searchPapers", () => {
  it("fans out, merges, and reports failed sources", async () => {
    const fetcher = (async (url: RequestInfo | URL) => {
      const href = String(url);
      if (href.includes("export.arxiv.org")) {
        return new Response(ARXIV_ATOM, { status: 200 });
      }
      if (href.includes("semanticscholar")) {
        return new Response("upstream down", { status: 500 });
      }
      if (href.includes("openalex")) {
        return new Response(JSON.stringify(OPENALEX_JSON), { status: 200 });
      }
      if (href.includes("wikipedia")) {
        return new Response(JSON.stringify(WIKI_JSON), { status: 200 });
      }
      throw new Error(`unexpected url ${href}`);
    }) as typeof fetch;

    const result = await searchPapers("conformal prediction", 5, {
      fetcher,
      nowYear: 2026,
    });
    expect(result.unavailable).toEqual(["Semantic Scholar"]);
    expect(result.background[0].title).toBe("Conformal prediction");
    const main = result.papers.find((p) => p.title.startsWith("Conformal"));
    expect(main?.source).toBe("arxiv+openalex");
  });

  it("returns deterministic fixtures in mock mode", async () => {
    process.env.CHATISA_MOCK_LLM = "1";
    const result = await searchPapers("anything", 5, {
      fetcher: (() => {
        throw new Error("mock mode must not fetch");
      }) as unknown as typeof fetch,
    });
    expect(result.papers[0].abstract).toContain("MOCK_ABSTRACT");
  });
});
