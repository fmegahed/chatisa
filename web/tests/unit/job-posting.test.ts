import { describe, expect, it } from "vitest";
import {
  extractJsonLdPosting,
  htmlToText,
  looksLikeListing,
  looksLikePosting,
  workdayCxsUrl,
  greenhouseApiUrl,
  leverApiUrl,
  smartRecruitersApiUrl,
  ashbyBoard,
  isBlockedAddress,
  isKnownBlocker,
  looksLikeLoginWall,
  fetchJobPosting,
} from "@/lib/jobs/fetch-posting";

describe("address safety", () => {
  it("refuses every internal IPv4 range", () => {
    // A server-side fetch of a user-supplied URL can otherwise be pointed at
    // anything the server can reach but the internet cannot.
    for (const address of [
      "127.0.0.1",
      "10.0.0.5",
      "172.16.0.1",
      "172.31.255.255",
      "192.168.1.1",
      "169.254.169.254", // cloud instance metadata, the classic target
      "0.0.0.0",
      "100.64.0.1",
      "224.0.0.1",
    ]) {
      expect(isBlockedAddress(address), address).toBe(true);
    }
  });

  it("refuses internal IPv6, including IPv4-mapped forms", () => {
    for (const address of [
      "::1",
      "fd00::1",
      "fe80::1",
      "::ffff:127.0.0.1",
      "::ffff:169.254.169.254",
    ]) {
      expect(isBlockedAddress(address), address).toBe(true);
    }
  });

  it("allows ordinary public addresses", () => {
    for (const address of ["93.184.216.34", "8.8.8.8", "2606:2800:220:1::"]) {
      expect(isBlockedAddress(address), address).toBe(false);
    }
  });

  it("refuses anything that is not an IP address", () => {
    expect(isBlockedAddress("not-an-ip")).toBe(true);
    expect(isBlockedAddress("")).toBe(true);
  });
});

describe("boards that refuse automated access", () => {
  it("recognises the big job boards, including subdomains", () => {
    for (const url of [
      "https://www.linkedin.com/jobs/view/123",
      "https://indeed.com/viewjob?jk=1",
      "https://www.glassdoor.com/job-listing/x",
    ]) {
      expect(isKnownBlocker(url), url).toBe(true);
    }
  });

  it("does not flag ordinary company career pages", () => {
    for (const url of [
      "https://boards.greenhouse.io/acme/jobs/1",
      "https://jobs.lever.co/acme/1",
      "https://careers.miamioh.edu/job/1",
    ]) {
      expect(isKnownBlocker(url), url).toBe(false);
    }
  });

  it("tells the student what to do instead of failing silently", async () => {
    const result = await fetchJobPosting("https://www.linkedin.com/jobs/view/1");
    expect(result.outcome).toBe("login_required");
    expect(result.text).toBeNull();
    expect(result.message).toMatch(/paste/i);
  });
});

describe("request guards", () => {
  it("refuses anything that is not https", async () => {
    for (const url of [
      "http://example.com/job",
      "file:///etc/passwd",
      "ftp://example.com/job",
    ]) {
      const result = await fetchJobPosting(url);
      expect(result.outcome, url).toBe("blocked_host");
      expect(result.text).toBeNull();
    }
  });

  it("refuses a URL that resolves to the server itself", async () => {
    const result = await fetchJobPosting("https://localhost/job");
    expect(result.outcome).toBe("blocked_host");
    expect(result.text).toBeNull();
  });

  it("handles a malformed address without throwing", async () => {
    const result = await fetchJobPosting("not a url at all");
    expect(result.outcome).toBe("unreachable");
    expect(result.message).toMatch(/paste/i);
  });

  it("never returns a raw error to the student", async () => {
    const results = await Promise.all([
      fetchJobPosting("not a url"),
      fetchJobPosting("http://example.com"),
      fetchJobPosting("https://127.0.0.1/x"),
    ]);
    for (const r of results) {
      expect(r.message).not.toMatch(/ECONN|ENOTFOUND|TypeError|undefined/i);
      expect(r.message.length).toBeGreaterThan(20);
    }
  });
});

describe("structured JobPosting data", () => {
  // Large employers (P&G, Workday, Greenhouse and similar) render the visible
  // page with JavaScript, so a plain fetch returns only a shell. The real
  // description lives in a schema.org JobPosting block for search engines.
  // Reading it is confirmed against live P&G and Deloitte postings on
  // 2026-07-21; these fixtures pin the behaviour.
  it("pulls the description out of a JobPosting block", () => {
    const html = `<html><head>
      <script type="application/ld+json">
        {"@type":"WebPage","name":"careers"}
      </script>
      <script type="application/ld+json">
        {"@type":"JobPosting","title":"Data Analyst Intern",
         "description":"You will build dashboards. Responsibilities include SQL and reporting."}
      </script>
      </head><body>Cookie information. Search jobs.</body></html>`;
    const text = extractJsonLdPosting(html);
    expect(text).toContain("Data Analyst Intern");
    expect(text).toContain("You will build dashboards");
    // Not the page furniture.
    expect(text).not.toContain("Cookie information");
  });

  it("decodes an entity-encoded description without leaving tags", () => {
    // This is exactly how P&G ships it: HTML inside the JSON is entity-encoded,
    // so the tags only become real after decoding. An earlier version decoded
    // after stripping and left "<p style=...>" in the output.
    const html = `<script type="application/ld+json">
      {"@type":"JobPosting","title":"Analyst",
       "description":"&lt;p style=\\"x\\"&gt;&lt;b&gt;Job Description&lt;/b&gt;&lt;/p&gt;&lt;p&gt;Build reports.&lt;/p&gt;"}
      </script>`;
    const text = extractJsonLdPosting(html) ?? "";
    expect(text).toContain("Job Description");
    expect(text).toContain("Build reports.");
    expect(text).not.toMatch(/<[a-z][^>]*>/i);
    expect(text).not.toContain("style=");
  });

it("reads PascalCase field names, not only lowercase", () => {
    // World Bank's Cornerstone site (csod.com) emits "Title" and "Description"
    // rather than the schema.org-conventional lowercase. JSON keys are
    // case-sensitive, so this slipped through as "empty" until 2026-07-21.
    const html = `<script type="application/ld+json">
      {"@type":"JobPosting","Title":"Program Assistant",
       "Description":"You will support the research team. Responsibilities include scheduling."}
      </script>`;
    const text = extractJsonLdPosting(html) ?? "";
    expect(text).toContain("Program Assistant");
    expect(text).toContain("support the research team");
  });

it("gathers responsibilities and qualifications, not just description", () => {
    // Meta splits its posting across separate schema.org fields, putting the
    // real substance in responsibilities and qualifications while description
    // holds only a short overview. Reading description alone dropped more than
    // half the posting (found live 2026-07-21).
    const html = `<script type="application/ld+json">
      {"@type":"JobPosting","title":"ML Scientist",
       "description":"Join our audio research team.",
       "responsibilities":"Develop novel algorithms and signal processing.",
       "qualifications":"PhD in ML or equivalent. Experience with acoustics."}
      </script>`;
    const text = extractJsonLdPosting(html) ?? "";
    expect(text).toContain("Join our audio research team");
    expect(text).toContain("Develop novel algorithms");
    expect(text).toContain("PhD in ML");
  });

  it("still works when only description is present", () => {
    // The common case: one description field, unchanged behaviour.
    const html = `<script type="application/ld+json">
      {"@type":"JobPosting","title":"Analyst","description":"Build dashboards in SQL."}
      </script>`;
    const text = extractJsonLdPosting(html) ?? "";
    expect(text).toBe("Analyst\n\nBuild dashboards in SQL.");
  });

  it("reads a JobPosting whose @type is an array", () => {
    const html = `<script type="application/ld+json">
      {"@type":["JobPosting","WebPage"],"title":"Analyst","description":"Build dashboards daily."}
      </script>`;
    expect(extractJsonLdPosting(html)).toContain("Build dashboards daily");
  });

it("finds a block whose type attribute encodes the plus sign", () => {
    // USAJobs writes type="application/ld&#x2B;json" because its nonce-based
    // CSP entity-encodes attribute values. A literal "+" match missed the
    // block and fell back to 24KB of nav-heavy page text (found live
    // 2026-07-21). Both raw and encoded forms must match.
    const encoded = `<script type="application/ld&#x2B;json">
      {"@type":"JobPosting","title":"Data Analyst","description":"Federal analytics role."}
      </script>`;
    expect(extractJsonLdPosting(encoded)).toContain("Federal analytics role");

    const raw = `<script type="application/ld+json">
      {"@type":"JobPosting","title":"Data Analyst","description":"Federal analytics role."}
      </script>`;
    expect(extractJsonLdPosting(raw)).toContain("Federal analytics role");
  });

  it("returns null when there is no JobPosting block", () => {
    expect(extractJsonLdPosting("<script type='application/ld+json'>{\"@type\":\"WebPage\"}</script>")).toBeNull();
    expect(extractJsonLdPosting("<html><body>no structured data</body></html>")).toBeNull();
  });

  it("ignores a JobPosting with an empty description", () => {
    const html = `<script type="application/ld+json">{"@type":"JobPosting","description":""}</script>`;
    expect(extractJsonLdPosting(html)).toBeNull();
  });
});

describe("Ashby", () => {
  // Ashby powers many AI/tech companies. Its single-job endpoint needs auth,
  // but the board endpoint is public. Confirmed live 2026-07-21 against Ashby's
  // own careers board (9,415 chars).
  it("reads the org and job id from Ashby's own careers URL", () => {
    expect(
      ashbyBoard("https://www.ashbyhq.com/careers?ashby_jid=1d637631-0d53-4e8c-9af7-38633bfc2723"),
    ).toEqual({ org: "ashby", jid: "1d637631-0d53-4e8c-9af7-38633bfc2723" });
  });

  it("reads org and jid from a jobs.ashbyhq.com path", () => {
    expect(ashbyBoard("https://jobs.ashbyhq.com/openai/1d637631-0d53-4e8c-9af7-38633bfc2723"))
      .toEqual({ org: "openai", jid: "1d637631-0d53-4e8c-9af7-38633bfc2723" });
    expect(
      ashbyBoard("https://jobs.ashbyhq.com/openai?ashby_jid=1d637631-0d53-4e8c-9af7-38633bfc2723"),
    ).toEqual({ org: "openai", jid: "1d637631-0d53-4e8c-9af7-38633bfc2723" });
  });

  it("returns null when there is no job id, or it is not Ashby", () => {
    // A company embedding Ashby on its own domain does not expose the org, so
    // it correctly falls through to the ordinary path.
    expect(ashbyBoard("https://openai.com/careers/some-role/")).toBeNull();
    expect(ashbyBoard("https://www.ashbyhq.com/careers")).toBeNull();
    expect(ashbyBoard("https://jobs.lever.co/acme/abc123def456")).toBeNull();
  });
});

describe("Lever and SmartRecruiters", () => {
  // Both have clean public per-job APIs behind JavaScript pages. Confirmed live
  // 2026-07-21 against Cleveland Construction (Lever) and Etihad (SmartRecruiters).
  it("maps a Lever job URL to its postings API", () => {
    expect(
      leverApiUrl("https://jobs.lever.co/clevelandconstruction/99bbad26-10ae-42af-9836-3cbdec82e608"),
    ).toBe(
      "https://api.lever.co/v0/postings/clevelandconstruction/99bbad26-10ae-42af-9836-3cbdec82e608?mode=json",
    );
  });

  it("maps a SmartRecruiters job URL to its postings API, taking the numeric id", () => {
    // The path is {postingId}-{slug}; only the numeric id addresses the API.
    expect(
      smartRecruitersApiUrl("https://jobs.smartrecruiters.com/EtihadAirways5/744000122345079-sales-representative-chicago"),
    ).toBe(
      "https://api.smartrecruiters.com/v1/companies/EtihadAirways5/postings/744000122345079",
    );
  });

  it("returns null for unrelated URLs", () => {
    expect(leverApiUrl("https://boards.greenhouse.io/acme/jobs/1")).toBeNull();
    expect(smartRecruitersApiUrl("https://jobs.lever.co/acme/abc123def456")).toBeNull();
  });
});

describe("Greenhouse", () => {
  // Greenhouse is one of the most common ATS for startups and tech companies
  // (Anthropic among them) and renders with JavaScript. Its public board API
  // returns the posting as clean JSON. Confirmed live 2026-07-21.
  it("maps both greenhouse host forms to the board API", () => {
    expect(
      greenhouseApiUrl("https://job-boards.greenhouse.io/anthropic/jobs/5230755008"),
    ).toBe(
      "https://boards-api.greenhouse.io/v1/boards/anthropic/jobs/5230755008?content=true",
    );
    expect(
      greenhouseApiUrl("https://boards.greenhouse.io/acme/jobs/12345"),
    ).toBe(
      "https://boards-api.greenhouse.io/v1/boards/acme/jobs/12345?content=true",
    );
  });

  it("returns null for a non-Greenhouse URL", () => {
    expect(greenhouseApiUrl("https://careers.cintas.com/job/x/1")).toBeNull();
  });
});

describe("Workday", () => {
  // Workday powers a huge share of enterprise hiring, Miami University's own
  // careers site included, and renders with JavaScript. Blocking it outright
  // (as an earlier version did) gave up on that whole platform, but its public
  // JSON API returns the full posting. Confirmed live against Smucker and Miami
  // on 2026-07-21.
  it("maps a human job URL to the JSON API on the same host", () => {
    const human =
      "https://smucker.wd5.myworkdayjobs.com/en-US/US_External_Careers/job/Orrville-OH/Operations-Management-Intern_115347?workerSubType=abc";
    expect(workdayCxsUrl(human)).toBe(
      "https://smucker.wd5.myworkdayjobs.com/wday/cxs/smucker/US_External_Careers/job/Orrville-OH/Operations-Management-Intern_115347",
    );
    // Same host, so the SSRF host check still governs it.
    expect(new URL(workdayCxsUrl(human)!).hostname).toBe(
      "smucker.wd5.myworkdayjobs.com",
    );
  });

  it("handles Miami's own Workday tenant", () => {
    const human =
      "https://miamioh.wd5.myworkdayjobs.com/en-US/miamioh-staff/job/Senior-Director-Technology_JR104377";
    expect(workdayCxsUrl(human)).toBe(
      "https://miamioh.wd5.myworkdayjobs.com/wday/cxs/miamioh/miamioh-staff/job/Senior-Director-Technology_JR104377",
    );
  });

  it("returns null for a non-Workday URL", () => {
    expect(workdayCxsUrl("https://boards.greenhouse.io/acme/jobs/1")).toBeNull();
    expect(workdayCxsUrl("https://careers.cintas.com/job/x/1")).toBeNull();
  });

  it("no longer treats Workday as an unreadable board", () => {
    expect(isKnownBlocker("https://acme.wd5.myworkdayjobs.com/en-US/x/job/y")).toBe(
      false,
    );
  });
});

describe("telling a single posting from a search-results grid", () => {
  // Found with a real Google Careers URL on 2026-07-21: the page served a grid
  // of unrelated openings that passed the posting heuristic, because a jobs
  // grid contains "responsibilities" and "qualifications" somewhere. Tailoring
  // a resume to a list of other people's jobs is worse than no help.
  it("rejects a results grid by its overflow marker", () => {
    const grid =
      "Business Program Manager Sunnyvale, CA, USA Machine Learning Engineer " +
      "Austin, TX, USA ; Chicago, IL, USA ; +24 more Site Reliability Manager " +
      "San Francisco, CA, USA ; +23 more Regulatory Counsel Seoul, South Korea";
    expect(looksLikeListing(grid)).toBe(true);
  });

  it("rejects a grid with many interleaved US locations", () => {
    const grid = Array.from(
      { length: 10 },
      (_, i) => `Analyst Role ${i} City${i}, OH, USA`,
    ).join(" ");
    expect(looksLikeListing(grid)).toBe(true);
  });

  it("does not flag a single posting that names a few offices", () => {
    // A real posting can list its own locations; that is not a grid. Deloitte
    // legitimately lists many, but reaches this only without structured data.
    const posting =
      "Data Analyst Intern, based in Cincinnati, OH, USA with occasional " +
      "travel to our Columbus, OH office. In this role you will build and " +
      "maintain dashboards used across the operations team, analyse shipment " +
      "and sales data to surface trends, and present your findings to " +
      "non-technical stakeholders each week. Responsibilities include owning " +
      "the weekly reporting cycle end to end. Qualifications: currently " +
      "pursuing a degree in Business Analytics or Statistics, comfortable " +
      "with SQL and either R or Python, with zero to two years of experience. " +
      "Preferred: familiarity with Tableau. We are looking for someone curious.";
    // One posting that names two of its own offices is not a grid.
    expect(looksLikeListing(posting)).toBe(false);
    expect(looksLikePosting(posting)).toBe(true);
  });
});

describe("turning a page into readable text", () => {
  it("drops scripts, styles and chrome", () => {
    const html = `
      <html><head><style>.a{color:red}</style></head>
      <body>
        <nav>Home Jobs About</nav>
        <script>window.x = 1</script>
        <h1>Data Analyst</h1>
        <p>You will build dashboards.</p>
        <footer>Privacy policy</footer>
      </body></html>`;
    const text = htmlToText(html);
    expect(text).toContain("Data Analyst");
    expect(text).toContain("You will build dashboards.");
    expect(text).not.toContain("color:red");
    expect(text).not.toContain("window.x");
    expect(text).not.toContain("Privacy policy");
    expect(text).not.toContain("Home Jobs About");
  });

  it("decodes entities and keeps paragraphs apart", () => {
    const text = htmlToText("<p>R &amp; Python</p><p>SQL</p>");
    expect(text).toContain("R & Python");
    expect(text.split("\n")).toContain("SQL");
  });

it("resolves double-encoded entities and named ones", () => {
    // Greenhouse ships "&amp;nbsp;" (double-encoded), and Amazon uses named
    // typographic entities like &mdash; and &times;. Both leaked verbatim until
    // 2026-07-21; the text decode now runs twice and covers named entities.
    expect(htmlToText("A&amp;nbsp;B &amp;mdash; C")).toBe("A B — C");
    expect(htmlToText("Sales &times; Marketing")).toBe("Sales x Marketing");
    // A single "&amp;" between tags still becomes one ampersand, not doubled.
    expect(htmlToText("<p>R &amp; D</p>")).toBe("R & D");
  });

  it("recognises a login wall rather than treating it as a posting", () => {
    expect(looksLikeLoginWall("Please sign in to continue")).toBe(true);
    expect(looksLikeLoginWall("Enable JavaScript to view this page")).toBe(true);
    // A long real posting that happens to mention signing in is not a wall.
    expect(
      looksLikeLoginWall(
        "You will sign in to our analytics tools daily. ".repeat(80),
      ),
    ).toBe(false);
  });
});
