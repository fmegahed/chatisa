import { describe, expect, it } from "vitest";
import { resolveDoc, referenceHome } from "@/lib/sandbox/help-docs/resolve";
import { symbolAt } from "@/lib/sandbox/help-docs/symbol-at";
import type { HelpRequest } from "@/lib/sandbox/help-docs/types";

function req(partial: Partial<HelpRequest> & { name: string }): HelpRequest {
  return {
    kind: "function",
    language: partial.language ?? "r",
    qualifier: partial.qualifier,
    name: partial.name,
  };
}

describe("resolveDoc: R", () => {
  it("maps summarise to the dplyr reference", () => {
    const d = resolveDoc(req({ name: "summarise", language: "r" }));
    expect(d?.source).toBe("dplyr");
    expect(d?.url).toBe("https://dplyr.tidyverse.org/reference/summarise.html");
    expect(d?.blurb).toBeTruthy();
  });

  it("accepts the American spelling summarize", () => {
    expect(resolveDoc(req({ name: "summarize", language: "r" }))?.url).toBe(
      "https://dplyr.tidyverse.org/reference/summarise.html",
    );
  });

  it("maps mean to base R", () => {
    const d = resolveDoc(req({ name: "mean", language: "r" }));
    expect(d?.source).toBe("base R");
    expect(d?.url).toBe(
      "https://stat.ethz.ch/R-manual/R-devel/library/base/html/mean.html",
    );
  });

  it("maps ggplot to ggplot2", () => {
    const d = resolveDoc(req({ name: "ggplot", language: "r" }));
    expect(d?.source).toBe("ggplot2");
    expect(d?.url).toBe("https://ggplot2.tidyverse.org/reference/ggplot.html");
  });

  it("returns null for an unknown R symbol", () => {
    expect(resolveDoc(req({ name: "no_such_fn_xyz", language: "r" }))).toBeNull();
  });

  it("keeps the curated entry when the qualifier matches (dplyr::filter)", () => {
    const d = resolveDoc(req({ name: "filter", qualifier: "dplyr", language: "r" }));
    expect(d?.source).toBe("dplyr");
    expect(d?.url).toBe("https://dplyr.tidyverse.org/reference/filter.html");
  });

  it("does not borrow dplyr's link for a different namespace (stats::filter)", () => {
    // The curated map only knows dplyr::filter; stats::filter is a different
    // function, so the curated entry must not be shown. The live help() text
    // (namespace-correct) still fills the HELP tab.
    expect(
      resolveDoc(req({ name: "filter", qualifier: "stats", language: "r" })),
    ).toBeNull();
  });
});

describe("resolveDoc: Python", () => {
  it("maps a DataFrame method to the pandas reference", () => {
    const d = resolveDoc(
      req({ name: "groupby", qualifier: "df", kind: "function", language: "python" }),
    );
    expect(d?.source).toBe("pandas");
    expect(d?.url).toBe(
      "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html",
    );
  });

  it("maps a pandas top-level function", () => {
    expect(
      resolveDoc(req({ name: "read_csv", qualifier: "pd", language: "python" }))?.url,
    ).toBe("https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html");
  });

  it("maps len to the Python builtins docs", () => {
    const d = resolveDoc(req({ name: "len", language: "python" }));
    expect(d?.source).toBe("Python");
    expect(d?.url).toBe("https://docs.python.org/3/library/functions.html#len");
  });

  it("returns null for an unknown Python symbol", () => {
    expect(
      resolveDoc(req({ name: "totally_made_up", language: "python" })),
    ).toBeNull();
  });
});

describe("resolveDoc: SQL (SQLite only)", () => {
  it("maps COUNT to the SQLite aggregate functions page", () => {
    const d = resolveDoc(req({ name: "COUNT", kind: "function", language: "sql" }));
    expect(d?.source).toBe("SQLite");
    expect(d?.url).toBe("https://www.sqlite.org/lang_aggfunc.html");
  });

  it("is case-insensitive for SQL (avg)", () => {
    expect(
      resolveDoc(req({ name: "avg", kind: "function", language: "sql" }))?.url,
    ).toBe("https://www.sqlite.org/lang_aggfunc.html");
  });

  it("maps JOIN and GROUP BY and WITH to SQLite syntax pages", () => {
    expect(
      resolveDoc(req({ name: "JOIN", kind: "keyword", language: "sql" }))?.url,
    ).toBe("https://www.sqlite.org/lang_select.html");
    expect(
      resolveDoc(req({ name: "GROUP BY", kind: "keyword", language: "sql" }))?.url,
    ).toBe("https://www.sqlite.org/lang_select.html");
    expect(
      resolveDoc(req({ name: "WITH", kind: "keyword", language: "sql" }))?.url,
    ).toBe("https://www.sqlite.org/lang_with.html");
  });

  it("handles DATE_TRUNC honestly (SQLite has none)", () => {
    const d = resolveDoc(req({ name: "DATE_TRUNC", kind: "function", language: "sql" }));
    expect(d?.source).toBe("SQLite");
    expect(d?.url).toBe("https://www.sqlite.org/lang_datefunc.html");
    expect(d?.note).toMatch(/SQLite has no DATE_TRUNC/i);
    expect(d?.note).toMatch(/strftime/i);
  });

  it("notes that only SQLite runs when another dialect is requested", () => {
    const d = resolveDoc(
      req({ name: "COUNT", kind: "function", language: "sql" }),
      { dialect: "postgres" },
    );
    // Still SQLite docs (the only engine that runs here), with an honest note.
    expect(d?.source).toBe("SQLite");
    expect(d?.url).toBe("https://www.sqlite.org/lang_aggfunc.html");
    expect(d?.note).toMatch(/only SQLite runs/i);
  });

  it("returns null for an unknown SQL token", () => {
    expect(
      resolveDoc(req({ name: "FLOORP", kind: "function", language: "sql" })),
    ).toBeNull();
  });
});

describe("referenceHome", () => {
  it("gives a per-language reference home for the unknown-symbol fallback", () => {
    expect(referenceHome("r").url).toContain("rdocumentation.org");
    expect(referenceHome("python").url).toContain("docs.python.org");
    expect(referenceHome("sql").url).toContain("sqlite.org");
  });
});

/** Resolve the symbol the finder returns for a cursor at `marker`. */
function at(src: string, marker: string, language: "r" | "python" | "sql") {
  const pos = src.indexOf(marker);
  return symbolAt(src, pos, language);
}

describe("symbolAt: R", () => {
  it("reads a bare function name", () => {
    expect(at("grades |> summarise(x = mean(g))", "summarise", "r")?.name).toBe(
      "summarise",
    );
  });

  it("reads the qualified name and package (dplyr::summarise)", () => {
    const s = at("dplyr::summarise(x)", "summarise", "r");
    expect(s?.name).toBe("summarise");
    expect(s?.qualifier).toBe("dplyr");
  });

  it("reads a dotted R name (theme_bw)", () => {
    expect(at("ggplot(d) + theme_bw()", "theme_bw", "r")?.name).toBe("theme_bw");
  });

  it("returns null inside a string", () => {
    expect(at('x <- "summarise here"', "summarise", "r")).toBeNull();
  });

  it("returns null inside a comment", () => {
    expect(at("x <- 1 # call summarise", "summarise", "r")).toBeNull();
  });
});

describe("symbolAt: Python", () => {
  it("reads a builtin call name", () => {
    expect(at("n = len(items)", "len", "python")?.name).toBe("len");
  });

  it("reads a method name and its receiver", () => {
    const s = at('out = df.groupby("k")', "groupby", "python");
    expect(s?.name).toBe("groupby");
    expect(s?.qualifier).toBe("df");
  });

  it("returns null inside a string", () => {
    expect(at('s = "please groupby"', "groupby", "python")).toBeNull();
  });
});

describe("symbolAt: SQL", () => {
  it("reads a function name", () => {
    expect(at("SELECT COUNT(*) FROM t;", "COUNT", "sql")?.name).toBe("COUNT");
  });

  it("combines GROUP BY into one symbol", () => {
    const s = at("SELECT k FROM t GROUP BY k;", "GROUP", "sql");
    expect(s?.name).toBe("GROUP BY");
    expect(s?.kind).toBe("keyword");
  });

  it("combines ORDER BY into one symbol", () => {
    expect(at("SELECT k FROM t ORDER BY k;", "ORDER", "sql")?.name).toBe(
      "ORDER BY",
    );
  });

  it("reads WITH", () => {
    expect(at("WITH a AS (SELECT 1) SELECT * FROM a;", "WITH", "sql")?.name).toBe(
      "WITH",
    );
  });

  it("returns null inside a string literal", () => {
    expect(at("SELECT 'COUNT me' AS s;", "COUNT", "sql")).toBeNull();
  });
});
