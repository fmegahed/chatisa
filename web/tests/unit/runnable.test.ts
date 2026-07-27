import { describe, expect, it } from "vitest";
import {
  pythonRequirements,
  rRequirements,
  requirementsFor,
} from "@/lib/sandbox/requirements";
import {
  assessRunnability,
  baseRIndex,
  buildRIndex,
  KNOWN_UNAVAILABLE_R,
} from "@/lib/sandbox/runnable";
import {
  BUNDLED_R_CLOSURE,
  buildPyodideIndex,
} from "@/lib/sandbox/packages";
import { existsSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";

/**
 * The Run button is a promise. These tests exist because of what happens when it
 * is wrong in each direction, and the two are not equally bad:
 *
 *   - Offering Run on code that cannot possibly work wastes the student's time
 *     and teaches them the app is unreliable.
 *   - HIDING Run on code that would have worked is worse: the feature is simply
 *     gone, and nothing tells them it should have been there.
 *
 * So the bar for blocking is positive evidence, and most of what follows checks
 * that nothing is blocked without it.
 */

const PY_LOCK = {
  packages: {
    numpy: { name: "numpy", imports: ["numpy"] },
    pandas: { name: "pandas", imports: ["pandas"] },
    matplotlib: { name: "matplotlib", imports: ["matplotlib", "pylab"] },
    "scikit-learn": { name: "scikit-learn", imports: ["sklearn"] },
    requests: { name: "requests", imports: ["requests"] },
    beautifulsoup4: { name: "beautifulsoup4", imports: ["bs4"] },
  },
};

const pyIndex = buildPyodideIndex(PY_LOCK);

describe("pythonRequirements", () => {
  it("reads every import form, keeping only the top-level package", () => {
    expect(
      pythonRequirements(
        [
          "import pandas as pd",
          "import matplotlib.pyplot as plt",
          "from sklearn.linear_model import LinearRegression",
          "import os, json, numpy",
        ].join("\n"),
      ),
    ).toEqual(["matplotlib", "numpy", "pandas", "sklearn"]);
  });

  it("drops the standard library, which is never installable", () => {
    expect(pythonRequirements("import os\nimport json\nimport re\nimport sys")).toEqual([]);
  });

  it("ignores imports inside comments and docstrings", () => {
    // A package named in prose must never hide a Run button.
    const code = [
      '"""This example would use import statsforecast if it were available."""',
      "# import rJava",
      "import pandas",
    ].join("\n");
    expect(pythonRequirements(code)).toEqual(["pandas"]);
  });

  it("counts an explicit micropip request", () => {
    expect(
      pythonRequirements('import micropip\nawait micropip.install("openpyxl")'),
    ).toEqual(["openpyxl"]);
  });

  it("skips relative imports, which name no package", () => {
    expect(pythonRequirements("from . import helper\nfrom .utils import x")).toEqual([]);
  });
});

describe("rRequirements", () => {
  it("reads library, require, and the pkg::fn form this app mandates", () => {
    // pkg::fn() is the house style, so a library()-only scan would see nothing
    // at all in most of this app's R code.
    expect(
      rRequirements(
        [
          'if(require(rvest)==FALSE) install.packages("rvest")',
          "doc <- rvest::read_html(url)",
          "out <- dplyr::mutate(df, x = 1)",
          'library("ggplot2")',
        ].join("\n"),
      ),
    ).toEqual(["dplyr", "ggplot2", "rvest"]);
  });

  it("drops base and recommended packages", () => {
    expect(rRequirements("stats::lm(y ~ x, df)\nutils::head(df)\nbase::sum(1:3)")).toEqual(
      [],
    );
  });

  it("reads a vector of install targets", () => {
    expect(rRequirements('install.packages(c("zoo", "forecast"))')).toEqual([
      "forecast",
      "zoo",
    ]);
  });

  it("ignores package names in comments", () => {
    expect(rRequirements("# rJava::init() would go here\ndplyr::tibble(x=1)")).toEqual([
      "dplyr",
    ]);
  });
});

describe("requirementsFor", () => {
  it("reports nothing for SQL, which has no packages", () => {
    expect(requirementsFor("sql", "SELECT * FROM t")).toEqual([]);
  });
});

describe("assessRunnability, Python", () => {
  const indexes = { python: pyIndex, r: baseRIndex() };

  it("is ready when everything is bundled", () => {
    const v = assessRunnability("python", "import pandas as pd\npd.DataFrame()", indexes);
    expect(v.status).toBe("ready");
    expect(v.message).toBeNull();
  });

  it("warns, but still runs, when a package installs on first use", () => {
    const v = assessRunnability("python", "import requests", indexes);
    expect(v.status).toBe("installable");
    expect(v.willInstall).toEqual(["requests"]);
    expect(v.message).toMatch(/first run installs requests/i);
  });

  it("blocks a package that provably needs compiling", () => {
    const v = assessRunnability("python", "import statsforecast", indexes);
    expect(v.status).toBe("blocked");
    expect(v.impossible).toEqual(["statsforecast"]);
    expect(v.message).toMatch(/cannot run here/i);
    // It must name the package and tell them what to do instead.
    expect(v.message).toContain("statsforecast");
    expect(v.message).toMatch(/on your computer/i);
  });

  it("does NOT block a package it merely does not recognise", () => {
    // micropip can still fetch a pure-Python package that is not in the lock.
    // Blocking here would remove Run from working code.
    const v = assessRunnability("python", "import some_obscure_helper", indexes);
    expect(v.status).toBe("unknown");
    expect(v.impossible).toEqual([]);
  });

  it("does not block when the index has not loaded yet", () => {
    const v = assessRunnability("python", "import pandas", {
      python: null,
      r: baseRIndex(),
    });
    expect(v.status).toBe("unknown");
    expect(v.impossible).toEqual([]);
  });

  it("blocks the whole snippet when any one package is impossible", () => {
    const v = assessRunnability(
      "python",
      "import pandas\nimport statsforecast",
      indexes,
    );
    expect(v.status).toBe("blocked");
  });
});

describe("assessRunnability, R", () => {
  it("is ready for the mirrored bundle", () => {
    const v = assessRunnability("r", "dplyr::tibble(x = 1)", {
      python: pyIndex,
      r: baseRIndex(),
    });
    expect(v.status).toBe("ready");
  });

  it("blocks a package needing a system component", () => {
    const v = assessRunnability("r", "rJava::.jinit()", {
      python: pyIndex,
      r: baseRIndex(),
    });
    expect(v.status).toBe("blocked");
    expect(v.message).toMatch(/system component/i);
  });

  it("stays unknown for an unmirrored package when no manifest exists", () => {
    // WebR's repository may well serve it. Without the manifest there is no
    // basis to block, and this is the common case on a machine where
    // npm run setup:runtimes has not regenerated the index.
    const v = assessRunnability("r", "zoo::zoo(1:3)", {
      python: pyIndex,
      r: baseRIndex(),
    });
    expect(v.status).toBe("unknown");
    expect(v.impossible).toEqual([]);
  });

  it("uses the manifest when it is there", () => {
    const index = buildRIndex({ mirrored: ["dplyr"], repo: ["dplyr", "zoo"] });
    expect(assessRunnability("r", "zoo::zoo(1:3)", { python: pyIndex, r: index }).status).toBe(
      "installable",
    );
    // Now absence IS evidence: the repository is known and does not have it.
    expect(
      assessRunnability("r", "notarealpkg::f()", { python: pyIndex, r: index }).status,
    ).toBe("blocked");
  });

  it("keeps the bundled packages known even if the manifest omits them", () => {
    // A truncated or stale manifest must not un-know tidyverse.
    const index = buildRIndex({ mirrored: [], repo: ["zoo"] });
    expect(index.mirrored.has("tidyverse")).toBe(true);
    expect(
      assessRunnability("r", "tidyverse::tidyverse_logo()", {
        python: pyIndex,
        r: index,
      }).status,
    ).toBe("ready");
  });

  it("only claims closure packages that are really on the mirror", () => {
    // BUNDLED_R_CLOSURE asserts these arrive with tidyverse. A wrong entry makes
    // the app promise "ready" for a package that is not there, which is the
    // failure mode this whole feature exists to remove. Checked against the
    // mirror when it is present; skipped on a machine that has not run
    // npm run setup:runtimes, because absence proves nothing.
    const mirror = resolve(
      __dirname,
      "..",
      "..",
      "public",
      "runtimes",
      "webr-packages",
      "bin",
      "emscripten",
      "contrib",
      "4.6",
    );
    if (!existsSync(mirror)) return;

    const onDisk = new Set(
      readdirSync(mirror)
        .filter((f) => f.endsWith(".tgz"))
        .map((f) => f.replace(/_.*$/, "")),
    );
    const missing = BUNDLED_R_CLOSURE.filter((pkg) => !onDisk.has(pkg));
    expect(missing, `named in BUNDLED_R_CLOSURE but not on the mirror`).toEqual([]);
    // Sanity on the scan itself: if the directory read produced nothing useful,
    // the check above would pass vacuously.
    expect(onDisk.has("tidyverse")).toBe(true);
    expect(join(mirror, "PACKAGES")).toBeTruthy();
  });

  it("keeps the impossible list short and specific", () => {
    // Every name here can hide a Run button, so the list is a liability if it
    // grows casually. Anything added needs a system-dependency reason.
    expect(KNOWN_UNAVAILABLE_R.size).toBeLessThanOrEqual(12);
    expect(KNOWN_UNAVAILABLE_R.has("ggplot2")).toBe(false);
    expect(KNOWN_UNAVAILABLE_R.has("dplyr")).toBe(false);
  });
});
