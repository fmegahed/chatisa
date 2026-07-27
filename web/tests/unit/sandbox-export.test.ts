import { describe, expect, it } from "vitest";
import {
  delimiterFor,
  extensionFor,
  mimeFor,
  exportFilename,
  workspaceExtensionFor,
  workspaceMimeFor,
  exportWorkspaceFilename,
} from "@/lib/sandbox/export";

describe("export helpers", () => {
  it("maps format to delimiter", () => {
    expect(delimiterFor("csv")).toBe(",");
    expect(delimiterFor("tsv")).toBe("\t");
  });

  it("maps format to extension and MIME", () => {
    expect(extensionFor("csv")).toBe("csv");
    expect(extensionFor("tsv")).toBe("tsv");
    expect(mimeFor("csv")).toBe("text/csv");
    expect(mimeFor("tsv")).toBe("text/tab-separated-values");
  });

  it("names the file after the student, the object, and the moment", () => {
    const date = new Date(2026, 6, 23, 14, 30); // 2026-07-23 14:30 local
    expect(
      exportFilename({ userEmail: "megahefm@miamioh.edu", name: "grades", format: "csv", date }),
    ).toBe("megahefm-grades-20260723-1430.csv");
  });

  it("sanitizes the object name and the email local part", () => {
    const date = new Date(2026, 6, 23, 9, 5);
    // Spaces and punctuation in the object name become underscores (trailing
    // underscores trimmed); a blank email falls back to "sandbox".
    expect(
      exportFilename({ userEmail: "", name: "my table!", format: "tsv", date }),
    ).toBe("sandbox-my_table-20260723-0905.tsv");
  });
});

describe("workspace export helpers", () => {
  it("maps each language to its whole-environment file extension", () => {
    expect(workspaceExtensionFor("r")).toBe("RData");
    expect(workspaceExtensionFor("sql")).toBe("sqlite");
    expect(workspaceExtensionFor("python")).toBe("pkl");
  });

  it("maps each language to a binary MIME type", () => {
    expect(workspaceMimeFor("r")).toBe("application/octet-stream");
    expect(workspaceMimeFor("sql")).toBe("application/vnd.sqlite3");
    expect(workspaceMimeFor("python")).toBe("application/octet-stream");
  });

  it("names the workspace file by language and moment", () => {
    const date = new Date(2026, 6, 23, 14, 30); // 2026-07-23 14:30 local
    expect(exportWorkspaceFilename({ lang: "r", date })).toBe(
      "chatisa-workspace-r-20260723-1430.RData",
    );
    expect(exportWorkspaceFilename({ lang: "sql", date })).toBe(
      "chatisa-workspace-sql-20260723-1430.sqlite",
    );
    expect(exportWorkspaceFilename({ lang: "python", date })).toBe(
      "chatisa-workspace-python-20260723-1430.pkl",
    );
  });
});
