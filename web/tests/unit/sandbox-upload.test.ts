import { describe, expect, it } from "vitest";
import {
  isWorkspaceFile,
  formatFromName,
  uniqueName,
  defaultRestoreOptions,
} from "@/lib/sandbox/upload";

describe("workspace detection", () => {
  it("recognises workspace files, not single objects", () => {
    expect(isWorkspaceFile("a.pkl")).toBe(true);
    expect(isWorkspaceFile("a.sqlite")).toBe(true);
    expect(isWorkspaceFile("a.RData")).toBe(true);
    expect(isWorkspaceFile("a.rda")).toBe(true);
    expect(isWorkspaceFile("a.rds")).toBe(false); // one object, not a workspace
    expect(isWorkspaceFile("a.csv")).toBe(false);
  });

  it("maps the new extensions to formats", () => {
    expect(formatFromName("w.pkl")).toBe("pkl");
    expect(formatFromName("w.sqlite")).toBe("sqlite");
    expect(formatFromName("w.db")).toBe("sqlite");
  });
});

describe("uniqueName", () => {
  it("returns the name when free, else the first free numeric suffix", () => {
    const taken = new Set(["grades", "grades_2"]);
    expect(uniqueName("courses", taken)).toBe("courses");
    expect(uniqueName("grades", taken)).toBe("grades_3");
  });
});

describe("defaultRestoreOptions", () => {
  it("defaults to a non-destructive rename", () => {
    expect(defaultRestoreOptions()).toEqual({ restore: true, conflict: "rename" });
  });
});
