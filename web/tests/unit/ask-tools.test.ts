import { describe, expect, it } from "vitest";
import {
  ASK_TOOL_NAMES,
  TOOL_OUTPUT_MAX,
  askToolDefs,
  enrichPythonError,
  missingModuleFrom,
  truncateToolOutput,
} from "@/lib/ask/tools";
import { buildPyodideIndex } from "@/lib/sandbox/packages";

describe("askToolDefs", () => {
  it("declares the three browser runtimes, none server-executed", () => {
    const defs = askToolDefs();
    expect(Object.keys(defs).sort()).toEqual([...ASK_TOOL_NAMES].sort());
    for (const name of ASK_TOOL_NAMES) {
      const def = defs[name] as { execute?: unknown; description?: string };
      // No execute: calls stream to the browser, which runs them (design 2026-07-24).
      expect(def.execute, name).toBeUndefined();
      expect(def.description ?? "").not.toContain("—"); // no em dashes
    }
  });

  it("run_python's description states the proxied-web contract", () => {
    // Updated twice on 2026-07-24: first when requests proved to work natively
    // (CORS-bound), then when /api/py-proxy gave Python full web parity with
    // R. The description must teach the model what works and how refusals
    // read, so it never treats a proxy refusal as a transient failure.
    const defs = askToolDefs() as Record<string, { description?: string }>;
    expect(defs.run_python.description).toMatch(/requests can reach the web/i);
    expect(defs.run_python.description).toMatch(/ChatISA proxy:/);
    expect(defs.run_python.description).toMatch(/private hosts blocked/i);
    expect(defs.run_r.description).toMatch(/can reach the web/i);
    expect(defs.run_sql.description).toMatch(/SQLite/);
  });
});

describe("truncateToolOutput", () => {
  it("passes short output through and caps long output with a note", () => {
    expect(truncateToolOutput("hello")).toEqual({ text: "hello", truncated: false });
    const long = "x".repeat(TOOL_OUTPUT_MAX + 500);
    const res = truncateToolOutput(long);
    expect(res.truncated).toBe(true);
    expect(res.text.length).toBeLessThanOrEqual(TOOL_OUTPUT_MAX + 100);
    expect(res.text).toMatch(/truncated/i);
  });
});

describe("missingModuleFrom", () => {
  it("extracts the module from a ModuleNotFoundError", () => {
    expect(
      missingModuleFrom("ModuleNotFoundError: No module named 'statsforecast'"),
    ).toBe("statsforecast");
    expect(missingModuleFrom('ImportError: No module named "prophet"')).toBe(
      "prophet",
    );
    expect(missingModuleFrom("ZeroDivisionError: division by zero")).toBeNull();
  });
});

describe("enrichPythonError", () => {
  const index = buildPyodideIndex({
    packages: { requests: { name: "requests", imports: ["requests"] } },
  });

  it("appends the package-checker verdict for an import failure", () => {
    const enriched = enrichPythonError(
      "ModuleNotFoundError: No module named 'statsforecast'",
      index,
    );
    expect(enriched).toContain("No module named");
    expect(enriched).toMatch(/needs compiling/i);
  });

  it("tells the model an available package just needs importing or micropip", () => {
    const enriched = enrichPythonError(
      "ModuleNotFoundError: No module named 'requests'",
      index,
    );
    expect(enriched).toMatch(/requests is available/i);
  });

  it("leaves non-import errors untouched", () => {
    const msg = "ZeroDivisionError: division by zero";
    expect(enrichPythonError(msg, index)).toBe(msg);
    expect(enrichPythonError(msg, null)).toBe(msg);
  });
});
