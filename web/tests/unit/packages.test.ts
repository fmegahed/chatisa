import { describe, expect, it } from "vitest";
import {
  normalizePkg,
  buildPyodideIndex,
  classifyPythonPackage,
} from "@/lib/sandbox/packages";

const lock = {
  packages: {
    numpy: { name: "numpy", imports: ["numpy"] },
    "scikit-learn": { name: "scikit-learn", imports: ["sklearn"] },
    beautifulsoup4: { name: "beautifulsoup4", imports: ["bs4"] },
    requests: { name: "requests", imports: ["requests"] },
    "requests-tests": { name: "requests-tests", imports: [] },
  },
};

describe("normalizePkg", () => {
  it("lowercases and collapses separators", () => {
    expect(normalizePkg("Scikit_Learn")).toBe("scikit-learn");
    expect(normalizePkg("beautiful.soup")).toBe("beautiful-soup");
  });
});

describe("buildPyodideIndex", () => {
  it("indexes package names and their import names, skipping -tests", () => {
    const idx = buildPyodideIndex(lock);
    expect(idx.packages.has("numpy")).toBe(true);
    expect(idx.packages.has("requests-tests")).toBe(false);
    expect(idx.byAlias.get("sklearn")).toBe("scikit-learn");
    expect(idx.byAlias.get("bs4")).toBe("beautifulsoup4");
  });
});

describe("classifyPythonPackage", () => {
  const idx = buildPyodideIndex(lock);

  it("marks a bundled package as ready", () => {
    const r = classifyPythonPackage("numpy", idx);
    expect(r?.status).toBe("ready");
  });

  it("marks a hosted-wheel package (not in Pyodide's lock) as ready", () => {
    // seaborn and openpyxl are our own hosted wheels, absent from pyodide-lock,
    // so they must be caught by the bundled set, not fall through to unavailable.
    expect(classifyPythonPackage("seaborn", idx)?.status).toBe("ready");
    expect(classifyPythonPackage("openpyxl", idx)?.status).toBe("ready");
  });

  it("resolves an import alias and marks a built-but-not-bundled package installable", () => {
    const r = classifyPythonPackage("sklearn", idx);
    expect(r?.status).toBe("ready"); // scikit-learn is bundled
    const req = classifyPythonPackage("requests", idx);
    expect(req?.status).toBe("installable");
    expect(req?.message).toMatch(/micropip\.install\("requests"\)/);
  });

  it("calls out a known-unavailable package by name", () => {
    const r = classifyPythonPackage("pyreadr", idx);
    expect(r?.status).toBe("unavailable");
    expect(r?.message).toMatch(/needs compiling/i);
  });

  it("marks the shipped forecasting stack ready (v6.3.0 wasm build)", () => {
    for (const name of ["statsforecast", "coreforecast", "utilsforecast"]) {
      const r = classifyPythonPackage(name, idx);
      expect(r?.status, name).toBe("ready");
    }
  });

  it("is honest about an unknown package", () => {
    const r = classifyPythonPackage("some_random_pkg", idx);
    expect(r?.status).toBe("unavailable");
    expect(r?.message).toMatch(/pure-Python/i);
  });

  it("returns null for an empty query", () => {
    expect(classifyPythonPackage("  ", idx)).toBeNull();
  });
});
