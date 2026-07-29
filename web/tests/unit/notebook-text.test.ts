import { describe, expect, it } from "vitest";
import { notebookToText } from "@/lib/files/notebook-text";

/** Minimal nbformat-4 notebook built as an object (no binary fixtures). */
function nb(cells: unknown[], metadata: Record<string, unknown> = {}): string {
  return JSON.stringify({ nbformat: 4, nbformat_minor: 5, metadata, cells });
}

const PIXEL =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

describe("notebookToText", () => {
  it("returns null for non-notebook input so callers fall back to plain text", () => {
    expect(notebookToText("def f():\n  return 1")).toBeNull();
    expect(notebookToText('{"nbformat": 4}')).toBeNull();
    expect(notebookToText('"just a string"')).toBeNull();
  });

  it("extracts markdown verbatim and fences code with the kernel language", () => {
    const raw = nb(
      [
        { cell_type: "markdown", source: ["# Sales forecast\n", "Weekly ARIMA."] },
        { cell_type: "code", source: "library(fpp3)\nfit <- model(ts, ARIMA(units))", outputs: [] },
      ],
      { kernelspec: { language: "R" } },
    );
    const result = notebookToText(raw);
    expect(result).not.toBeNull();
    expect(result!.language).toBe("r");
    expect(result!.cellCount).toBe(2);
    expect(result!.text).toContain("# Sales forecast\nWeekly ARIMA.");
    expect(result!.text).toContain("```r\nlibrary(fpp3)\nfit <- model(ts, ARIMA(units))\n```");
  });

  it("joins string-array sources and defaults the language to python", () => {
    const raw = nb([
      { cell_type: "code", source: ["import pandas as pd\n", "df.head()"], outputs: [] },
    ]);
    const result = notebookToText(raw)!;
    expect(result.language).toBe("python");
    expect(result.text).toContain("```python\nimport pandas as pd\ndf.head()\n```");
  });

  it("keeps stream and text/plain outputs, capped per output", () => {
    const raw = nb([
      {
        cell_type: "code",
        source: "print(df.shape)",
        outputs: [
          { output_type: "stream", name: "stdout", text: ["(120, 5)\n"] },
          {
            output_type: "execute_result",
            data: { "text/plain": ["   week  units\n0     1     10"] },
          },
          { output_type: "stream", name: "stdout", text: "x".repeat(3000) },
        ],
      },
    ]);
    const text = notebookToText(raw)!.text;
    expect(text).toContain("Output:\n(120, 5)");
    expect(text).toContain("   week  units\n0     1     10");
    expect(text).toContain("[output truncated]");
  });

  it("collects plot images up to the cap and marks the overflow", () => {
    const plot = {
      output_type: "display_data",
      data: { "image/png": PIXEL },
    };
    const raw = nb([
      { cell_type: "code", source: "plots()", outputs: [plot, plot, plot] },
    ]);
    const capped = notebookToText(raw, { maxImages: 2 })!;
    expect(capped.images).toHaveLength(2);
    expect(capped.images[0]).toMatchObject({ mediaType: "image/png", cellIndex: 0 });
    expect(capped.images[0].base64).toBe(PIXEL);
    expect(capped.text).toContain("[plot 1 from cell 1]");
    expect(capped.text).toContain("[plot 2 from cell 1]");
    expect(capped.text).toContain("[plot output omitted]");

    const none = notebookToText(raw, { maxImages: 0 })!;
    expect(none.images).toHaveLength(0);
    expect(none.text).not.toContain("from cell");
    expect(none.text).toContain("[plot output omitted]");
  });

  it("prefers the image over the text/plain repr of the same output", () => {
    const raw = nb([
      {
        cell_type: "code",
        source: "fig",
        outputs: [
          {
            output_type: "execute_result",
            data: {
              "image/png": PIXEL,
              "text/plain": "<Figure size 640x480 with 1 Axes>",
            },
          },
        ],
      },
    ]);
    const result = notebookToText(raw)!;
    expect(result.images).toHaveLength(1);
    expect(result.text).not.toContain("Figure size");
  });

  it("keeps error tracebacks with ANSI colors stripped", () => {
    const esc = String.fromCharCode(27);
    const raw = nb([
      {
        cell_type: "code",
        source: "1/0",
        outputs: [
          {
            output_type: "error",
            ename: "ZeroDivisionError",
            evalue: "division by zero",
            traceback: [`${esc}[0;31mZeroDivisionError${esc}[0m: division by zero`],
          },
        ],
      },
    ]);
    const text = notebookToText(raw)!.text;
    expect(text).toContain("ZeroDivisionError: division by zero");
    expect(text).not.toContain(esc);
    // Bracketed plain text must survive the ANSI strip.
    expect(notebookToText(nb([
      { cell_type: "code", source: "x", outputs: [{ output_type: "stream", text: "[1] 0.5" }] },
    ]))!.text).toContain("[1] 0.5");
  });
});
