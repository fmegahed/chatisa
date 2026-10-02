import { describe, expect, it } from "vitest";
import {
  CARET_POSITION_KEY,
  PIPE_TOKEN,
  buildPipeInsertion,
  caretAnnouncement,
  caretPositionAt,
  caretPositionKeysLabel,
  caretStatusText,
} from "@/lib/sandbox/editor-keys";

describe("native pipe insertion", () => {
  it("uses the native pipe with surrounding spaces, not magrittr", () => {
    expect(PIPE_TOKEN).toBe(" |> ");
    expect(PIPE_TOKEN).not.toContain("%>%");
  });

  it("inserts at an empty caret and places the caret after the pipe", () => {
    // caret at offset 5, nothing selected
    expect(buildPipeInsertion({ from: 5, to: 5 })).toEqual({
      from: 5,
      to: 5,
      insert: " |> ",
      anchor: 9, // 5 + 4
    });
  });

  it("replaces a selection and places the caret after the pipe", () => {
    // "df|filter" style: a 5-char selection [2,7) is replaced by the pipe
    expect(buildPipeInsertion({ from: 2, to: 7 })).toEqual({
      from: 2,
      to: 7,
      insert: " |> ",
      anchor: 6, // 2 + 4
    });
  });
});

describe("caret position (#21)", () => {
  // A tiny line table for "ab\ncde\n": lines start at 0, 3 and 7.
  const starts = [0, 3, 7];
  const lineAt = (pos: number) => {
    let i = starts.length - 1;
    while (starts[i] > pos) i--;
    return { number: i + 1, from: starts[i] };
  };

  it("reports a 1-based line and column", () => {
    expect(caretPositionAt(lineAt, 3, 0)).toEqual({ line: 1, column: 1, lines: 3 });
    expect(caretPositionAt(lineAt, 3, 5)).toEqual({ line: 2, column: 3, lines: 3 });
    expect(caretPositionAt(lineAt, 3, 7)).toEqual({ line: 3, column: 1, lines: 3 });
  });

  it("formats the status line and the announcement", () => {
    const pos = { line: 2, column: 3, lines: 10 };
    expect(caretStatusText(pos)).toBe("Ln 2, Col 3");
    expect(caretAnnouncement(pos)).toBe("Line 2 of 10, column 3.");
    expect(caretAnnouncement(pos, ["Error: Unmatched ("])).toBe(
      "Line 2 of 10, column 3. Error: Unmatched (",
    );
  });

  it("uses a binding CodeMirror does not already own, labelled per platform", () => {
    // Alt-l is CodeMirror's selectLine; Shift keeps them apart.
    expect(CARET_POSITION_KEY).toBe("Alt-Shift-l");
    expect(caretPositionKeysLabel(false)).toBe("Alt+Shift+L");
    expect(caretPositionKeysLabel(true)).toBe("Option+Shift+L");
  });
});
