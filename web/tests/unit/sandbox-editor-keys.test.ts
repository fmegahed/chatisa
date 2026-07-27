import { describe, expect, it } from "vitest";
import { PIPE_TOKEN, buildPipeInsertion } from "@/lib/sandbox/editor-keys";

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
