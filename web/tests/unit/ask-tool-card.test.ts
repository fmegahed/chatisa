import { describe, expect, it } from "vitest";
import {
  createdFileIds,
  hostOf,
  openaiContainerId,
  toolSummary,
} from "@/lib/ask/tool-card";

describe("toolSummary", () => {
  it("names the tool that failed in a whole sentence", () => {
    // The 2026-07-25 report: the summary was built by taking the first word of
    // the success label, so a failed lookup read "Looked failed".
    expect(toolSummary({ toolName: "get_paper", failed: true })).toBe(
      "The paper lookup failed",
    );
    expect(toolSummary({ toolName: "search_papers", failed: true })).toBe(
      "The literature search failed",
    );
    expect(toolSummary({ toolName: "code_execution", failed: true })).toBe(
      "The run on Anthropic's servers failed",
    );
    for (const toolName of [
      "get_paper",
      "search_papers",
      "read_url",
      "get_miami_style",
      "code_execution",
      "code_interpreter",
    ]) {
      expect(toolSummary({ toolName, failed: true })).not.toMatch(
        /^(Looked|Searched|Ran|Read|Fetched) failed$/,
      );
    }
  });

  it("falls back to the tool label for an unknown tool", () => {
    expect(toolSummary({ toolName: "run_python", failed: true })).toBe(
      "The Python run failed",
    );
  });

  it("describes the running and finished states", () => {
    expect(toolSummary({ toolName: "get_paper", running: true })).toBe(
      "Looking up the paper...",
    );
    expect(
      toolSummary({
        toolName: "search_papers",
        output: { papers: [{}, {}] },
      }),
    ).toBe("Searched the literature (2 papers)");
    expect(toolSummary({ toolName: "code_execution" })).toBe(
      "Ran on Anthropic's servers",
    );
    expect(toolSummary({ toolName: "run_python", output: { ms: 1500 } })).toBe(
      "Ran Python in 1.5s",
    );
  });
});

describe("createdFileIds", () => {
  it("reads file ids out of an Anthropic code execution result", () => {
    expect(
      createdFileIds("code_execution", {
        type: "code_execution_result",
        stdout: "Saved deck.pptx",
        content: [
          { type: "code_execution_output", file_id: "file_abc12345" },
          { type: "code_execution_output", file_id: "file_def67890" },
        ],
      }),
    ).toEqual(["file_abc12345", "file_def67890"]);
  });

  it("reads them from the encrypted and bash result shapes too", () => {
    // code_execution_20260120 can return encrypted_code_execution_result or
    // bash_code_execution_result; both still carry content[].file_id.
    for (const type of [
      "encrypted_code_execution_result",
      "bash_code_execution_result",
    ]) {
      expect(
        createdFileIds("code_execution", {
          type,
          content: [{ file_id: "file_abc12345" }],
        }),
      ).toEqual(["file_abc12345"]);
    }
  });

  it("returns nothing for runs that made no files or other tools", () => {
    expect(
      createdFileIds("code_execution", { stdout: "hi", content: [] }),
    ).toEqual([]);
    expect(createdFileIds("code_execution", undefined)).toEqual([]);
    expect(
      createdFileIds("run_python", { content: [{ file_id: "file_abc12345" }] }),
    ).toEqual([]);
  });

  it("drops ids the download route would refuse", () => {
    expect(
      createdFileIds("code_execution", {
        content: [{ file_id: "../../secret" }, { file_id: 7 }, { file_id: "" }],
      }),
    ).toEqual([]);
  });
});

describe("openaiContainerId", () => {
  it("returns the container id only for the interpreter tool", () => {
    expect(
      openaiContainerId("code_interpreter", { containerId: "cntr_abc12345" }),
    ).toBe("cntr_abc12345");
    expect(openaiContainerId("code_interpreter", { containerId: "nope" })).toBeNull();
    expect(
      openaiContainerId("code_execution", { containerId: "cntr_abc12345" }),
    ).toBeNull();
  });
});

describe("hostOf", () => {
  it("names the host without the www prefix", () => {
    expect(hostOf("https://www.example.com/a/b")).toBe("example.com");
    expect(hostOf("not a url")).toBe("page");
  });
});
