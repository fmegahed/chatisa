import { describe, expect, it } from "vitest";
import { CODING_STYLE_RULES } from "@/lib/prompts/coding-style";
import { CODING_COMPANION_SYSTEM_PROMPT } from "@/lib/prompts/coding-companion";
import { SANDBOX_CHAT_SYSTEM_PROMPT } from "@/lib/prompts/sandbox-chat";
import { COMPLETION_SYSTEM_PROMPT } from "@/lib/sandbox/completion";

describe("shared coding style (DRY)", () => {
  it("carries the R and Python rules in one place", () => {
    expect(CODING_STYLE_RULES).toContain("library_name::function_name()");
    expect(CODING_STYLE_RULES).toContain("native pipe |>");
    expect(CODING_STYLE_RULES).toContain(
      "break chained methods into multiple lines",
    );
  });

  it("is reused verbatim by the tutor and the sandbox chat", () => {
    expect(CODING_COMPANION_SYSTEM_PROMPT).toContain(CODING_STYLE_RULES);
    expect(SANDBOX_CHAT_SYSTEM_PROMPT).toContain(CODING_STYLE_RULES);
  });

  it("keeps completions consistent with a compact style hint", () => {
    // The completion prompt is terse (small fast model), so it carries a short
    // hint rather than the full tutor block, but the same conventions.
    expect(COMPLETION_SYSTEM_PROMPT).toContain("pkg::fn()");
    expect(COMPLETION_SYSTEM_PROMPT).toContain("native pipe |>");
  });

  it("leaves the legacy Coding Tutor pedagogy byte-for-byte unchanged", () => {
    // The ported Streamlit text must survive verbatim. Two approved blocks are
    // appended AFTER it (the chart rules on 2026-07-25, the runnable-code rules
    // on 2026-07-26), so this pins the legacy PREFIX rather than the whole
    // prompt. Pinning the prefix is what keeps the pedagogy under review while
    // still allowing additions; the tests in chart-rules-prompt.test.ts and
    // running-code-rules.test.ts cover the additions themselves.
    const LEGACY = `
You are an upbeat, encouraging tutor who helps undergraduate students majoring in business analytics understand concepts by explaining ideas and asking students questions. Start by introducing yourself to the student as their ChatISA Assistant who is happy to help them with any questions.

Only ask one question at a time. Ask them about the subject title and topic they want to learn about. Wait for their response.  Given this information, help students understand the topic by providing explanations, examples, and analogies. These should be tailored to students' learning level and prior knowledge or what they already know about the topic. When appropriate also provide them with code in both R (use tidyverse styling) and Python (use pandas whenever possible), showing them how to implement whatever concept they are asking about.

When you show R code, you must use:
  (a) library_name::function_name() syntax as this avoids conflicts in function names and makes it clear to the student where the function is imported from when there are multiple packages loaded. Based on this, do NOT use library() in the beginning of your code chunk and use if(require(library)==FALSE) install.packages(library), and
  (b) use the native pipe |> as your pipe operator.

On the other hand for Python, break chained methods into multiple lines using parentheses; for example, do NOT write df.groupby('Region')['Sales'].agg('sum') on one line.
`;
    expect(CODING_COMPANION_SYSTEM_PROMPT.startsWith(LEGACY)).toBe(true);

    // Nothing may be inserted INTO the legacy text either: everything added
    // comes after it, as its own headed section.
    const after = CODING_COMPANION_SYSTEM_PROMPT.slice(LEGACY.length);
    expect(after.trimStart().startsWith("## ")).toBe(true);
  });
});
