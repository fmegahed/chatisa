import { describe, expect, it } from "vitest";
import {
  MIAMI_HIGHLIGHT,
  MIAMI_LINK_TEAL,
  MIAMI_SERIES,
  PIE_POLICY,
  paletteRules,
  portableRules,
} from "@/lib/ask/chart-style";
import { chartRulesForPrompt } from "@/lib/prompts/chart-rules";
import { CODING_COMPANION_SYSTEM_PROMPT } from "@/lib/prompts/coding-companion";
import { SANDBOX_CHAT_SYSTEM_PROMPT } from "@/lib/prompts/sandbox-chat";
import { ASK_ANYTHING_SYSTEM_PROMPT } from "@/lib/prompts/ask-anything";

/**
 * The Coding Tutor and the Sandbox chat were given Ask Anything's plotting and
 * styling instructions on 2026-07-25 (professor's instruction). The risk that
 * creates is drift: three modules stating a palette, two of them by hand. These
 * tests exist so a palette edit in lib/ask/chart-style either reaches every
 * module or fails the build, and so nobody can "fix" one module's colours in
 * isolation.
 */
describe("chartRulesForPrompt", () => {
  it("is composed from the shared contract, not restated", () => {
    const rules = chartRulesForPrompt();
    // Substring equality, so an edit to either source function flows through.
    expect(rules).toContain(paletteRules());
    expect(rules).toContain(portableRules());
    expect(rules).toContain(PIE_POLICY);
  });

  it("carries every series colour in assignment order", () => {
    const rules = chartRulesForPrompt();
    const positions = MIAMI_SERIES.map((hex) => rules.indexOf(hex));
    for (const at of positions) expect(at).toBeGreaterThan(-1);
    expect([...positions]).toEqual([...positions].sort((a, b) => a - b));
  });

  it("keeps corn yellow and link teal off the series list", () => {
    const rules = chartRulesForPrompt();
    expect(rules).toMatch(new RegExp(`${MIAMI_HIGHLIGHT}[^\\n]*fill only`, "i"));
    expect(rules).toMatch(
      new RegExp(`${MIAMI_LINK_TEAL}[^\\n]*hyperlinks only`, "i"),
    );
  });

  it("reconciles the chart code with this app's R style rules", () => {
    // The shared exemplar in lib/ask/chart-examples opens with library(ggplot2),
    // which the Coding Tutor and Sandbox style block forbids. The prompt block
    // must therefore state the package-qualified form explicitly, or the two
    // rule sets contradict each other inside the same system prompt.
    const rules = chartRulesForPrompt();
    expect(rules).toContain("ggplot2::ggplot()");
    expect(rules).toContain("ggrepel::geom_text_repel()");
    expect(rules).toMatch(/never library\(ggplot2\)/i);
  });

  it("names the label packages for both languages", () => {
    const rules = chartRulesForPrompt();
    for (const pkg of ["ggrepel", "ggtext", "adjustText", "highlight_text"]) {
      expect(rules).toContain(pkg);
    }
  });
});

describe("modules that plot", () => {
  it("gives the Coding Tutor and the Sandbox chat the same block", () => {
    const rules = chartRulesForPrompt();
    expect(CODING_COMPANION_SYSTEM_PROMPT).toContain(rules);
    expect(SANDBOX_CHAT_SYSTEM_PROMPT).toContain(rules);
  });

  it("agrees with Ask Anything on the palette and the title contract", () => {
    // Ask Anything states the short version inline and fetches the rest through
    // get_miami_style, so the texts differ by design. The load-bearing claims
    // must not: same hexes, same escalation, same pie policy, same title rule.
    for (const hex of MIAMI_SERIES) {
      expect(ASK_ANYTHING_SYSTEM_PROMPT).toContain(hex);
      expect(CODING_COMPANION_SYSTEM_PROMPT).toContain(hex);
    }
    for (const prompt of [
      ASK_ANYTHING_SYSTEM_PROMPT,
      CODING_COMPANION_SYSTEM_PROMPT,
      SANDBOX_CHAT_SYSTEM_PROMPT,
    ]) {
      expect(prompt).toMatch(/Dark2/);
      expect(prompt).toMatch(/states the finding/i);
      expect(prompt).toMatch(/suboptimal/i);
      expect(prompt).toMatch(/never a second y axis|never a second y axis/i);
    }
  });

  it("does not inline the code exemplar into either chat prompt", () => {
    // Deliberate: the exemplar roughly quadruples the block and is sent on
    // every turn, including the many that never mention a chart. If this ever
    // becomes desirable, it belongs behind a tool call, not in the prompt.
    for (const prompt of [
      CODING_COMPANION_SYSTEM_PROMPT,
      SANDBOX_CHAT_SYSTEM_PROMPT,
    ]) {
      expect(prompt).not.toContain("geom_text_repel(aes(label = grade)");
      expect(prompt).not.toContain("import matplotlib");
    }
  });

  it("keeps both prompts inside a sane per-turn budget", () => {
    // These ride on every request in their module, including the many turns
    // that never mention a chart, so the ceiling is deliberate rather than
    // generous. A regression that pastes the code exemplars in (roughly 4,500
    // extra characters) breaks this before it shows up on the invoice.
    //
    // Measured 2026-07-26: tutor 6,686 and sandbox 4,588. The tutor carries the
    // runnable-code block as well, which the sandbox chat does not need because
    // its own prompt already describes its live workspace.
    expect(CODING_COMPANION_SYSTEM_PROMPT.length).toBeLessThan(8_000);
    expect(SANDBOX_CHAT_SYSTEM_PROMPT.length).toBeLessThan(6_000);
  });
});
