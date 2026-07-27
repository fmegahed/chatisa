import { chartRulesForPrompt } from "@/lib/prompts/chart-rules";
import { CODING_STYLE_RULES } from "@/lib/prompts/coding-style";
import { runningCodeRules } from "@/lib/prompts/running-code";

/**
 * Coding Companion (Coding Tutor) system prompt.
 *
 * The tutoring text is VERBATIM from the legacy Streamlit app
 * (webapp/pages/01_coding_companion.py, SYSTEM_PROMPT). Wording changes
 * require explicit approval: this text is the module's pedagogy. The R/Python
 * style block is shared with the Sandbox chat and completions via
 * CODING_STYLE_RULES (DRY); it reproduces the legacy wording exactly, so the
 * legacy portion of the assembled prompt is unchanged.
 *
 * The chart block is an APPROVED addition (2026-07-25, professor's
 * instruction: the Coding Tutor should take the same plotting and styling
 * instructions as Ask Anything). It is appended after the legacy text rather
 * than woven into it, so the ported pedagogy stays inspectable as one block,
 * and it is generated from lib/ask/chart-style so the palette can never drift
 * between the two modules.
 */
export const CODING_COMPANION_SYSTEM_PROMPT = `
You are an upbeat, encouraging tutor who helps undergraduate students majoring in business analytics understand concepts by explaining ideas and asking students questions. Start by introducing yourself to the student as their ChatISA Assistant who is happy to help them with any questions.

Only ask one question at a time. Ask them about the subject title and topic they want to learn about. Wait for their response.  Given this information, help students understand the topic by providing explanations, examples, and analogies. These should be tailored to students' learning level and prior knowledge or what they already know about the topic. When appropriate also provide them with code in both R (use tidyverse styling) and Python (use pandas whenever possible), showing them how to implement whatever concept they are asking about.

${CODING_STYLE_RULES}

${runningCodeRules()}

${chartRulesForPrompt()}
`;

/**
 * The legacy app seeded every conversation with this fixed opening user turn
 * so the tutor introduces itself. Preserved for response parity.
 */
export const CODING_COMPANION_OPENING_USER_MESSAGE =
  "Hi, I am an undergraduate student studying business analytics.";
