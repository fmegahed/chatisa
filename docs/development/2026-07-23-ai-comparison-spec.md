# AI Comparison module — spec (from professor, 2026-07-23)

New module (`ai-comparisons` slug, "AI Comparison"), currently a placeholder. A
blind, side-by-side comparison of two LLMs, voted on by the student.

## Behavior

- **Two models side by side.** Left answer and right answer to the same prompt.
- **Two setup modes:**
  - **Anonymous (default):** a time-based seed picks 2 of the available models at
    random. The student does not know which model is which (blind).
  - **Pick models:** the student chooses the two models to compare.
- **Number of trials n:** the student picks how many prompts to compare by.
  Default n = 1, maximum n = 5.
- **One prompt at a time:** for each of the n trials, the student enters one
  prompt; both models answer; the two answers show side by side (left / right).
- **Vote:** the student clicks whether they prefer the left or the right answer.
- **After the n trials, report:** reveal both models (highlight the winning
  model) and the other model, and the number of votes each received.

## Design notes / decisions to settle in the plan

- **Available models:** reuse the model catalog. There is already an
  `ai_comparisons` page-models key in `lib/config/models.ts`. Anonymous mode
  seeds from that available set.
- **Blind fairness:** in anonymous mode, randomize which model is left vs right
  per trial (or keep consistent) — decide and document. The models stay hidden
  until the final report.
- **Both answers:** both models answer the same prompt (stream both, or generate
  both). Reuse the streaming chat infrastructure; two concurrent calls.
- **Privacy / retention:** consistent with the chat-retention decision, the
  comparison is ephemeral (no server-side persistence of prompts or answers);
  only content-free usage events. The report is computed in-session.
- **Ties:** define tie handling in the report (equal votes).
- **Reuse:** auth guard, model picker (for pick-models mode), streaming + error
  handling, Miami brand tokens, WCAG 2.1 AA, no em dashes.

## Non-goals

- No leaderboard or cross-user aggregation (each session is standalone).
- No persistence of which model won across sessions.

Status: spec captured; implementation plan to be written (grounded in the chat /
model infrastructure) and reviewed before building.
