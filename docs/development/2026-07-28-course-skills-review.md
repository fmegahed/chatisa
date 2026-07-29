# Course-to-skill mapping — instructor review (2026-07-28)

Source: `web/lib/scout/course-skills.ts`, authored by Claude Fable 5 from the
bulletin descriptions (snapshot 2026-07-28). Review request: scan each
course's links; fix anything that misstates what students actually do. Edits
go directly in `course-skills.ts`; the integrity tests
(`tests/unit/scout-taxonomy.test.ts`) enforce that every skill id exists,
every mapped course keeps at least one anchor with evidence, and no pair is
duplicated.

Levels: **anchor** = graded deliverables demonstrate it · **applied** = used
repeatedly as a working tool · **exposure** = introduced.

## Flagged for your judgment (tool inferences beyond the bulletin text)

The bulletin rarely names software. These links assert a tool anyway, based
on how FSB teaches the course; strike any that are wrong:

| Course | Link | Basis |
|---|---|---|
| ISA 242 | python (applied) | FSB programming sequence is Python |
| ISA 381 | python (anchor) | continuation of 242 |
| ISA 419 | python (applied) | "extensive programming with real datasets" |
| ISA 630 | python, deep_learning_frameworks (applied) | ML tooling norm |
| ISA 444 | r (applied) | course is taught in R |
| ISA 616 | r (exposure) | reproducible reporting workflow |
| ISA 401 | tableau, power_bi (applied) | BI tooling norm |
| ISA 632 | spark (applied) | "in-memory cluster computing" |
| ISA 414 | spark, nosql (applied) | big-data tooling norm |

## Full mapping

Rendered from the source of truth; each row is course → skill (level),
with evidence on anchors. See `course-skills.ts` for the complete list —
this file intentionally does not duplicate all ~190 rows; the sections
below call out only the judgment calls.

### Judgment calls beyond tools

- **ISA 225**: forecasting, data_mining, classification kept at *exposure*
  (survey coverage), regression at *applied*. If projects go deeper, promote.
- **ISA 235**: generative_ai at *exposure* ("problem-solving using ... AI").
  If the AI unit has graded deliverables, promote to applied.
- **ISA 365**: ab_testing at *exposure*; ISA 633 owns it at anchor.
- **ISA 491**: deep_learning at *exposure* ("neural nets" appears once).
- **ISA 495/496/650**: professional skills (consulting, stakeholder
  management, presentation) carry the anchors; that is deliberate — these
  courses evidence professional competencies more than methods.
- **ISA 628/629/641**: kept lightweight (1.5–2 credits scale their weight
  down automatically); anchors chosen to reflect the single strongest claim.
- Cross-listed pairs (401/501, 414/514, 444/544, 491/591) map once under
  the undergraduate code; alt codes resolve automatically, and 5xx students
  get identical links.

## What happens after your review

Match strength = noisy-OR across your courses (anchor 1.0, applied 0.6,
exposure 0.25, scaled by credits/3), so over-generous *exposure* links have
limited effect, but a wrong *anchor* overstates a student meaningfully.
Anchor evidence phrases also seed resume bullets, so they must be claims a
student in that course can defend in an interview.
