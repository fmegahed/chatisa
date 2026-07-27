/**
 * Spreading the correct answer across positions.
 *
 * Reported by the user on 2026-07-21: within one exam the correct answer sat in
 * the same slot for most questions, so noticing the first one gave away the
 * rest. This is a well documented behaviour of language models rather than a
 * bug in the prompt: they favour particular positions, and asking them to
 * "vary the answer position" does not reliably fix it. So the position is
 * decided here, after generation, where it is arithmetic rather than a request.
 *
 * Balanced rather than merely random. Shuffling each question independently
 * still clusters by chance often enough to be noticeable: across 5 questions
 * with 4 options, three or more sharing a position happens surprisingly often.
 * Instead every position is dealt out as evenly as the question count allows,
 * then that assignment is shuffled, so counts differ by at most one and no
 * position is ever predictable.
 */

export interface ShufflableQuestion {
  type: string;
  options: string[] | null;
  correctIndex: number | null;
}

/** Injectable so tests are deterministic; defaults to Math.random. */
export type Rng = () => number;

/**
 * Options that must stay where they are.
 *
 * "All of the above" only makes sense last, and an option that names another
 * option ("Both A and C") stops meaning anything once the letters move. The
 * second case is a poorly formed question rather than a shuffling problem, but
 * silently scrambling it would turn a weak question into an incorrect one.
 */
const TRAILING_OPTION = /\b(all|none) of the above\b/i;
const REFERS_TO_OTHER_OPTIONS =
  /\b(both\s+)?\(?[a-d]\)?\s+and\s+\(?[a-d]\)?\b|\boptions?\s+\(?[a-d]\)?/i;

export function mustStayLast(option: string): boolean {
  return TRAILING_OPTION.test(option);
}

export function referencesOtherOptions(options: string[]): boolean {
  return options.some(
    (o) => REFERS_TO_OTHER_OPTIONS.test(o) && !TRAILING_OPTION.test(o),
  );
}

function shuffle<T>(items: T[], rng: Rng): T[] {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

/**
 * Positions to place correct answers in, as evenly spread as the counts allow.
 *
 * For 10 questions and 4 options this yields each of 0-3 either two or three
 * times, in random order, rather than whatever the model happened to prefer.
 */
export function balancedPositions(
  questionCount: number,
  optionCount: number,
  rng: Rng = Math.random,
): number[] {
  if (questionCount <= 0 || optionCount <= 0) return [];
  const positions: number[] = [];
  for (let i = 0; i < questionCount; i += 1) positions.push(i % optionCount);
  return shuffle(positions, rng);
}

export interface RepositionResult<T> {
  questions: T[];
  /** How many were moved, for the record. */
  shuffled: number;
  /** Left alone because their options refer to each other by letter. */
  skipped: number;
}

/**
 * Rewrites each multiple-choice question so its correct answer lands on an
 * assigned position, keeping `correctIndex` in step.
 *
 * Non-MCQ questions pass through untouched.
 */
export function repositionAnswers<T extends ShufflableQuestion>(
  questions: T[],
  rng: Rng = Math.random,
): RepositionResult<T> {
  const eligible: number[] = [];
  for (let i = 0; i < questions.length; i += 1) {
    const q = questions[i];
    if (
      q.type !== "multiple_choice" ||
      !q.options ||
      q.options.length < 2 ||
      q.correctIndex === null ||
      q.correctIndex < 0 ||
      q.correctIndex >= q.options.length ||
      referencesOtherOptions(q.options)
    ) {
      continue;
    }
    eligible.push(i);
  }

  const optionCount = questions[eligible[0]]?.options?.length ?? 4;
  const targets = balancedPositions(eligible.length, optionCount, rng);

  const out = [...questions];
  let shuffled = 0;

  eligible.forEach((questionIndex, nth) => {
    const q = out[questionIndex];
    const options = q.options as string[];
    const correct = options[q.correctIndex as number];

    // Options that only make sense last are pinned there, and the correct
    // answer is placed among the remaining slots.
    const pinned = options.filter((o) => mustStayLast(o));
    const movable = options.filter((o) => !mustStayLast(o));

    if (pinned.includes(correct)) {
      // "All of the above" is the answer: it has to stay last, so the only
      // thing to vary is the order of the distractors above it.
      const reordered = [...shuffle(movable, rng), ...pinned];
      out[questionIndex] = {
        ...q,
        options: reordered,
        correctIndex: reordered.indexOf(correct),
      };
      shuffled += 1;
      return;
    }

    const distractors = shuffle(
      movable.filter((o) => o !== correct),
      rng,
    );
    // Clamp in case pinned options shrink the available slots.
    const target = Math.min(targets[nth] ?? 0, movable.length - 1);

    const reordered: string[] = [];
    let d = 0;
    for (let slot = 0; slot < movable.length; slot += 1) {
      reordered.push(slot === target ? correct : distractors[d++]);
    }
    reordered.push(...pinned);

    out[questionIndex] = {
      ...q,
      options: reordered,
      correctIndex: target,
    };
    shuffled += 1;
  });

  return {
    questions: out,
    shuffled,
    skipped: questions.filter(
      (q) =>
        q.type === "multiple_choice" &&
        q.options !== null &&
        referencesOtherOptions(q.options),
    ).length,
  };
}
