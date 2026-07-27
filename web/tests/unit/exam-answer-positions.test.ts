import { describe, expect, it } from "vitest";
import {
  balancedPositions,
  mustStayLast,
  referencesOtherOptions,
  repositionAnswers,
  type Rng,
} from "@/lib/exam/answer-positions";

/** Deterministic RNG so these assert behaviour, not luck. */
function seeded(seed: number): Rng {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
}

function mcq(correctIndex: number, options?: string[]) {
  return {
    type: "multiple_choice",
    options: options ?? ["Alpha", "Bravo", "Charlie", "Delta"],
    correctIndex,
  };
}

/** Position of the correct answer, read back from the rewritten question. */
function positionsOf(questions: ReturnType<typeof mcq>[]): number[] {
  return questions.map((q) => q.correctIndex as number);
}

describe("balanced position assignment", () => {
  it("spreads positions as evenly as the counts allow", () => {
    for (const count of [4, 5, 8, 10, 13]) {
      const positions = balancedPositions(count, 4, seeded(count));
      expect(positions).toHaveLength(count);
      const tally = [0, 0, 0, 0];
      for (const p of positions) tally[p] += 1;
      // Every position used, and no position used more than one extra time.
      expect(Math.max(...tally) - Math.min(...tally)).toBeLessThanOrEqual(1);
    }
  });

  it("does not produce the same order every time", () => {
    const a = balancedPositions(10, 4, seeded(1)).join("");
    const b = balancedPositions(10, 4, seeded(99)).join("");
    expect(a).not.toBe(b);
  });
});

describe("repositioning correct answers", () => {
  it("keeps the correct option correct after moving it", () => {
    // The failure that matters most: a shuffle that moves options without
    // moving correctIndex silently marks right answers wrong.
    const questions = [mcq(0), mcq(0), mcq(0), mcq(0), mcq(0)];
    const before = questions.map((q) => q.options[q.correctIndex]);

    const { questions: after } = repositionAnswers(questions, seeded(7));

    after.forEach((q, i) => {
      expect(q.options![q.correctIndex!]).toBe(before[i]);
    });
  });

  it("preserves the full set of options, losing and inventing none", () => {
    const questions = [mcq(1), mcq(2), mcq(3)];
    const { questions: after } = repositionAnswers(questions, seeded(3));
    after.forEach((q, i) => {
      expect([...q.options!].sort()).toEqual([...questions[i].options].sort());
    });
  });

  it("breaks up the clustering the user reported", () => {
    // Every correct answer generated in position A, which is exactly the
    // pattern reported: "if the first correct answer is A it tends to be A for
    // 3 of the remaining 4 questions".
    const questions = Array.from({ length: 12 }, () => mcq(0));
    const { questions: after } = repositionAnswers(questions, seeded(42));

    const tally = [0, 0, 0, 0];
    for (const p of positionsOf(after)) tally[p] += 1;
    expect(Math.max(...tally) - Math.min(...tally)).toBeLessThanOrEqual(1);
    // And specifically, position A is no longer dominant.
    expect(tally[0]).toBeLessThanOrEqual(3);
  });

  it("stays balanced across many different exams, not just a lucky seed", () => {
    // Guards against a change that makes each question independently random,
    // which clusters by chance far more often than people expect.
    for (let seed = 1; seed <= 40; seed += 1) {
      const questions = Array.from({ length: 5 }, () => mcq(0));
      const { questions: after } = repositionAnswers(questions, seeded(seed));
      const tally = [0, 0, 0, 0];
      for (const p of positionsOf(after)) tally[p] += 1;
      // With 5 questions and 4 slots the best possible is one slot twice.
      expect(Math.max(...tally), `seed ${seed}`).toBeLessThanOrEqual(2);
    }
  });

  it("leaves non multiple-choice questions untouched", () => {
    const written = { type: "short_answer", options: null, correctIndex: null };
    const { questions: after } = repositionAnswers([written], seeded(1));
    expect(after[0]).toEqual(written);
  });
});

describe("options that must not move", () => {
  it("recognises trailing options", () => {
    expect(mustStayLast("All of the above")).toBe(true);
    expect(mustStayLast("none of the above")).toBe(true);
    expect(mustStayLast("A normalized schema")).toBe(false);
  });

  it("keeps 'all of the above' last while still varying the others", () => {
    const options = ["Alpha", "Bravo", "Charlie", "All of the above"];
    const questions = [{ type: "multiple_choice", options, correctIndex: 0 }];
    const { questions: after } = repositionAnswers(questions, seeded(11));

    expect(after[0].options![3]).toBe("All of the above");
    expect(after[0].options![after[0].correctIndex!]).toBe("Alpha");
  });

  it("keeps 'all of the above' last even when it is the answer", () => {
    const options = ["Alpha", "Bravo", "Charlie", "All of the above"];
    const questions = [{ type: "multiple_choice", options, correctIndex: 3 }];
    const { questions: after } = repositionAnswers(questions, seeded(5));

    expect(after[0].options![3]).toBe("All of the above");
    expect(after[0].correctIndex).toBe(3);
  });

  it("does not scramble options that name other options", () => {
    // "Both A and C" stops meaning anything once the letters move. Such a
    // question is poorly formed already; shuffling would turn it from weak
    // into wrong.
    expect(referencesOtherOptions(["Alpha", "Bravo", "Both A and B", "Delta"]))
      .toBe(true);
    expect(referencesOtherOptions(["Alpha", "Bravo", "Charlie", "Delta"]))
      .toBe(false);

    const options = ["Alpha", "Bravo", "Both A and B", "Delta"];
    const questions = [{ type: "multiple_choice", options, correctIndex: 2 }];
    const { questions: after, skipped } = repositionAnswers(
      questions,
      seeded(2),
    );

    expect(skipped).toBe(1);
    expect(after[0].options).toEqual(options);
    expect(after[0].correctIndex).toBe(2);
  });

  it("does not mistake an ordinary option for a reference", () => {
    // "A and B testing" or a stray capital letter must not disable shuffling.
    expect(referencesOtherOptions(["Randomised trials", "Observational data"]))
      .toBe(false);
  });
});
