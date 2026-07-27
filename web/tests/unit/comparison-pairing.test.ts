import { describe, expect, it } from "vitest";
import {
  pickPair,
  leftSlotForTrial,
  resolveVote,
  decideOutcome,
  NotEnoughModelsError,
} from "@/lib/comparison/pairing";

const MODELS = ["a", "b", "c", "d", "e"];

describe("pickPair", () => {
  it("is deterministic for a given seed and list", () => {
    expect(pickPair(MODELS, 12345)).toEqual(pickPair(MODELS, 12345));
  });

  it("returns two distinct ids, both from the list", () => {
    for (let seed = 0; seed < 200; seed++) {
      const [x, y] = pickPair(MODELS, seed);
      expect(x).not.toBe(y);
      expect(MODELS).toContain(x);
      expect(MODELS).toContain(y);
    }
  });

  it("can reach every model across seeds, so no id is unpickable", () => {
    const seen = new Set<string>();
    for (let seed = 0; seed < 500; seed++) {
      const [x, y] = pickPair(MODELS, seed);
      seen.add(x);
      seen.add(y);
    }
    expect(seen).toEqual(new Set(MODELS));
  });

  it("ignores duplicate ids in the input", () => {
    const [x, y] = pickPair(["a", "a", "b"], 7);
    expect(x).not.toBe(y);
  });

  it("throws when fewer than two distinct models are available", () => {
    expect(() => pickPair(["a"], 1)).toThrow(NotEnoughModelsError);
    expect(() => pickPair(["a", "a"], 1)).toThrow(NotEnoughModelsError);
  });
});

describe("leftSlotForTrial", () => {
  it("is deterministic for a given seed and trial", () => {
    expect(leftSlotForTrial(999, 3)).toBe(leftSlotForTrial(999, 3));
  });

  it("alternates sides on consecutive trials", () => {
    for (const seed of [0, 1, 42, 100000]) {
      expect(leftSlotForTrial(seed, 0)).not.toBe(leftSlotForTrial(seed, 1));
      expect(leftSlotForTrial(seed, 1)).not.toBe(leftSlotForTrial(seed, 2));
    }
  });

  it("gives each slot the left side equally over an even trial count", () => {
    const seed = 555;
    const lefts = [0, 1, 2, 3].map((t) => leftSlotForTrial(seed, t));
    expect(lefts.filter((s) => s === 0).length).toBe(2);
    expect(lefts.filter((s) => s === 1).length).toBe(2);
  });
});

describe("resolveVote", () => {
  it("maps a left vote to the slot on the left", () => {
    expect(resolveVote("left", 0)).toBe(0);
    expect(resolveVote("left", 1)).toBe(1);
  });
  it("maps a right vote to the other slot", () => {
    expect(resolveVote("right", 0)).toBe(1);
    expect(resolveVote("right", 1)).toBe(0);
  });
});

describe("decideOutcome", () => {
  it("names the slot with more votes", () => {
    expect(decideOutcome([0, 0, 1]).winner).toBe(0);
    expect(decideOutcome([1, 1, 0]).winner).toBe(1);
  });
  it("returns a tie for equal votes, including zero to zero", () => {
    expect(decideOutcome([0, 1]).winner).toBeNull();
    expect(decideOutcome([]).winner).toBeNull();
  });
  it("counts votes per slot", () => {
    const out = decideOutcome([0, 0, 1]);
    expect(out.votesSlot0).toBe(2);
    expect(out.votesSlot1).toBe(1);
  });
});
