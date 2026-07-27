/**
 * Pure logic for the AI Comparison module: choosing the pair, deciding which
 * model sits on which side per trial, mapping a vote back to a model, and
 * tallying the result. No React and no app imports, so it is unit-testable in
 * isolation and carries the parts of the module that must be provably correct.
 */

/**
 * mulberry32: a small, well-distributed 32-bit PRNG. Deterministic given its
 * seed, which is the whole point here: an anonymous pairing seeded from the
 * clock must be reproducible within a session and testable. It is not
 * cryptographic and does not need to be, since it only shuffles a public list
 * of model ids.
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The two model ids under comparison, in fixed slot order [slot0, slot1]. */
export type ComparisonPair = readonly [string, string];

export class NotEnoughModelsError extends Error {
  constructor() {
    super("At least two different models are needed to run a comparison.");
    this.name = "NotEnoughModelsError";
  }
}

/**
 * Two distinct model ids drawn from `available` using `seed`. The same seed and
 * list always yield the same pair, which is what makes the anonymous
 * time-based seed reproducible and testable. Duplicates in the input are
 * collapsed first, so a short list padded with repeats cannot pick the same
 * model twice.
 */
export function pickPair(available: string[], seed: number): ComparisonPair {
  const unique = [...new Set(available)];
  if (unique.length < 2) throw new NotEnoughModelsError();
  const rand = mulberry32(seed);
  const first = Math.floor(rand() * unique.length);
  // Draw the second from the remaining n-1 positions, then map back around the
  // hole at `first`, so the two indices are always distinct.
  let second = Math.floor(rand() * (unique.length - 1));
  if (second >= first) second += 1;
  return [unique[first], unique[second]] as const;
}

/** A stable per-seed coin flip, deciding which slot starts on the left. */
function seedParity(seed: number): 0 | 1 {
  return mulberry32(seed)() < 0.5 ? 0 : 1;
}

/**
 * Which slot (0 or 1) is shown on the LEFT for a given trial. The starting side
 * is a seed-derived coin flip, then the two models swap sides every trial. This
 * blinds the student (a consistent left position never reveals identity because
 * the starting side flips per session) and balances position bias (each slot
 * spends an equal number of trials on the left for an even trial count).
 */
export function leftSlotForTrial(seed: number, trialIndex: number): 0 | 1 {
  return (((seedParity(seed) + trialIndex) % 2) as 0 | 1);
}

/**
 * Maps a vote for a screen side back to the slot that actually received it,
 * given which slot was on the left for that trial.
 */
export function resolveVote(side: "left" | "right", leftSlot: 0 | 1): 0 | 1 {
  const rightSlot: 0 | 1 = leftSlot === 0 ? 1 : 0;
  return side === "left" ? leftSlot : rightSlot;
}

export interface Outcome {
  votesSlot0: number;
  votesSlot1: number;
  /** Winning slot, or null for a tie (including zero to zero). */
  winner: 0 | 1 | null;
}

/** Tallies a list of per-slot votes into a final outcome. */
export function decideOutcome(slotVotes: (0 | 1)[]): Outcome {
  const votesSlot0 = slotVotes.filter((s) => s === 0).length;
  const votesSlot1 = slotVotes.filter((s) => s === 1).length;
  const winner =
    votesSlot0 === votesSlot1 ? null : votesSlot0 > votesSlot1 ? 0 : 1;
  return { votesSlot0, votesSlot1, winner };
}
