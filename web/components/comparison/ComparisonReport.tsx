"use client";

import { getModelDisplayName } from "@/lib/config/models";
import type { ComparisonPair, Outcome } from "@/lib/comparison/pairing";

export function ComparisonReport({
  pair,
  outcome,
  onRestart,
}: {
  pair: ComparisonPair;
  outcome: Outcome;
  onRestart: () => void;
}) {
  const names = [getModelDisplayName(pair[0]), getModelDisplayName(pair[1])];
  const votes = [outcome.votesSlot0, outcome.votesSlot1];
  const tie = outcome.winner === null;
  const heading = tie ? "It is a tie" : `${names[outcome.winner as 0 | 1]} won`;

  return (
    <section aria-label="Comparison result" className="flex flex-col gap-4">
      <p className="ribbon">Result</p>
      <h2 className="text-2xl">{heading}</h2>

      <ul className="flex flex-col gap-3">
        {[0, 1].map((slot) => {
          const isWinner = outcome.winner === slot;
          return (
            <li
              key={slot}
              className={
                isWinner
                  ? "rounded-card border-2 border-miami-red bg-light-tan p-4"
                  : "rounded-card border border-medium-tan bg-paper p-4"
              }
            >
              <p className="font-bold">
                {names[slot]}
                {isWinner ? (
                  <span className="ml-2 text-miami-red">Winner</span>
                ) : null}
              </p>
              <p className="text-sm text-dark-tan">
                {votes[slot]} {votes[slot] === 1 ? "vote" : "votes"}
              </p>
            </li>
          );
        })}
      </ul>

      {tie ? (
        <p>Both models received the same number of votes.</p>
      ) : null}

      <button
        type="button"
        onClick={onRestart}
        className="self-start rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red"
      >
        Run another comparison
      </button>
    </section>
  );
}
