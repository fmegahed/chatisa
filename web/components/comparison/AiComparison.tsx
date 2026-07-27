"use client";

import { useState } from "react";
import { ComparisonSetup, type SetupMode } from "./ComparisonSetup";
import { ComparisonTrial } from "./ComparisonTrial";
import { ComparisonReport } from "./ComparisonReport";
import { pickPair, decideOutcome, type ComparisonPair } from "@/lib/comparison/pairing";
import type { ModelOption } from "@/lib/config/models";

type Phase =
  | { name: "setup" }
  | {
      name: "trials";
      pair: ComparisonPair;
      seed: number;
      trials: number;
      index: number;
      votes: (0 | 1)[];
    }
  | { name: "report"; pair: ComparisonPair; votes: (0 | 1)[] };

export function AiComparison({ models }: { models: ModelOption[] }) {
  const [phase, setPhase] = useState<Phase>({ name: "setup" });
  const ids = models.map((m) => m.id);

  function start(config: {
    mode: SetupMode;
    trials: number;
    leftPick: string;
    rightPick: string;
  }) {
    const seed = Date.now();
    const pair: ComparisonPair =
      config.mode === "anonymous"
        ? pickPair(ids, seed)
        : [config.leftPick, config.rightPick];
    setPhase({
      name: "trials",
      pair,
      seed,
      trials: config.trials,
      index: 0,
      votes: [],
    });
  }

  function vote(slot: 0 | 1) {
    setPhase((prev) => {
      if (prev.name !== "trials") return prev;
      const votes = [...prev.votes, slot];
      if (prev.index + 1 >= prev.trials) {
        return { name: "report", pair: prev.pair, votes };
      }
      return { ...prev, index: prev.index + 1, votes };
    });
  }

  if (phase.name === "setup") {
    return <ComparisonSetup models={models} onStart={start} />;
  }

  if (phase.name === "trials") {
    return (
      <ComparisonTrial
        // A fresh key per trial resets both useChat instances between prompts.
        key={phase.index}
        pair={phase.pair}
        seed={phase.seed}
        trialIndex={phase.index}
        trialCount={phase.trials}
        onVote={vote}
      />
    );
  }

  return (
    <ComparisonReport
      pair={phase.pair}
      outcome={decideOutcome(phase.votes)}
      onRestart={() => setPhase({ name: "setup" })}
    />
  );
}
