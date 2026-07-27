"use client";

import { useState } from "react";
import { ModelChooser } from "@/components/ModelChooser";
import { DEFAULT_TRIALS, MAX_TRIALS } from "@/lib/comparison/config";
import type { ModelOption } from "@/lib/config/models";

export type SetupMode = "anonymous" | "pick";

export function ComparisonSetup({
  models,
  onStart,
}: {
  models: ModelOption[];
  onStart: (config: {
    mode: SetupMode;
    trials: number;
    leftPick: string;
    rightPick: string;
  }) => void;
}) {
  const [mode, setMode] = useState<SetupMode>("anonymous");
  const [trials, setTrials] = useState(DEFAULT_TRIALS);
  const [leftPick, setLeftPick] = useState(models[0].id);
  const [rightPick, setRightPick] = useState(models[1].id);

  const samePick = mode === "pick" && leftPick === rightPick;

  function start(event: React.FormEvent) {
    event.preventDefault();
    if (samePick) return;
    onStart({ mode, trials, leftPick, rightPick });
  }

  return (
    <form onSubmit={start} className="flex flex-col gap-6">
      <fieldset>
        <legend className="text-sm font-bold">
          How should the two models be chosen?
        </legend>
        <div className="mt-2 flex flex-col gap-2">
          <label className="flex items-start gap-2">
            <input
              type="radio"
              name="comparison-mode"
              checked={mode === "anonymous"}
              onChange={() => setMode("anonymous")}
              className="mt-1.5"
            />
            <span>
              <strong>Surprise me (blind)</strong>
              <span className="block text-sm text-dark-tan">
                Two models are chosen at random and stay hidden until the end.
              </span>
            </span>
          </label>
          <label className="flex items-start gap-2">
            <input
              type="radio"
              name="comparison-mode"
              checked={mode === "pick"}
              onChange={() => setMode("pick")}
              className="mt-1.5"
            />
            <span>
              <strong>Pick the two models</strong>
              <span className="block text-sm text-dark-tan">
                Choose both models yourself. Their answers are still shown left
                and right without labels.
              </span>
            </span>
          </label>
        </div>
      </fieldset>

      {mode === "pick" ? (
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <p className="text-sm font-bold">First model</p>
            <ModelChooser options={models} value={leftPick} onChange={setLeftPick} />
          </div>
          <div>
            <p className="text-sm font-bold">Second model</p>
            <ModelChooser
              options={models}
              value={rightPick}
              onChange={setRightPick}
            />
          </div>
        </div>
      ) : null}

      {samePick ? (
        <p role="alert" className="text-miami-red">
          Choose two different models.
        </p>
      ) : null}

      <div>
        <label htmlFor="comparison-trials" className="text-sm font-bold">
          How many questions? (1 to {MAX_TRIALS})
        </label>
        <input
          id="comparison-trials"
          type="number"
          min={1}
          max={MAX_TRIALS}
          value={trials}
          onChange={(e) =>
            setTrials(Math.max(1, Math.min(MAX_TRIALS, Number(e.target.value) || 1)))
          }
          className="mt-1 block w-24 rounded-card border border-medium-tan bg-paper p-2"
        />
      </div>

      <button
        type="submit"
        disabled={samePick}
        className="self-start rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:cursor-not-allowed disabled:bg-medium-gray"
      >
        Start comparing
      </button>
    </form>
  );
}
