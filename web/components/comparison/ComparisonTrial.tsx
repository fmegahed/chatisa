"use client";

import { useMemo, useState } from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { ComparisonPane } from "./ComparisonPane";
import {
  leftSlotForTrial,
  resolveVote,
  type ComparisonPair,
} from "@/lib/comparison/pairing";

/** Concatenated text of the most recent assistant message. */
function assistantText(messages: UIMessage[]): string {
  const last = [...messages].reverse().find((m) => m.role === "assistant");
  if (!last) return "";
  return last.parts
    .filter((p) => p.type === "text")
    .map((p) => ("text" in p ? p.text : ""))
    .join("");
}

export function ComparisonTrial({
  pair,
  seed,
  trialIndex,
  trialCount,
  onVote,
}: {
  pair: ComparisonPair;
  seed: number;
  trialIndex: number;
  trialCount: number;
  onVote: (slot: 0 | 1) => void;
}) {
  const [input, setInput] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const leftSlot = useMemo(
    () => leftSlotForTrial(seed, trialIndex),
    [seed, trialIndex],
  );
  const rightSlot: 0 | 1 = leftSlot === 0 ? 1 : 0;
  const leftModelId = pair[leftSlot];
  const rightModelId = pair[rightSlot];

  // Two transports, two conversations, both aimed at the shared chat route.
  const [leftTransport] = useState(
    () => new DefaultChatTransport({ api: "/api/chat" }),
  );
  const [rightTransport] = useState(
    () => new DefaultChatTransport({ api: "/api/chat" }),
  );
  const left = useChat({ transport: leftTransport });
  const right = useChat({ transport: rightTransport });

  const busy =
    left.status === "submitted" ||
    left.status === "streaming" ||
    right.status === "submitted" ||
    right.status === "streaming";

  const leftText = assistantText(left.messages);
  const rightText = assistantText(right.messages);
  const bothReady =
    submitted &&
    !busy &&
    left.status === "ready" &&
    right.status === "ready" &&
    leftText.length > 0 &&
    rightText.length > 0;

  function ask(event: React.FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || busy || submitted) return;
    setSubmitted(true);
    left.sendMessage(
      { text },
      { body: { module: "ai_comparisons", modelId: leftModelId } },
    );
    right.sendMessage(
      { text },
      { body: { module: "ai_comparisons", modelId: rightModelId } },
    );
  }

  return (
    <section
      aria-label={`Trial ${trialIndex + 1} of ${trialCount}`}
      className="flex flex-col gap-4"
    >
      <p className="ribbon">
        Trial {trialIndex + 1} of {trialCount}
      </p>

      {!submitted ? (
        <form onSubmit={ask} className="flex flex-col gap-2">
          <label htmlFor="comparison-prompt" className="text-sm font-bold">
            Your question for both models
          </label>
          <textarea
            id="comparison-prompt"
            rows={3}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="w-full rounded-card border border-medium-tan bg-paper p-3"
          />
          <button
            type="submit"
            disabled={input.trim().length === 0}
            className="self-start rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:cursor-not-allowed disabled:bg-medium-gray"
          >
            Ask both models
          </button>
        </form>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-2">
            <ComparisonPane
              side="left"
              text={leftText}
              status={left.status}
              error={left.error?.message}
            />
            <ComparisonPane
              side="right"
              text={rightText}
              status={right.status}
              error={right.error?.message}
            />
          </div>

          <p role="status" className="text-sm text-dark-tan">
            {busy
              ? "Both models are answering."
              : bothReady
                ? "Both answers are ready. Choose the one you prefer."
                : ""}
          </p>

          <fieldset disabled={!bothReady} className="flex flex-col gap-2">
            <legend className="text-sm font-bold">
              Which answer do you prefer?
            </legend>
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => onVote(resolveVote("left", leftSlot))}
                className="rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red disabled:cursor-not-allowed disabled:text-medium-gray"
              >
                Prefer the left answer
              </button>
              <button
                type="button"
                onClick={() => onVote(resolveVote("right", leftSlot))}
                className="rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red disabled:cursor-not-allowed disabled:text-medium-gray"
              >
                Prefer the right answer
              </button>
            </div>
          </fieldset>
        </>
      )}
    </section>
  );
}
