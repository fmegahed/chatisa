"use client";

import { Markdown } from "@/components/chat/Markdown";

/**
 * One answer, kept blind. The heading names the side only ("left" or "right"),
 * never the model, so a student cannot learn which model they are reading until
 * the report reveals it.
 */
export function ComparisonPane({
  side,
  text,
  status,
  error,
}: {
  side: "left" | "right";
  text: string;
  status: string;
  error?: string;
}) {
  const heading = side === "left" ? "Answer on the left" : "Answer on the right";
  const headingId = `pane-${side}`;
  const streaming = status === "submitted" || status === "streaming";
  return (
    <article
      aria-labelledby={headingId}
      className="rounded-card border border-medium-tan bg-paper p-4"
    >
      <h2 id={headingId} className="mb-1 text-sm font-bold text-dark-tan">
        {heading}
      </h2>
      {error ? (
        <p role="alert" className="text-miami-red">
          This model could not answer. {error}
        </p>
      ) : text ? (
        <Markdown>{text}</Markdown>
      ) : (
        <p className="text-dark-tan">{streaming ? "Thinking." : "Waiting."}</p>
      )}
    </article>
  );
}
